import os, csv, time
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
from roi_extractor import ROIExtractor


def pick_preprocess(backend: str):
    b = backend.lower()
    if b == "mobilenet_v2":
        from tensorflow.keras.applications import mobilenet_v2
        return mobilenet_v2.preprocess_input
    if b in ("resnet50v2", "resnet_v2", "resnet"):
        from tensorflow.keras.applications import resnet_v2
        return resnet_v2.preprocess_input
    raise ValueError(f"Unsupported backend: {backend}")


def guess_backend(model, model_path):
    name = (getattr(model, "name", "") + " " + os.path.basename(model_path)).lower()
    if "mobilenet_v2" in name or "mobilenetv2" in name:
        return "mobilenet_v2"
    if "resnet50v2" in name or "resnet_v2" in name or "resnet" in name:
        return "resnet50v2"
    return "resnet50v2"


class VideoProcessor:
    def __init__(self, args, class_names):
        self.args = args
        self.class_names = class_names

        self.model = tf.keras.models.load_model(args.model, compile=False)
        backend = args.backend if args.backend != "auto" else guess_backend(self.model, args.model)
        self.backend = backend
        self.preproc = pick_preprocess(backend)

        in_shape = self.model.inputs[0].shape  # (None, H, W, C)
        self.H, self.W, self.C = int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
        if self.C != 3:
            raise ValueError(f"Model expects {self.C} channels. Need 3 (RGB).")
        self.img_size = (self.W, self.H)  # (W, H) for cv2.resize

        self.roi = ROIExtractor(use_mp=bool(args.use_mp_roi), margin=args.margin)

        self.cap = None
        self.writer = None
        self.csvf = None
        self.csvw = None

        self.smooth = deque(maxlen=max(1, args.smooth_n))
        self.grasp_idx = next((i for i, c in enumerate(class_names) if "grasp" in c.lower()), None)
        self.release_idx = next((i for i, c in enumerate(class_names) if "release" in c.lower()), None)
        self.grasp_cnt = 0
        self.release_cnt = 0
        self.last_event = None  # "grasp" / "release" / None

        self.frame_id = 0
        self.fps = 30.0
        self.t_last = time.time()
        self.disp_fps = 0.0
        self.hud_offset = 0

        os.makedirs(args.events_dir, exist_ok=True)
        log_dir = os.path.dirname(args.log_csv)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def set_hud_offset(self, offset_px: int):
        self.hud_offset = int(max(0, offset_px))

    def open(self):
        src = 0 if str(self.args.video).strip() == "0" else self.args.video
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.args.video}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            w, h = 1280, 720

        self.writer = None
        out_path = getattr(self.args, "out_video", "")
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))
            print(f"[REC] Recording this session to: {out_path}")

        self.csvf = open(self.args.log_csv, "w", newline="", encoding="utf-8")
        self.csvw = csv.writer(self.csvf)
        self.csvw.writerow(
            ["frame_id", "time_sec", "top_idx", "top_class", "conf", "margin"]
            + [f"p_{c}" for c in self.class_names]
        )
    def close(self):
        try:
            if self.writer:
                self.writer.release()
                print("[REC] Video saved.")
        except:
            pass
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        try:
            if self.csvf:
                self.csvf.flush()
                self.csvf.close()
        except:
            pass

    def read_and_infer(self):
        """
        :return:
            overlay_bgr: np.ndarray | None
            info: (cls:str, conf:float, event:str|None, margin:float) | None
            snap_frame_bgr: np.ndarray | None
        """
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None

        self.frame_id += 1
        overlay = frame.copy()

        roi = self.roi.extract(frame, out_size=self.img_size, draw_on=overlay)
        x = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = self.preproc(x)
        x = np.expand_dims(x, 0)

        probs = self.model(x, training=False).numpy()[0]
        self.smooth.append(probs)
        if len(self.smooth) > 1:
            probs = np.mean(self.smooth, axis=0)

        top2 = np.argsort(probs)[-2:][::-1]
        p1, p2 = float(probs[top2[0]]), float(probs[top2[1]])
        margin = p1 - p2
        ci = int(top2[0])
        cls = self.class_names[ci]
        conf = p1

        if margin >= self.args.ambiguity_margin:
            if self.grasp_idx is not None and ci == self.grasp_idx:
                self.grasp_cnt += 1
                self.release_cnt = 0
            elif self.release_idx is not None and ci == self.release_idx:
                self.release_cnt += 1
                self.grasp_cnt = 0
            else:
                self.grasp_cnt = self.release_cnt = 0
        else:
            self.grasp_cnt = self.release_cnt = 0

        t2 = time.time()
        dt = t2 - self.t_last
        if dt > 0:
            self.disp_fps = 0.9 * self.disp_fps + 0.1 * (1.0 / dt)
        self.t_last = t2

        y_off = self.hud_offset
        color = (0, 200, 0) if (conf >= self.args.thr_color and margin >= self.args.ambiguity_margin) else (0, 0, 200)
        self._put_text(overlay, f"{cls} ({conf:.2f})  m={margin:.2f}", (10, 30 + y_off), 0.9, color)
        self._put_text(overlay, f"deb: G={self.grasp_cnt}  R={self.release_cnt}", (10, 62 + y_off), 0.8, (255, 200, 0))
        self._put_text(overlay, f"FPS: {self.disp_fps:.1f}", (10, 94 + y_off), 0.8, (200, 255, 200))

        self._bars(overlay, probs, self.class_names, origin=(10, 134 + y_off))

        snap_frame = None
        event = None

        trigger_grasp = (
            self.grasp_idx is not None and ci == self.grasp_idx
            and margin >= self.args.ambiguity_margin
            and conf >= self.args.thr_color
            and self.grasp_cnt >= self.args.debounce_n
            and self.last_event != "grasp"
        )

        trigger_release = (
            self.release_idx is not None and ci == self.release_idx
            and margin >= self.args.ambiguity_margin
            and conf >= self.args.thr_color
            and self.release_cnt >= self.args.debounce_n
            and self.last_event != "release"
        )

        if trigger_grasp:
            event = "grasp"
            self.last_event = "grasp"
            snap_frame = overlay.copy()
            cv2.imwrite(os.path.join(self.args.events_dir, f"grasp_{self.frame_id:06d}.jpg"), snap_frame)

        elif trigger_release:
            event = "release"
            self.last_event = "release"
            snap_frame = overlay.copy()
            cv2.imwrite(os.path.join(self.args.events_dir, f"release_{self.frame_id:06d}.jpg"), snap_frame)

        if self.csvw:
            self.csvw.writerow(
                [self.frame_id, self.frame_id / max(self.fps, 1e-9), ci, cls, f"{conf:.4f}", f"{margin:.4f}"]
                + [f"{float(p):.4f}" for p in probs]
            )

        if self.writer is not None:
                self.writer.write(overlay)
        return overlay, (cls, conf, event, margin), snap_frame
    # --------- drawing helpers ----------
    @staticmethod
    def _put_text(img, txt, org, scale=0.7, color=(255, 255, 255), thick=2):
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _bars(img, probs, labels, origin=(10, 130), w=260, h=16, gap=6):

        x0, y0 = origin
        for i, (p, lab) in enumerate(zip(probs, labels)):
            y = y0 + i * (h + gap)
            cv2.rectangle(img, (x0, y), (x0 + w, y + h), (30, 30, 30), 1)
            cv2.rectangle(img, (x0, y), (x0 + int(w * float(p)), y + h), (60, 140, 60), -1)
            cv2.putText(
                img, f"{lab}: {p:.2f}",
                (x0 + w + 10, y + h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
