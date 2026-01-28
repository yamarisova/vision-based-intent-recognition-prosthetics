"""
Universal validation of a .keras model on video (MobileNetV2 / ResNet50V2).
Automatic detection of the model input layer size.
Correct preprocess_input according to the backbone.
ROI via MediaPipe Hands (as used during training).
Overlays: class name, confidence score, top-2 margin, probability bars, FPS.
Saving annotated video and CSV.
Snapshots of event moments (Grasp/Release) with display in additional windows.
Optional sending of commands to Arduino with automatic port detection..
"""

import os, time, csv, glob, argparse
import numpy as np
import cv2
import tensorflow as tf
from collections import deque

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Шлях до .keras моделі")
    ap.add_argument("--classes", default="", help="JSON з CLASS_NAMES. Якщо порожньо — з моделі/індекси")
    ap.add_argument("--video", default="0", help="Шлях до відео або '0' для вебкамери")
    ap.add_argument("--out_video", default="annotated_out.mp4", help="Вихідне анотоване відео")
    ap.add_argument("--out_csv", default="pred_log.csv", help="CSV-лог покадрових предиктів")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "mobilenet_v2", "resnet50v2"],
                    help="Який preprocess використати. 'auto' спробує вгадати")
    ap.add_argument("--use_mp_roi", type=int, default=1, help="1=MediaPipe ROI (рекомендовано), 0=центр-кроп")
    ap.add_argument("--margin", type=float, default=1.8, help="Множник bbox навколо руки для ROI")
    ap.add_argument("--smooth_n", type=int, default=3, help="Кадрів для усереднення softmax")
    ap.add_argument("--debounce_n", type=int, default=2, help="Лічильник підтвердження події")
    ap.add_argument("--ambiguity_margin", type=float, default=0.1, help="Поріг близькості топ-2 (0..1)")
    ap.add_argument("--thr_color", type=float, default=0.4, help="Лише для кольору тексту (не логіка)")
    ap.add_argument("--show", type=int, default=1, help="1=показувати live-вікно")
    # події/серіал
    ap.add_argument("--events_dir", default="events_snaps", help="Куди зберігати знімки моментів подій")
    ap.add_argument("--send_serial", type=int, default=0, help="1=вмикає надсилання команд GRASP/RELEASE")
    ap.add_argument("--baud", type=int, default=115200, help="Baudrate для серійного порту")
    ap.add_argument("--min_gap_frames", type=int, default=6,
                    help="К-сть non-event кадрів, щоб дозволити новий тригер того ж типу")

    return ap.parse_args()

def get_preprocess(backend: str):
    if backend == "mobilenet_v2":
        from tensorflow.keras.applications import mobilenet_v2
        return mobilenet_v2.preprocess_input
    if backend == "resnet50v2":
        from tensorflow.keras.applications import resnet_v2
        return resnet_v2.preprocess_input
    raise ValueError("Unknown backend for preprocess")

def guess_backend_from_model(model: tf.keras.Model, model_path: str) -> str:
    name = (getattr(model, "name", "") + " " + os.path.basename(model_path)).lower()
    if "mobilenet_v2" in name or "mobilenetv2" in name:
        return "mobilenet_v2"
    if "resnet" in name or "resnet50v2" in name or "resnet_v2" in name:
        return "resnet50v2"
    return "resnet50v2"

def open_serial_auto(baud: int):
    try:
        import serial
        from serial.tools import list_ports
    except Exception as e:
        print("[SER] pyserial недоступний:", e)
        return None

    ports = [p.device for p in list_ports.comports()]
    prefer = [p for p in ports if any(k in p.lower() for k in ["usbmodem", "usbserial", "cu.usb", "tty.usb"])]
    cands = prefer or ports

    if not cands:
        patterns = ["/dev/tty.usbmodem*", "/dev/tty.usbserial*", "/dev/cu.usbmodem*",
                    "/dev/cu.usbserial*", "COM*", "/dev/serial/by-id/*"]
        for pat in patterns:
            cands += glob.glob(pat)

    print("[SER] Кандидати портів:", cands if cands else "нема")
    for port in cands:
        try:
            print("[SER] Відкриваємо:", port)
            ser = serial.Serial(port, baud, timeout=0.5, write_timeout=0.5)
            time.sleep(2.0)
            # простий пінг
            try:
                ser.reset_input_buffer()
                ser.write(b"PING\n")
                resp = ser.readline().decode("utf-8", errors="ignore").strip()
                print("[SER] PING ->", resp)
            except Exception:
                pass
            return ser
        except Exception as e:
            print(f"[SER] Не вдалося відкрити {port}:", e)
    print("[SER] Працюємо без Arduino.")
    return None

def put_text(img, txt, org, scale=0.7, color=(255,255,255), thick=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def bar_probs(img, probs, labels, origin=(10, 150), w=260, h=16, gap=6):
    x0, y0 = origin
    for i, (p, lab) in enumerate(zip(probs, labels)):
        y = y0 + i * (h + gap)
        cv2.rectangle(img, (x0, y), (x0 + w, y + h), (30,30,30), 1)
        fill = int(w * float(p))
        cv2.rectangle(img, (x0, y), (x0 + fill, y + h), (60,140,60), -1)
        put_text(img, f"{lab}: {p:.2f}", (x0 + w + 10, y + h - 3), 0.55, (255,255,255), 2)

# ---------------- ROI (MediaPipe) ----------------
class ROIExtractor:
    def __init__(self, use_mp=True, margin=1.8):
        self.use_mp = use_mp
        self.margin = margin
        self.hands = None
        if use_mp:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                                 min_detection_confidence=0.4, min_tracking_confidence=0.4)
            except Exception as e:
                print("[WARN] MediaPipe недоступний → центр-кроп. Причина:", e)
                self.use_mp = False

    def _largest_hand_bbox(self, img, lms_list):
        h, w = img.shape[:2]
        best = None
        for lm in lms_list:
            xs = [int(p.x*w) for p in lm.landmark]
            ys = [int(p.y*h) for p in lm.landmark]
            x1, y1, x2, y2 = max(min(xs),0), max(min(ys),0), min(max(xs),w-1), min(max(ys),h-1)
            area = max(1, (x2-x1)*(y2-y1))
            if best is None or area > best[-1]:
                best = (x1, y1, x2, y2, area)
        if best is None: return None
        x1,y1,x2,y2,_ = best
        cx, cy = (x1+x2)/2, (y1+y2)/2
        bw, bh = (x2-x1)*self.margin, (y2-y1)*self.margin
        x1n, y1n = int(max(0, cx-bw/2)), int(max(0, cy-bh/2))
        x2n, y2n = int(min(w-1, cx+bw/2)), int(min(h-1, cy+bh/2))
        return x1n, y1n, x2n, y2n

    @staticmethod
    def _center_square_crop(img):
        h, w = img.shape[:2]
        side = min(h, w)
        y1 = (h - side)//2; x1 = (w - side)//2
        return img[y1:y1+side, x1:x1+side]

    def extract(self, frame, out_size, draw_on=None):
        if self.use_mp and self.hands is not None:
            res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                box = self._largest_hand_bbox(frame, res.multi_hand_landmarks)
                if box is not None:
                    x1,y1,x2,y2 = box
                    if draw_on is not None:
                        cv2.rectangle(draw_on, (x1,y1), (x2,y2), (0,255,0), 2)
                    roi = frame[y1:y2, x1:x2]
                else:
                    roi = self._center_square_crop(frame)
            else:
                roi = self._center_square_crop(frame)
        else:
            roi = self._center_square_crop(frame)

        h, w = roi.shape[:2]
        side = max(h, w)
        top = (side-h)//2; bottom = side-h-top
        left = (side-w)//2; right = side-w-left
        roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        roi = cv2.resize(roi, out_size, interpolation=cv2.INTER_AREA)
        return roi

def main():
    args = parse_args()

    ser = None
    if args.send_serial:
        ser = open_serial_auto(args.baud)

    print("[INFO] Loading model:", args.model)
    model = tf.keras.models.load_model(args.model, compile=False)

    backend = args.backend
    if backend == "auto":
        backend = guess_backend_from_model(model, args.model)
        print(f"[INFO] Auto-detected backend → {backend}")
    PREPROC = get_preprocess(backend)

    in_shape = model.inputs[0].shape  # (None, H, W, C)
    H, W, C = int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
    if C != 3:
        raise ValueError(f"[FATAL] Модель очікує {C} канал(-и). Потрібно 3 (RGB). Пересейв/перетренуй модель.")
    IMG_SIZE = (W, H)  # порядок (W,H) для cv2.resize

    CLASS_NAMES = None
    if args.classes and os.path.exists(args.classes):
        import json
        with open(args.classes, "r", encoding="utf-8") as f:
            CLASS_NAMES = json.load(f)
    if not CLASS_NAMES:
        CLASS_NAMES = [f"class_{i}" for i in range(int(model.outputs[0].shape[-1]))]
    if int(model.outputs[0].shape[-1]) != len(CLASS_NAMES):
        raise ValueError("К-сть виходів моделі не збігається з довжиною CLASS_NAMES.")

    print("[INFO] CLASS_NAMES:", CLASS_NAMES)

    src = 0 if str(args.video).strip() == "0" else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити відеоджерело: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))
    roi_extractor = ROIExtractor(use_mp=bool(args.use_mp_roi), margin=args.margin)
    os.makedirs(args.events_dir, exist_ok=True)
    csvf = open(args.out_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(csvf)
    #writer.writerow(["frame_id","time_sec","top_idx","top_class","conf","margin"] + [f"p_{c}" for c in CLASS_NAMES])
    writer.writerow(
        ["timestamp", "frame_id", "pred_class", "pred_label", "confidence", "state", "action"]
        + [f"p_{c}" for c in CLASS_NAMES]
    )
    grasp_idx   = next((i for i,c in enumerate(CLASS_NAMES) if "grasp"   in c.lower()), None)
    release_idx = next((i for i,c in enumerate(CLASS_NAMES) if "release" in c.lower()), None)

    smooth_buf = deque(maxlen=max(1, args.smooth_n))
    grasp_cnt = release_cnt = 0
    last_event = None   # 'grasp' або 'release'
    frame_id = 0
    t_last = time.time(); disp_fps = 0.0

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        overlay = frame.copy()

        roi = roi_extractor.extract(frame, out_size=IMG_SIZE, draw_on=overlay)
        x = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = PREPROC(x)
        x = np.expand_dims(x, 0)

        probs = model.predict(x, verbose=0)[0]
        smooth_buf.append(probs)
        if len(smooth_buf) > 1:
            probs = np.mean(smooth_buf, axis=0)

        top2 = np.argsort(probs)[-2:][::-1]
        p1, p2 = float(probs[top2[0]]), float(probs[top2[1]])
        margin = p1 - p2
        ci = int(top2[0]); cls = CLASS_NAMES[ci]; conf = p1

        if margin >= args.ambiguity_margin:
            if grasp_idx is not None and ci == grasp_idx:
                grasp_cnt += 1; release_cnt = 0
            elif release_idx is not None and ci == release_idx:
                release_cnt += 1; grasp_cnt = 0
            else:
                grasp_cnt = release_cnt = 0
        else:
            grasp_cnt = release_cnt = 0

        t2 = time.time(); dt = t2 - t_last
        if dt > 0: disp_fps = 0.9*disp_fps + 0.1*(1.0/dt)
        t_last = t2

        color = (0,200,0) if (conf >= args.thr_color and margin >= args.ambiguity_margin) else (0,0,200)
        put_text(overlay, f"{cls} ({conf:.2f})  margin={margin:.2f}", (10, 30), 0.9, color)
        put_text(overlay, f"debounce: G={grasp_cnt} R={release_cnt}", (10, 60), 0.8, (255,200,0))
        put_text(overlay, f"FPS: {disp_fps:.1f}", (10, 90), 0.8, (200,255,200))
        bar_probs(overlay, probs, CLASS_NAMES, origin=(10, 130))

        trigger_grasp = (
            grasp_idx is not None and ci == grasp_idx
            and margin >= args.ambiguity_margin
            and conf   >= args.thr_color
            and grasp_cnt >= args.debounce_n
            and last_event != "grasp"
        )

        trigger_release = (
            release_idx is not None and ci == release_idx
            and margin >= args.ambiguity_margin
            and conf   >= args.thr_color
            and release_cnt >= args.debounce_n
            and last_event != "release"
        )

        if trigger_grasp:
            snap_path = os.path.join(args.events_dir, f"grasp_{frame_id:06d}.jpg")
            cv2.imwrite(snap_path, overlay)
            if args.show:
                cv2.imshow("Grasp SNAP", overlay)
            last_event = "grasp"
            if ser:
                try: ser.write(b"GRASP\n")
                except Exception as e: print("[WARN] Arduino write failed:", e)

        elif trigger_release:
            snap_path = os.path.join(args.events_dir, f"release_{frame_id:06d}.jpg")
            cv2.imwrite(snap_path, overlay)
            if args.show:
                cv2.imshow("Release SNAP", overlay)
            last_event = "release"
            if ser:
                try: ser.write(b"RELEASE\n")
                except Exception as e: print("[WARN] Arduino write failed:", e)

        out.write(overlay)
        if args.show:
            cv2.imshow("video_pred", overlay)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break
        writer.writerow([frame_id, frame_id/max(fps,1e-9), ci, cls, f"{conf:.4f}", f"{margin:.4f}"]
                        + [f"{float(p):.4f}" for p in probs])

    try: out.release()
    except: pass
    try: cap.release()
    except: pass
    try: csvf.flush(); csvf.close()
    except: pass
    try: cv2.destroyAllWindows()
    except: pass
    if ser:
        try: ser.close()
        except: pass

    print(f"[DONE] Saved video → {args.out_video}")
    print(f"[DONE] Saved CSV   → {args.out_csv}")
    print(f"[DONE] Snaps dir   → {args.events_dir}")

if __name__ == "__main__":
    main()
