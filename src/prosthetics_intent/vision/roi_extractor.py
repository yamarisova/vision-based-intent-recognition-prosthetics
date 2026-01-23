# roi_extractor.py
import cv2
import numpy as np

class ROIExtractor:
    def __init__(self, use_mp=True, margin=1.8):
        self.use_mp = use_mp
        self.margin = margin
        self.hands = None
        if use_mp:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5
                )
            except Exception as e:
                print("[WARN] MediaPipe unavailable → center crop. Reason:", e)
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
                best = (x1,y1,x2,y2,area)
        if not best: return None
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
