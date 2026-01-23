# main.py
import argparse, json, os
from src.prosthetics_intent.app.ui_app import App

def parse_args():
    ap = argparse.ArgumentParser("HandGestureUI")
    ap.add_argument("--model", required=True, help="Path to .keras")
    ap.add_argument("--classes", required=True, help="class_names.json")
    ap.add_argument("--video", default="0", help="'0' webcam or path to video")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "mobilenet_v2", "resnet50v2"],
                    help="Preprocess backend (auto tries to guess)")
    ap.add_argument("--use_mp_roi", type=int, default=1, help="1=MediaPipe ROI")
    ap.add_argument("--margin", type=float, default=1.8, help="ROI margin scale")
    ap.add_argument("--smooth_n", type=int, default=5, help="Smoothing window")
    ap.add_argument("--debounce_n", type=int, default=2, help="Debounce N")
    ap.add_argument("--ambiguity_margin", type=float, default=0.06)
    ap.add_argument("--thr_color", type=float, default=0.40)
    ap.add_argument("--send_serial", type=int, default=0, help="1=use Arduino")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--events_dir", default="events_snaps")
    ap.add_argument("--log_csv", default="logs/pred_log.csv")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.classes, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    os.makedirs(args.events_dir, exist_ok=True)

    app = App(args, class_names)
    app.run()

