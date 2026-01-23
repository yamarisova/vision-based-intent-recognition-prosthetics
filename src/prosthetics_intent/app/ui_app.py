# ui_app.py
import datetime
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

from src.prosthetics_intent.vision.video_processor import VideoProcessor
from src.prosthetics_intent.control.arduino_comm import ArduinoComm


class App:
    def __init__(self, args, class_names):
        self.args = args
        self.class_names = class_names

        # ------- window & layout sizing (≈ 92% of screen) -------
        self.root = tk.Tk()
        self.root.title("Hand Gesture Inference")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        win_w, win_h = int(sw * 0.92), int(sh * 0.92)
        self.root.geometry(f"{win_w}x{win_h}")

        # proportions
        self.RIGHT_COL_W = int(win_w * 0.30)      # right panel
        self.VIDEO_W     = win_w - self.RIGHT_COL_W - 40
        self.VIDEO_H     = int(self.VIDEO_W * 9 / 16)
        self.SNAP_W, self.SNAP_H = 420, 300       # equal snap frames

        # ------- left: video -------
        left = tk.Frame(self.root, bg="#0f0f0f")
        left.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.video_lbl = tk.Label(left, bg="#111")
        self.video_lbl.pack(fill="both", expand=True)

        # ------- right: snaps -------
        right = tk.Frame(self.root, width=self.RIGHT_COL_W, bg="#2b2b2b")
        right.grid(row=0, column=1, padx=8, pady=8, sticky="ns")
        right.grid_propagate(False)

        # GRASP
        tk.Label(right, text="GRASP SNAP", fg="#ddd", bg="#2b2b2b",
                 font=("Arial", 12, "bold")).pack(anchor="w", pady=(6, 4))
        grasp_box = tk.Frame(right, width=self.SNAP_W, height=self.SNAP_H,
                             bg="#222", highlightthickness=1, highlightbackground="#444")
        grasp_box.pack_propagate(False)
        grasp_box.pack(pady=(0, 12))
        self.snap_grasp_img = tk.Label(grasp_box, bg="#222")
        self.snap_grasp_img.pack(fill="both", expand=True)

        # RELEASE
        tk.Label(right, text="RELEASE SNAP", fg="#ddd", bg="#2b2b2b",
                 font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 4))
        release_box = tk.Frame(right, width=self.SNAP_W, height=self.SNAP_H,
                               bg="#222", highlightthickness=1, highlightbackground="#444")
        release_box.pack_propagate(False)
        release_box.pack()
        self.snap_release_img = tk.Label(release_box, bg="#222")
        self.snap_release_img.pack(fill="both", expand=True)

        # ------- bottom: logs + buttons -------
        bottom = tk.Frame(self.root)
        bottom.grid(row=1, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 8))
        bottom.grid_columnconfigure(0, weight=1)

        self.logs = tk.Listbox(bottom, height=10, font=("Menlo", 11))
        self.logs.grid(row=0, column=0, sticky="we")

        btns = tk.Frame(bottom)
        btns.grid(row=0, column=1, padx=(8, 0))
        self.btn_start  = ttk.Button(btns, text="Start",  command=self.on_start, width=12)
        self.btn_pause  = ttk.Button(btns, text="Pause",  command=self.on_pause, width=12, state="disabled")
        self.btn_resume = ttk.Button(btns, text="Resume", command=self.on_resume, width=12, state="disabled")
        self.btn_quit   = ttk.Button(btns, text="Quit",   command=self.on_quit, width=12)
        for i, b in enumerate((self.btn_start, self.btn_pause, self.btn_resume, self.btn_quit)):
            b.grid(row=i, column=0, pady=3)

        # state
        self.running = False
        self.proc: VideoProcessor | None = None
        self.arduino: ArduinoComm | None = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

    # ---------- control handlers ----------
    def on_start(self):
        if self.proc is None:
            self.proc = VideoProcessor(self.args, self.class_names)
            self.proc.open()
            # опустити HUD, якщо підтримується у VideoProcessor
            if hasattr(self.proc, "set_hud_offset"):
                try:
                    self.proc.set_hud_offset(60)
                except Exception:
                    pass
        self.running = True
        if getattr(self.args, "send_serial", 0) and self.arduino is None:
            self.arduino = ArduinoComm(self.args.baud)
            self.arduino.connect_auto()
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_resume.config(state="disabled")
        self.loop()

    def on_pause(self):
        self.running = False
        self.btn_pause.config(state="disabled")
        self.btn_resume.config(state="normal")

    def on_resume(self):
        self.running = True
        self.btn_pause.config(state="normal")
        self.btn_resume.config(state="disabled")
        self.loop()

    def on_quit(self):
        self.running = False
        try:
            if self.proc:
                self.proc.close()
        except Exception:
            pass
        try:
            if self.arduino:
                self.arduino.close()
        except Exception:
            pass
        self.root.destroy()

    # ---------- UI loop ----------
    def loop(self):
        if not self.running:
            return

        try:
            # read_and_infer -> (bgr_frame, (cls, conf, event, margin) | None, event_frame_bgr)
            frame_bgr, info, event_frame_bgr = self.proc.read_and_infer()
        except Exception as e:
            self.log_line(f"[ERR] {e}")
            # навіть при помилці не зависаємо — плануємо наступний кадр
            self.root.after(10, self.loop)
            return

        # закінчився стрім -> ставимо на паузу
        if frame_bgr is None:
            self.log_line("Stream ended.")
            self.on_pause()
            return

        # показуємо відео (letterbox у задане вікно)
        self._show_bgr_on(self.video_lbl, frame_bgr, fit=(self.VIDEO_W, self.VIDEO_H))

        # логи / знімки / arduino
        if info:
            cls, conf, event, margin = info
            now = datetime.datetime.now().strftime("%H:%M:%S")
            tvid = f"{self.proc.frame_id / max(self.proc.fps, 1e-9):06.2f}s"
            self.log_line(f"{now} | t={tvid} | {cls} ({conf:.2f}) | m={margin:.2f} | action={event or '-'}")

            # важливо: НЕ ставимо на паузу під час подій
            if event == "grasp":
                snap = (event_frame_bgr if event_frame_bgr is not None else frame_bgr).copy()
                self._show_bgr_on(self.snap_grasp_img, snap, fit=(self.SNAP_W, self.SNAP_H))
                self._send_arduino_async("GRASP")

            elif event == "release":
                snap = (event_frame_bgr if event_frame_bgr is not None else frame_bgr).copy()
                self._show_bgr_on(self.snap_release_img, snap, fit=(self.SNAP_W, self.SNAP_H))
                self._send_arduino_async("RELEASE")

        self.root.after(10, self.loop)

    # ---------- helpers ----------
    def _send_arduino_async(self, msg: str):
        if not self.arduino:
            return
        # відправка у фоні, щоб не блокувати UI
        def _send():
            try:
                self.arduino.send(msg)
            except Exception as e:
                self.log_line(f"[SER] {e}")
        t = threading.Thread(target=_send, daemon=True)
        t.start()

    def _show_bgr_on(self, widget: tk.Label, bgr, fit=(640, 360)):
        """Показати BGR-картинку у Label із збереженням пропорцій (letterbox)."""
        if bgr is None:
            return
        target_w, target_h = fit
        h, w = bgr.shape[:2]
        scale = min(target_w / float(w), target_h / float(h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        widget.imgtk = imgtk
        widget.configure(image=imgtk)

    def log_line(self, s):
        self.logs.insert(tk.END, s)
        # тримаємо 10 свіжих
        while self.logs.size() > 10:
            self.logs.delete(0)

    # ---------- entry ----------
    def run(self):
        self.root.mainloop()
