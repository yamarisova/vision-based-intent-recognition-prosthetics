import cv2
import os
import math
import numpy as np
from pathlib import Path

# ----------------- Налаштування -----------------
INPUT_DIR   = Path("/")          # де лежать відео
OUTPUT_DIR  = Path("//photos")   # куди класти кадри

# Спосіб вибірки кадрів: або КОЖЕН N-Й КАДР, або КОЖНІ S СЕКУНД
SELECT_MODE = "every_n"      # "every_n" або "every_s"
EVERY_N     = 5              # зберігати кожен N-й кадр (для режиму "every_n")
EVERY_S     = 0.1            # інтервал у секундах між кадрами (для режиму "every_s")

# Якість збереження
OUTPUT_FORMAT = "jpg"        # "jpg" або "png"
JPEG_QUALITY  = 95           # 0..100 (лише для jpg), 95 — майже без втрат
PNG_COMPRESSION = 3          # 0..9 (лише для png), 0 — без компресії

MAKE_SUBDIR_PER_VIDEO = True # True -> кадри кожного відео в окрему папку
INCLUDE_TIMESTAMP_IN_NAME = True  # додати мс-час з відео до імені файлу

# Покращення кадру
APPLY_SHARPEN = True         # різкість
APPLY_CLAHE   = True         # підвищення контрасту по L-каналу (LAB)
SKIP_BLURRY   = True         # пропускати дуже розмиті кадри
BLUR_THRESHOLD = 50.0        # поріг різкості (варіація Лапласіана); чим більше — тим різкіше
# -----------------------------------------------

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}  # підтримувані розширення (lowercase)


def improve_frame(frame: np.ndarray) -> np.ndarray:
    """Швидке покращення кадру: різкість + контраст (CLAHE)."""
    out = frame

    if APPLY_SHARPEN:
        # м'яке ядро різкості (не агресивне, щоб не шуміло)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)

    if APPLY_CLAHE:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return out


def is_blurry(frame: np.ndarray, threshold: float) -> bool:
    """Оцінка «змазу» через варіацію Лапласіана: нижче порогу — розмите."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


def extract_frames(video_path: Path, out_dir: Path, select_mode="every_n",
                   every_n=5, every_s=0.5, prefix=None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Не вдалося відкрити відео: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"📹 {video_path.name}: кадрів={total_frames}, fps≈{fps:.2f}")

    # для режиму "every_s" рахуємо, кожен скільки кадрів брати
    step_n = max(1, int(round(every_s * fps))) if select_mode == "every_s" else max(1, every_n)
    if select_mode == "every_s":
        print(f"⏱️ Вибірка кожні {every_s}s → кожен {step_n}-й кадр")

    if prefix is None:
        prefix = video_path.stem  # ім'я файлу без розширення

    saved = 0
    frame_idx = 0

    # Налаштування параметрів збереження
    imwrite_params = []
    ext = OUTPUT_FORMAT.lower()
    if ext == "jpg":
        imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)]
        out_ext = ".jpg"
    else:
        imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, int(PNG_COMPRESSION)]
        out_ext = ".png"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step_n == 0:
            # (без ресайзу) — працюємо з оригінальною роздільною здатністю
            proc = improve_frame(frame)

            if SKIP_BLURRY and is_blurry(proc, BLUR_THRESHOLD):
                frame_idx += 1
                continue

            # час у мс від початку відео
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) if INCLUDE_TIMESTAMP_IN_NAME else None

            if INCLUDE_TIMESTAMP_IN_NAME and ts_ms is not None:
                frame_filename = out_dir / f"{prefix}_frame_{saved:04d}_t{ts_ms}ms{out_ext}"
            else:
                frame_filename = out_dir / f"{prefix}_frame_{saved:04d}{out_ext}"

            cv2.imwrite(str(frame_filename), proc, imwrite_params)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"✅ {video_path.name}: збережено {saved} кадрів → {out_dir}")
    return saved


def find_videos(input_dir: Path):
    """Знаходимо всі відео (рекурсивно) з підтримуваними розширеннями."""
    vids = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
    vids.sort(key=lambda x: str(x).lower())
    return vids


def main():
    videos = find_videos(INPUT_DIR)
    if not videos:
        print(f"⚠️ Відео не знайдено у: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for vid in videos:
        # папка виводу: або окремо на кожне відео, або все в одну
        out_dir = (OUTPUT_DIR / vid.stem) if MAKE_SUBDIR_PER_VIDEO else OUTPUT_DIR

        saved = extract_frames(
            video_path=vid,
            out_dir=out_dir,
            select_mode=SELECT_MODE,
            every_n=EVERY_N,
            every_s=EVERY_S,
            prefix=vid.stem  # ім’я для фото = назва відео
        )
        total_saved += saved

    print(f"\n🎉 Готово! Загалом збережено кадрів: {total_saved}")


if __name__ == "__main__":
    main()
