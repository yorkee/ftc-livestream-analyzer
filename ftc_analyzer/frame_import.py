"""Read existing captured images without modifying the source directory."""
import re
from pathlib import Path
import cv2
from ftc_analyzer.video_ocr import format_timestamp


def import_frame_folder(folder: Path, interval: int, offset: int = 0):
    if interval <= 0 or offset < 0:
        raise ValueError("Invalid frame interval or offset")
    files = [p for p in Path(folder).iterdir() if p.is_file()
             and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    def natural(p):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", p.name)]
    files.sort(key=natural)
    numbers = [re.fullmatch(r"frame_(\d+)", p.stem, re.I) for p in files]
    numbered = all(numbers)
    rows, skipped = [], 0
    for index, path in enumerate(files):
        frame_index = int(numbers[index].group(1)) if numbered else index
        image = cv2.imread(str(path))
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]
        seconds = offset + frame_index * interval
        rows.append(dict(frame_index=frame_index, timestamp_seconds=seconds,
                         timestamp=format_timestamp(seconds), frame_path=str(path),
                         width=width, height=height))
    if not rows:
        raise ValueError("No readable images in selected folder")
    return rows, skipped
