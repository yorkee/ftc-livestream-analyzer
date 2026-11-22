"""Video download, 10-second frame sampling, and scoreboard OCR helpers."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
import pytesseract
from PIL import Image
from pytesseract import Output


def download_video(url: str, video_id: str, video_dir: Path, max_height: int = 720) -> Path:
    """Download a single-file video stream suitable for local frame extraction."""
    import yt_dlp

    video_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in video_dir.glob(f"{video_id}.*")
                if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
                and not p.name.endswith(".part")]
    if existing:
        return max(existing, key=lambda p: p.stat().st_size)

    output = video_dir / f"{video_id}.%(ext)s"
    opts = {
        # Prefer a single muxed stream: no audio is needed for this workflow.
        # "format": f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        "format": f"bestvideo[height<={max_height}]/bestvideo/best","outtmpl": str(output),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "continuedl": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    downloaded = [p for p in video_dir.glob(f"{video_id}.*")
                   if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
                   and not p.name.endswith(".part")]
    if not downloaded:
        raise FileNotFoundError("yt-dlp did not produce a video file.")
    return max(downloaded, key=lambda p: p.stat().st_size)


def extract_frames(video_path: Path, frame_dir: Path, interval_seconds: int = 10,
                   max_width: int = 1280) -> list[dict]:
    """Use FFmpeg to save one JPEG every N seconds and return frame metadata."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.jpg"):
        old.unlink()

    pattern = frame_dir / "frame_%08d.jpg"
    scale = f"scale='min({max_width},iw)':-2"
    vf = f"fps=1/{interval_seconds},{scale}"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vf", vf, "-q:v", "3", "-start_number", "0", str(pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr[-2000:]}")

    files = sorted(frame_dir.glob("frame_*.jpg"))
    if not files:
        raise RuntimeError("No frames were extracted. Check that FFmpeg can read this video.")
    rows = []
    for index, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        rows.append({
            "frame_index": index,
            "timestamp_seconds": index * interval_seconds,
            "timestamp": format_timestamp(index * interval_seconds),
            "frame_path": str(path),
            "width": width,
            "height": height,
        })
    return rows


def format_timestamp(seconds: int | float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _ocr_text(image_path: str) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    # Upscale smaller frames slightly; OCR works better on small overlay text.
    if image.width < 1000:
        scale = 1000 / image.width
        image = image.resize((1000, int(image.height * scale)))
    data = pytesseract.image_to_data(image, config="--psm 11", output_type=Output.DICT)
    words, confidences = [], []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        word = word.strip()
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            confidence = -1
        if word:
            words.append(word)
        if confidence >= 0:
            confidences.append(confidence)
    average = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
    return " ".join(words), average


MATCH_PATTERNS = [
    re.compile(r"\b(?:qualification|qualifying|qual|q)\s*(?:match\s*)?(?:no\.?\s*)?(\d{1,3})\b", re.I),
    re.compile(r"\bmatch\s*(?:no\.?\s*)?(\d{1,3})\b", re.I),
]


def detect_match_number(text: str) -> int | None:
    normalized = re.sub(r"[|:;,_]", " ", text)
    for pattern in MATCH_PATTERNS:
        match = pattern.search(normalized)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
    # Common compact scoreboard label, e.g. Q-12 or Q12.
    match = re.search(r"\bQ\s*[-#]?\s*(\d{1,3})\b", normalized, re.I)
    return int(match.group(1)) if match else None


def detect_team_numbers(text: str, known_team_numbers: Iterable[int] | None = None) -> list[int]:
    """Prefer exact event roster matches to avoid mistaking scores for team IDs."""
    digits = {int(value) for value in re.findall(r"(?<!\d)(\d{3,6})(?!\d)", text)}
    if known_team_numbers is not None:
        roster = {int(n) for n in known_team_numbers}
        return sorted(digits & roster)
    return sorted(digits)


def _ocr_team_panel(image_path: str, alliance: str) -> tuple[list[int], str, float]:
    """Read the two team-number cards from one alliance's scoreboard panel.

    The current template targets the FTC Canopy-style 16:9 overlay shown in the
    Saline Preview sample. Coordinates are proportional, so 720p and 1080p
    frames use the same crop. Other overlay layouts will need another template.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    # Team-number cards sit in the lower scoreboard strip, left of each
    # alliance's score. Keep each alliance isolated to avoid cross-reading.
    regions = {
        "red": (0.235, 0.845, 0.333, 1.0),
        "blue": (0.667, 0.845, 0.765, 1.0),
    }
    if alliance not in regions:
        raise ValueError("alliance must be 'red' or 'blue'")
    left, top, right, bottom = regions[alliance]
    crop = image.crop((int(left * width), int(top * height),
                       int(right * width), int(bottom * height)))
    # Enlarge the relatively small cards, then use a high-contrast grayscale
    # image. Tesseract is constrained to digits to reduce icon/score noise.
    crop = crop.resize((crop.width * 4, crop.height * 4))
    import numpy as np
    gray = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(gray, config=config, output_type=Output.DICT)
    words, confidences = [], []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        word = word.strip()
        try:
            score = float(conf)
        except (TypeError, ValueError):
            score = -1
        if word:
            words.append(word)
        if score >= 0:
            confidences.append(score)
    raw = " ".join(words)
    numbers = []
    for value in re.findall(r"(?<!\\d)(\\d{3,6})(?!\\d)", raw):
        number = int(value)
        if number not in numbers:
            numbers.append(number)
    average = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
    return numbers[:2], raw, average


def run_ocr(frame_rows: list[dict], known_team_numbers: Iterable[int] | None = None,
            progress_callback=None) -> list[dict]:
    results = []
    total = len(frame_rows)
    roster = {int(n) for n in known_team_numbers} if known_team_numbers is not None else None
    for index, frame in enumerate(frame_rows, start=1):
        text, confidence = _ocr_text(frame["frame_path"])
        red_teams, red_raw, red_conf = _ocr_team_panel(frame["frame_path"], "red")
        blue_teams, blue_raw, blue_conf = _ocr_team_panel(frame["frame_path"], "blue")
        if roster is not None:
            red_teams = [n for n in red_teams if n in roster]
            blue_teams = [n for n in blue_teams if n in roster]
        alliance_teams = sorted(set(red_teams + blue_teams))
        results.append({
            **frame,
            "ocr_text": text,
            "ocr_confidence": confidence,
            "match_number_candidate": detect_match_number(text),
            "red_team_numbers": red_teams,
            "blue_team_numbers": blue_teams,
            "red_panel_ocr": red_raw,
            "blue_panel_ocr": blue_raw,
            "red_panel_confidence": red_conf,
            "blue_panel_confidence": blue_conf,
            "team_number_candidates": alliance_teams,
        })
        if progress_callback:
            progress_callback(index / total, index, total)
    return results


def build_observed_match_windows(ocr_rows: list[dict], interval_seconds: int) -> list[dict]:
    """Group consecutive frames with the same detected match number.

    Boundaries are observed windows, not exact match start/end times. A scoreboard
    can show a match before play starts or after play ends.
    """
    windows = []
    current = None
    for row in ocr_rows:
        number = row.get("match_number_candidate")
        if number is None:
            if current:
                windows.append(current)
                current = None
            continue
        teams = row.get("team_number_candidates") or []
        if current and current["match_number"] == number:
            current["last_seen_seconds"] = row["timestamp_seconds"]
            current["frames_seen"] += 1
            current["team_numbers_seen"] = sorted(set(current["team_numbers_seen"]) | set(teams))
        else:
            if current:
                windows.append(current)
            current = {
                "match_number": number,
                "first_seen_seconds": row["timestamp_seconds"],
                "last_seen_seconds": row["timestamp_seconds"],
                "observed_until_seconds": row["timestamp_seconds"] + interval_seconds,
                "frames_seen": 1,
                "team_numbers_seen": teams,
                "boundary_note": "Observed scoreboard window; not an exact play start/end.",
            }
    if current:
        windows.append(current)
    for item in windows:
        item["observed_until_seconds"] = item["last_seen_seconds"] + interval_seconds
        item["first_seen"] = format_timestamp(item["first_seen_seconds"])
        item["observed_until"] = format_timestamp(item["observed_until_seconds"])
    return windows


def save_ocr_results(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
