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
    """Read match/event text only from the top and bottom banner search bands."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    band_height = max(1, int(height * 0.25))
    regions = [
        image.crop((0, 0, width, band_height)),
        image.crop((0, height - band_height, width, height)),
    ]
    words, confidences = [], []
    for crop in regions:
        if crop.width < 1000:
            scale = 1000 / crop.width
            crop = crop.resize((1000, int(crop.height * scale)))
        data = pytesseract.image_to_data(crop, config="--psm 11", output_type=Output.DICT)
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


def _ocr_team_numbers_by_side(image_path: str,
                              known_team_numbers: Iterable[int] | None = None
                              ) -> tuple[list[int], list[int], str, float]:
    """Read team IDs from the four fixed alliance cells in the FTC banner.

    The banner can be at the top or bottom. Each of the four team cells is
    OCR'd independently so alliance icons, scores, and clock digits are not
    mixed into team IDs.
    """
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    frame_width, frame_height = image.size
    band_height = max(1, int(frame_height * 0.25))
    regions = [
        ("top", image.crop((0, 0, frame_width, band_height))),
        ("bottom", image.crop((0, frame_height - band_height, frame_width, frame_height))),
    ]

    # Calibrated against both supplied Canopy/BioBuzz examples. Include the
    # complete number glyphs while excluding the adjacent team-logo artwork.
    x_cells = {
        "red": (0.265, 0.335),
        "blue": (0.670, 0.735),
    }
    y_cells = [
        (0.35, 0.69),  # upper team-number cell
        (0.66, 1.00),  # lower team-number cell
    ]
    roster = {int(n) for n in known_team_numbers} if known_team_numbers is not None else None
    region_results = []
    raw_words = []
    all_confidences = []

    for region_name, band in regions:
        candidates = {"red": [], "blue": []}
        for side, (x0, x1) in x_cells.items():
            for row_index, (y0, y1) in enumerate(y_cells):
                left = max(0, int(frame_width * x0))
                right = min(frame_width, int(frame_width * x1))
                top = max(0, int(band_height * y0))
                bottom = min(band_height, int(band_height * y1))
                cell = band.crop((left, top, right, bottom))
                if cell.width == 0 or cell.height == 0:
                    continue

                # The source cell is small; enlarge it while preserving aspect.
                cell = cell.resize((cell.width * 8, cell.height * 8))
                gray = cv2.cvtColor(np.array(cell), cv2.COLOR_RGB2GRAY)
                gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
                variants = [
                    ("gray", gray),
                    ("otsu", cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )[1]),
                ]

                attempts = []
                for variant_name, processed in variants:
                    # PSM 6/7/8 cover the broadcast font's single-line and
                    # single-word interpretations. Do not reject valid IDs
                    # solely because Tesseract reports confidence 0.
                    for psm in (6, 7, 8):
                        data = pytesseract.image_to_data(
                            processed,
                            config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789",
                            output_type=Output.DICT,
                        )
                        for token, conf in zip(data.get("text", []), data.get("conf", [])):
                            token = token.strip()
                            try:
                                confidence = float(conf)
                            except (TypeError, ValueError):
                                confidence = -1.0
                            if not token:
                                continue
                            raw_words.append(
                                f"{region_name}:{side}{row_index + 1}:{variant_name}:psm{psm}:{token}"
                            )
                            if confidence >= 0:
                                all_confidences.append(confidence)
                            digits = re.sub(r"[^0-9]", "", token)
                            if not re.fullmatch(r"\d{3,5}", digits):
                                continue
                            number = int(digits)
                            if roster is not None and number not in roster:
                                continue
                            attempts.append((number, confidence, psm, variant_name))

                if attempts:
                    # Prefer higher confidence, then the more constrained
                    # single-word mode when confidence scores are tied.
                    best = max(attempts, key=lambda item: (item[1], item[2] == 8))
                    candidates[side].append({
                        "number": best[0],
                        "confidence": best[1],
                        "row": row_index,
                    })

        selected = {}
        for side in ("red", "blue"):
            side_items = sorted(
                candidates[side],
                key=lambda item: item["confidence"],
                reverse=True,
            )
            values = []
            for item in side_items:
                if item["number"] not in values:
                    values.append(item["number"])
            selected[side] = values[:2]

        count = len(selected["red"]) + len(selected["blue"])
        confidence = sum(
            item["confidence"]
            for side in candidates.values()
            for item in side
        )
        region_results.append((count, confidence, selected))

    _, _, best_result = max(region_results, key=lambda result: (result[0], result[1]))
    average = round(sum(all_confidences) / len(all_confidences), 1) if all_confidences else 0.0
    return (
        best_result["red"],
        best_result["blue"],
        " ".join(raw_words),
        average,
    )
def _ocr_match_clock(image_path: str) -> tuple[str | None, int | None, float]:
    """Read the large mm:ss countdown in the center of the scoreboard banner."""
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    band_height = max(1, int(height * 0.25))
    regions = [
        image.crop((0, 0, width, band_height)),
        image.crop((0, height - band_height, width, height)),
    ]
    readings = []
    for band in regions:
        # The large countdown sits between the red and blue alliance panels.
        cell = band.crop((
            int(width * 0.445),
            int(band_height * 0.30),
            int(width * 0.555),
            band_height,
        ))
        if cell.width == 0 or cell.height == 0:
            continue
        cell = cell.resize((cell.width * 5, cell.height * 5))
        gray = cv2.cvtColor(np.array(cell), cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        variants = [
            gray,
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        ]
        for processed in variants:
            for psm in (7, 8):
                text = pytesseract.image_to_string(
                    processed,
                    config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789:",
                ).strip()
                match = re.search(r"(?<!\d)(\d{1,2})\s*[:;.,]?\s*(\d{2})(?!\d)", text)
                if not match:
                    continue
                minutes, seconds = int(match.group(1)), int(match.group(2))
                if seconds >= 60:
                    continue
                total_seconds = minutes * 60 + seconds
                if 0 <= total_seconds <= 150:
                    readings.append((total_seconds, text))
    if not readings:
        return None, None, 0.0
    # The center countdown has only one plausible value in the valid range;
    # choose the most frequently recognized value across preprocessing modes.
    counts = {}
    for value, raw in readings:
        counts.setdefault(value, []).append(raw)
    best_value = max(counts, key=lambda value: (len(counts[value]), value))
    raw_text = counts[best_value][0]
    return f"{best_value // 60}:{best_value % 60:02d}", best_value, round(
        len(counts[best_value]) / len(readings) * 100, 1
    )


def run_ocr(frame_rows: list[dict], known_team_numbers: Iterable[int] | None = None,
            progress_callback=None) -> list[dict]:
    results = []
    total = len(frame_rows)
    for index, frame in enumerate(frame_rows, start=1):
        text, confidence = _ocr_text(frame["frame_path"])
        match_clock, match_clock_seconds, match_clock_confidence = _ocr_match_clock(
            frame["frame_path"]
        )
        red_teams, blue_teams, team_raw, team_conf = _ocr_team_numbers_by_side(
            frame["frame_path"], known_team_numbers
        )
        alliance_teams = sorted(set(red_teams + blue_teams))
        results.append({
            **frame,
            "ocr_text": text,
            "ocr_confidence": confidence,
            "match_number_candidate": detect_match_number(text),
            "match_clock": match_clock,
            "match_clock_seconds": match_clock_seconds,
            "match_clock_confidence": match_clock_confidence,
            "red_team_numbers": red_teams,
            "blue_team_numbers": blue_teams,
            "team_number_ocr": team_raw,
            "team_number_ocr_confidence": team_conf,
            "team_number_candidates": alliance_teams,
        })
        if progress_callback:
            progress_callback(index / total, index, total)
    return results

def build_observed_match_windows(ocr_rows: list[dict], interval_seconds: int) -> list[dict]:
    """Group match observations and estimate the actual start from the 2:30 clock.

    For every in-progress clock reading, estimated start = video timestamp -
    (150 seconds - displayed clock). The median estimate reduces OCR jitter.
    """
    windows = []
    current = None

    def finish(item):
        if not item:
            return
        item["observed_until_seconds"] = item["last_seen_seconds"] + interval_seconds
        item["first_seen"] = format_timestamp(item["first_seen_seconds"])
        item["observed_until"] = format_timestamp(item["observed_until_seconds"])
        estimates = item.pop("start_estimates", [])[:5]
        if estimates:
            estimates.sort()
            middle = len(estimates) // 2
            if len(estimates) % 2:
                estimate = estimates[middle]
            else:
                estimate = (estimates[middle - 1] + estimates[middle]) / 2
            item["estimated_start_seconds"] = max(0, round(estimate))
            item["estimated_start"] = format_timestamp(item["estimated_start_seconds"])
            item["start_estimate_samples"] = len(estimates)
            item["boundary_note"] = (
                "Estimated from the 2:30 countdown across in-progress frames; "
                "verify around this timestamp for the exact first motion/clock transition."
            )
        else:
            item["estimated_start_seconds"] = None
            item["estimated_start"] = None
            item["start_estimate_samples"] = 0
            item["boundary_note"] = (
                "No reliable in-progress countdown readings; only the observed scoreboard window is available."
            )
        windows.append(item)

    for row in ocr_rows:
        number = row.get("match_number_candidate")
        if number is None:
            finish(current)
            current = None
            continue

        teams = row.get("team_number_candidates") or []
        if current and current["match_number"] == number:
            current["last_seen_seconds"] = row["timestamp_seconds"]
            current["frames_seen"] += 1
            current["team_numbers_seen"] = sorted(set(current["team_numbers_seen"]) | set(teams))
        else:
            finish(current)
            current = {
                "match_number": number,
                "first_seen_seconds": row["timestamp_seconds"],
                "last_seen_seconds": row["timestamp_seconds"],
                "observed_until_seconds": row["timestamp_seconds"] + interval_seconds,
                "frames_seen": 1,
                "team_numbers_seen": teams,
                "start_estimates": [],
            }

        clock_seconds = row.get("match_clock_seconds")
        if clock_seconds is not None and 0 < int(clock_seconds) < 150:
            elapsed = 150 - int(clock_seconds)
            current["start_estimates"].append(
                float(row["timestamp_seconds"]) - elapsed
            )

    finish(current)
    return windows

def save_ocr_results(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
