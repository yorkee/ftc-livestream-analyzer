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

from ftc_analyzer.scoreboard import read_team_slots, stabilize_team_numbers, live_boxes, safe_ocr, OCR_VERSION


def segment_key(video_id: str, max_height: int, start_seconds: int = 0,
                end_seconds: int | None = None) -> str:
    if start_seconds < 0 or (end_seconds is not None and end_seconds <= start_seconds):
        raise ValueError("Invalid video time range")
    if end_seconds is None and start_seconds != 0:
        raise ValueError("Full video must start at zero")
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", video_id)
    if not safe_id:
        raise ValueError("Missing video ID")
    suffix = "full" if end_seconds is None else f"{start_seconds}-{end_seconds}"
    return f"{safe_id}_{max_height}p_{suffix}"


def download_video(url: str, video_id: str, video_dir: Path, max_height: int = 720,
                   start_seconds: int = 0, end_seconds: int | None = None) -> Path:
    """Download a single-file video stream suitable for local frame extraction."""
    import yt_dlp

    video_id = segment_key(video_id, max_height, start_seconds, end_seconds)
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
    if end_seconds is not None:
        opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(start_seconds, end_seconds)])
        opts["force_keyframes_at_cuts"] = True
    with yt_dlp.YoutubeDL(opts) as ydl:
        status = ydl.download([url])
        if status:
            raise RuntimeError(f"Video download failed: {status}")

    downloaded = [p for p in video_dir.glob(f"{video_id}.*")
                   if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
                   and not p.name.endswith(".part")]
    if not downloaded:
        raise FileNotFoundError("yt-dlp did not produce a video file.")
    return max(downloaded, key=lambda p: p.stat().st_size)


def extract_frames(video_path: Path, frame_dir: Path, interval_seconds: int = 10,
                   max_width: int = 1280, start_offset_seconds: int = 0,
                   duration_seconds: int | None = None) -> list[dict]:
    """Use FFmpeg to save one JPEG every N seconds and return frame metadata."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.jpg"):
        old.unlink()

    pattern = frame_dir / "frame_%08d.jpg"
    scale = f"scale='min({max_width},iw)':-2"
    vf = f"fps=1/{interval_seconds},{scale}"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vf", vf, "-q:v", "3", "-start_number", "0",
        *([] if duration_seconds is None else ["-t", str(duration_seconds)]), str(pattern),
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
            "timestamp_seconds": start_offset_seconds + index * interval_seconds,
            "timestamp": format_timestamp(start_offset_seconds + index * interval_seconds),
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


def _ocr_text(image_path: str, errors=None) -> tuple[str, float]:
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
        data = safe_ocr(pytesseract.image_to_data, crop, errors, "match_text", config="--psm 11", output_type=Output.DICT)
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
                              known_team_numbers: Iterable[int] | None = None, errors=None
                              ) -> tuple[list[int], list[int], str, float, dict]:
    """Find colored scoreboard panels and OCR their four team slots."""
    image = cv2.imread(image_path)
    if image is None:
        return [], [], "could not open frame", 0.0, {}
    slots, raw, confidence = read_team_slots(image, known_team_numbers, errors)
    red = [slots[key] for key in ("red1", "red2") if slots.get(key) is not None]
    blue = [slots[key] for key in ("blue1", "blue2") if slots.get(key) is not None]
    return red, blue, raw, confidence, slots


def _ocr_match_clock(image_path: str, errors=None) -> tuple[str | None, int | None, float]:
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
        for variant_index, processed in enumerate(variants):
            if int(processed.max()) - int(processed.min()) < 8:
                continue
            for psm in (7, 8):
                text = safe_ocr(pytesseract.image_to_string, processed, errors,
                    f"match_clock:variant{variant_index}:psm{psm}",
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
            progress_callback=None, schedule: list[dict] | None = None, *, enable_voting=True, enable_schedule=True) -> list[dict]:
    roster = set(map(int, known_team_numbers)) if known_team_numbers is not None else None
    results = []
    total = len(frame_rows)
    for index, frame in enumerate(frame_rows, start=1):
        boxes = live_boxes(cv2.imread(frame["frame_path"]))
        if not boxes:
            results.append({**frame, "scoreboard_format": "skipped", "ocr_errors": [], "ocr_version": OCR_VERSION, "ocr_text": "",
                "ocr_confidence": 0., "match_number_candidate": None, "match_clock": None,
                "match_clock_seconds": None, "match_clock_confidence": 0.,
                "red_team_numbers": [], "blue_team_numbers": [], "team_number_ocr": "non-live layout",
                "team_number_ocr_confidence": 0., "team_number_candidates": [],
                "team_slots_raw": {}, "team_slots": {}, "team_number_source": "skipped",
                "teams_complete": False})
            if progress_callback:
                progress_callback(index / total, index, total)
            continue
        errors = []
        text, confidence = _ocr_text(frame["frame_path"], errors)
        match_clock, match_clock_seconds, match_clock_confidence = _ocr_match_clock(
            frame["frame_path"], errors
        )
        red_teams, blue_teams, team_raw, team_conf, team_slots = _ocr_team_numbers_by_side(
            frame["frame_path"], None, errors
        )
        raw_slots = team_slots.copy()
        if roster is not None:
            team_slots = {k: v if v in roster else None for k, v in team_slots.items()}
        red_teams = [team_slots[k] for k in ('red1', 'red2') if team_slots.get(k) is not None]
        blue_teams = [team_slots[k] for k in ('blue1', 'blue2') if team_slots.get(k) is not None]
        alliance_teams = sorted(set(red_teams + blue_teams))
        results.append({
            **frame,
            "scoreboard_format": "live",
            "ocr_errors": errors,
            "ocr_version": OCR_VERSION,
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
            "team_slots_raw": raw_slots,
            "team_slots_filtered": team_slots.copy(),
            "team_slots": team_slots.copy(),
            "team_number_source": "single_frame_ocr",
            "teams_complete": len(alliance_teams) == 4 and len(red_teams) == 2 and len(blue_teams) == 2,
        })
        if progress_callback:
            progress_callback(index / total, index, total)
    if enable_voting or (enable_schedule and schedule):
        return stabilize_team_numbers(results, schedule if enable_schedule else None, enable_voting=enable_voting)
    return results

MATCH_WINDOW_VERSION = "2"

def build_observed_match_windows(ocr_rows: list[dict], interval_seconds: int) -> list[dict]:
    """Consolidate nearby observations using match identity and teleop anchors."""
    from collections import Counter
    from statistics import median

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    rows = sorted(ocr_rows, key=lambda r: r["timestamp_seconds"])
    gap = max(60, 3 * interval_seconds)

    def identity(row):
        text = row.get("ocr_text") or ""
        phase = ("playoff" if re.search(r"\b(?:playoff|final|semifinal)\b", text, re.I)
                 else "qualification" if re.search(r"\b(?:qualification|qualifying|qual)\b", text, re.I)
                 else None)
        return row.get("match_number_candidate"), phase

    def teams(row):
        return set(row.get("red_team_numbers") or []) | set(row.get("blue_team_numbers") or [])

    def anchor(row):
        clock = row.get("match_clock_seconds")
        if clock is not None and 45 <= clock <= 75 and len(teams(row)) == 4:
            return row["timestamp_seconds"] - (158 - clock)
        return None

    groups = []
    for row in rows:
        number, phase = identity(row)
        if number is None:
            continue
        t = row["timestamp_seconds"]
        estimate = anchor(row)
        group = groups[-1] if groups else None
        compatible = False
        if group:
            previous = group[-1]
            pn, pp = identity(previous)
            compatible = (number == pn and not (phase and pp and phase != pp)
                          and t - previous["timestamp_seconds"] <= gap)
            estimates = [anchor(x) for x in group if anchor(x) is not None]
            if compatible and estimate is not None and estimates:
                compatible = abs(estimate - median(estimates)) <= max(20, interval_seconds * 2)
            if compatible and len(teams(row)) == len(teams(previous)) == 4:
                compatible = len(teams(row) & teams(previous)) >= 2
        if compatible:
            group.append(row)
        else:
            groups.append([row])

    # Bridge an isolated incorrect label between two compatible fragments.
    i = 1
    while i + 1 < len(groups):
        left, middle, right = groups[i-1:i+2]
        ln, lp = identity(left[-1])
        rn, rp = identity(right[0])
        if (len(middle) == 1 and anchor(middle[0]) is None and ln == rn
                and not (lp and rp and lp != rp)
                and right[0]["timestamp_seconds"] - left[-1]["timestamp_seconds"] <= gap
                and len(teams(left[-1]) & teams(right[0])) >= 3):
            left.extend(right)
            del groups[i:i+2]
        else:
            i += 1

    windows = []
    for group in groups:
        number = identity(group[0])[0]
        phases = Counter(p for _, p in map(identity, group) if p)
        phase = phases.most_common(1)[0][0] if phases else None
        first, last = group[0]["timestamp_seconds"], group[-1]["timestamp_seconds"]
        # Recover unlabeled observations inside the window only when teams agree.
        votes = Counter(t for row in group for t in teams(row))
        consensus = {t for t, _ in votes.most_common(4)}
        recovered = [r for r in rows if first <= r["timestamp_seconds"] <= last
                     and identity(r)[0] is None and len(teams(r) & consensus) >= 3]
        observations = group + recovered
        votes = Counter(t for row in observations for t in teams(row))
        consensus = sorted(t for t, count in votes.most_common(4)
                           if count * 2 >= len(observations))
        estimates = [anchor(r) for r in observations if anchor(r) is not None
                     and len(teams(r) & set(consensus)) >= 3]
        center = median(estimates) if estimates else None
        estimates = [e for e in estimates if abs(e-center) <= max(20, interval_seconds*2)]
        start = max(0, round(median(estimates))) if estimates else None
        clocks = sorted((r["timestamp_seconds"], r["match_clock_seconds"])
                        for r in observations if r.get("match_clock_seconds") is not None)
        moving = any(0 < c1-c2 and abs((c1-c2)-(t2-t1)) <= max(4, interval_seconds)
                     for (t1,c1),(t2,c2) in zip(clocks, clocks[1:]))
        review = start is None or len(consensus) != 4 or not moving
        windows.append({
            "match_number": number, "match_phase": phase,
            "match_window_version": MATCH_WINDOW_VERSION,
            "first_seen_seconds": first, "last_seen_seconds": last,
            "observed_until_seconds": last + interval_seconds,
            "first_seen": format_timestamp(first),
            "observed_until": format_timestamp(last + interval_seconds),
            "frames_seen": len(observations), "recovered_unlabeled_frames": len(recovered),
            "team_numbers_seen": consensus,
            "team_number_votes": dict(votes),
            "estimated_start_seconds": start,
            "estimated_start": format_timestamp(start) if start is not None else None,
            "start_offset_seconds": start,
            "start_offset": format_timestamp(start) if start is not None else None,
            "start_estimate_samples": len(estimates),
            "countdown_progression": moving,
            "needs_review": review,
            "boundary_note": ("Teleop consensus: timestamp - (158 - clock)."
                              if start is not None else "No reliable teleop anchor; review required."),
        })
    return windows

def save_ocr_results(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
