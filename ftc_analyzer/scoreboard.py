"""Color-based FTC scoreboard OCR and conservative temporal validation."""
from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

SLOTS = ("red1", "red2", "blue1", "blue2")


def _panel_candidates(frame: np.ndarray, alliance: str) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if alliance == "red":
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 105, 50), (12, 255, 255)),
            cv2.inRange(hsv, (165, 105, 50), (179, 255, 255)),
        )
    else:
        mask = cv2.inRange(hsv, (85, 90, 40), (135, 255, 255))
    # Exclude most of the field, where robots and jerseys introduce false colors.
    mask[int(height * .35):int(height * .65)] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, round(width * .013)), 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w >= width * .17 and .065 * height <= h <= .35 * height
                and w / h >= 1.6
                and (y + h <= height * .36 or y >= height * .64)):
            panels.append((x, y, w, h))
    return sorted(panels, key=lambda box: box[2] * box[3], reverse=True)


def detect_alliance_panels(frame: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    """Locate aligned red and blue alliance panels at either vertical edge."""
    if frame is None or frame.size == 0:
        return {}
    red = _panel_candidates(frame, "red")
    blue = _panel_candidates(frame, "blue")
    height, width = frame.shape[:2]
    pairs = []
    for r in red:
        for b in blue:
            if abs((r[1] + r[3] / 2) - (b[1] + b[3] / 2)) > .07 * height:
                continue
            if abs(r[3] - b[3]) > .08 * height:
                continue
            overlap = max(0, min(r[0] + r[2], b[0] + b[2]) - max(r[0], b[0]))
            if overlap > .1 * min(r[2], b[2]):
                continue
            if r[2] + b[2] < width * .60:
                continue
            pairs.append((r[2] * r[3] + b[2] * b[3], r, b))
    if not pairs:
        return {}
    _, red_panel, blue_panel = max(pairs, key=lambda pair: pair[0])
    return {"red": red_panel, "blue": blue_panel}


def _read_digits(crop: np.ndarray, roster: set[int] | None) -> tuple[int | None, float, str]:
    if crop.size == 0:
        return None, 0., ""
    enlarged = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(enlarged, cv2.COLOR_BGR2HSV)
    light = cv2.inRange(hsv, (0, 0, 145), (179, 110, 255))
    variants = (gray, cv2.bitwise_not(light))
    attempts = []
    debug = []
    for variant_index, img in enumerate(variants):
        padded = cv2.copyMakeBorder(
            img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255
        )
        for psm in (7, 8):
            data = pytesseract.image_to_data(
                padded, config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789",
                output_type=Output.DICT,
            )
            tokens = [str(t).strip() for t in data.get("text", []) if str(t).strip()]
            raw = "".join(tokens)
            debug.append(f"v{variant_index}:psm{psm}:{raw}")
            if not re.fullmatch(r"[1-9][0-9]{2,4}", raw):
                continue
            number = int(raw)
            if roster is not None and number not in roster:
                continue
            confidence = max(
                (float(c) for c in data.get("conf", []) if float(c) >= 0),
                default=0.,
            )
            attempts.append((number, confidence))
    if not attempts:
        return None, 0., " ".join(debug)
    counts = Counter(num for num, _ in attempts)
    # Confidence can legitimately be zero; agreement across variants wins.
    best = max(counts, key=lambda n: (counts[n], max(c for v, c in attempts if v == n)))
    conf = max(c for n, c in attempts if n == best)
    return best, round(conf, 1), " ".join(debug)


def read_team_slots(frame: np.ndarray, known_team_numbers: Iterable[int] | None = None
                    ) -> tuple[dict[str, int | None], str, float]:
    """Read exactly four slots, preserving red/blue and first/second position."""
    slots: dict[str, int | None] = {key: None for key in SLOTS}
    panels = detect_alliance_panels(frame)
    if not panels:
        return slots, "no aligned red/blue scoreboard panels", 0.
    roster = {int(n) for n in known_team_numbers} if known_team_numbers is not None else None
    observations = []
    confidences = []
    frame_height, frame_width = frame.shape[:2]
    for side in ("red", "blue"):
        x, y, w, h = panels[side]
        # Column positions follow alliance icons, relative to the detected panel.
        left = max(0, round(x + w * .07))
        right = min(frame_width, round(x + w * .205))
        for row, (top_ratio, bottom_ratio) in enumerate(((.045, .49), (.49, .99)), 1):
            top = max(0, round(y + h * top_ratio))
            bottom = min(frame_height, round(y + h * bottom_ratio))
            value, confidence, raw = _read_digits(frame[top:bottom, left:right], roster)
            slot = f"{side}{row}"
            slots[slot] = value
            observations.append(f"{slot}={value or '?'} ({raw})")
            if value is not None:
                confidences.append(confidence)
    mean_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0.
    return slots, " | ".join(observations), mean_conf


def _schedule_slots(schedule: list[dict] | None) -> dict[int, dict[str, int]]:
    """Index unique official matches with all four station assignments."""
    matches: dict[int, list[dict[str, int]]] = {}
    for item in schedule or []:
        try:
            number = int(item.get("matchNumber", item.get("match")))
        except (TypeError, ValueError):
            continue
        slots: dict[str, int] = {}
        for team in item.get("teams") or []:
            station = re.sub(r"[^a-z0-9]", "", str(team.get("station", "")).lower())
            raw_number = team.get("teamNumber", team.get("team"))
            if station not in SLOTS:
                continue
            try:
                slots[station] = int(raw_number)
            except (ValueError, TypeError):
                pass
        if len(slots) == 4 and len(set(slots.values())) == 4:
            matches.setdefault(number, []).append(slots)
    # Repeated match numbers from multiple divisions must not be conflated.
    return {number: options[0] for number, options in matches.items()
            if len(options) == 1 or all(option == options[0] for option in options)}


def stabilize_team_numbers(rows: list[dict], schedule: list[dict] | None = None) -> list[dict]:
    """Vote within consecutive match windows, with optional schedule verification.

    A unique official schedule entry is used only when independent OCR evidence
    agrees. Raw OCR values remain available in team_slots_raw.
    """
    expected = _schedule_slots(schedule)
    groups: list[list[dict]] = []
    active: list[dict] = []
    active_number = None
    blanks = 0
    for row in rows:
        number = row.get("match_number_candidate")
        if number is None:
            if active and blanks < 2:
                active.append(row)
                blanks += 1
            else:
                if active:
                    groups.append(active)
                active = []
                active_number = None
            continue
        if active and number != active_number:
            groups.append(active)
            active = []
        active.append(row)
        active_number = number
        blanks = 0
    if active:
        groups.append(active)

    for group in groups:
        matches = [r for r in group if r.get("match_number_candidate") is not None]
        if not matches:
            continue
        number = matches[0]["match_number_candidate"]
        voted: dict[str, int] = {}
        for slot in SLOTS:
            votes = Counter(row.get("team_slots_raw", {}).get(slot)
                            for row in matches if row.get("team_slots_raw", {}).get(slot) is not None)
            if not votes:
                continue
            best, count = votes.most_common(1)[0]
            if sum(1 for n in votes if votes[n] == count) > 1:
                continue  # Never break a tie arbitrarily.
            if count >= 2 or (len(matches) == 1 and len(votes) == 1):
                voted[slot] = best

        official = expected.get(number)
        agreements = sum(voted.get(slot) == team for slot, team in (official or {}).items())
        disagreements = sum(slot in voted and voted[slot] != team
                            for slot, team in (official or {}).items())
        consistent = agreements >= 1 and (
            disagreements == 0 or (agreements >= 2 and disagreements <= 1)
        )
        if official and consistent:
            resolved = official.copy()
            source = "first_schedule_verified"
        else:
            resolved = voted
            source = "temporal_ocr" if voted else "unresolved"
        for row in matches:
            row["team_slots"] = {slot: resolved.get(slot) for slot in SLOTS}
            row["red_team_numbers"] = [resolved[slot] for slot in ("red1", "red2") if slot in resolved]
            row["blue_team_numbers"] = [resolved[slot] for slot in ("blue1", "blue2") if slot in resolved]
            row["team_number_candidates"] = sorted(set(resolved.values()))
            row["team_number_source"] = source
            row["teams_complete"] = len(resolved) == 4 and len(set(resolved.values())) == 4
    return rows
