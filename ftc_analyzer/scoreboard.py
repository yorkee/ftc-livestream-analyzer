"""Color-based FTC scoreboard OCR and conservative temporal validation."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
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


def live_boxes(frame):
    """Discover white clock and paired team cells in the live scoreboard."""
    if frame is None:
        return {}
    h, w = frame.shape[:2]
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (0, 0, 180), (179, 65, 255))
    mask[int(h*.36):int(h*.64)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(2, round(w*.006)), max(2, round(h*.015)))))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > w*h*.001]
    clocks = [(x,y,a,b) for x,y,a,b in boxes if abs(x+a/2-w/2)<w*.045
              and .07*w<a<.16*w and .10*h<b<.23*h and (y+b<h*.36 or y>h*.64)]
    for x,y,a,b in clocks:
        out = {"clock": (x,y,a,b)}
        for side in ("red", "blue"):
            candidates = [(u,v,p,q) for u,v,p,q in boxes if .065*w<p<.19*w
                          and .045*h<q<.14*h and y-.02*h<v<y+b
                          and v+q<y+b+.06*h and (u+p<x if side=="red" else u>x+a)]
            pairs = []
            for first, second in combinations(sorted(candidates,key=lambda z:z[1]),2):
                u,v,p,q = first
                U,V,P,Q = second
                if (abs(u-U)<.025*w and abs(p-P)<.04*w and abs(q-Q)<.045*h
                    and -.008*h<V-v-q<.025*h and abs(V+q-y-b)<.035*h):
                    pairs.append((abs(v-y)+abs(V+Q-y-b),first,second))
            if not pairs:
                break
            _, first, second = min(pairs)
            out[side+"1"] = first
            out[side+"2"] = (second[0],second[1],second[2],min(first[3],second[3]))
        if len(out)==5:
            return out
    return {}


def digit_crop(frame, slot, box):
    x,y,w,h = box
    # Red logo is on the left; blue logo is on the right.
    left, right = (.28,.97) if slot.startswith("red") else (.03,.72)
    pad = max(1, round(h*.06))
    return frame[y+pad:y+h-pad,x+int(w*left):x+int(w*right)]


def read_team_slots(frame, known_team_numbers=None):
    slots = dict.fromkeys(SLOTS)
    boxes = live_boxes(frame)
    if not boxes:
        return slots, "skipped: non-live scoreboard", 0.
    roster = set(known_team_numbers) if known_team_numbers is not None else None
    raw, confidences = [], []
    for slot in SLOTS:
        crop = digit_crop(frame, slot, boxes[slot])
        enlarged = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(enlarged, config="--psm 7 -c tessedit_char_whitelist=0123456789", output_type=Output.DICT)
        token = "".join(t.strip() for t in data["text"] if t.strip())
        confidence = max((float(c) for c in data["conf"] if float(c)>=0), default=0.)
        value = int(token) if re.fullmatch(r"[1-9][0-9]{2,4}", token) else None
        if value is None or (roster is not None and value not in roster):
            value, confidence, extra = _read_digits(crop, roster)
            token += " " + extra
        slots[slot] = value
        raw.append(f"{slot}: {token}")
        if value is not None:
            confidences.append(confidence)
    return slots, " | ".join(raw), round(sum(confidences)/len(confidences),1) if confidences else 0.


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


def stabilize_team_numbers(rows: list[dict], schedule: list[dict] | None = None, *, enable_voting=True) -> list[dict]:
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
            votes = Counter(row.get("team_slots_filtered", row.get("team_slots_raw", {})).get(slot)
                            for row in matches if row.get("team_slots_raw", {}).get(slot) is not None)
            if not votes:
                continue
            best, count = votes.most_common(1)[0]
            if sum(1 for n in votes if votes[n] == count) > 1:
                continue  # Never break a tie arbitrarily.
            if len(votes) == 1 or count >= 2:
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
            individual = {k:v for k,v in row.get("team_slots_filtered", row.get("team_slots_raw", {})).items() if v is not None}
            if not enable_voting:
                official_agreements = sum(individual.get(k)==v for k,v in (official or {}).items())
                conflicts = sum(k in individual and individual[k]!=v for k,v in (official or {}).items())
                resolved = official.copy() if official and official_agreements >= 2 and not conflicts else individual
                source = "first_schedule_verified" if official and resolved == official and official_agreements >= 2 and not conflicts else "single_frame_ocr"
            row["team_slots_voted"] = {slot: (voted if enable_voting else individual).get(slot) for slot in SLOTS}
            row["team_slots"] = {slot: resolved.get(slot) for slot in SLOTS}
            row["red_team_numbers"] = [resolved[slot] for slot in ("red1", "red2") if slot in resolved]
            row["blue_team_numbers"] = [resolved[slot] for slot in ("blue1", "blue2") if slot in resolved]
            row["team_number_candidates"] = sorted(set(resolved.values()))
            row["team_number_source"] = source
            row["teams_complete"] = len(resolved) == 4 and len(set(resolved.values())) == 4
    return rows
