import re

# Deliberately conservative patterns: only explicit match references are candidates.
PATTERNS = [
    re.compile(r"\b(?:qualification|qualifying|qual)\s+(?:match\s+)?(?:number\s+)?(\d{1,3})\b", re.I),
    re.compile(r"\bmatch\s+(?:number\s+)?(\d{1,3})\b", re.I),
    re.compile(r"\bq\s*[-#]?\s*(\d{1,3})\b", re.I),
]

def detect_match_mentions(transcript_rows):
    mentions = []
    for row in transcript_rows:
        text = row.get("text", "")
        found = []
        for pattern in PATTERNS:
            found.extend(int(m.group(1)) for m in pattern.finditer(text))
        for number in sorted(set(found)):
            mentions.append({
                "match_number": number,
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "transcript_text": text,
                "detection_method": "spoken_match_number",
                "confidence": "medium",
            })
    # Collapse repeated mentions of the same match number within 90 seconds.
    collapsed = []
    for item in sorted(mentions, key=lambda x: x["start_seconds"]):
        previous = next((p for p in reversed(collapsed)
                         if p["match_number"] == item["match_number"]), None)
        if previous and item["start_seconds"] - previous["start_seconds"] < 90:
            continue
        collapsed.append(item)
    return collapsed

def _extract_teams(match):
    teams = []
    for team in match.get("teams") or []:
        number = team.get("teamNumber")
        if number is None:
            number = team.get("team")
        if number is not None:
            try:
                teams.append(int(number))
            except (TypeError, ValueError):
                pass
    return teams

def _match_number(match):
    value = match.get("matchNumber")
    if value is None:
        value = match.get("match")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def map_mentions_to_schedule(mentions, schedule, youtube_url):
    from urllib.parse import urlparse, parse_qs

    base = youtube_url.split("&t=")[0].split("?t=")[0]
    sep = "&" if "?" in base else "?"

    by_number = {}
    for match in schedule:
        number = _match_number(match)
        if number is not None:
            by_number.setdefault(number, []).append(match)

    results = []
    for mention in mentions:
        candidates = by_number.get(mention["match_number"], [])
        if not candidates:
            results.append({
                **mention,
                "tournament_level": None,
                "team_number": None,
                "youtube_url": f"{base}{sep}t={int(mention['start_seconds'])}s",
                "mapping_status": "no_schedule_match",
            })
            continue
        # MVP selects all matching schedule entries, preserving ambiguity if a
        # tournament's match number repeats across sub-divisions/fields.
        for match in candidates:
            level = match.get("tournamentLevel") or match.get("level") or "unknown"
            timestamp = int(mention["start_seconds"])
            link = f"{base}{sep}t={timestamp}s"
            teams = _extract_teams(match)
            if not teams:
                results.append({
                    **mention,
                    "tournament_level": level,
                    "team_number": None,
                    "youtube_url": link,
                    "mapping_status": "schedule_entry_has_no_team_list",
                })
            for team_number in teams:
                results.append({
                    **mention,
                    "tournament_level": level,
                    "team_number": team_number,
                    "youtube_url": link,
                    "mapping_status": "candidate_needs_video_verification",
                })
    return results
