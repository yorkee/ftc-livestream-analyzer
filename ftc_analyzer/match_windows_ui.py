import json
import hashlib
from collections import Counter
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pandas as pd
import streamlit as st


def render_match_windows(windows, youtube_url, key):
    st.subheader("Candidate match windows")
    if not windows:
        st.warning("No match numbers were recognized yet.")
        return
    rows = []
    for item in windows:
        start = item.get("start_offset_seconds")
        if start is None:
            start = int(item["first_seen_seconds"])
        base = clean_video_url(youtube_url) if youtube_url else ""
        sep = "&" if "?" in base else "?"
        rows.append({**item, "youtube_link": f"{base}{sep}t={max(0, int(start))}s" if base else item.get("youtube_link")})
    df = pd.DataFrame(rows)
    selection = st.dataframe(df, use_container_width=True, height=320,
        key=key + "_windows", on_select="rerun", selection_mode="single-row",
        column_config={"youtube_link": st.column_config.LinkColumn("YouTube link")})
    if selection.selection.rows:
        link = rows[selection.selection.rows[0]]["youtube_link"]
        if link:
            st.caption("Copy selected match link using the copy icon:")
            st.code(link, language=None)
    st.download_button("Download candidate match windows (CSV)", df.to_csv(index=False),
        file_name="candidate_match_windows.csv", mime="text/csv", key=key + "_csv")
    st.download_button("Download candidate match windows (JSON)", json.dumps(rows, indent=2),
        file_name="windows.json", mime="application/json", key=key + "_json")
    st.caption("Scoreboard-observation windows, not verified match start/end times. The scoreboard may show an upcoming match before play begins.")


def render_saved_windows():
    import json
    import hashlib
    st.caption("Open saved candidate windows without downloading video or running OCR.")
    uploaded = st.file_uploader("Upload windows.json", type=["json"], key="windows_import_file")
    youtube_url = st.text_input("Original YouTube URL (optional for older files)", key="windows_import_url")
    if uploaded is None:
        return
    try:
        raw = uploaded.getvalue()
        windows = json.loads(raw)
        if not isinstance(windows, list) or any(not isinstance(row, dict) for row in windows):
            raise ValueError("Expected a JSON array of candidate match objects.")
        for row in windows:
            start = row.get("start_offset_seconds")
            if start is None:
                start = row.get("first_seen_seconds")
            int(start)
        if not youtube_url and any(not row.get("youtube_link") for row in windows):
            st.info("Enter the original YouTube URL to generate links for older exports.")
        render_match_windows(windows, youtube_url.strip(), "import_" + hashlib.sha256(raw).hexdigest()[:20])
    except (ValueError, TypeError, OverflowError, KeyError) as exc:
        st.error(f"Invalid windows.json: {exc}")


def clean_video_url(url):
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k not in {"t", "start"}])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def load_rows(upload, required):
    rows = json.loads(upload.getvalue())
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not required.issubset(row) for row in rows
    ):
        raise ValueError("Expected a JSON array with fields: " + ", ".join(sorted(required)))
    return rows


def window_stats(windows):
    counts = Counter(row.get("match_number") for row in windows)
    return {
        "Candidate windows": len(windows),
        "With estimated start": sum(row.get("estimated_start_seconds") is not None for row in windows),
        "Without estimated start": sum(row.get("estimated_start_seconds") is None for row in windows),
        "Repeated-number windows": sum(n - 1 for n in counts.values()),
    }


def render_rebuild_windows():
    from ftc_analyzer.video_ocr import build_observed_match_windows

    st.caption("Rebuild from saved OCR observations. No video, Tesseract, FIRST API or database calls.")
    source = st.file_uploader("Upload OCR results.json", type=["json"], key="rebuild_ocr")
    baseline = st.file_uploader("Original windows.json (optional comparison)", type=["json"], key="rebuild_baseline")
    if source is None:
        return
    try:
        rows = load_rows(source, {"timestamp_seconds", "match_number_candidate"})
        if not rows:
            raise ValueError("OCR results are empty.")
        times = [row["timestamp_seconds"] for row in rows]
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t < float("inf") for t in times):
            raise ValueError("Invalid timestamp_seconds.")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("OCR timestamps must be strictly increasing.")
        gaps = Counter(round(b - a) for a, b in zip(times, times[1:]) if b - a >= 1)
        inferred = gaps.most_common(1)[0][0] if gaps else 10
        old = load_rows(baseline, {"match_number", "first_seen_seconds"}) if baseline else None
        urls = {clean_video_url(row["youtube_link"]) for row in (old or []) if row.get("youtube_link")}
        default_url = next(iter(urls)) if len(urls) == 1 else ""
        identity = hashlib.sha256(source.getvalue() + (baseline.getvalue() if baseline else b"")).hexdigest()[:20]
        interval = st.number_input("Frame interval (seconds)", min_value=1, value=inferred, step=1, key="rebuild_interval_" + identity)
        url = st.text_input("Original YouTube URL", value=default_url, key="rebuild_url_" + identity).strip()
        if len(urls) > 1:
            st.warning("Baseline contains multiple video URLs; verify that it belongs to this recording.")
        st.caption(f"{len(rows):,} OCR frames. Detected OCR versions: " + ", ".join(sorted({str(row.get("ocr_version", "legacy")) for row in rows})))
        st.caption("Uses the existing segmentation algorithm. Repeated match numbers may be valid playoff matches or replays.")
        if not url:
            st.info("Enter a YouTube URL to include timestamp links in exports.")
        windows = build_observed_match_windows(rows, int(interval))
        stats = window_stats(windows)
        for col, (label, value) in zip(st.columns(4), stats.items()):
            col.metric(label, value, delta=value - window_stats(old)[label] if old is not None else None, delta_color="off")
        if old is not None:
            st.dataframe(pd.DataFrame({"Original": window_stats(old), "Rebuilt": stats}), use_container_width=True)
            def signature(row):
                return (row.get("match_number"), row.get("first_seen_seconds"), row.get("last_seen_seconds"), row.get("estimated_start_seconds"), tuple(sorted(row.get("team_numbers_seen") or [])))
            old_counts = Counter(map(signature, old))
            new_counts = Counter(map(signature, windows))
            with st.expander("Changed windows (exact observation comparison)"):
                for label, changes in [("Original only", old_counts - new_counts), ("Rebuilt only", new_counts - old_counts)]:
                    st.write(label)
                    st.dataframe(pd.DataFrame([dict(match_number=s[0], first_seen_seconds=s[1], last_seen_seconds=s[2], estimated_start_seconds=s[3], team_numbers_seen=list(s[4])) for s in changes.elements()]), use_container_width=True)
        counts = Counter(row["match_number"] for row in windows)
        review = [row for row in windows if counts[row["match_number"]] > 1 or row.get("estimated_start_seconds") is None or len(row.get("team_numbers_seen") or []) != 4]
        with st.expander(f"Review candidates ({len(review)})"):
            st.caption("Repeated number, missing start estimate or team count other than four. These are review flags, not confirmed duplicates or rejected matches.")
            st.dataframe(pd.DataFrame(review), use_container_width=True)
        render_match_windows(windows, url, "rebuild_" + identity)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        st.error(f"Invalid input: {exc}")


def render_windows_import():
    st.header("Match Windows Viewer")
    view, rebuild = st.tabs(["View saved windows", "Rebuild from OCR"])
    with view:
        render_saved_windows()
    with rebuild:
        render_rebuild_windows()
