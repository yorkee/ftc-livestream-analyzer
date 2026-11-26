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
        base = youtube_url.split("&t=")[0].split("?t=")[0] if youtube_url else ""
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
    st.caption("Scoreboard-observation windows, not verified match start/end times. The scoreboard may show an upcoming match before play begins.")


def render_windows_import():
    import json
    import hashlib
    st.header("Match Windows Viewer")
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
