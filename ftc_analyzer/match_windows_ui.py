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
        rows.append({**item, "youtube_link": f"{base}{sep}t={max(0, int(start))}s" if base else None})
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
