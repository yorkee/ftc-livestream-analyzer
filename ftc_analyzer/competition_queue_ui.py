import json
from pathlib import Path
import streamlit as st
from .match_windows_ui import render_match_windows
from .competition_queue import CompetitionQueue, process_job


def render_competition_queue(data_dir, interval, max_height):
    st.header("Competition Queue")
    st.caption("Verified recordings queued by the directory. Processing and retries are manual; results remain local.")
    status = st.selectbox("Job status", ["All", "pending", "processing", "completed", "failed"])
    page = st.number_input("Queue page", min_value=1, value=1, step=1)
    st.button("Refresh queue")
    try:
        queue = CompetitionQueue()
        jobs = queue.list_jobs(None if status == "All" else status, offset=(page - 1) * 200)
    except Exception as exc:
        st.warning(str(exc))
        return
    if not jobs:
        st.info("No jobs on this page.")
        return
    fields = ("id", "season_year", "event_code", "event_name", "date_start", "date_end", "video_url", "status",
              "priority", "attempts", "progress_percent", "lease_expires_at", "last_error")
    st.caption("Select one competition recording from the table.")
    # Reset positional selection when the visible job IDs or ordering change.
    table_key = "queue_table_" + "_".join(str(j["id"]) for j in jobs)
    selection = st.dataframe([{k: job.get(k) for k in fields} for job in jobs],
                 use_container_width=True, key=table_key, on_select="rerun",
                 selection_mode="single-row",
                 column_config={"date_start": "Event start", "date_end": "Event end"})
    if not selection.selection.rows:
        return
    job = jobs[selection.selection.rows[0]]
    selected = job["id"]
    st.write(f"Selected: #{selected} · {job['event_name']}")
    if job.get("result_summary"):
        st.json(job["result_summary"])
    def show_results():
        folder = Path(data_dir) / "competition_queue" / str(selected) / str(job["attempts"]) / "ocr"
        path = folder / "windows.json"
        if not path.is_file():
            st.info("Candidate match results are not available on this computer.")
            return
        try:
            windows = json.loads(path.read_text(encoding="utf-8"))
            render_match_windows(windows, job["video_url"], f"queue_{selected}_{job['attempts']}")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.warning(f"Could not read candidate match windows: {type(exc).__name__}")
    if job["status"] == "completed":
        show_results()
    st.caption("Full recording; raw OCR defaults. Sampling and resolution use the sidebar settings.")
    confirm = st.checkbox("Confirm manual processing of this full recording", key=f"queue_confirm_{selected}")
    if st.button("Process selected job", disabled=not confirm or job["status"] != "pending" or job["attempts"] >= 5):
        bar = st.progress(0, text="Claiming job")
        try:
            process_job(queue, job, data_dir, int(interval), int(max_height),
                        lambda p, message: bar.progress(p, text=message))
            st.rerun()
        except Exception as exc:
            st.error(f"Processing stopped: {type(exc).__name__}. Check local logs and refresh.")
    retry = st.checkbox("Confirm manual retry", key=f"queue_retry_{selected}")
    if st.button("Requeue failed job", disabled=not retry or job["status"] != "failed" or job["attempts"] >= 5):
        try:
            queue.retry(job)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
