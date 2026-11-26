import streamlit as st
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
    st.dataframe([{k: job.get(k) for k in fields} for job in jobs], use_container_width=True,
                 column_config={"date_start": "Event start", "date_end": "Event end"})
    by_id = {job["id"]: job for job in jobs}
    selected = st.selectbox("Select job", list(by_id),
                            format_func=lambda i: f"#{i} · {by_id[i]['event_code']} · {by_id[i]['status']}")
    job = by_id[selected]
    if job.get("result_summary"):
        st.json(job["result_summary"])
    st.caption("Full recording; raw OCR defaults. Sampling and resolution use the sidebar settings.")
    confirm = st.checkbox("Confirm manual processing of this full recording", key=f"queue_confirm_{selected}")
    if st.button("Process selected job", disabled=not confirm or job["status"] != "pending" or job["attempts"] >= 5):
        bar = st.progress(0, text="Claiming job")
        try:
            process_job(queue, job, data_dir, int(interval), int(max_height),
                        lambda p, message: bar.progress(p, text=message))
            st.success("Completed. Refresh to see the updated queue.")
        except Exception as exc:
            st.error(f"Processing stopped: {type(exc).__name__}. Check local logs and refresh.")
    retry = st.checkbox("Confirm manual retry", key=f"queue_retry_{selected}")
    if st.button("Requeue failed job", disabled=not retry or job["status"] != "failed" or job["attempts"] >= 5):
        try:
            queue.retry(job)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
