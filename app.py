import os
import hashlib
import re
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import pandas as pd
import streamlit as st
import yt_dlp
from dotenv import load_dotenv

from ftc_analyzer.first_api import fetch_event_schedule
from ftc_analyzer.scoreboard import live_boxes, digit_crop, read_team_slots, SLOTS
from ftc_analyzer.frame_import import import_frame_folder
from ftc_analyzer.video_ocr import (
    download_video, extract_frames, run_ocr, build_observed_match_windows,
    save_ocr_results, format_timestamp, segment_key,
)

load_dotenv()

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
FRAME_DIR = DATA_DIR / "frames"
OCR_DIR = DATA_DIR / "ocr"
for folder in (DATA_DIR, VIDEO_DIR, FRAME_DIR, OCR_DIR):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="FTC Livestream Analyzer", page_icon="🤖", layout="wide")
st.title("FTC Livestream Analyzer")
st.caption("Local MVP · archived YouTube livestreams · video-frame scoreboard OCR")

with st.sidebar:
    st.header("FIRST API credentials")
    first_username = st.text_input("FIRST API username", value=os.getenv("FIRST_API_USERNAME", ""))
    first_token = st.text_input("FIRST API authorization key", value=os.getenv("FIRST_API_TOKEN", ""), type="password")
    st.caption("Credentials are used only by this local app. Do not share your .env file.")
    st.divider()
    st.header("Frame sampling")
    interval = st.select_slider("Capture one frame every", options=[1, 2, 5, 10, 15, 30], value=10, format_func=lambda x: f"{x} seconds")
    max_height = st.selectbox("Maximum video resolution", [480, 720, 1080], index=1)
    st.caption("10 seconds is the initial default. This downloads video and samples frames; audio transcription is not used.")

from ftc_analyzer.competition_queue_ui import render_competition_queue
from ftc_analyzer.match_windows_ui import render_match_windows, render_windows_import

manual_tab, queue_tab, viewer_tab = st.tabs(["Manual Analyzer", "Competition Queue", "Match Windows Viewer"])
with viewer_tab:
    render_windows_import()
with queue_tab:
    render_competition_queue(DATA_DIR, interval, max_height)
with manual_tab:
    st.subheader("1. Video and event")
    source = st.radio("Video source", ["YouTube", "Local video file", "Captured frame folder"], horizontal=True)
    local_path = None
    local_offset = 0
    frame_folder = None
    if source == "YouTube":
        youtube_url = st.text_input("Archived YouTube livestream URL", placeholder="https://www.youtube.com/watch?v=...")
    elif source == "Local video file":
        youtube_url = st.text_input("Original YouTube URL (optional, for match links)")
        method = st.radio("Local file input", ["Drag and drop", "Existing file path"], horizontal=True)
        if method == "Drag and drop":
            upload = st.file_uploader("Drop a downloaded video here", type=["mp4","webm","mkv","mov","avi","m4v"])
            if upload is not None:
                buffer = upload.getbuffer()
                digest = hashlib.sha256(buffer).hexdigest()
                local_path = VIDEO_DIR / f"upload_{digest}{Path(upload.name).suffix.lower()}"
                if not local_path.exists() or local_path.stat().st_size != upload.size:
                    temporary = local_path.with_suffix(".part")
                    with temporary.open("wb") as handle:
                        handle.write(buffer)
                    temporary.replace(local_path)
        else:
            entered = st.text_input("Existing video path", placeholder="/home/user/Downloads/match.mp4")
            if entered:
                candidate = Path(entered).expanduser().resolve()
                if candidate.is_file() and candidate.suffix.lower() in {".mp4",".webm",".mkv",".mov",".avi",".m4v"}:
                    local_path = candidate
                else:
                    st.warning("Enter a valid local video file path.")
        local_offset = round(st.number_input("Original livestream offset (seconds)", min_value=0, value=0, step=1))
        st.caption("For a previously downloaded segment, enter its original start time. For a complete video, leave this at zero. Large files are best opened by path, avoiding upload limits and copying.")

    else:
        youtube_url = st.text_input("Original YouTube URL (optional, for match links)")
        folders = sorted(p for p in FRAME_DIR.iterdir() if p.is_dir())
        selected = st.selectbox("Previously captured folder", ["Enter another folder"] + [str(p) for p in folders])
        entered = st.text_input("Frame folder path", placeholder="data/frames/...") if selected == "Enter another folder" else selected
        if entered:
            candidate = Path(entered).expanduser().resolve()
            if candidate.is_dir():
                frame_folder = candidate
            else:
                st.warning("Folder does not exist.")
        interval = int(st.number_input("Original frame interval (seconds)", min_value=1, value=int(interval), step=1))
        local_offset = int(st.number_input("Original livestream offset (seconds)", min_value=0, value=0, step=1))
        st.caption("Use the original capture interval and segment start. Numbered frame filenames preserve gaps; no images are copied or modified.")

    c1, c2, c3 = st.columns(3)
    season = c1.number_input("FTC season", min_value=2023, max_value=2035, value=2026, step=1)
    event_code = c2.text_input("FIRST event code", placeholder="e.g. MIxxxx")
    tournament = c3.selectbox("Tournament level", ["qual", "playoff"], index=0)


    def get_video_id(url: str) -> str:
        parsed = urlparse(url)
        if parsed.hostname in ("youtu.be", "www.youtu.be"):
            return parsed.path.strip("/")
        if parsed.hostname and "youtube.com" in parsed.hostname:
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [""])[0]
            if parsed.path.startswith(("/live/", "/shorts/")):
                return parsed.path.split("/")[2]
        return re.sub(r"[^A-Za-z0-9_-]", "", url)[-20:] or "video"


    def get_video_info(url: str):
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            return ydl.extract_info(url, download=False)


    def yt_timestamp_url(url: str, seconds: int) -> str:
        base = url.split("&t=")[0].split("?t=")[0]
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}t={max(0, int(seconds))}s"


    def schedule_team_numbers(schedule: list[dict]) -> set[int]:
        numbers = set()
        for match in schedule:
            for team in match.get("teams") or []:
                value = team.get("teamNumber", team.get("team"))
                try:
                    if value is not None:
                        numbers.add(int(value))
                except (TypeError, ValueError):
                    continue
        return numbers


    if "video_info" not in st.session_state:
        st.session_state.video_info = None
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "frame_rows" not in st.session_state:
        st.session_state.frame_rows = None
    if "ocr_rows" not in st.session_state:
        st.session_state.ocr_rows = None
    if "match_windows" not in st.session_state:
        st.session_state.match_windows = None

    if source == "YouTube" and st.button("Inspect YouTube video", type="secondary", disabled=not youtube_url):
        try:
            with st.spinner("Reading YouTube metadata..."):
                info = get_video_info(youtube_url)
            st.session_state.video_info = info
            st.session_state.video_url = youtube_url
            st.success("Video metadata retrieved.")
        except Exception as exc:
            st.error(f"Could not inspect this video: {exc}")

    info = st.session_state.video_info
    if info and source == "YouTube" and st.session_state.get("video_url") == youtube_url:
        st.markdown(f"**{info.get('title', 'Untitled')}**")
        m1, m2, m3 = st.columns(3)
        duration = int(info.get("duration") or 0)
        m1.metric("Duration", f"{duration // 3600}h {(duration % 3600) // 60}m")
        m2.metric("Channel", info.get("channel") or info.get("uploader") or "Unknown")
        m3.metric("Video ID", info.get("id", "Unknown"))

    st.divider()
    st.subheader("2. Download video and capture frames")
    mode = st.radio("Download mode", ["Quick test", "Custom range", "Full video"], horizontal=True) if source == "YouTube" else "Full video"
    start_seconds = 0
    end_seconds = None
    if mode != "Full video":
        c_start, c_length = st.columns(2)
        start_minutes = c_start.number_input("Start (minutes into livestream)", min_value=0.0, value=0.0, step=0.5)
        length_minutes = c_length.number_input("Duration (minutes)", min_value=0.5, value=5.0, step=0.5)
        start_seconds = round(start_minutes * 60)
        end_seconds = start_seconds + round(length_minutes * 60)
        st.caption(f"Selected segment: {format_timestamp(start_seconds)} to {format_timestamp(end_seconds)}. Only this section is requested from YouTube.")
    elif source == "YouTube":
        st.warning("Full livestream downloads may take a long time and use several GB.")
    if source == "Captured frame folder":
        key = "frames_" + hashlib.sha256(f"{frame_folder}:{local_offset}".encode()).hexdigest()[:24] if frame_folder else None
    elif source == "Local video file":
        start_seconds = local_offset
        if local_path:
            stat = local_path.stat()
            identity = f"{local_path}:{stat.st_size}:{stat.st_mtime_ns}:{local_offset}"
            key = "local_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        else:
            key = None
    else:
        key = segment_key(get_video_id(youtube_url), int(max_height), start_seconds, end_seconds) if youtube_url else None
    # Do not show or process frames from a different video, range or interval.
    selection = (key, int(interval))
    if st.session_state.get("selection") != selection:
        st.session_state.selection = selection
        st.session_state.video_path = None
        st.session_state.frame_rows = None
        st.session_state.ocr_rows = None
        st.session_state.match_windows = None
    st.info(f"Sampling every {interval} seconds. Cached downloads are reused automatically.")
    if source == "Local video file":
        st.session_state.video_path = str(local_path) if local_path else None
    if source == "YouTube" and st.button("Download selected video", type="primary", disabled=not youtube_url):
        try:
            with st.spinner("Downloading selected video range..."):
                path = download_video(youtube_url, get_video_id(youtube_url), VIDEO_DIR,
                                      int(max_height), start_seconds, end_seconds)
            st.session_state.video_path = str(path)
            st.success(f"Video ready: {path.name}")
        except Exception as exc:
            st.exception(exc)

    if st.session_state.video_path:
        st.caption(f"Ready: {st.session_state.video_path}")
    if source == "Captured frame folder":
        if st.button("Import captured frames", disabled=frame_folder is None):
            try:
                with st.spinner("Reading existing images..."):
                    rows, skipped = import_frame_folder(frame_folder, interval, local_offset)
                st.session_state.frame_rows = rows
                st.session_state.ocr_rows = None
                st.session_state.match_windows = None
                st.success(f"Imported {len(rows):,} frames without copying.")
                if skipped:
                    st.warning(f"Skipped {skipped} unreadable images.")
            except Exception as exc:
                st.exception(exc)
    elif st.button("Capture frames", disabled=not st.session_state.video_path):
        try:
            frame_dir = FRAME_DIR / f"{key}_{interval}s"
            with st.spinner("Extracting frames..."):
                frames = extract_frames(Path(st.session_state.video_path), frame_dir, int(interval),
                                        start_offset_seconds=start_seconds,
                                        duration_seconds=None if end_seconds is None else end_seconds-start_seconds)
            st.session_state.frame_rows = frames
            st.session_state.ocr_rows = None
            st.session_state.match_windows = None
            st.success(f"Captured {len(frames):,} frames.")
        except Exception as exc:
            st.exception(exc)

    frame_rows = st.session_state.frame_rows
    if frame_rows:
        st.subheader("Captured frame preview")
        preview_count = min(12, len(frame_rows))
        preview_indices = sorted(set(round(i * (len(frame_rows) - 1) / max(1, preview_count - 1)) for i in range(preview_count)))
        cols = st.columns(3)
        for i, frame_index in enumerate(preview_indices):
            row = frame_rows[frame_index]
            image = cv2.imread(row["frame_path"])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                with cols[i % 3]:
                    st.image(image, caption=f"{row['timestamp']} · frame {frame_index}", use_container_width=True)
        st.caption("Preview is sampled across the recording. Check that the match scoreboard is visible and readable before running OCR.")

    st.divider()
    st.subheader("3. Read scoreboard text with OCR")
    filter_roster = st.checkbox("Enable FIRST roster filtering", value=False)
    vote_frames = st.checkbox("Enable multi-frame voting", value=False)
    correct_schedule = st.checkbox("Enable FIRST schedule correction", value=False)
    show_raw = st.checkbox("Show raw OCR results", value=True)
    show_crops = st.checkbox("Show detected crop regions", value=True)
    if (filter_roster or correct_schedule) and not (event_code and first_username and first_token):
        st.warning("FIRST validation requires an event code and API credentials.")

    if frame_rows and show_crops:
        with st.expander("Inspect a single frame / OCR crops"):
            idx = st.selectbox("Frame", range(len(frame_rows)), format_func=lambda i: f"{frame_rows[i]['frame_index']} · {frame_rows[i]['timestamp']}")
            if st.button("Inspect selected frame"):
                image = cv2.imread(frame_rows[idx]["frame_path"])
                boxes = live_boxes(image)
                if not boxes:
                    st.warning("Skipped: no live-match scoreboard detected.")
                else:
                    slots, raw, confidence = read_team_slots(image)
                    preview = image.copy()
                    for label, (x,y,w,h) in boxes.items():
                        cv2.rectangle(preview,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.putText(preview,label,(x,y+12),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,128,0),1)
                    st.image(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB))
                    for col, slot in zip(st.columns(4), SLOTS):
                        col.image(cv2.cvtColor(digit_crop(image,slot,boxes[slot]),cv2.COLOR_BGR2RGB),caption=f"{slot}: {slots[slot]}")
                    st.write(raw)
                    st.caption(f"Mean OCR confidence: {confidence}")

    if st.button("Run OCR on captured frames", disabled=not frame_rows):
        try:
            known_teams = None
            schedule = None
            if (filter_roster or correct_schedule) and event_code and first_username and first_token:
                with st.spinner("Loading the FIRST event schedule to validate recognized team numbers..."):
                    schedule = fetch_event_schedule(int(season), event_code.strip(), tournament,
                                                    first_username.strip(), first_token.strip())
                known_teams = (schedule_team_numbers(schedule) or None) if filter_roster else None
                if filter_roster and known_teams is None:
                    st.warning('Empty FIRST roster; filtering disabled.')
                st.session_state.schedule = schedule
            progress = st.progress(0, text="Starting OCR...")
            def update_progress(fraction, current, total):
                progress.progress(fraction, text=f"Reading frame {current:,} of {total:,}...")
            with st.spinner("Running Tesseract OCR on each captured frame..."):
                ocr_rows = run_ocr(frame_rows, known_teams, update_progress, schedule=schedule, enable_voting=vote_frames, enable_schedule=correct_schedule)
            st.session_state.ocr_rows = ocr_rows
            st.session_state.match_windows = build_observed_match_windows(ocr_rows, int(interval))
            export_key = f"{key}_{interval}s"
            save_ocr_results(ocr_rows, OCR_DIR / f"{export_key}.json", OCR_DIR / f"{export_key}.csv")
            st.success(f"OCR complete: processed {len(ocr_rows):,} frames. Results saved under data/ocr/.")
        except Exception as exc:
            st.exception(exc)

    ocr_rows = st.session_state.ocr_rows
    if ocr_rows:
        st.subheader("OCR results")
        ocr_df = pd.DataFrame(ocr_rows)
        display_columns = ["timestamp", "scoreboard_format", "match_number_candidate", "match_clock", "match_clock_confidence", "red_team_numbers", "blue_team_numbers", "team_slots", "team_number_source", "teams_complete", "ocr_confidence", "team_number_ocr_confidence", "team_number_candidates", "ocr_text", "team_number_ocr"]
        if show_raw:
            display_columns = ["frame_index", "team_slots_raw", "team_slots_filtered", "team_slots_voted"] + display_columns
        st.dataframe(ocr_df.reindex(columns=display_columns), use_container_width=True, height=420)
        st.caption("Team numbers are read from dynamically detected red/blue overlay panels at the top or bottom of a frame. Temporal voting and optional FIRST match-schedule verification are reflected in team_number_source; team_slots_raw retains uncorrected reads in exports.")
        st.download_button("Download OCR results (CSV)", ocr_df.to_csv(index=False),
                           file_name="ftc_ocr_results.csv", mime="text/csv")
        st.download_button("Download OCR results (JSON)", json.dumps(ocr_rows, indent=2),
                           file_name="ftc_ocr_results.json", mime="application/json")

        render_match_windows(st.session_state.match_windows or [], youtube_url, "manual")

    with st.expander("Troubleshooting and limitations"):
        st.markdown("""
        - This version analyzes video frames, not audio. Whisper and PyAV are no longer required.
        - Install the system OCR engine with `sudo apt install tesseract-ocr` and ensure `ffmpeg` is installed.
        - The default is one frame every 10 seconds; 5, 15, and 30 seconds are also available.
        - Downloading a multi-hour video can require substantial disk space and time. The selected maximum resolution limits the stream when YouTube offers a suitable format.
        - OCR output is raw recognition. Different events use different scoreboard layouts, so text recognition and team-number parsing may need tuning for each stream.
        - Match windows show when a scoreboard label was observed. They are not exact play boundaries until we add visual match-state/clock transition detection.
        - Use only videos you are authorized to access. YouTube availability and download restrictions can vary.
        """)
        st.markdown("FIRST API information: https://ftc-events.firstinspires.org/services/API")
