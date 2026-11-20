import os
import re
import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel
from dotenv import load_dotenv

from ftc_analyzer.first_api import fetch_event_schedule
from ftc_analyzer.match_detection import detect_match_mentions, map_mentions_to_schedule

load_dotenv()

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
for folder in (DATA_DIR, AUDIO_DIR, TRANSCRIPT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="FTC Livestream Analyzer", page_icon="🤖", layout="wide")
st.title("FTC Livestream Analyzer")
st.caption("Local MVP · archived YouTube livestreams · audio-first match detection")

with st.sidebar:
    st.header("FIRST API credentials")
    first_username = st.text_input("FIRST API username", value=os.getenv("FIRST_API_USERNAME", ""))
    first_token = st.text_input("FIRST API authorization key", value=os.getenv("FIRST_API_TOKEN", ""), type="password")
    st.caption("Credentials are used only by this local app. Do not share your .env file.")

st.subheader("1. Video and event")
youtube_url = st.text_input("Archived YouTube livestream URL", placeholder="https://www.youtube.com/watch?v=...")
c1, c2, c3 = st.columns(3)
season = c1.number_input("FTC season", min_value=2023, max_value=2035, value=2026, step=1)
event_code = c2.text_input("FIRST event code", placeholder="e.g. MIxxxx")
tournament = c3.selectbox("Tournament level", ["qual", "playoff"], index=0)

st.subheader("2. Transcription settings")
c4, c5, c6 = st.columns(3)
model_name = c4.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=2,
                          help="Start with small. Larger models are slower but may transcribe announcements more accurately.")
compute_type = c5.selectbox("Compute type", ["int8", "float32"], index=0)
language = c6.selectbox("Audio language", ["Auto-detect", "English"], index=0)
st.caption("The first run downloads the selected Whisper model. Processing a multi-hour recording can take a while, especially on CPU.")

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
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)

def extract_audio(url: str, video_id: str) -> Path:
    output = AUDIO_DIR / f"{video_id}.%(ext)s"
    expected = AUDIO_DIR / f"{video_id}.mp3"
    if expected.exists() and expected.stat().st_size > 0:
        return expected
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output),
        "noplaylist": True,
        "quiet": False,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "96",
        }],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not expected.exists():
        candidates = list(AUDIO_DIR.glob(f"{video_id}.*"))
        if not candidates:
            raise FileNotFoundError("yt-dlp did not produce an audio file.")
        # Convert any downloaded audio to a predictable MP3 using FFmpeg.
        source = candidates[0]
        subprocess.run(["ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1",
                        "-ar", "16000", str(expected)], check=True, capture_output=True)
    return expected

def transcribe_audio(audio_path: Path, model_name: str, compute_type: str, language_choice: str):
    model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
    kwargs = {"beam_size": 5, "vad_filter": True}
    if language_choice == "English":
        kwargs["language"] = "en"
    segments, info = model.transcribe(str(audio_path), **kwargs)
    rows = []
    for seg in segments:
        rows.append({
            "start_seconds": round(float(seg.start), 2),
            "end_seconds": round(float(seg.end), 2),
            "text": seg.text.strip(),
        })
    return rows, info

def yt_timestamp_url(url: str, seconds: int) -> str:
    base = url.split("&t=")[0].split("?t=")[0]
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}t={max(0, int(seconds))}s"

if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "transcript_rows" not in st.session_state:
    st.session_state.transcript_rows = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "mapped_rows" not in st.session_state:
    st.session_state.mapped_rows = None

if st.button("Inspect YouTube video", type="secondary", disabled=not youtube_url):
    try:
        with st.spinner("Reading YouTube metadata..."):
            info = get_video_info(youtube_url)
        st.session_state.video_info = info
        st.session_state.video_url = youtube_url
        st.success("Video metadata retrieved.")
    except Exception as exc:
        st.error(f"Could not inspect this video: {exc}")

info = st.session_state.video_info
if info:
    st.markdown(f"**{info.get('title', 'Untitled')}**")
    m1, m2, m3 = st.columns(3)
    duration = int(info.get("duration") or 0)
    m1.metric("Duration", f"{duration // 3600}h {(duration % 3600) // 60}m")
    m2.metric("Channel", info.get("channel") or info.get("uploader") or "Unknown")
    m3.metric("Video ID", info.get("id", "Unknown"))

st.divider()
st.subheader("3. Extract audio and transcribe")
if st.button("Extract audio + transcribe", type="primary", disabled=not youtube_url):
    try:
        video_id = get_video_id(youtube_url)
        with st.spinner("Extracting audio with yt-dlp and FFmpeg. This may take several minutes..."):
            audio_path = extract_audio(youtube_url, video_id)
        st.session_state.audio_path = str(audio_path)
        st.success(f"Audio ready: {audio_path.name}")
        with st.spinner(f"Transcribing with Whisper '{model_name}'. Long recordings may take substantial time..."):
            rows, whisper_info = transcribe_audio(audio_path, model_name, compute_type, language)
        st.session_state.transcript_rows = rows
        transcript_file = TRANSCRIPT_DIR / f"{video_id}.json"
        transcript_file.write_text(json.dumps({
            "video_url": youtube_url,
            "video_id": video_id,
            "language": getattr(whisper_info, "language", None),
            "segments": rows,
        }, indent=2), encoding="utf-8")
        st.success(f"Transcription complete: {len(rows)} speech segments.")
    except Exception as exc:
        st.exception(exc)

rows = st.session_state.transcript_rows
if rows is not None:
    st.subheader("Transcript preview")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)
    st.download_button("Download transcript JSON",
                       data=json.dumps(rows, indent=2),
                       file_name="ftc_transcript.json",
                       mime="application/json")

st.divider()
st.subheader("4. Detect match numbers and map teams")
if st.button("Detect matches + fetch FIRST schedule",
             disabled=not rows or not event_code or not first_username or not first_token):
    try:
        with st.spinner("Fetching official FIRST hybrid schedule..."):
            schedule = fetch_event_schedule(int(season), event_code.strip(),
                                             tournament, first_username.strip(), first_token.strip())
        mentions = detect_match_mentions(rows)
        mapped = map_mentions_to_schedule(mentions, schedule, youtube_url)
        st.session_state.mapped_rows = mapped
        st.session_state.schedule_count = len(schedule)
        st.session_state.mentions = mentions
        st.success(f"Loaded {len(schedule)} schedule entries; detected {len(mentions)} possible match announcements.")
    except Exception as exc:
        st.exception(exc)

mapped = st.session_state.mapped_rows
if mapped is not None:
    st.subheader("Team / match / video mapping")
    st.caption("A match mention is a candidate, not proof. Review confidence and verify the timestamp against the recording.")
    df = pd.DataFrame(mapped)
    st.dataframe(df, use_container_width=True, height=420)
    st.download_button("Download team-match CSV", df.to_csv(index=False),
                       file_name="ftc_team_match_timestamps.csv", mime="text/csv")
    st.download_button("Download mapping JSON", json.dumps(mapped, indent=2),
                       file_name="ftc_team_match_timestamps.json", mime="application/json")

with st.expander("Troubleshooting and limitations"):
    st.markdown("""
    - This MVP detects spoken match-number phrases in the transcript. If announcers do not say match numbers clearly, detections can be missed.
    - FIRST's hybrid schedule provides official match/team associations, but its schedule times are **not** used as YouTube timestamps.
    - The initial detector is audio-first. Scoreboard OCR and visual boundary verification are the next iteration.
    - FIRST API data is generally made available after an event concludes and syncs.
    - Use only videos you are authorized to access. YouTube availability and download restrictions can vary.
    """)
    st.markdown("FIRST API information: https://ftc-events.firstinspires.org/services/API")
