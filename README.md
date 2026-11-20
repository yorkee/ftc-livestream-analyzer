# FTC Livestream Analyzer (local MVP)

## What it does
- Accepts an archived YouTube URL.
- Uses yt-dlp + FFmpeg to extract an MP3 audio track.
- Uses faster-whisper to transcribe speech locally.
- Detects explicit spoken match-number mentions.
- Fetches the FIRST FTC hybrid schedule for a season/event/tournament level.
- Maps detected match numbers to the teams in FIRST's schedule.
- Produces YouTube `?t=...s` links and CSV/JSON exports.

**Important:** This first version does not yet use visual scoreboard OCR. A spoken match mention is only a candidate timestamp and should be checked against the video. FIRST schedule times are not used to invent video timestamps.

## Requirements
- Ubuntu/Linux
- Python 3.11 or 3.12 recommended
- FFmpeg installed and available on PATH
- Internet access for YouTube retrieval, Whisper model download, and FIRST API calls

Install FFmpeg if needed:
```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Setup
```bash
cd ~/ftc-livestream-analyzer
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your FIRST API username and authorization key. You can also enter these in the app sidebar. Do not commit or share `.env`.

Get FIRST API credentials:
https://ftc-events.firstinspires.org/services/API/register

## Run
```bash
streamlit run app.py
```
Open http://localhost:8501

## Workflow
1. Paste the YouTube archive URL and click **Inspect YouTube video**.
2. Enter season, FIRST event code, and tournament level (`qual` or `playoff`).
3. Enter FIRST API username/key in the sidebar.
4. Click **Extract audio + transcribe**.
5. Click **Detect matches + fetch FIRST schedule**.
6. Review/export the CSV or JSON.

## Notes
- First Whisper run downloads its model. `small` is a reasonable starting point; `base` is faster, `medium` may improve accuracy but needs more compute.
- Processing is CPU-based in this MVP. Several-hour recordings can take a long time.
- The app caches extracted MP3 and transcript JSON files under `data/`.
- If YouTube blocks or restricts access, yt-dlp may fail. Use only videos you are authorized to access.
- FIRST says its API data is generally available after the tournament concludes and has synced. The app uses HTTP Basic authentication.
- FIRST requests that applications displaying FIRST API data include a link to its API information page.
