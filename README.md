# FTC Livestream Analyzer — video OCR MVP

## What this version does
- Accepts an archived YouTube livestream URL.
- Downloads a video stream (no audio extraction or transcription).
- Uses FFmpeg to capture one frame every 10 seconds by default; 5, 15, and 30 seconds are configurable.
- Runs Tesseract OCR on every captured frame. Detects matching red and blue alliance panels at the **top or bottom** of the picture, then reads exactly four ordered team slots instead of guessing numbers from the whole frame.
- Attempts to recognize match-number labels and event team numbers from OCR text.
- Votes on each alliance/team slot across consecutive frames belonging to the same observed match. An optional FIRST hybrid schedule filters OCR candidates and can confirm or correct the four official teams when the match has corroborating OCR evidence; uncertain results remain unresolved.
- Groups consecutive frames that show the same match number into **observed scoreboard windows**.
- Saves frame images and OCR results locally and provides CSV/JSON exports.

**Important:** This is an OCR prototype, not yet a verified match-boundary detector. A scoreboard can show the next match before play begins or continue showing a match after play ends. The reported first/last observations are not exact match start/end times. We will add match-state/clock transition detection after validating OCR against a real event stream.

## System requirements (Ubuntu)
```bash
sudo apt update
sudo apt install -y ffmpeg tesseract-ocr python3-venv python3-full
```

## Setup
Use the Python version available on your system. A virtual environment prevents Ubuntu's externally-managed Python restriction:
```bash
cd ~/workspace/ftc-livestream-analyzer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your FIRST API username and authorization key, or enter them in the app sidebar. Do not commit or share `.env`.

FIRST API credentials: https://ftc-events.firstinspires.org/services/API/register

## Run
```bash
cd ~/workspace/ftc-livestream-analyzer
source .venv/bin/activate
streamlit run app.py
```
Open http://localhost:8501

## Workflow
1. Paste the archived YouTube URL and click **Inspect YouTube video**.
2. Enter the season, FIRST event code, and tournament level (`qual` or `playoff`).
3. Set frame interval to 10 seconds (default).
4. Click **Download video + capture frames**.
5. Review the frame previews to make sure the scoreboard is visible.
6. Click **Run OCR on captured frames**.
7. Review OCR text, match-number candidates, team-number candidates, and observed match windows. Export CSV/JSON as needed.

## Local data
- `data/videos/`: downloaded video files
- `data/frames/<video_id>/`: extracted JPEG frames
- `data/ocr/`: OCR results in JSON and CSV

These files can become large, especially the downloaded video and frame directory. Delete a video's folder under `data/` when you no longer need it.

## Notes and limitations
- Whisper/faster-whisper and PyAV are not dependencies in this video-first version.
- Install `tesseract-ocr` with apt; the Python `pytesseract` package is only a wrapper.
- YouTube may not offer the requested resolution as a single-file stream. yt-dlp selects the best available compatible format.
- Color panel detection expects the broadcast to use recognizable red and blue horizontal alliance panels. Different broadcasters may use different colors and team-number column positions. The four slots are fixed, but their content is not invented when OCR or schedule validation is inconclusive.
- A 10-second sample interval gives roughly 15 frames during a 2:30 match, but match boundaries can fall between samples. We will refine actual start/end timestamps after validating scoreboard state/clock detection.
- FIRST API data is generally available after an event concludes and syncs. The app uses HTTP Basic authentication and FIRST asks applications displaying API data to link to its API information page.
- Use only videos you are authorized to access. YouTube availability and download restrictions can vary.

## Testing team-number detection

```bash
python -m unittest discover -s tests -v
```

The tests create synthetic top/bottom scoreboards and exercise four-team OCR, per-slot temporal voting, and optional FIRST schedule verification. The additional original Athens Qualifier screenshot can be tested locally by placing it at `tests/frame_00000129.jpg` (not committed).

## Quick test mode

Run `streamlit run app.py` as usual. Select Quick test (default: first five minutes), Custom range, or Full video. Enter a start and duration in minutes, then click **Download selected video**, **Capture frames**, and **Run OCR on captured frames** independently. For fast iteration choose a two-second interval and a segment containing actual matches. Downloads are cached by video ID, requested resolution and range; frames and OCR exports also include the interval. Segment timestamps are offset to the original livestream. Actual cut precision depends on the source and FFmpeg. Existing full-video caches from older versions are not automatically reused. Download cancellation, pause/resume and automatic crop calibration are not implemented in this PR.
