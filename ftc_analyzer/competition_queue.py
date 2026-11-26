"""Server-side queue client. Never expose the service role key to browsers."""
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests


def now():
    return datetime.now(timezone.utc)


class QueueConflict(RuntimeError):
    pass


class CompetitionQueue:
    def __init__(self, url=None, key=None):
        url = url or os.getenv("SUPABASE_URL")
        key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the local .env.")
        self.url = url.rstrip("/") + "/rest/v1/ftc_video_processing_jobs"
        self.headers = {"apikey": key, "Authorization": "Bearer " + key,
                        "Prefer": "return=representation"}

    def _request(self, method, params, payload=None):
        response = requests.request(method, self.url, params=params, json=payload,
                                    headers=self.headers, timeout=30)
        if not response.ok:
            raise RuntimeError(f"Queue API returned HTTP {response.status_code}")
        return response.json()

    def list_jobs(self, status=None, limit=200, offset=0):
        params = {"select": "*", "order": "priority.desc,created_at.asc,id.asc",
                  "limit": limit, "offset": offset}
        if status:
            params["status"] = "eq." + status
        rows = self._request("GET", params)
        events = {}
        for year in {row["season_year"] for row in rows}:
            codes = sorted({row["event_code"] for row in rows if row["season_year"] == year})
            for i in range(0, len(codes), 50):
                import json
                response = requests.get(self.url.rsplit("/", 1)[0] + "/ftc_events",
                    params={"select": "event_code,date_start,date_end", "season_year": f"eq.{year}",
                            "event_code": "in.(" + ",".join(json.dumps(code) for code in codes[i:i+50]) + ")"},
                    headers=self.headers, timeout=30)
                if not response.ok:
                    raise RuntimeError(f"Event dates API returned HTTP {response.status_code}")
                for event in response.json():
                    events[(year, event["event_code"])] = event
        for row in rows:
            event = events.get((row["season_year"], row["event_code"]), {})
            row["date_start"] = event.get("date_start")
            row["date_end"] = event.get("date_end")
        return rows

    def _patch(self, filters, values):
        rows = self._request("PATCH", filters, {**values, "updated_at": now().isoformat()})
        if len(rows) != 1:
            raise QueueConflict("Job changed or ownership was lost. Refresh the queue.")
        return rows[0]

    def claim(self, job):
        if job["status"] != "pending" or job["attempts"] >= 5:
            raise QueueConflict("Only pending jobs with fewer than five attempts can run.")
        stamp = now()
        return self._patch({"id": f"eq.{job['id']}", "status": "eq.pending",
                            "attempts": f"eq.{job['attempts']}"},
                           {"status": "processing", "attempts": job["attempts"] + 1,
                            "locked_by": str(uuid.uuid4()), "locked_at": stamp.isoformat(),
                            "lease_expires_at": (stamp + timedelta(hours=6)).isoformat(),
                            "progress_percent": 0, "last_error": None,
                            "completed_at": None, "result_summary": None})

    def owned_update(self, job, **values):
        stamp = now()
        return self._patch({"id": f"eq.{job['id']}", "status": "eq.processing",
                            "locked_by": "eq." + job["locked_by"],
                            "lease_expires_at": "gt." + stamp.isoformat()},
                           values)

    def progress(self, job, percent):
        stamp = now()
        return self.owned_update(job, progress_percent=int(percent),
                                 locked_at=stamp.isoformat(),
                                 lease_expires_at=(stamp + timedelta(hours=6)).isoformat())

    def finish(self, job, summary=None, error=None):
        return self.owned_update(job, status="failed" if error else "completed",
                                 progress_percent=0 if error else 100,
                                 last_error=str(error)[:4000] if error else None,
                                 result_summary=summary, completed_at=now().isoformat(),
                                 locked_by=None, locked_at=None, lease_expires_at=None)

    def retry(self, job):
        if job["status"] != "failed" or job["attempts"] >= 5:
            raise QueueConflict("Only failed jobs below the attempt limit can be retried.")
        return self._patch({"id": f"eq.{job['id']}", "status": "eq.failed",
                            "attempts": f"eq.{job['attempts']}"},
                           {"status": "pending", "progress_percent": 0,
                            "completed_at": None, "locked_by": None,
                            "locked_at": None, "lease_expires_at": None})


def process_job(queue, selected, data_dir, interval=10, max_height=720,
                notify=lambda p, s: None, preview=lambda rows, windows, stats: None):
    """Checkpoint raw OCR locally; never publish unverified matches."""
    import json
    import re
    import time
    import traceback
    import pytesseract
    from .video_ocr import download_video, extract_frames, run_ocr, build_observed_match_windows, save_ocr_results
    video_id = selected.get("youtube_video_id") or ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Job needs a valid normalized YouTube ID.")
    job = queue.claim(selected)
    try:
        base = Path(data_dir).resolve() / "competition_queue" / str(job["id"])
        root = base / str(job["attempts"])
        cache = base / f"cache_v1_{video_id}_{interval}_{max_height}"
        for folder in (cache / "videos", cache / "frames", root / "ocr"):
            folder.mkdir(parents=True, exist_ok=True)
        def atomic(path, value):
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
            tmp.replace(path)
        def report(p, message):
            queue.progress(job, p)
            notify(p, message)
        report(5, "Downloading or reusing recording")
        # Existing downloader validates the expected filename and reuses completed files.
        path = download_video("https://www.youtube.com/watch?v=" + video_id, video_id,
                              cache / "videos", max_height, 0, None)
        report(25, "Capturing or reusing frames")
        manifest = cache / "frames.json"
        frames = json.loads(manifest.read_text()) if manifest.exists() else []
        if not frames or not all(Path(f["frame_path"]).is_file() for f in frames):
            checkpoint = cache / "checkpoint.json"
            checkpoint.unlink(missing_ok=True)
            frames = extract_frames(path, cache / "frames", interval, start_offset_seconds=0,
                                    duration_seconds=None)
            atomic(manifest, frames)
        checkpoint = cache / "checkpoint.json"
        rows = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
        if len(rows) > len(frames) or any(r["frame_index"] != f["frame_index"]
                                         for r, f in zip(rows, frames)):
            raise RuntimeError("Checkpoint does not match frame manifest")
        initial = len(rows)
        started = time.monotonic()
        consecutive_errors = 0
        def snapshot():
            atomic(checkpoint, rows)
            windows = build_observed_match_windows(rows, interval)
            for w in windows:
                offset = w.get("start_offset_seconds")
                if offset is None:
                    offset = w["first_seen_seconds"]
                w["youtube_link"] = f"https://www.youtube.com/watch?v={video_id}&t={int(offset)}s"
                w.update(season_year=job["season_year"], event_code=job["event_code"],
                         youtube_video_id=video_id)
            atomic(root / "ocr/windows.json", windows)
            failed = sum(bool(r.get("ocr_error")) for r in rows)
            speed = (len(rows) - initial) / max(time.monotonic() - started, 0.001)
            stats = {"processed": len(rows), "total": len(frames), "failed": failed,
                     "successful": len(rows) - failed, "checkpoint": len(rows),
                     "frames_per_second": round(speed, 2),
                     "eta_seconds": round((len(frames) - len(rows)) / speed) if speed else None}
            preview(rows, windows, stats)
            return windows, stats
        report(45 + int(50 * len(rows) / len(frames)), f"Resuming OCR at {len(rows)}/{len(frames)}")
        snapshot()
        for frame in frames[len(rows):]:
            try:
                for attempt in range(2):
                    try:
                        result = run_ocr([frame], None, schedule=None,
                                         enable_voting=False, enable_schedule=False)[0]
                        break
                    except pytesseract.TesseractError:
                        if attempt:
                            raise
                        time.sleep(1)
                consecutive_errors = 0
            except pytesseract.TesseractError as exc:
                consecutive_errors += 1
                traceback.print_exc()
                if consecutive_errors >= 3:
                    snapshot()
                    raise RuntimeError(f"Three consecutive OCR errors; last frame {frame['frame_index']}: {exc}") from exc
                result = {**frame, "scoreboard_format": "error", "ocr_error": str(exc),
                          "match_number_candidate": None}
            rows.append(result)
            if len(rows) % 50 == 0 or len(rows) == len(frames):
                snapshot()
                report(45 + int(50 * len(rows) / len(frames)), f"OCR frame {len(rows)}/{len(frames)}")
            else:
                notify(45 + int(50 * len(rows) / len(frames)), f"OCR frame {len(rows)}/{len(frames)}")
        windows, stats = snapshot()
        save_ocr_results(rows, root / "ocr/results.json", root / "ocr/results.csv")
        summary = {"frames": len(frames), "ocr_rows": len(rows), "candidate_windows": len(windows),
                   "failed_frames": stats["failed"], "artifact_dir": str(root / "ocr"),
                   "published": False, "interval_seconds": interval, "max_height": max_height}
        queue.finish(job, summary=summary)
        notify(100, "Completed; results saved locally")
        return summary
    except Exception as exc:
        traceback.print_exc()
        try:
            queue.finish(job, error=f"{type(exc).__name__}: {exc}")
        except Exception:
            traceback.print_exc()
        raise
