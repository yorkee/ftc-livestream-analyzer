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
        params = {"select": "*,ftc_events(date_start,date_end)", "order": "priority.desc,created_at.asc,id.asc",
                  "limit": limit, "offset": offset}
        if status:
            params["status"] = "eq." + status
        rows = self._request("GET", params)
        for row in rows:
            event = row.pop("ftc_events", None) or {}
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


def process_job(queue, selected, data_dir, interval=10, max_height=720, notify=lambda p, s: None):
    """Run existing OCR pipeline; save local artifacts only, never publish matches."""
    import re
    from .video_ocr import download_video, extract_frames, run_ocr, build_observed_match_windows, save_ocr_results
    video_id = selected.get("youtube_video_id") or ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Job needs a valid normalized YouTube ID.")
    job = queue.claim(selected)
    try:
        root = Path(data_dir) / "competition_queue" / str(job["id"]) / str(job["attempts"])
        for folder in ("videos", "frames", "ocr"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        def report(p, message):
            queue.progress(job, p)
            notify(p, message)
        report(5, "Downloading full recording")
        path = download_video("https://www.youtube.com/watch?v=" + video_id, video_id,
                              root / "videos", max_height, 0, None)
        report(25, "Capturing frames")
        frames = extract_frames(path, root / "frames", interval, start_offset_seconds=0,
                                duration_seconds=None)
        if not frames:
            raise RuntimeError("No frames captured")
        report(45, "Running OCR")
        last = 45
        def callback(fraction, current, total):
            nonlocal last
            percent = min(95, 45 + int(50 * fraction))
            if percent > last:
                report(percent, f"OCR frame {current}/{total}")
                last = percent
        rows = run_ocr(frames, None, callback, schedule=None,
                       enable_voting=False, enable_schedule=False)
        windows = build_observed_match_windows(rows, interval)
        import json
        save_ocr_results(rows, root / "ocr/results.json", root / "ocr/results.csv")
        (root / "ocr/windows.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")
        summary = {"frames": len(frames), "ocr_rows": len(rows), "candidate_windows": len(windows),
                   "artifact_dir": str(root / "ocr"), "published": False,
                   "interval_seconds": interval, "max_height": max_height}
        queue.finish(job, summary=summary)
        notify(100, "Completed; results saved locally")
        return summary
    except Exception as exc:
        try:
            queue.finish(job, error=type(exc).__name__ + ": processing failed; see local application error")
        except Exception:
            pass
        raise
