import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "https://ftc-api.firstinspires.org/v2.0"

def fetch_event_schedule(season: int, event_code: str, tournament_level: str,
                         username: str, authorization_key: str):
    """Fetch FIRST's hybrid schedule. FIRST uses HTTP Basic auth with username:key."""
    url = f"{BASE_URL}/{season}/schedule/{event_code}/{tournament_level}/hybrid"
    response = requests.get(
        url,
        auth=HTTPBasicAuth(username, authorization_key),
        headers={"Accept": "application/json", "User-Agent": "FTC-Livestream-Analyzer/0.1"},
        timeout=45,
    )
    if response.status_code == 401:
        raise RuntimeError("FIRST API returned 401. Check your FIRST API username and authorization key.")
    if response.status_code == 404:
        raise RuntimeError(f"FIRST event not found: season={season}, event_code={event_code}. Check the event code.")
    response.raise_for_status()
    payload = response.json()
    schedule = payload.get("schedule", [])
    if not isinstance(schedule, list):
        raise RuntimeError("Unexpected FIRST API response: 'schedule' was not a list.")
    return schedule
