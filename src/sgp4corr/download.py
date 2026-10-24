"""Dataset acquisition for the SGP4 residual-correction project.
 
Three steps, each cached on disk so re-running skips work already done:
 
1. ``select_objects``      - choose candidate LEO debris / rocket bodies from the CelesTrak SATCAT.
2. ``fetch_gp_history``    - download each object's element-set history from Space-Track.
3. ``fetch_space_weather`` - download the CelesTrak space weather file (F10.7, Ap).
 
A final ``trim_objects`` step keeps the objects with the densest history in each
altitude band and writes the definitive object list to ``data/objects.csv``.
 
Usage (from the repository root; Windows PowerShell shown, see the README
for macOS/Linux):

    $env:SPACETRACK_USER="you@example.com"
    $env:SPACETRACK_PASS="your-password"
    python src/sgp4corr/download.py --start 2024-07-01 --end 2026-07-01

Space-Track credentials are read from environment variables and must never be
committed. Raw Space-Track data stays in ``data/raw/`` (git-ignored), because
Space-Track's user agreement restricts redistribution.
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import time
from pathlib import Path
 
import pandas as pd
import requests
 
SATCAT_URL = "https://celestrak.org/pub/satcat.csv"
SPACE_WEATHER_URL = "https://celestrak.org/SpaceData/SW-All.csv"
SPACETRACK_BASE = "https://www.space-track.org"
SPACETRACK_LOGIN = f"{SPACETRACK_BASE}/ajaxauth/login"
 
# Perigee-altitude band edges in km. Also used later for the generalisation test.
ALTITUDE_BANDS_KM = [300, 500, 700, 900, 1200, 2000]
 
# Space-Track allows roughly 30 requests/minute and 300/hour; stay well under.
SECONDS_BETWEEN_REQUESTS = 15
OBJECTS_PER_REQUEST = 10
 
 
def select_objects(
    window_start: str,
    n_per_band: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample candidate LEO debris and rocket bodies from the CelesTrak SATCAT.
 
    Debris and rocket bodies are used because they do not manoeuvre, so any
    SGP4 error is due to force modelling rather than thruster firings.
 
    Args:
        window_start: ISO date (YYYY-MM-DD). Objects must have launched before this.
        n_per_band: Objects to sample per altitude band. Oversample here
            (default 60) and let ``trim_objects`` cut back to the final count.
        seed: Random seed, fixed so the selection is reproducible.
 
    Returns:
        DataFrame of candidates with a ``BAND`` column giving the altitude band.
 
    Raises:
        requests.HTTPError: If the SATCAT download fails.
        ValueError: If the filters leave no objects.
    """
    sat = pd.read_csv(SATCAT_URL)
 
    leo = sat[
        sat["OBJECT_TYPE"].isin(["DEB", "R/B"])
        & sat["DECAY_DATE"].isna()            # still in orbit
        & sat["DATA_STATUS_CODE"].isna()      # has current elements
        & (sat["ORBIT_CENTER"] == "EA")       # Earth-orbiting
        & (sat["PERIGEE"] >= ALTITUDE_BANDS_KM[0])
        & (sat["APOGEE"] <= ALTITUDE_BANDS_KM[-1])
        & (sat["LAUNCH_DATE"] < window_start)
    ].copy()
 
    if leo.empty:
        raise ValueError("No objects passed the SATCAT filters; check column names.")
 
    leo["BAND"] = pd.cut(leo["PERIGEE"], ALTITUDE_BANDS_KM, include_lowest=True).astype(str)
 
    # Sample per band, taking everything if a band has fewer than n_per_band objects.
    parts = [
        group.sample(n=min(n_per_band, len(group)), random_state=seed)
        for _, group in leo.groupby("BAND")
    ]
    columns = [
        "NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE",
        "PERIGEE", "APOGEE", "INCLINATION", "RCS", "BAND",
    ]
    return pd.concat(parts)[columns].sort_values("NORAD_CAT_ID").reset_index(drop=True)
 
 
def spacetrack_session() -> requests.Session:
    """Log in to Space-Track and return an authenticated session.
 
    Returns:
        A ``requests.Session`` carrying the Space-Track login cookie.
 
    Raises:
        RuntimeError: If credentials are missing or the login is rejected.
    """
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")
    if not user or not password:
        raise RuntimeError("Set SPACETRACK_USER and SPACETRACK_PASS environment variables.")
 
    session = requests.Session()
    response = session.post(
        SPACETRACK_LOGIN,
        data={"identity": user, "password": password},
        timeout=60,
    )
    response.raise_for_status()
    # Space-Track returns HTTP 200 with a "Failed" body on bad credentials.
    if "Failed" in response.text:
        raise RuntimeError("Space-Track login failed; check your credentials.")
    return session
 
 
def fetch_gp_history(
    norad_ids: list[int],
    start: str,
    end: str,
    out_dir: Path,
) -> None:
    """Download element-set history from Space-Track, one JSON file per object.
 
    Objects whose file already exists are skipped, so an interrupted run can
    simply be restarted. Requests are batched and throttled to respect
    Space-Track's rate limits.
 
    Args:
        norad_ids: NORAD catalogue numbers to download.
        start: ISO start date of the epoch window (inclusive).
        end: ISO end date of the epoch window.
        out_dir: Directory for the per-object ``<norad_id>.json`` files.
 
    Raises:
        RuntimeError: If Space-Track login fails.
        requests.HTTPError: If a query is rejected (e.g. rate limiting).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in norad_ids if not (out_dir / f"{i}.json").exists()]
    if not todo:
        print("GP history: all objects already cached.")
        return
 
    session = spacetrack_session()
    batches = [todo[i:i + OBJECTS_PER_REQUEST] for i in range(0, len(todo), OBJECTS_PER_REQUEST)]
 
    for n, batch in enumerate(batches, start=1):
        ids = ",".join(str(i) for i in batch)
        url = (
            f"{SPACETRACK_BASE}/basicspacedata/query/class/gp_history"
            f"/NORAD_CAT_ID/{ids}/EPOCH/{start}--{end}"
            f"/orderby/EPOCH asc/format/json"
        )
        response = session.get(url, timeout=600)
        response.raise_for_status()
        records = response.json()
 
        # Split the batch response into one file per object. An object with no
        # records still gets an (empty) file so it is not requested again.
        by_object: dict[int, list[dict]] = {i: [] for i in batch}
        for record in records:
            by_object[int(record["NORAD_CAT_ID"])].append(record)
        for norad_id, object_records in by_object.items():
            (out_dir / f"{norad_id}.json").write_text(json.dumps(object_records))
 
        print(f"GP history: batch {n}/{len(batches)} done ({len(records)} element sets).")
        if n < len(batches):
            time.sleep(SECONDS_BETWEEN_REQUESTS)
 
 
def fetch_space_weather(out_dir: Path) -> Path:
    """Download the CelesTrak space weather file (daily F10.7 and Ap).
 
    Args:
        out_dir: Directory to save ``SW-All.csv`` into.
 
    Returns:
        Path to the saved file.
 
    Raises:
        requests.HTTPError: If the download fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "SW-All.csv"
    response = requests.get(SPACE_WEATHER_URL, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path
 
 
def trim_objects(
    candidates: pd.DataFrame,
    history_dir: Path,
    n_per_band: int = 50,
    min_element_sets: int = 200,
) -> pd.DataFrame:
    """Keep the objects with the densest element-set history in each band.
 
    Removes objects with sparse history (e.g. fragments created part-way
    through the window, or poorly tracked objects).
 
    Args:
        candidates: Output of ``select_objects``.
        history_dir: Directory of per-object JSON files from ``fetch_gp_history``.
        n_per_band: Final number of objects to keep per altitude band.
        min_element_sets: Objects with fewer element sets than this are dropped.
 
    Returns:
        The final object list with an added ``N_ELEMENT_SETS`` column.
    """
    counts = []
    for norad_id in candidates["NORAD_CAT_ID"]:
        path = history_dir / f"{norad_id}.json"
        counts.append(len(json.loads(path.read_text())) if path.exists() else 0)
 
    result = candidates.assign(N_ELEMENT_SETS=counts)
    result = result[result["N_ELEMENT_SETS"] >= min_element_sets]
    result = (
        result.sort_values("N_ELEMENT_SETS", ascending=False)
        .groupby("BAND")
        .head(n_per_band)
    )
    return result.sort_values("NORAD_CAT_ID").reset_index(drop=True)
 
 
def main() -> None:
    """Run the full download pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Download the project dataset.")
    parser.add_argument("--start", default="2024-07-01", help="Window start (YYYY-MM-DD).")
    parser.add_argument("--end", default="2026-07-01", help="Window end (YYYY-MM-DD).")
    parser.add_argument("--data-dir", default="data", type=Path, help="Data directory.")
    args = parser.parse_args()
 
    raw_dir = args.data_dir / "raw"
    history_dir = raw_dir / "gp_history"
    candidates_path = args.data_dir / "candidates.csv"
 
    # Reuse an existing candidate list so re-runs fetch the same objects even
    # if the live SATCAT has changed since.
    if candidates_path.exists():
        candidates = pd.read_csv(candidates_path)
        print(f"Loaded {len(candidates)} candidates from {candidates_path}.")
    else:
        candidates = select_objects(args.start)
        args.data_dir.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidates_path, index=False)
        print(f"Selected {len(candidates)} candidates.")
 
    fetch_gp_history(candidates["NORAD_CAT_ID"].tolist(), args.start, args.end, history_dir)
    fetch_space_weather(raw_dir)
 
    objects = trim_objects(candidates, history_dir)
    objects.to_csv(args.data_dir / "objects.csv", index=False)
    print(f"Final dataset: {len(objects)} objects.")
    print(objects.groupby("BAND")["N_ELEMENT_SETS"].agg(["count", "median"]))
 
 
if __name__ == "__main__":
    main()
 