"""
Data acquisition — Steam Storefront appdetails fetch
Downloads the full per-game store payload (text descriptions + all structured
fields) for every game in the corpus and caches one JSON file per app_id under
data/appdetails/. One-time, resumable, polite-rate-limited.

The Storefront endpoint (store.steampowered.com/api/appdetails) is unofficial:
no auth, ~200 requests / 5 min before it starts returning 429. We pace at
REQUEST_INTERVAL_SECONDS and back off on 429. Every app_id writes exactly one
file — successful payloads store the full {appid: {success, data}} response;
delisted / region-locked ids (success=False) write a {"success": false} marker
so a re-run skips them instead of re-hitting the API forever.

Only short_description + about_the_game are promoted downstream (see CLAUDE.md
"text tower"); the full payload is cached so other fields (categories,
metacritic, supported_languages, …) can be mined later without re-fetching.

Usage:
    python -m src.fetch_appdetails          # fetch all corpus games, skip cached
    python -m src.fetch_appdetails --force  # re-fetch even if a file exists
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

BASE_GAMES_PARQUET       = "data/base_games.parquet"
OUTPUT_DIR               = "data/appdetails"
APPDETAILS_URL           = "https://store.steampowered.com/api/appdetails?appids={appid}&l=english"
USER_AGENT               = "Mozilla/5.0 (recommender-dataset-build)"

REQUEST_INTERVAL_SECONDS = 1.5    # ~200 req / 5 min ceiling — stay under it
REQUEST_TIMEOUT_SECONDS  = 30
MAX_RETRIES              = 5      # per app_id, on 429 / transient network errors
BACKOFF_BASE_SECONDS     = 5.0    # 429 backoff = BACKOFF_BASE * 2**attempt


# ── Fetch ───────────────────────────────────────────────────────────────────

def _fetch_one(appid):
    """
    Fetch a single app_id with retry/backoff. Returns the parsed response dict
    on success (whether or not success==True), or None if all retries failed.
    A 429 sleeps with exponential backoff and retries; other HTTP/network errors
    retry a few times then give up (caller logs and moves on).
    """
    url = APPDETAILS_URL.format(appid=appid)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                backoff = BACKOFF_BASE_SECONDS * (2 ** attempt)
                tqdm.write(f"  429 on {appid} — backing off {backoff:.0f}s")
                time.sleep(backoff)
                continue
            tqdm.write(f"  HTTP {e.code} on {appid} (attempt {attempt + 1})")
            time.sleep(BACKOFF_BASE_SECONDS)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            tqdm.write(f"  {type(e).__name__} on {appid} (attempt {attempt + 1})")
            time.sleep(BACKOFF_BASE_SECONDS)

    return None


def fetch_all(force=False):
    """
    Walk the corpus app_ids, fetch each that is not already cached, write one
    JSON per app_id to OUTPUT_DIR. Resumable: existing files are skipped unless
    --force. Prints a coverage summary (found / delisted / failed) at the end.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    games = pd.read_parquet(BASE_GAMES_PARQUET, columns=["item_id"])
    appids = [str(x) for x in games["item_id"].tolist()]

    n_found = n_delisted = n_failed = n_skipped = 0

    for appid in tqdm(appids, desc="appdetails"):
        out_path = os.path.join(OUTPUT_DIR, f"{appid}.json")
        if os.path.exists(out_path) and not force:
            n_skipped += 1
            continue

        data = _fetch_one(appid)
        time.sleep(REQUEST_INTERVAL_SECONDS)

        if data is None:
            n_failed += 1
            continue

        entry = data.get(appid, {})
        if entry.get("success"):
            n_found += 1
        else:
            # Delisted / region-locked — write a marker so a re-run skips it.
            data = {appid: {"success": False}}
            n_delisted += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    print(
        f"\nDone. cached_this_run: found={n_found} delisted={n_delisted} "
        f"failed={n_failed} | skipped(existing)={n_skipped} | "
        f"total_on_disk={len(os.listdir(OUTPUT_DIR))}/{len(appids)}"
    )
    if n_failed:
        print("Re-run to retry the failed ids (they were not written, so they retry).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Steam Storefront appdetails for the corpus.")
    parser.add_argument("--force", action="store_true", help="re-fetch even if a cached file exists")
    args = parser.parse_args()
    fetch_all(force=args.force)
