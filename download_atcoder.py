# Download AtCoder rated algorithm contests as a new dataset (not used in the paper)
import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "data", "atcoder", "cache")
LIST_URL = "https://kenkoooo.com/atcoder/resources/contests.json"
RESULTS_URL = "https://atcoder.jp/contests/{}/results/json"


def fetch(url, retries=5):
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip",
                                               "User-Agent": "elommr-research/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except Exception as e:  # rate limiting or transient network errors
            wait = 5 * (attempt + 1)
            print(f"  retry {url} in {wait}s ({e})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}")


def to_standings(rows):
    # Re-rank rated rows (sorted by Place) into (handle, lo, hi) tie ranges
    rows = sorted(rows, key=lambda x: x["Place"])
    n = len(rows)
    standings, old = [], []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and rows[j + 1]["Place"] == rows[i]["Place"]:
            j += 1
        for k in range(i, j + 1):
            standings.append([rows[k]["UserScreenName"], i, j])
            old.append(rows[k]["OldRating"])
        i = j + 1
    return standings, old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=dt.date.today().isoformat())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=1.5)
    args = ap.parse_args()
    cutoff = dt.datetime.fromisoformat(args.cutoff).replace(tzinfo=dt.timezone.utc).timestamp()

    os.makedirs(CACHE, exist_ok=True)
    contests = [c for c in fetch(LIST_URL)
                if c["rate_change"] != "-" and 0 < c["start_epoch_second"] < cutoff]
    contests.sort(key=lambda c: c["start_epoch_second"])
    if args.limit:
        contests = contests[: args.limit]
    print(f"{len(contests)} rated contests listed", flush=True)
    for idx, c in enumerate(contests):
        path = os.path.join(CACHE, f"{idx}.json.gz")
        skip = path + ".skip"
        if os.path.exists(path) or os.path.exists(skip):
            continue
        rows = [r for r in fetch(RESULTS_URL.format(c["id"]))
                if r["IsRated"] and r.get("ContestType", 0) == 0]
        if len(rows) < 2:
            print(f"[{idx}] {c['id']}: no rated algorithm results, skipping", flush=True)
            open(skip, "w").close()
            time.sleep(args.delay)
            continue
        standings, old = to_standings(rows)
        contest = {"id": c["id"], "name": c["title"],
                   "time_seconds": c["start_epoch_second"] + c["duration_second"],
                   "standings": standings, "official_old_rating": old}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(contest, f)
        print(f"[{idx}] {c['id']}: {len(standings)} rated players", flush=True)
        time.sleep(args.delay)


if __name__ == "__main__":
    sys.exit(main())
