# Download Codeforces contest standings for the Elo-MMR dataset
import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
IDS_FILE = os.path.join(ROOT, "data", "codeforces", "contest_ids.csv")
CACHE = os.path.join(ROOT, "data", "codeforces", "cache")
API = "https://codeforces.com/api/contest.ratingChanges?contestId={}"


def fetch(contest_id, retries=5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(API.format(contest_id), timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # rate limiting or transient network errors
            wait = 5 * (attempt + 1)
            print(f"  retry {contest_id} in {wait}s ({e})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch contest {contest_id}")


def to_standings(changes):
    # Convert API rank numbers into (handle, lo, hi) tie ranges
    n = len(changes)
    standings, seen = [], set()
    i = 0
    while i < n:
        j = i
        while j + 1 < n and changes[j + 1]["rank"] == changes[i]["rank"]:
            j += 1
        for k in range(i, j + 1):
            h = changes[k]["handle"]
            while h in seen:  # a few contests contain duplicate users
                h += "_clone"
            seen.add(h)
            standings.append([h, i, j])
        i = j + 1
    return standings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2020-10-12", help="last date, as in the paper")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=2.1)
    args = ap.parse_args()
    cutoff = dt.datetime.fromisoformat(args.cutoff).replace(tzinfo=dt.timezone.utc).timestamp() + 86400

    os.makedirs(CACHE, exist_ok=True)
    ids = [int(x) for x in open(IDS_FILE) if x.strip()]
    if args.limit:
        ids = ids[: args.limit]
    for idx, cid in enumerate(ids):
        path = os.path.join(CACHE, f"{idx}.json.gz")
        if os.path.exists(path):
            continue
        packet = fetch(cid)
        if packet.get("status") != "OK" or not packet["result"]:
            print(f"[{idx}] contest {cid}: no rated standings, skipping", flush=True)
            continue
        res = packet["result"]
        t = res[0]["ratingUpdateTimeSeconds"]
        if t > cutoff:
            print(f"[{idx}] contest {cid} is after cutoff; stopping", flush=True)
            break
        contest = {"id": cid, "name": res[0]["contestName"], "time_seconds": t,
                   "standings": to_standings(res)}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(contest, f)
        print(f"[{idx}] contest {cid}: {len(res)} players", flush=True)
        time.sleep(args.delay)


if __name__ == "__main__":
    sys.exit(main())
