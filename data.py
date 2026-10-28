# Datasets
import glob
import gzip
import json
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Contest:
    index: int
    name: str
    time: int
    ids: np.ndarray   # player ids, in standings order (first place first)
    lo: np.ndarray    # 0-indexed rank range of the tie block containing each player
    hi: np.ndarray
    official: np.ndarray = None  # AtCoder only: platform's own pre-contest rating per player


@dataclass
class Dataset:
    name: str
    contests: list
    num_players: int
    handles: list = field(default_factory=list)
    true_skill: np.ndarray = None   # synthetic only: skill at each round [rounds, players]

    def __post_init__(self):
        counts = np.zeros(self.num_players, dtype=np.int64)
        for c in self.contests:
            counts[c.ids] += 1
        self.player_counts = counts

    def __len__(self):
        return len(self.contests)

    def summary(self):
        sizes = [len(c.ids) for c in self.contests]
        return {"dataset": self.name, "contests": len(self.contests),
                "players": int((self.player_counts > 0).sum()),
                "avg_participants": float(np.mean(sizes)), "max_participants": int(np.max(sizes))}


def load_codeforces(cache_dir=None, max_contests=None):
    return _load_cache("Codeforces", cache_dir or os.path.join(ROOT, "data", "codeforces", "cache"),
                       max_contests)


def load_atcoder(cache_dir=None, max_contests=None):
    # AtCoder rated algorithm contests (download_atcoder.py), not in the paper
    return _load_cache("AtCoder", cache_dir or os.path.join(ROOT, "data", "atcoder", "cache"),
                       max_contests)


def _load_cache(name, cache_dir, max_contests):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.json.gz")),
                   key=lambda f: int(os.path.basename(f).split(".")[0]))
    if max_contests:
        files = files[:max_contests]
    handle_id, handles, contests = {}, [], []
    for i, f in enumerate(files):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            raw = json.load(fh)
        st = raw["standings"]
        ids = np.empty(len(st), dtype=np.int64)
        for k, (h, _, _) in enumerate(st):
            if h not in handle_id:
                handle_id[h] = len(handles)
                handles.append(h)
            ids[k] = handle_id[h]
        lo = np.array([s[1] for s in st], dtype=np.int64)
        hi = np.array([s[2] for s in st], dtype=np.int64)
        official = raw.get("official_old_rating")
        if official is not None:
            official = np.array(official, dtype=np.float64)
        contests.append(Contest(i, raw["name"], int(raw["time_seconds"]), ids, lo, hi, official))
    return Dataset(name, contests, len(handles), handles)


def make_synthetic(num_players=10000, rounds=50, participation=0.25, seed=0,
                   skill_mu=1500.0, skill_sd=300.0, perf_sd=200.0, drift_sd=35.0):
    # Gaussian generative model of Section 2 / 6.1
    rng = np.random.default_rng(seed)
    skill = rng.normal(skill_mu, skill_sd, num_players)
    contests, truth = [], []
    t0 = 1_600_000_000
    for r in range(rounds):
        if r > 0:
            skill = skill + rng.normal(0, drift_sd, num_players)
        truth.append(skill.copy())
        players = np.flatnonzero(rng.random(num_players) < participation)
        perf = skill[players] + rng.normal(0, perf_sd, len(players))
        order = players[np.argsort(-perf)]
        ar = np.arange(len(order), dtype=np.int64)
        contests.append(Contest(r, f"synthetic-{r}", t0 + r * 7 * 86400, order.astype(np.int64), ar, ar.copy()))
    return Dataset("Synthetic", contests, num_players, true_skill=np.array(truth))
