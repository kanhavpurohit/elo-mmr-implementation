# Reproduce Table 2 / Table 3 of the paper and evaluate our improvements
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

from data import load_atcoder, load_codeforces, make_synthetic
from evaluate import run
from systems import AtCoderOfficial, Codeforces, EloMMR, Glicko, TopCoder

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

GRIDS = {
    "Codeforces": (Codeforces, {"beta": [50, 65, 100, 150, 200, 315], "weight": [0.3, 0.631, 1.0, 1.5]}),
    "TopCoder": (TopCoder, {"weight_noob": [0.4, 0.6, 0.9], "weight_limit": [0.12, 0.18, 0.25, 0.35]}),
    "Glicko": (Glicko, {"beta": [75, 100, 150, 200, 300], "sig_drift": [15, 25, 40, 60, 90]}),
    "Elo-MMchi": (EloMMR, {"gaussian": [True], "weight_limit": [0.1, 0.2, 0.3, 0.45, 0.6],
                           "sig_limit": [40, 60, 80, 110, 150]}),
    "Elo-MMR(rho)": (EloMMR, {"weight_limit": [0.1, 0.2, 0.3, 0.45, 0.6], "sig_limit": [40, 60, 80, 110, 150],
                              "rho": [0.04, 1.0]}),
}


def grid_search(cls, grid, ds, tune_range, base=None, label=""):
    base = base or {}
    keys = list(grid)
    best, log = None, []
    for vals in itertools.product(*(grid[k] for k in keys)):
        kw = dict(base, **dict(zip(keys, vals)))
        r = run(cls(**kw), ds, eval_from=tune_range[0], eval_to=tune_range[1])
        log.append({"params": kw, "pair_inv": r["pair_inv"], "rank_dev": r["rank_dev"]})
        if best is None or r["pair_inv"] > best[1]["pair_inv"]:
            best = (kw, r)
        print(f"  [{label}] {kw} -> pair {r['pair_inv']:.3f} dev {r['rank_dev']:.3f}", flush=True)
    return best[0], log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["codeforces", "atcoder", "synthetic"])
    ap.add_argument("--max-contests", type=int, default=None)
    args = ap.parse_args()

    if args.dataset == "codeforces":
        ds = load_codeforces(max_contests=args.max_contests)
    elif args.dataset == "atcoder":
        ds = load_atcoder(max_contests=args.max_contests)
    else:
        ds = make_synthetic()
    n_train = max(1, len(ds) // 10)
    print(ds.summary(), "train contests:", n_train, flush=True)
    os.makedirs(RESULTS, exist_ok=True)

    if args.dataset == "synthetic":
        # The first 10% of synthetic rounds contain no player with 5+ contests, so we tune on an
        # independent draw from the same generative model instead (seed 1; test uses seed 0).
        tune_ds, tune_range = make_synthetic(seed=1), (n_train, None)
    else:
        tune_ds, tune_range = ds, (0, n_train)
    out = {"summary": ds.summary(), "n_train": n_train, "systems": {}, "search": {}}
    chosen = {}
    for name, (cls, grid) in GRIDS.items():
        kw, log = grid_search(cls, grid, tune_ds, tune_range, label=name)
        chosen[name] = (cls, kw)
        out["search"][name] = log

    # ---- improvements, tuned on the same training split, starting from tuned Elo-MMR
    mmr_kw = chosen["Elo-MMR(rho)"][1]
    # time drift interacts with the per-contest drift, so search it jointly with (w, sig_limit)
    joint = {"weight_limit": GRIDS["Elo-MMR(rho)"][1]["weight_limit"],
             "sig_limit": GRIDS["Elo-MMR(rho)"][1]["sig_limit"],
             "drift_per_day": [3, 10, 30, 100, 300]}
    kw_t, log = grid_search(EloMMR, joint, tune_ds, tune_range, {"rho": mmr_kw["rho"]}, "time-drift")
    out["search"]["+time-drift"] = log
    alphas = {"adaptive_alpha": [0.5, 1.0, 2.0, 3.0]}
    kw_a, log = grid_search(EloMMR, alphas, tune_ds, tune_range, mmr_kw, "adaptive")
    out["search"]["+adaptive"] = log
    kw_b, log = grid_search(EloMMR, alphas, tune_ds, tune_range, kw_t, "both")
    out["search"]["+both"] = log
    chosen["Elo-MMR + time drift"] = (EloMMR, kw_t)
    chosen["Elo-MMR + adaptive weight"] = (EloMMR, kw_a)
    chosen["Elo-MMR + both"] = (EloMMR, kw_b)
    if args.dataset == "atcoder":
        chosen["AtCoder (official)"] = (AtCoderOfficial, {})

    famous = ["tourist", "Petr", "Um_nik", "Radewoosh", "Benq", "ecnerwala", "jiangly", "maroonrk"]
    handle_ids = {h: i for i, h in enumerate(ds.handles)} if ds.handles else {}
    record = [handle_ids[h] for h in famous if h in handle_ids]
    for name, (cls, kw) in chosen.items():
        t0 = time.perf_counter()
        system = cls(**kw)
        r = run(system, ds, eval_from=n_train, record=record if handle_ids else None)
        if handle_ids:
            r["trajectories"] = {ds.handles[p]: v for p, v in r["trajectories"].items()}
            active = np.flatnonzero(system.n_played >= 30)
            top = active[np.argsort(-system.mu[active])[:20]]
            r["leaderboard"] = [(ds.handles[p], float(system.mu[p]), int(system.n_played[p])) for p in top]
        elif ds.true_skill is not None:
            # synthetic: distance of the final ratings from the true skills
            played = system.n_played > 0
            err = system.mu[played] - ds.true_skill[-1][played]
            r["rmse_vs_true_skill"] = float(np.sqrt(np.mean((err - err.mean()) ** 2)))
        r["params"] = kw
        r["wall"] = time.perf_counter() - t0
        if getattr(system, "round_weights", None):
            r["round_weights"] = system.round_weights
        out["systems"][name] = r
        print(f"TEST {name}: pair {r['pair_inv']:.3f}  dev {r['rank_dev']:.3f}  time {r['seconds']:.1f}s",
              flush=True)

    #pilot runs go to a separate file so they never overwrite the full results
    tag = f"_pilot{args.max_contests}" if args.max_contests else ""
    with open(os.path.join(RESULTS, f"{args.dataset}{tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
