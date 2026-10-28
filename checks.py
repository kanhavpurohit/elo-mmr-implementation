# Checks on why our numbers differ from the paper, and how solid our improvements are
import itertools
import json
import math
import os
import sys
import time

import numpy as np

from data import load_atcoder, load_codeforces, make_synthetic
from evaluate import run
from systems import AtCoderOfficial, Codeforces, EloMMR, TopCoder

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FINE_GRIDS = {
    "Codeforces": (Codeforces, {"beta": [30, 40, 50, 65, 80, 100], "weight": [0.4, 0.631, 1.0, 1.5, 2.0]}),
    # TopCoder needs weight < 1, otherwise w / (1 - w) changes sign
    "TopCoder": (TopCoder, {"weight_noob": [0.6, 0.75, 0.9, 0.95], "weight_limit": [0.06, 0.09, 0.12, 0.18, 0.25]}),
    "Elo-MMchi": (EloMMR, {"gaussian": [True], "weight_limit": [0.05, 0.075, 0.1, 0.15, 0.2, 0.23, 0.3, 0.45],
                           "sig_limit": [20, 25, 30, 40, 54, 60, 74, 80, 110]}),
    "Elo-MMR(rho)": (EloMMR, {"rho": [0.04], "weight_limit": [0.05, 0.075, 0.1, 0.15, 0.2, 0.23, 0.3, 0.45],
                              "sig_limit": [20, 25, 30, 40, 54, 60, 74, 80, 110]}),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tune(cls, grid, ds, n_train, label):
    keys = list(grid)
    best = None
    for vals in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, vals))
        r = run(cls(**kw), ds, eval_from=0, eval_to=n_train)
        if best is None or r["pair_inv"] > best[1]:
            best = (kw, r["pair_inv"])
    on_edge = [k for k in keys if len(grid[k]) > 1 and best[0][k] in (grid[k][0], grid[k][-1])]
    log(f"  tuned {label}: {best[0]} train={best[1]:.3f} edge={on_edge}")
    return best[0], best[1], on_edge


def bootstrap(per_a, per_b, reps=2000, seed=0):
    # 95% CI of pair_inv(a) - pair_inv(b), resampling test contests (weighted by size)
    a = {c: (n, p) for c, n, p, _ in per_a}
    b = {c: (n, p) for c, n, p, _ in per_b}
    common = sorted(set(a) & set(b))
    n = np.array([a[c][0] for c in common])
    pa = np.array([a[c][1] for c in common])
    pb = np.array([b[c][1] for c in common])
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(common), len(common))
        diffs[i] = pa[idx].sum() / n[idx].sum() - pb[idx].sum() / n[idx].sum()
    point = pa.sum() / n.sum() - pb.sum() / n.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)], "significant": bool(lo > 0 or hi < 0)}


def real_dataset(name):
    ds = load_codeforces() if name == "codeforces" else load_atcoder()
    n_train = len(ds) // 10
    log(name, ds.summary(), "train contests:", n_train)
    out = {"summary": ds.summary(), "fine_grid": {}, "test": {}, "bootstrap": {}}

    configs = {}
    for label, (cls, grid) in FINE_GRIDS.items():
        kw, train, edge = tune(cls, grid, ds, n_train, label)
        out["fine_grid"][label] = {"params": kw, "train_pair": train, "on_grid_edge": edge}
        configs[label + " (fine grid)"] = (cls, kw)
    mmr = out["fine_grid"]["Elo-MMR(rho)"]["params"]
    mmx = out["fine_grid"]["Elo-MMchi"]["params"]
    drift_grid = {"drift_per_day": [1, 3, 10, 30, 100]}
    kw, _, _ = tune(EloMMR, {**{k: [v] for k, v in mmr.items()}, **drift_grid}, ds, n_train, "MMR + drift")
    configs["Elo-MMR(rho) + drift (fine grid)"] = (EloMMR, kw)
    kw, _, _ = tune(EloMMR, {**{k: [v] for k, v in mmx.items()}, **drift_grid}, ds, n_train, "MMchi + drift")
    configs["Elo-MMchi + drift (fine grid)"] = (EloMMR, kw)
    # the authors' published Codeforces settings, as a fixed reference on both datasets
    configs["Elo-MMR(rho) authors"] = (EloMMR, dict(weight_limit=0.23, sig_limit=54, rho=0.04))
    configs["Elo-MMR(rho) authors + drift10"] = (EloMMR, dict(weight_limit=0.23, sig_limit=54, rho=0.04,
                                                              drift_per_day=10))
    configs["Elo-MMchi authors"] = (EloMMR, dict(weight_limit=0.16, sig_limit=74, gaussian=True))
    configs["Elo-MMchi authors + drift10"] = (EloMMR, dict(weight_limit=0.16, sig_limit=74, gaussian=True,
                                                           drift_per_day=10))
    if name == "atcoder":
        configs["AtCoder (official)"] = (AtCoderOfficial, {})

    per = {}
    for label, (cls, kw) in configs.items():
        r = run(cls(**kw), ds, eval_from=n_train)
        per[label] = r["per_contest"]
        out["test"][label] = {"params": kw, "pair_inv": r["pair_inv"], "rank_dev": r["rank_dev"],
                              "seconds": r["seconds"]}
        log(f"  TEST {label}: pair {r['pair_inv']:.3f} dev {r['rank_dev']:.3f}")

    pairs = [("Elo-MMR(rho) (fine grid)", "TopCoder (fine grid)"),
             ("Elo-MMR(rho) (fine grid)", "Codeforces (fine grid)"),
             ("Elo-MMchi (fine grid)", "TopCoder (fine grid)"),
             ("Elo-MMR(rho) + drift (fine grid)", "Elo-MMR(rho) (fine grid)"),
             ("Elo-MMchi + drift (fine grid)", "Elo-MMchi (fine grid)"),
             ("Elo-MMR(rho) authors + drift10", "Elo-MMR(rho) authors"),
             ("Elo-MMchi authors + drift10", "Elo-MMchi authors")]
    if name == "atcoder":
        pairs.append(("Elo-MMR(rho) (fine grid)", "AtCoder (official)"))
    for a, b in pairs:
        out["bootstrap"][f"{a}  minus  {b}"] = bs = bootstrap(per[a], per[b])
        log(f"  {a} - {b}: {bs['diff']:+.3f}  CI [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]")
    return out


def synthetic_variants():
    # Which reading of the paper's synthetic setup reproduces its 81.7%?
    variants = {
        "sd, 25% participation (ours)": dict(participation=0.25, skill_sd=300, perf_sd=200, drift_sd=35),
        "sd, all players every round": dict(participation=1.0, skill_sd=300, perf_sd=200, drift_sd=35),
        "variance, 25% participation": dict(participation=0.25, skill_sd=math.sqrt(300),
                                            perf_sd=math.sqrt(200), drift_sd=math.sqrt(35)),
        "variance, all players every round": dict(participation=1.0, skill_sd=math.sqrt(300),
                                                  perf_sd=math.sqrt(200), drift_sd=math.sqrt(35)),
    }
    out = {}
    for label, spec in variants.items():
        scale = spec["skill_sd"] / 300.0
        tune_ds, test_ds = make_synthetic(seed=1, **spec), make_synthetic(seed=0, **spec)
        res = {}
        for sys_label, base in (("Elo-MMR(rho)", {"rho": 1.0}), ("Elo-MMchi", {"gaussian": True})):
            best = None
            for wl, sl in itertools.product([0.1, 0.2, 0.3, 0.45, 0.6], [40, 60, 80, 110, 150]):
                s = EloMMR(weight_limit=wl, sig_limit=sl * scale, **base)
                s.sig_noob = 350.0 * scale
                r = run(s, tune_ds, eval_from=5)
                if best is None or r["pair_inv"] > best[1]:
                    best = ((wl, sl), r["pair_inv"])
            s = EloMMR(weight_limit=best[0][0], sig_limit=best[0][1] * scale, **base)
            s.sig_noob = 350.0 * scale
            r = run(s, test_ds, eval_from=5)
            res[sys_label] = {"weight_limit": best[0][0], "sig_limit": best[0][1] * scale,
                              "pair_inv": r["pair_inv"], "rank_dev": r["rank_dev"]}
            log(f"  {label} / {sys_label}: pair {r['pair_inv']:.3f} dev {r['rank_dev']:.3f}")
        out[label] = res
    out["paper"] = {"pair_inv": 81.7, "rank_dev": 12.8}
    return out


def synthetic_seeds(seeds=(0, 1, 2, 3, 4)):
    # Time drift on the default synthetic set, repeated over independent draws
    base = dict(weight_limit=0.3, sig_limit=150, rho=1.0)       # tuned Elo-MMR(rho) on synthetic
    drift = dict(weight_limit=0.3, sig_limit=80, rho=1.0, drift_per_day=100)  # tuned + time drift
    rows = []
    for sd in seeds:
        ds = make_synthetic(seed=sd + 10)
        a = run(EloMMR(**base), ds, eval_from=5)["pair_inv"]
        b = run(EloMMR(**drift), ds, eval_from=5)["pair_inv"]
        rows.append({"seed": sd + 10, "base": a, "drift": b, "gain": b - a})
        log(f"  seed {sd + 10}: base {a:.3f}  drift {b:.3f}  gain {b - a:+.3f}")
    g = np.array([r["gain"] for r in rows])
    return {"runs": rows, "mean_gain": float(g.mean()), "sd_gain": float(g.std(ddof=1)),
            "all_positive": bool((g > 0).all())}


def main():
    parts = sys.argv[1:] or ["codeforces", "atcoder", "synthetic", "seeds"]
    os.makedirs(RESULTS, exist_ok=True)
    for part in parts:
        log(f"=== {part}")
        if part in ("codeforces", "atcoder"):
            res = real_dataset(part)
        elif part == "synthetic":
            res = synthetic_variants()
        elif part == "seeds":
            res = synthetic_seeds()
        else:
            raise SystemExit(f"unknown part {part}")
        with open(os.path.join(RESULTS, f"replication_checks_{part}.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
