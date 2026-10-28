# Figures from results/*.json
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
from theory_figures import C, FIG, save

PAPER = {  # Table 2 of the paper (pair inv., rank dev.)
    "Codeforces": {"Codeforces": (78.3, 14.9), "TopCoder": (78.5, 15.1), "Elo-MMchi": (78.5, 14.8),
                   "Elo-MMR(rho)": (78.6, 14.7)},
    "Synthetic": {"Codeforces": (81.7, 12.9), "TopCoder": (81.7, 12.8), "Elo-MMchi": (81.7, 12.8),
                  "Elo-MMR(rho)": (81.7, 12.8)},
}
BASE = ["Glicko", "Codeforces", "TopCoder", "Elo-MMchi", "Elo-MMR(rho)"]
IMPR = ["Elo-MMR + time drift", "Elo-MMR + adaptive weight", "Elo-MMR + both"]
LABEL = {"Elo-MMchi": "Elo-MMχ", "Elo-MMR(rho)": "Elo-MMR(ρ)", "Elo-MMR + time drift": "+ time drift",
         "Elo-MMR + adaptive weight": "+ adaptive weight", "Elo-MMR + both": "+ both"}


def bars(res, dataset):
    names = BASE + IMPR
    sys_ = res["systems"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, title, better in [(axes[0], "pair_inv", "Pair inversion accuracy (%)", "higher"),
                                   (axes[1], "rank_dev", "Rank deviation (%)", "lower")]:
        vals = [sys_[n][key] for n in names]
        cols = [C["blue"] if n in BASE[:3] else C["orange"] if n in BASE else C["aqua"] for n in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=cols, height=0.62)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.6 + 0.02
        ax.set_xlim(lo - pad, hi + pad * 0.6)
        for yy, v in zip(y, vals):
            ax.text(v + (hi - lo) * 0.02, yy, f"{v:.3f}", va="center", fontsize=10, color=C["ink"])
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL.get(n, n) for n in names])
        ax.set_title(f"{title} — {better} is better", loc="left", fontsize=12)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"{dataset}: test split (last 90% of rounds), players with ≥ 5 prior rounds",
                 x=0.01, ha="left", fontsize=13)
    save(fig, f"results_{dataset.lower()}.png")


CATS = [("April Fools", r"april fools"), ("Educational", r"educational"), ("Div. 4", r"div\. ?4"),
        ("Div. 3", r"div\. ?3"), ("Global", r"global"), ("Div. 1", r"div\. ?1(?!\s*\+)"),
        ("Div. 1 + 2", r"div\. ?1\s*\+\s*div\. ?2|combined"), ("Div. 2", r"div\. ?2"),
        ("Other", r".")]


def category(name):
    n = name.lower()
    for cat, pat in CATS:
        if re.search(pat, n):
            return cat
    return "Other"


def adaptive_weights(res, contest_names):
    rw = res["systems"]["Elo-MMR + both"].get("round_weights") or res["systems"]["Elo-MMR + adaptive weight"]["round_weights"]
    by = {}
    for idx, s, w in rw:
        if s > 0:
            by.setdefault(category(contest_names[int(idx)]), []).append(w)
    order = [c for c, _ in CATS if c in by]
    order.sort(key=lambda c: np.median(by[c]))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.boxplot([by[c] for c in order], vert=False, widths=0.55, patch_artist=True,
               boxprops=dict(facecolor="#d9e7f8", color=C["blue"]), medianprops=dict(color=C["orange"], lw=2),
               whiskerprops=dict(color=C["muted"]), capprops=dict(color=C["muted"]),
               flierprops=dict(marker="o", ms=3, markerfacecolor=C["muted"], markeredgecolor="none"))
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{c}  (n={len(by[c])})" for c in order])
    ax.axvline(1.0, color=C["muted"], lw=1, ls="--")
    ax.set_xlabel("learned round weight  (1 = paper's fixed weight)")
    ax.set_title("Adaptive round weighting rediscovers which contest types are noisy", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "adaptive_weights.png")
    return {c: float(np.median(by[c])) for c in order}


def trajectories(res, times):
    tr = res["systems"]["Elo-MMR(rho)"]["trajectories"]
    import datetime as dt
    years = [dt.datetime.utcfromtimestamp(t).year + dt.datetime.utcfromtimestamp(t).timetuple().tm_yday / 366
             for t in times]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    palette = [C["blue"], C["orange"], C["aqua"], C["yellow"], "#e87ba4", "#008300", C["violet"], C["red"]]
    for (h, v), col in zip(tr.items(), palette):
        v = np.array(v, dtype=float)
        ax.plot(years, v, color=col, lw=1.8, label=h)
    ax.set_ylabel("Elo-MMR rating μ")
    ax.set_title("Elo-MMR rating trajectories of top Codeforces competitors", loc="left")
    ax.legend(ncol=4, loc="lower right", fontsize=10)
    save(fig, "trajectories.png")


if __name__ == "__main__":
    for ds in ["synthetic", "codeforces"]:
        path = os.path.join(ROOT, "results", f"{ds}.json")
        if not os.path.exists(path):
            continue
        res = json.load(open(path))
        bars(res, ds.capitalize())
        if ds == "codeforces":
            from data import load_codeforces
            data = load_codeforces()
            names = [c.name for c in data.contests]
            med = adaptive_weights(res, names)
            print("median adaptive weight by category:", med)
            json.dump(med, open(os.path.join(ROOT, "results", "adaptive_by_category.json"), "w"), indent=1)
            trajectories(res, [c.time for c in data.contests])
