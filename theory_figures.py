# Figures that illustrate / empirically verify the paper's theory (Sections 3-5)
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.abspath(__file__))
import kernels as K
from data import Contest, Dataset, make_synthetic
from evaluate import run
from systems import Codeforces, EloMMR, TopCoder

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "violet": "#6250d6", "red": "#e34948", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": C["muted"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["muted"], "ytick.color": C["muted"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": C["grid"], "grid.linewidth": 0.8, "lines.linewidth": 2.2,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
TM = np.pi / np.sqrt(3)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig_loss_pdf():
    x = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, 0.5 * x ** 2, color=C["blue"], label="$L_2(x)=x^2/2$  (Gaussian)")
    ax[0].plot(x, 2 * np.log(np.cosh(np.pi * x / np.sqrt(12))), color=C["orange"],
               label="$L_R(x)=2\\ln\\cosh(\\pi x/\\sqrt{12})$  (logistic)")
    ax[0].set_title("Penalty per performance (negative log-density)", loc="left")
    ax[0].set_xlabel("standardised deviation $(s-p_k)/\\beta_k$")
    ax[0].legend(loc="upper center")
    ax[1].plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color=C["blue"], label="Normal")
    s = np.sqrt(3) / np.pi
    ax[1].plot(x, 1 / (4 * s) / np.cosh(x / (2 * s)) ** 2, color=C["orange"], label="Logistic (same variance)")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-4, 1)
    ax[1].set_title("Densities (log scale): logistic has heavier tails", loc="left")
    ax[1].legend(loc="lower center")
    save(fig, "loss_pdf.png")


def posterior_mode(prior_mu, prior_sig, perfs, betas):
    # Zero of Eq
    def dL(s):
        v = (s - prior_mu) / prior_sig ** 2
        for p, b in zip(perfs, betas):
            v += np.pi / (b * np.sqrt(3)) * np.tanh(np.pi * (s - p) / (b * np.sqrt(12)))
        return v
    return brentq(dL, -1e5, 1e5)


def fig_robustness():
    beta, sig0 = 200.0, 120.0
    p = np.linspace(-2500, 5500, 300)
    hist_consistent = [1500.0] * 8
    rng = np.random.default_rng(3)
    hist_volatile = list(1500 + rng.choice([-1, 1], 8) * 450.0)
    mu_c = posterior_mode(1500, sig0, hist_consistent, [beta] * 8)
    mu_v = posterior_mode(1500, sig0, hist_volatile, [beta] * 8)
    d_c = [posterior_mode(1500, sig0, hist_consistent + [q], [beta] * 9) - mu_c for q in p]
    d_v = [posterior_mode(1500, sig0, hist_volatile + [q], [beta] * 9) - mu_v for q in p]
    # Gaussian (Elo-MMchi) update with the same total weight: linear, unbounded
    w_prior = 1 / sig0 ** 2 + 8 / beta ** 2
    d_g = (p - 1500) * (1 / beta ** 2) / (w_prior + 1 / beta ** 2)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(p - 1500, d_g, color=C["blue"], label="Gaussian model (Elo-MM$\\chi$): unbounded")
    ax.plot(p - 1500, d_v, color=C["yellow"], label="Elo-MMR, volatile history: larger cap")
    ax.plot(p - 1500, d_c, color=C["orange"], label="Elo-MMR, consistent history: small cap")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xlabel("performance in new round minus current rating")
    ax.set_ylabel("rating change $\\mu_t-\\mu_{t-1}$")
    ax.set_title("Theorem 5.7 — robust response: one extreme round has a bounded effect", loc="left")
    ax.legend(loc="upper left")
    save(fig, "robustness.png")


def fig_monotonicity():
    # Theorem 5.5
    base = make_synthetic(num_players=3000, rounds=30, participation=0.5, seed=5)
    c10 = base.contests[10]
    x = int(c10.ids[len(c10.ids) // 2])
    others = c10.ids[c10.ids != x]
    finals, finals_x = [], []
    positions = np.linspace(0, len(others), 41).astype(int)
    for pos in positions:
        ids = np.insert(others, pos, x)
        contests = list(base.contests)
        ar = np.arange(len(ids), dtype=np.int64)
        contests[10] = Contest(10, c10.name, c10.time, ids.astype(np.int64), ar, ar.copy())
        ds = Dataset("mono", contests, base.num_players)
        # exact opponent sets: subsampling may break monotonicity by a few tenths of a point
        s = EloMMR(weight_limit=0.2, sig_limit=80, rho=1.0, subsample=10 ** 9)
        run(s, ds, eval_from=10 ** 9)
        finals.append(s.mu[x])
        s2 = EloMMR(weight_limit=0.2, sig_limit=80, rho=1.0, subsample=10 ** 9)
        run(s2, Dataset("mono", contests[:11], base.num_players), eval_from=10 ** 9)
        finals_x.append(s2.mu[x])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(positions + 1, finals_x, color=C["orange"], marker="o", ms=4, label="rating right after round 10")
    ax.plot(positions + 1, finals, color=C["blue"], marker="o", ms=4, label="rating after round 30")
    ax.invert_xaxis()
    ax.set_xlabel("player X's place in round 10  (better →)")
    ax.set_ylabel("rating of player X")
    ax.set_title("Theorem 5.5 — aligned incentives: placing higher never lowers your rating", loc="left")
    ax.legend(loc="upper left")
    d1, d2 = np.diff(finals_x), np.diff(finals)
    print("monotone (all diffs <= 0 as place grows):", bool((d1 <= 1e-9).all()), bool((d2 <= 1e-9).all()))
    save(fig, "monotonicity.png")


def fig_runtime():
    # width=1e-6 disables rating-bucket grouping, so every system runs its exact algorithm
    sizes = [1000, 2000, 5000, 10000, 20000]
    ex = 1e-6
    systems = [("Elo-MMR (500 nearest opponents)", lambda: EloMMR(subsample=500, width=ex), C["orange"]),
               ("Elo-MMR (all opponents)", lambda: EloMMR(subsample=10 ** 9, width=ex), C["yellow"]),
               ("Codeforces", lambda: Codeforces(width=ex), C["blue"]),
               ("TopCoder", lambda: TopCoder(width=ex), C["aqua"])]
    res = {name: [] for name, _, _ in systems}
    warm = make_synthetic(num_players=300, rounds=2, participation=1.0, seed=1)
    for _, mk, _ in systems:
        run(mk(), warm, eval_from=10 ** 9)   # trigger JIT compilation before timing
    for n in sizes:
        ds = make_synthetic(num_players=n, rounds=4, participation=1.0, seed=1)
        for name, mk, _ in systems:
            r = run(mk(), ds, eval_from=10 ** 9)
            res[name].append(r["seconds"] / 4)
        print(n, {k: round(v[-1], 3) for k, v in res.items()}, flush=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for name, _, col in systems:
        ax.plot(sizes, res[name], color=col, marker="o", ms=6, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("participants in one round")
    ax.set_ylabel("seconds per round (12 threads)")
    ax.set_title("Runtime per round: subsampled Elo-MMR scales linearly", loc="left")
    ax.legend(loc="upper left")
    save(fig, "runtime.png")
    return dict(sizes=sizes, **res)


def fig_pseudodiffusion():
    # Weight left on one logistic factor (its 'memory') after m more rounds, kappa^((1+rho) m)
    w, = 0.2,
    kappa = 1 / (1 + w)   # steady state: sigma^2 -> sig_lim^2 and gamma^2 = w sig_lim^2
    m = np.arange(0, 31)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for rho, col in [(0.0, C["blue"]), (0.04, C["aqua"]), (1.0, C["orange"]), (4.0, C["violet"])]:
        ax.plot(m, kappa ** ((1 + rho) * m), color=col, marker="o", ms=3, label=f"$\\rho$ = {rho}")
    ax.set_xlabel("rounds since the performance")
    ax.set_ylabel("remaining logistic weight (fraction)")
    ax.set_title("Pseudodiffusion: old performances decay and transfer to the Gaussian prior", loc="left")
    ax.legend(loc="upper right")
    save(fig, "pseudodiffusion.png")


if __name__ == "__main__":
    only = sys.argv[1:]
    if not only or "theory" in only:
        fig_loss_pdf()
        fig_robustness()
        fig_pseudodiffusion()
    if not only or "mono" in only:
        fig_monotonicity()
    if only and "runtime" not in only:
        sys.exit()
    import json
    rt = fig_runtime()
    with open(os.path.join(ROOT, "results", "runtime.json"), "w") as f:
        json.dump(rt, f, indent=1)
