import math
import numpy as np

import kernels as K
from data import make_synthetic
from evaluate import run
from systems import Codeforces, EloMMR

TM = math.pi / math.sqrt(3)


def bisect(f, lo=-6000.0, hi=9000.0):
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if f(m) > 0 else (lo, m)
    return 0.5 * (lo + hi)


def reference(ds, wl, sl, rho):
    #plain python version of algorithms 1-3 straight from the paper, no grouping or subsampling
    beta, gam = math.sqrt(1 + 1 / wl) * sl, math.sqrt(wl) * sl
    st = {}
    for c in ds.contests:
        ids = list(c.ids)
        for i in ids:
            st.setdefault(i, dict(mu=1500.0, sig=350.0, p0=1500.0, w0=350.0 ** -2, hist=[]))
        for i in ids:  #pseudodiffusion (alg 2)
            s = st[i]
            kappa = 1 / (1 + gam ** 2 / s["sig"] ** 2)
            tot = s["w0"] + sum(o / b ** 2 for _, b, o in s["hist"])
            wg, wl_ = kappa ** rho * s["w0"], (1 - kappa ** rho) * tot
            s["p0"] = (wg * s["p0"] + wl_ * s["mu"]) / (wg + wl_)
            s["w0"] = kappa * (wg + wl_)
            s["hist"] = [(p, b, o * kappa ** (1 + rho)) for p, b, o in s["hist"]]
            s["sig"] /= math.sqrt(kappa)
        mus = np.array([st[i]["mu"] for i in ids])
        sc = np.array([math.sqrt(st[i]["sig"] ** 2 + beta ** 2) for i in ids]) / TM
        perf = []
        for k in range(len(ids)):  #phase 1, theorem 3.2
            lost, won, tie = c.lo < c.lo[k], c.lo > c.lo[k], c.lo == c.lo[k]
            def q(p):
                F = 1 / (1 + np.exp(-(p - mus) / sc))
                return (np.where(lost, -F, 0) + np.where(won, 1 - F, 0) + np.where(tie, 1 - 2 * F, 0)) @ (1 / sc)
            perf.append(bisect(q))
        for k, i in enumerate(ids):  #phase 2, eq 6
            s = st[i]
            s["hist"].append((perf[k], beta, 1.0))
            def lp(x):
                return -(s["w0"] * (x - s["p0"]) + sum(o * TM / b * math.tanh((x - p) * TM / (2 * b)) for p, b, o in s["hist"]))
            s["mu"] = bisect(lp)
            s["sig"] = (s["sig"] ** -2 + beta ** -2) ** -0.5
    return st


def test_matches_paper_reference():
    ds = make_synthetic(num_players=150, rounds=5, participation=0.7, seed=3)
    s = EloMMR(weight_limit=0.2, sig_limit=80, rho=1.0, subsample=10 ** 6, history=100, width=0.05)
    run(s, ds)
    ref = reference(ds, 0.2, 80, 1.0)
    assert max(abs(s.mu[i] - r["mu"]) for i, r in ref.items()) < 1e-2
    assert max(abs(s.sig[i] - r["sig"]) for i, r in ref.items()) < 1e-6


def test_exact_codeforces_matches_bucketed():
    ds = make_synthetic(num_players=400, rounds=4, participation=0.5, seed=2)
    a, b = Codeforces(beta=200), Codeforces(beta=200, exact=True)
    run(a, ds), run(b, ds)
    m = a.n_played > 0
    assert np.abs(a.mu[m] - b.mu[m]).max() < 0.01


def test_metrics_perfect_order():
    r = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ar = np.arange(5)
    n, pair, dev = K.contest_metrics(r, ar, ar.copy())
    assert n == 5 and abs(pair / n - 100) < 1e-9 and dev == 0


def test_better_rank_never_hurts():
    #theorem 5.5 on a tiny case: moving one player up a place raises their rating
    ds = make_synthetic(num_players=60, rounds=3, participation=1.0, seed=1)
    base = EloMMR(subsample=10 ** 6); run(base, ds)
    c = ds.contests[-1]
    i = 10  #swap places 10 and 9 in the last round
    c.ids[[i - 1, i]] = c.ids[[i, i - 1]]
    p = c.ids[i - 1]
    better = EloMMR(subsample=10 ** 6); run(better, ds)
    assert better.mu[p] > base.mu[p]
