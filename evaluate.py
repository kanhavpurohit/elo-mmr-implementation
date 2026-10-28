# Run a rating system over a dataset and score its predictions (Section 6.2)
import time

import numpy as np

import kernels as K


def run(system, dataset, eval_from=0, eval_to=None, min_history=5, record=None):
    # Process every contest in order
    eval_to = len(dataset) if eval_to is None else eval_to
    system.reset(dataset)
    tot_n = tot_pair = tot_dev = 0.0
    per_contest = []  # (contest index, n, pair_sum, dev_sum), for bootstrap error bars
    traj = {p: [] for p in (record or [])}
    elapsed = 0.0
    for c in dataset.contests:
        if c.index >= eval_to:
            break  # nothing later is scored
        system.prepare(c)
        if eval_from <= c.index < eval_to:
            n, pair, dev = K.eval_subset(c.ids, c.lo, c.hi, system.ratings(), system.n_played,
                                         min_history)
            tot_n += n
            tot_pair += pair
            tot_dev += dev
            if n > 0:
                per_contest.append((c.index, n, pair, dev))
        t0 = time.perf_counter()
        system.round(c)
        elapsed += time.perf_counter() - t0
        for p in traj:
            traj[p].append(system.ratings()[p] if system.n_played[p] else np.nan)
    out = {"system": system.name, "pair_inv": tot_pair / max(tot_n, 1),
           "rank_dev": tot_dev / max(tot_n, 1), "evaluated": int(tot_n), "seconds": elapsed,
           "per_contest": per_contest}
    if record:
        out["trajectories"] = traj
    return out
