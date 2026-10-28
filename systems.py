# Rating systems
import math

import numpy as np

import kernels as K


class RatingSystem:
    name = "base"
    mu_noob = 1500.0
    sig_noob = 350.0

    def reset(self, dataset):
        P = dataset.num_players
        self.mu = np.zeros(P)
        self.sig = np.zeros(P)
        self.n_played = np.zeros(P, dtype=np.int64)

    def prepare(self, contest):
        pass  # Called before a contest is scored; most systems need nothing here

    def round(self, contest):
        raise NotImplementedError

    def ratings(self):
        return self.mu


class EloMMR(RatingSystem):
    # Elo-MMR(rho) with logistic performances, or Elo-MMchi when gaussian=True

    def __init__(self, weight_limit=0.2, sig_limit=80.0, rho=1.0, gaussian=False,
                 split_ties=False, subsample=500, history=32, width=2.0,
                 drift_per_day=0.0, contest_weights=None, adaptive_alpha=0.0, name=None):
        self.weight_limit = weight_limit
        self.sig_limit = sig_limit
        self.rho = rho
        self.gaussian = gaussian
        self.split_ties = split_ties
        self.subsample = subsample
        self.history = history
        self.width = width
        self.drift_per_day = drift_per_day
        self.contest_weights = contest_weights
        self.adaptive_alpha = adaptive_alpha
        self.name = name or ("Elo-MMchi" if gaussian else f"Elo-MMR(rho={rho})")

    def reset(self, dataset):
        super().reset(dataset)
        P = dataset.num_players
        self.nmu = np.zeros(P)
        self.nsig = np.zeros(P)
        cap = np.minimum(dataset.player_counts, self.history).astype(np.int64)
        cap = np.maximum(cap, 1)
        self.h_cap = cap
        self.h_off = np.concatenate([[0], np.cumsum(cap)[:-1]]).astype(np.int64)
        tot = int(cap.sum())
        self.h_mu = np.zeros(tot)
        self.h_warg = np.zeros(tot)
        self.h_wout = np.zeros(tot)
        self.h_start = np.zeros(P, dtype=np.int64)
        self.h_len = np.zeros(P, dtype=np.int64)
        self.last_time = np.zeros(P, dtype=np.int64)
        self.log_s_hist = []
        self.round_weights = []

    def round(self, c):
        cw = 1.0
        if self.contest_weights is not None:
            cw = float(self.contest_weights[c.index])
        if self.adaptive_alpha > 0:
            cw *= self.adaptive_weight(c)
        perf = np.empty(len(c.ids))
        K.elo_mmr_round(c.ids, c.lo, c.time, cw,
                        self.mu, self.sig, self.nmu, self.nsig,
                        self.h_mu, self.h_warg, self.h_wout, self.h_off, self.h_cap,
                        self.h_start, self.h_len, self.n_played, self.last_time, perf,
                        self.mu_noob, self.sig_noob, self.weight_limit, self.sig_limit,
                        self.drift_per_day, self.rho, self.gaussian, self.split_ties,
                        self.subsample, self.width)
        return perf


    def adaptive_weight(self, c):
        # Improvement
        s, m = K.round_noise_scale(c.ids, c.lo, self.mu, self.n_played, 5, 600)
        w = 1.0
        if s > 0:
            if len(self.log_s_hist) >= 5:
                ref = float(np.mean(self.log_s_hist[-200:]))
                w = float(np.clip(np.exp(self.adaptive_alpha * (ref - np.log(s))), 0.3, 3.0))
            self.log_s_hist.append(np.log(s))
        self.round_weights.append((c.index, s, w))
        return w


class Codeforces(RatingSystem):
    def __init__(self, beta=400 * K.TANH_MULT / math.log(10), weight=1.0, width=2.0, exact=False):
        self.beta, self.weight, self.width, self.exact = beta, weight, width, exact
        self.name = "Codeforces (exact O(n^2))" if exact else "Codeforces"

    def round(self, c):
        K.codeforces_round(c.ids, c.lo, c.hi, self.mu, self.sig, self.n_played,
                           self.mu_noob, self.sig_noob, self.beta, self.weight, self.width,
                           self.exact)


class TopCoder(RatingSystem):
    def __init__(self, weight_noob=0.6, weight_limit=0.18, width=2.0):
        self.weight_noob, self.weight_limit, self.width = weight_noob, weight_limit, width
        self.name = "TopCoder"

    def round(self, c):
        K.topcoder_round(c.ids, c.lo, c.hi, self.mu, self.sig, self.n_played,
                         self.mu_noob, self.sig_noob, self.weight_noob, self.weight_limit,
                         self.width)


class AtCoderOfficial(RatingSystem):
    # AtCoder's own published ratings, read from the results (no hyperparameters)
    name = "AtCoder (official)"

    def prepare(self, c):
        self.mu[c.ids] = c.official

    def round(self, c):
        self.n_played[c.ids] += 1


class Glicko(RatingSystem):
    def __init__(self, beta=400 * K.TANH_MULT / math.log(10), sig_drift=35.0, width=2.0):
        self.beta, self.sig_drift, self.width = beta, sig_drift, width
        self.name = "Glicko"

    def round(self, c):
        K.glicko_round(c.ids, c.lo, self.mu, self.sig, self.n_played,
                       self.mu_noob, self.sig_noob, self.beta, self.sig_drift, self.width)
