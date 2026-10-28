# Numba kernels shared by all rating systems
import math

import numpy as np
from numba import njit, prange

TANH_MULT = math.pi / math.sqrt(3.0)  # converts std-dev to logistic scale: s = sqrt(3)/pi * sigma
BOUNDS_LO, BOUNDS_HI = -6000.0, 9000.0
SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# Basic numerics
@njit
def norm_pdf(z):
    return INV_SQRT_2PI * math.exp(-0.5 * z * z)


@njit
def norm_cdf(z):
    return 0.5 * math.erfc(-z / SQRT2)


@njit
def norm_cdf_inv(p):
    # Acklam's rational approximation followed by one Halley refinement step.
    if p <= 0.0:
        return -40.0
    if p >= 1.0:
        return 40.0
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


@njit
def logistic_cdf(z):
    # CDF of a logistic with unit standard deviation
    return 0.5 + 0.5 * math.tanh(0.5 * TANH_MULT * z)


@njit
def mills_ratio(z):
    # Pdf(z)/cdf(z), numerically stable for very negative z
    if z > -30.0:
        return norm_pdf(z) / norm_cdf(z)
    # asymptotic expansion: phi/Phi ~ -z / (1 - 1/z^2 + 3/z^4)
    z2 = z * z
    return -z / (1.0 - 1.0 / z2 + 3.0 / (z2 * z2))


@njit
def lower_bound(arr, lo, hi, x):
    # First index in arr[lo:hi] with arr[idx] >= x
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit
def upper_bound(arr, lo, hi, x):
    # First index in arr[lo:hi] with arr[idx] > x
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit
def bucket(a, width):
    return int(round(a / width))


# Grouping opponents into (mu, sigma) buckets
@njit
def build_groups(term_mu, term_sig, lo, width):
    # Sort terms by (mu bucket, sigma bucket, rank) and merge equal buckets
    n = term_mu.shape[0]
    keys = np.empty(n, dtype=np.int64)
    for k in range(n):
        bm = bucket(term_mu[k], width) + 100000
        bs = bucket(term_sig[k], width)
        keys[k] = (bm * 20000 + bs) * 1000000 + lo[k]
    order = np.argsort(keys, kind="mergesort")

    g_mu = np.empty(n)
    g_sig = np.empty(n)
    g_bmu = np.empty(n, dtype=np.int64)
    g_off = np.empty(n + 1, dtype=np.int64)
    ranks = np.empty(n, dtype=np.int64)
    ng = 0
    last = -1
    cnt = 0
    for idx in range(n):
        k = order[idx]
        gk = keys[k] // 1000000
        if gk != last:
            if ng > 0:
                g_mu[ng - 1] /= cnt
                g_sig[ng - 1] /= cnt
            g_off[ng] = idx
            g_mu[ng] = 0.0
            g_sig[ng] = 0.0
            g_bmu[ng] = bucket(term_mu[k], width)
            ng += 1
            cnt = 0
            last = gk
        g_mu[ng - 1] += term_mu[k]
        g_sig[ng - 1] += term_sig[k]
        ranks[idx] = lo[k]
        cnt += 1
    g_mu[ng - 1] /= cnt
    g_sig[ng - 1] /= cnt
    g_off[ng] = n
    return g_mu[:ng], g_sig[:ng], g_bmu[:ng], g_off[:ng + 1], ranks


@njit
def subsample_window(g_bmu, rating, width, size):
    # Contiguous window of `size` groups whose ratings are closest to `rating`
    ng = g_bmu.shape[0]
    if size >= ng:
        return 0, ng
    b = bucket(rating, width)
    beg = lower_bound(g_bmu, 0, ng, b)
    end = beg + 1
    expand = (size - (end - beg) + 1) // 2
    beg = max(0, beg - expand)
    end = min(ng, end + expand)
    expand = size - (end - beg)
    if expand > 0:
        beg = max(0, beg - expand)
        end = min(ng, end + expand)
    return beg, end


# Phase 1: performance estimation (Theorem 3.2): zero of
#     Q_i(p) = sum_{j lost to} l_j(p) + sum_{j tied} d_j(p) + sum_{j beaten} v_j(p)
@njit
def q_logistic(x, g_mu, g_sig, g_off, ranks, beg, end, my_rank, split_ties):
    # Logistic performance model
    val = 0.0
    der = 0.0
    for g in range(beg, end):
        w_out = TANH_MULT / g_sig[g]
        w_arg = 0.5 * w_out
        z = (x - g_mu[g]) * w_arg
        t = math.tanh(z)
        base = -t * w_out
        base_p = -(1.0 - t * t) * w_arg * w_out
        a = g_off[g]
        b = g_off[g + 1]
        s = lower_bound(ranks, a, b, my_rank)
        e = upper_bound(ranks, s, b, my_rank)
        less = s - a          # opponents ranked above me (I lost)
        equal = e - s         # tied (includes myself)
        greater = b - e       # opponents ranked below me (I won)
        total = float(b - a)
        if not split_ties:
            total += equal
        val += base * total + w_out * (greater - less)
        der += base_p * total
    return val, der


@njit
def gauss_term(x, mu, sig, order, split_ties):
    # (value, derivative) of the l / d / v function of one Gaussian opponent
    z = (x - mu) / sig
    s2 = sig * sig
    if order == -1:
        # l_j = -f/(1-F);  written via the Mills ratio for numerical stability
        m = mills_ratio(-z)
        return -m / sig, (z * m - m * m) / s2
    if order == 1:
        # v_j = f/F
        m = mills_ratio(z)
        return m / sig, (-z * m - m * m) / s2
    if split_ties:
        m0 = mills_ratio(z)
        m1 = mills_ratio(-z)
        return 0.5 * (m0 - m1) / sig, 0.5 * ((-z * m0 - m0 * m0) + (z * m1 - m1 * m1)) / s2
    # d_j = f'/f
    return -z / sig, -1.0 / s2


@njit
def q_gauss(x, g_mu, g_sig, g_off, ranks, beg, end, my_rank, split_ties):
    val = 0.0
    der = 0.0
    for g in range(beg, end):
        a = g_off[g]
        b = g_off[g + 1]
        s = lower_bound(ranks, a, b, my_rank)
        e = upper_bound(ranks, s, b, my_rank)
        if s > a:
            v, d = gauss_term(x, g_mu[g], g_sig[g], -1, split_ties)
            val += (s - a) * v
            der += (s - a) * d
        if e > s:
            v, d = gauss_term(x, g_mu[g], g_sig[g], 0, split_ties)
            val += (e - s) * v
            der += (e - s) * d
        if b > e:
            v, d = gauss_term(x, g_mu[g], g_sig[g], 1, split_ties)
            val += (b - e) * v
            der += (b - e) * d
    return val, der


@njit
def solve_perf(gaussian, g_mu, g_sig, g_off, ranks, beg, end, my_rank, split_ties):
    # Safeguarded Newton's method (port of numerical::solve_newton)
    lo, hi = BOUNDS_LO, BOUNDS_HI
    guess = 0.5 * (lo + hi)
    for _ in range(200):
        if gaussian:
            s, sp = q_gauss(guess, g_mu, g_sig, g_off, ranks, beg, end, my_rank, split_ties)
        else:
            s, sp = q_logistic(guess, g_mu, g_sig, g_off, ranks, beg, end, my_rank, split_ties)
        if sp == 0.0:
            sp = -1e-12
        ext = guess - s / sp
        if ext < guess:
            hi = guess
            guess = min(max(ext, hi - 0.75 * (hi - lo)), hi)
        else:
            lo = guess
            guess = min(max(ext, lo), lo + 0.75 * (hi - lo))
        if lo >= guess or guess >= hi or abs(s) < 1e-12:
            break
    return guess


# Phase 2 helper: robust average  (zero of  offset + slope*x + sum_k w_out_k tanh((x-mu_k) w_arg_k))
# Equation (6)/(7) of the paper.
@njit
def robust_average(h_mu, h_warg, h_wout, start, length, cap, off, offset, slope):
    lo, hi = BOUNDS_LO, BOUNDS_HI
    guess = 0.5 * (lo + hi)
    for _ in range(200):
        s = offset + slope * guess
        sp = slope
        for q in range(length):
            k = off + (start + q) % cap
            t = math.tanh((guess - h_mu[k]) * h_warg[k])
            s += t * h_wout[k]
            sp += (1.0 - t * t) * h_warg[k] * h_wout[k]
        if sp == 0.0:
            sp = 1e-12
        ext = guess - s / sp
        if ext < guess:
            hi = guess
            guess = min(max(ext, hi - 0.75 * (hi - lo)), hi)
        else:
            lo = guess
            guess = min(max(ext, lo), lo + 0.75 * (hi - lo))
        if lo >= guess or guess >= hi or abs(s) < 1e-12:
            break
    return guess


@njit
def robust_average_groups(g_mu, g_w, g_cnt, offset):
    # Zero of offset + sum_g cnt_g * w_g * tanh((x - mu_g) * w_g / 2) (Codeforces helper)
    lo, hi = BOUNDS_LO, BOUNDS_HI
    guess = 0.5 * (lo + hi)
    for _ in range(200):
        s = offset
        sp = 0.0
        for g in range(g_mu.shape[0]):
            t = math.tanh((guess - g_mu[g]) * 0.5 * g_w[g])
            s += g_cnt[g] * t * g_w[g]
            sp += g_cnt[g] * (1.0 - t * t) * 0.5 * g_w[g] * g_w[g]
        if sp == 0.0:
            sp = 1e-12
        ext = guess - s / sp
        if ext < guess:
            hi = guess
            guess = min(max(ext, hi - 0.75 * (hi - lo)), hi)
        else:
            lo = guess
            guess = min(max(ext, lo), lo + 0.75 * (hi - lo))
        if lo >= guess or guess >= hi or abs(s) < 1e-12:
            break
    return guess


# Elo-MMR round update (Algorithms 1-3)
@njit
def _add_noise_best(p, sig_noise, rho, mu, sig, nmu, nsig, h_mu, h_warg, h_wout,
                    h_off, h_cap, h_start, h_len):
    # Pseudodiffusion (Algorithm 2)
    new_sig = math.hypot(sig[p], sig_noise)
    kappa = (sig[p] / new_sig) ** 2
    transfer = kappa ** rho
    sig[p] = new_sig
    wn_old = nsig[p] ** -2
    wsum = 0.0
    for q in range(h_len[p]):
        k = h_off[p] + (h_start[p] + q) % h_cap[p]
        wsum += h_warg[k] * h_wout[k] * 2.0 / (TANH_MULT * TANH_MULT)
    w_from_norm = transfer * wn_old
    w_from_transfer = (1.0 - transfer) * (wn_old + wsum)
    w_total = w_from_norm + w_from_transfer
    nmu[p] = (w_from_norm * nmu[p] + w_from_transfer * mu[p]) / w_total
    nsig[p] = (kappa * w_total) ** -0.5
    for q in range(h_len[p]):
        k = h_off[p] + (h_start[p] + q) % h_cap[p]
        h_wout[k] *= transfer * kappa


@njit(parallel=True)
def elo_mmr_round(ids, lo, t_sec, contest_weight,
                  mu, sig, nmu, nsig, h_mu, h_warg, h_wout, h_off, h_cap, h_start, h_len,
                  n_played, last_time, perf_out,
                  mu_noob, sig_noob, weight_limit, sig_limit, drift_per_day, rho,
                  gaussian, split_ties, subsample, width):
    n = ids.shape[0]
    term_mu = np.empty(n)
    term_sig = np.empty(n)
    sig_perf_arr = np.empty(n)

    # ---- skill drift (Algorithm 2) and construction of opponent terms
    for k in prange(n):
        p = ids[k]
        if n_played[p] == 0:
            mu[p] = mu_noob
            sig[p] = sig_noob
            nmu[p] = mu_noob
            nsig[p] = sig_noob
            h_len[p] = 0
            h_start[p] = 0
            last_time[p] = t_sec
        w = weight_limit * contest_weight
        sig_perf = math.sqrt(1.0 + 1.0 / w) * sig_limit          # beta
        dt_days = (t_sec - last_time[p]) / 86400.0
        sig_drift = math.sqrt(w * sig_limit * sig_limit + drift_per_day * dt_days)  # gamma
        if gaussian or rho == np.inf:
            # Elo-MMchi / Elo-MMR(inf): collapse the posterior into one Gaussian
            sig[p] = math.hypot(sig[p], sig_drift)
            nmu[p] = mu[p]
            nsig[p] = sig[p]
            h_len[p] = 0
            h_start[p] = 0
        else:
            _add_noise_best(p, sig_drift, rho, mu, sig, nmu, nsig, h_mu, h_warg, h_wout,
                            h_off, h_cap, h_start, h_len)
        term_mu[k] = mu[p]
        term_sig[k] = math.hypot(sig[p], sig_perf)               # delta_j
        sig_perf_arr[k] = sig_perf

    g_mu, g_sig, g_bmu, g_off, ranks = build_groups(term_mu, term_sig, lo, width)

    # ---- phase 1 (performance) and phase 2 (belief update), embarrassingly parallel
    for k in prange(n):
        p = ids[k]
        beg, end = subsample_window(g_bmu, mu[p], width, subsample)
        perf = solve_perf(gaussian, g_mu, g_sig, g_off, ranks, beg, end, lo[k], split_ties)
        perf_out[k] = perf
        sp = sig_perf_arr[k]
        if gaussian:
            wn = nsig[p] ** -2
            wp = sp ** -2
            nmu[p] = (wn * nmu[p] + wp * perf) / (wn + wp)
            nsig[p] = (wn + wp) ** -0.5
            mu[p] = nmu[p]
            sig[p] = nsig[p]
        else:
            cap = h_cap[p]
            off = h_off[p]
            if h_len[p] >= cap:
                # history compression: fold the oldest logistic factor into the Gaussian
                k0 = off + h_start[p]
                wn = nsig[p] ** -2
                wl = h_warg[k0] * h_wout[k0] * 2.0 / (TANH_MULT * TANH_MULT)
                nmu[p] = (wn * nmu[p] + wl * h_mu[k0]) / (wn + wl)
                nsig[p] = (wn + wl) ** -0.5
                h_start[p] = (h_start[p] + 1) % cap
                h_len[p] -= 1
            kk = off + (h_start[p] + h_len[p]) % cap
            h_mu[kk] = perf
            h_wout[kk] = TANH_MULT / sp
            h_warg[kk] = 0.5 * TANH_MULT / sp
            h_len[p] += 1
            wn = nsig[p] ** -2
            mu[p] = robust_average(h_mu, h_warg, h_wout, h_start[p], h_len[p], cap, off,
                                   -nmu[p] * wn, wn)
            sig[p] = (sig[p] ** -2 + sp ** -2) ** -0.5
        n_played[p] += 1
        last_time[p] = t_sec


# Baselines: Codeforces, TopCoder, Glicko (ports of the authors' Rust baselines)
@njit(parallel=True)
def codeforces_round(ids, lo, hi, mu, sig, n_played, mu_noob, sig_noob, beta, weight, width,
                     exact=False):
    # Exact=False buckets opponents by rating (O(n * groups)); exact=True treats every opponent individually, which is the O(n^2) per-round cost of the unoptimised system
    n = ids.shape[0]
    for k in range(n):
        p = ids[k]
        if n_played[p] == 0:
            mu[p] = mu_noob
            sig[p] = sig_noob
    term_mu = np.empty(n)
    term_sig = np.full(n, beta)
    for k in range(n):
        term_mu[k] = mu[ids[k]]
    if exact:
        g_mu = term_mu.copy()
        ng = n
        g_cnt = np.ones(n)
        g_w = np.full(n, TANH_MULT / beta)
    else:
        g_mu, g_sig, g_bmu, g_off, ranks = build_groups(term_mu, term_sig, lo, width)
        ng = g_mu.shape[0]
        g_cnt = np.empty(ng)
        g_w = np.empty(ng)
        for g in range(ng):
            g_cnt[g] = g_off[g + 1] - g_off[g]
            g_w[g] = TANH_MULT / beta
    inv = 1.0 / beta
    all_off = n * inv
    new_mu = np.empty(n)
    for k in prange(n):
        my = term_mu[k]
        pos = lo[k] * inv
        neg = (n - 1 - hi[k]) * inv
        ac_rank = 0.5 * (pos - neg + all_off + inv)
        ex_rank = 0.5 * inv
        for g in range(ng):
            ex_rank += g_cnt[g] * logistic_cdf((g_mu[g] - my) / beta) * inv
        geo_rank = math.sqrt(ac_rank * ex_rank)
        geo_off = 2.0 * geo_rank - inv - all_off
        geo = robust_average_groups(g_mu, g_w, g_cnt, TANH_MULT * geo_off)
        new_mu[k] = (my + weight * geo) / (1.0 + weight)
    for k in range(n):
        mu[ids[k]] = new_mu[k]
        n_played[ids[k]] += 1


@njit(parallel=True)
def topcoder_round(ids, lo, hi, mu, sig, n_played, mu_noob, sig_noob,
                   weight_noob, weight_limit, width):
    n = ids.shape[0]
    for k in range(n):
        p = ids[k]
        if n_played[p] == 0:
            mu[p] = mu_noob
            sig[p] = sig_noob
    term_mu = np.empty(n)
    term_sig = np.empty(n)
    for k in range(n):
        term_mu[k] = mu[ids[k]]
        term_sig[k] = sig[ids[k]]
    ave = term_mu.mean()
    mean_vol_sq = (term_sig ** 2).mean()
    if n > 1:
        mean_vol_sq += ((term_mu - ave) ** 2).sum() / (n - 1)
    c_factor = math.sqrt(mean_vol_sq)
    g_mu, g_sig, g_bmu, g_off, ranks = build_groups(term_mu, term_sig, lo, width)
    ng = g_mu.shape[0]
    w_extra = weight_noob - weight_limit
    new_mu = np.empty(n)
    new_sig = np.empty(n)
    for k in prange(n):
        my_mu = term_mu[k]
        my_sig = term_sig[k]
        ex_rank = 0.0
        for g in range(ng):
            z = (g_mu[g] - my_mu) / math.hypot(g_sig[g], my_sig)
            ex_rank += (g_off[g + 1] - g_off[g]) * norm_cdf(z)
        ac_rank = 0.5 * (1 + lo[k] + hi[k])
        ex_perf = -norm_cdf_inv(ex_rank / n)
        ac_perf = -norm_cdf_inv(ac_rank / n)
        perf_as = my_mu + c_factor * (ac_perf - ex_perf)
        nc = n_played[ids[k]] + 1.0
        w = weight_limit + w_extra / nc
        cap = 150.0 + 1500.0 / (nc + 1.0)
        cap *= w / (0.18 + 0.42 / nc)
        w *= 1.0 / (1.0 - w)
        if my_mu >= 2500:
            w *= 0.8
        elif my_mu >= 2000:
            w *= 0.9
        try_r = (my_mu + w * perf_as) / (1.0 + w)
        new_mu[k] = min(max(try_r, my_mu - cap), my_mu + cap)
        new_sig[k] = math.sqrt((try_r - my_mu) ** 2 / w + my_sig ** 2 / (1.0 + w))
    for k in range(n):
        mu[ids[k]] = new_mu[k]
        sig[ids[k]] = new_sig[k]
        n_played[ids[k]] += 1


@njit(parallel=True)
def glicko_round(ids, lo, mu, sig, n_played, mu_noob, sig_noob, beta, sig_drift, width):
    n = ids.shape[0]
    for k in range(n):
        p = ids[k]
        if n_played[p] == 0:
            mu[p] = mu_noob
            sig[p] = sig_noob
        sig[p] = math.hypot(sig[p], sig_drift)
    term_mu = np.empty(n)
    term_sig = np.empty(n)
    for k in range(n):
        term_mu[k] = mu[ids[k]]
        term_sig[k] = sig[ids[k]]
    g_mu, g_sig, g_bmu, g_off, ranks = build_groups(term_mu, term_sig, lo, width)
    ng = g_mu.shape[0]
    q = TANH_MULT / beta
    new_mu = np.empty(n)
    new_sig = np.empty(n)
    for k in prange(n):
        my_mu = term_mu[k]
        my_sig = term_sig[k]
        upd = 0.0
        for g in range(ng):
            gg = 1.0 / math.hypot(1.0, g_sig[g] / beta)
            prob = logistic_cdf((my_mu - g_mu[g]) / math.hypot(g_sig[g], beta))
            a = g_off[g]
            b = g_off[g + 1]
            s = lower_bound(ranks, a, b, lo[k])
            e = upper_bound(ranks, s, b, lo[k])
            outcome_sum = (b - e) * 1.0 + (e - s) * 0.5
            upd += gg * (outcome_sum - (b - a) * prob)
        upd /= n
        info = 0.25 * q * q
        s2 = 1.0 / (my_sig ** -2 + info)
        new_sig[k] = math.sqrt(s2)
        new_mu[k] = my_mu + q * s2 * upd
    for k in range(n):
        mu[ids[k]] = new_mu[k]
        sig[ids[k]] = new_sig[k]
        n_played[ids[k]] += 1


# Evaluation metrics (Section 6.2), ports of metrics.rs
@njit
def _merge_count(a, tmp, lo, hi):
    if hi - lo < 2:
        return 0
    mid = (lo + hi) >> 1
    inv = _merge_count(a, tmp, lo, mid) + _merge_count(a, tmp, mid, hi)
    i, j, k = lo, mid, lo
    while i < mid and j < hi:
        if a[i] >= a[j]:
            tmp[k] = a[i]
            i += 1
        else:
            tmp[k] = a[j]
            inv += mid - i
            j += 1
        k += 1
    while i < mid:
        tmp[k] = a[i]
        i += 1
        k += 1
    while j < hi:
        tmp[k] = a[j]
        j += 1
        k += 1
    for q in range(lo, hi):
        a[q] = tmp[q]
    return inv


@njit
def contest_metrics(ratings, lo, hi):
    # Ratings/lo/hi are for the evaluated subset, already re-ranked 0..n-1 and sorted by lo
    n = ratings.shape[0]
    if n < 2 or hi[0] + 1 >= n:
        return 0.0, 0.0, 0.0
    # pairwise: sort each tie block by rating descending so ties count as correct
    a = ratings.copy()
    i = 0
    while i < n:
        j = hi[i]
        if j > i:
            blk = -np.sort(-a[i:j + 1])
            a[i:j + 1] = blk
        i = j + 1
    tmp = np.empty(n)
    invs = _merge_count(a, tmp, 0, n)
    pair = 100.0 * (n - invs * 2.0 / (n - 1.0))
    # rank deviation
    order = np.argsort(-ratings, kind="mergesort")
    err = 0.0
    for pos in range(n):
        k = order[pos]
        c = min(max(pos, lo[k]), hi[k])
        err += abs(pos - c)
    return float(n), pair, 100.0 * err / (n - 1.0)


@njit
def eval_subset(ids, lo, hi, mu, n_played, min_history):
    # Select experienced players and renormalise their tie ranges (get_participant_ratings)
    n = ids.shape[0]
    sel = np.empty(n, dtype=np.int64)
    m = 0
    for k in range(n):
        if n_played[ids[k]] >= min_history:
            sel[m] = k
            m += 1
    r = np.empty(m)
    nlo = np.empty(m, dtype=np.int64)
    nhi = np.empty(m, dtype=np.int64)
    last_k = -1
    last_v = 0
    for i in range(m):
        k = sel[i]
        r[i] = mu[ids[k]]
        if lo[k] != last_k:
            last_k = lo[k]
            last_v = i
        nlo[i] = last_v
    last_k = -1
    for i in range(m - 1, -1, -1):
        k = sel[i]
        if hi[k] != last_k:
            last_k = hi[k]
            last_v = i
        nhi[i] = last_v
    return contest_metrics(r, nlo, nhi)


# Improvement: adaptive round weight from how predictable the round was
@njit
def round_noise_scale(ids, lo, mu, n_played, min_history, max_players):
    # MLE of the logistic scale s in P(i beats j) = 1 / (1 + exp(-(mu_i - mu_j)/s)) over all pairs of experienced players (subsampled evenly to max_players)
    n = ids.shape[0]
    sel = np.empty(n, dtype=np.int64)
    m = 0
    for k in range(n):
        if n_played[ids[k]] >= min_history:
            sel[m] = k
            m += 1
    if m < 30:
        return -1.0, m
    step = max(1, m // max_players)
    cnt = (m + step - 1) // step
    r = np.empty(cnt)
    rk = np.empty(cnt, dtype=np.int64)
    for q in range(cnt):
        k = sel[q * step]
        r[q] = mu[ids[k]]
        rk[q] = lo[k]
    # Newton on u = 1/s: maximise sum log sigmoid(u * d_ij) where d_ij = mu_winner - mu_loser
    u = 1.0 / 200.0
    for _ in range(50):
        g = 0.0
        h = 0.0
        for a in range(cnt):
            for b in range(a + 1, cnt):
                if rk[a] == rk[b]:
                    continue
                d = r[a] - r[b] if rk[a] < rk[b] else r[b] - r[a]
                pr = 1.0 / (1.0 + math.exp(u * d))      # = 1 - sigmoid(u d)
                g += d * pr
                h -= d * d * pr * (1.0 - pr)
        if h == 0.0:
            break
        step_u = g / h
        u_new = u - step_u
        if u_new <= 1e-5:
            u_new = 0.5 * u
        if abs(u_new - u) < 1e-9:
            u = u_new
            break
        u = u_new
    return 1.0 / u, m
