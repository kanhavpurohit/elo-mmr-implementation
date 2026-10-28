# Elo-MMR (Ebtekar & Liu, WWW 2021) - Replication

**Authors:** Kanhav Purohit (240150014), Arnav Sinha (240150005)

This repository provides an empirical replication and extension of Elo-MMR based on the paper:

**An Elo-like System for Massive Multiplayer Competitions**
Aram Ebtekar, Paul Liu (The Web Conference 2021, arXiv:2101.00400)

## Overview

Elo-MMR rates players from rankings alone, for contests with thousands of players per round (Codeforces, TopCoder, races, judged sports). Skills $S_{i,t}$ are hidden and drift between rounds, each round a player performs at $P_{i,t} = S_{i,t} + \text{noise}$, and only the ranking $E_t$ of the performances is observed:

$$\Pr(S,P,E) = \prod_i \Pr(S_{i,0}) \prod_{i,t} \Pr(S_{i,t} \mid S_{i,t-1}) \prod_{i,t} \Pr(P_{i,t} \mid S_{i,t}) \prod_t \Pr(E_t \mid P_{\cdot,t})$$

With thousands of players the ranking pins down every performance almost exactly (Doob's consistency theorem), so each player is updated on their own in two 1-D steps:

* **Phase 1 (performance):** $p_i$ is the zero of $Q_i(p) = \sum_{j \succ i} l_j(p) + \sum_{j \sim i} d_j(p) + \sum_{j \prec i} v_j(p)$. With logistic performances, $l_j = -F_j/\bar\delta_j$ and $v_j = (1-F_j)/\bar\delta_j$, where $F_j$ is the logistic cdf centred at the opponent's rating with variance $\delta_j^2 = \sigma_j^2 + \beta^2$.
* **Phase 2 (rating):** $\mu_t$ is the zero of $L'(s) = \frac{s-p_0}{\beta_0^2} + \sum_k \frac{\pi}{\beta_k\sqrt{3}} \tanh\left(\frac{\pi(s-p_k)}{\beta_k\sqrt{12}}\right)$. This is a robust average of past performances: a freak round sits in the flat tail of tanh and barely moves the rating.
* **Between rounds (pseudodiffusion):** all weights decay by $\kappa = (1+\gamma^2/\sigma^2)^{-1}$, old performances fade as $\kappa^{1+\rho}$ and their weight is moved onto a Gaussian centred at the current rating.

The paper proves the system is massive (linear time), monotonic (it never pays to lose) and robust (one round can only move a rating by a bounded amount).

## Setup

```bash
pip install -r requirements.txt
pytest -q
```

The tests check our kernels against a plain Python version of Algorithms 1-3 written straight from the paper, the exact against the bucketed Codeforces baseline, the metrics, and Theorem 5.5 on a small case.

## Running Experiments

```bash
# Print all result tables from results/ (no computation)
python run.py --show

# Paper protocol on one or more datasets (grid search on first 10% of rounds, test on last 90%)
python run.py --dataset codeforces atcoder synthetic

# Quick pilot run on the first 100 contests only (saved as results/codeforces_pilot100.json)
python run.py --dataset codeforces --max-contests 100

# Replication checks: finer grid, bootstrap error bars, synthetic setup variants, extra synthetic seeds
python run.py --checks codeforces atcoder synthetic seeds

# Full dataset runtime of exact O(n^2) Codeforces vs Elo-MMR
python run.py --runtime

# Elo-MMR with the authors' published Codeforces hyperparameters
python run.py --sensitivity

# Figures (theory checks + result plots)
python run.py --figures
```

* **Datasets:** Codeforces and AtCoder contests are already cached in `data/`. To rebuild them: `python run.py --download codeforces` (about 1.5 h, API rate limited) or `python run.py --download atcoder` (about 40 min). Downloads resume where they stopped.
* **Synthetic data** is generated on the fly (Section 6.1 of the paper).

## Project Structure

* `kernels.py`: Numba kernels. Performance estimation (Thm 3.2), robust average belief update (Eq. 6), pseudodiffusion (Alg. 2), Codeforces / TopCoder / Glicko baselines, metrics (Sec. 6.2), adaptive round noise estimator.
* `systems.py`: `EloMMR` (logistic = Elo-MMR(ρ), `gaussian=True` = Elo-MMχ), `Codeforces` (`exact=True` for the unbucketed O(n²) version), `TopCoder`, `Glicko`, `AtCoderOfficial`.
* `data.py`: Codeforces and AtCoder loaders, synthetic generator.
* `evaluate.py`: runs a system over all rounds and scores prior ratings of players with ≥ 5 past rounds.
* `experiment.py`: grid search on the first 10% of rounds, test on the last 90%, writes `results/<dataset>.json`.
* `checks.py`: finer grid, synthetic variants, bootstrap confidence intervals, multi-seed test, writes `results/replication_checks_*.json`.
* `download_codeforces.py`, `download_atcoder.py`: dataset builders.
* `benchmark.py`, `sensitivity.py`: runtime benchmark and authors' hyperparameter run.
* `theory_figures.py`, `result_figures.py`: figures in `figures/`.
* `show.py`: prints the result tables and confidence intervals.
* `run.py`: command line entry point.
* `tests/`: unit tests.

## Experimental Protocol & Design Decisions

### 1. Same protocol as the paper

Hyperparameters are picked by grid search on the first 10% of rounds and the chosen setting is scored on the last 90%. Only players with at least 5 earlier rounds are scored. Metrics are pair inversion accuracy and rank deviation (Section 6.2). Codeforces contests come from the authors' contest list (`data/codeforces/contest_ids.csv`, cut off at 2020-10-12 like the paper), standings from the Codeforces API. This gives 1,089 rounds with 3,009 players per round (paper: 1,087 and 2,999).

### 2. Parametrisation

Like the authors' code, one contest weight $w$ sets both the performance noise and the per-round drift: $\beta^2 = (1 + 1/w)\,\sigma_{lim}^2$ and $\gamma^2 = w\,\sigma_{lim}^2$, so $\sigma_{lim}$ is the uncertainty a regular player settles at.

### 3. Speed optimisations

* Opponents are grouped into $(\mu, \sigma)$ buckets of width 2, so a round costs O(n · groups) instead of O(n²).
* Phase 1 uses only the 500 groups nearest to the player's rating (the paper's "adaptive subsampling").
* At most 32 logistic terms are kept per player. Older ones are folded into the Gaussian term (the paper's "history compression").

The Codeforces baseline gets the same bucketing, which makes it near linear. `Codeforces(exact=True)` turns it off for a fair speed comparison with the paper.

### 4. Datasets not reproduced

The authors never published the TopCoder data or the per-thread Reddit rankings (their `reddit` folder only has bot karma totals), so those rows and TrueSkill are not reproduced. Instead we added a new dataset, AtCoder.

### 5. Our extensions

* **Time-aware drift** (`drift_per_day`): $\gamma_{i,t}^2 = w\,\sigma_{lim}^2 + c\,\Delta\text{days}_{i,t}$. The paper only adds drift per round played, so a player coming back after two years is treated like one who played yesterday. The extra term depends only on time, never on performances, so the aligned incentives proof (Thm 5.5) still holds.
* **Adaptive round weighting** (`adaptive_alpha`): estimate each round's noise scale $s_t$ by maximum likelihood on the prior ratings and scale the round's weight by $(\bar s / s_t)^\alpha$, clipped to [0.3, 3]. A data driven version of the paper's suggestion to vary β per contest.
* **AtCoder dataset**: 735 rated algorithm contests (2016 to Oct 2026), 337,433 players, 4,743 per round, with AtCoder's own published ratings as an extra baseline.

## Key Results & Replication Summary

### Replication

| Claim | Paper | Ours | Condition |
|---|---|---|---|
| Codeforces pair accuracy, Elo-MMR(ρ) | 78.6 | 78.61 | authors' hyperparameters |
| Codeforces rank deviation, Elo-MMR(ρ) | 14.7 | 14.75 | authors' hyperparameters |
| Synthetic pair accuracy | 81.7 | 81.63 | every player in every round |
| Synthetic rank deviation | 12.8 | 12.90 | every player in every round |
| Elo-MMR vs Codeforces, full dataset | 35 s vs 213 s | 39 s vs 519 s | exact Codeforces |

### Why our first numbers were different

* **Tuning split:** tuning on 2010-11 rounds picks weight 0.1 and σ_lim 40 for Elo-MMR(ρ), which transfer badly to later years (78.37). A finer and wider grid picks the same values, so it is the split and not the grid. Under this protocol TopCoder beats Elo-MMR(ρ) on Codeforces (-0.17, 95% CI [-0.20, -0.15]), but Elo-MMχ still beats TopCoder (+0.06, CI [+0.05, +0.07]).
* **Synthetic participation:** we first followed the paper's Table 1 (about 2,500 of 10,000 players per round, 80.49). The text says every player plays every round, which gives 81.63.
* **Speed:** our bucketed Codeforces runs the whole dataset in 7 s with the same accuracy, so Elo-MMR looked slower. Against the exact formula it is about 13x faster, same as the paper.

### New dataset: AtCoder

| System | Pair inv. | Rank dev. |
|---|---|---|
| AtCoder (official ratings) | 82.01 | 12.33 |
| Glicko | 81.96 | 12.39 |
| TopCoder | 82.61 | 11.87 |
| Codeforces | 82.65 | 11.90 |
| Elo-MMχ | 82.78 | 11.79 |
| Elo-MMR(ρ) | 82.80 | 11.77 |
| Elo-MMR + time drift | **82.87** | **11.72** |

All Elo-MMR margins are significant (bootstrap over rounds): over TopCoder +0.17 (CI [+0.15, +0.19]), over Codeforces +0.08 (CI [+0.06, +0.11]), over AtCoder's own ratings +0.72 (CI [+0.69, +0.75]). AtCoder's displayed rating is lowered for players with fewer than about 10 contests, which is probably part of its gap.

### Time-aware drift

| Gain from time drift | Codeforces | AtCoder |
|---|---|---|
| Elo-MMR(ρ), authors' settings | +0.022 [+0.018, +0.026] | +0.050 [+0.045, +0.054] |
| Elo-MMχ, authors' settings | +0.032 [+0.029, +0.035] | +0.054 [+0.050, +0.057] |
| Elo-MMχ, tuned | +0.047 [+0.042, +0.051] | +0.069 [+0.065, +0.073] |
| Elo-MMR(ρ), tuned | +0.259 [+0.243, +0.277] | +0.095 [+0.091, +0.099] |

On 5 fresh synthetic draws the gain is +0.04 to +0.06 every time. Small, but positive in every setting. The +0.26 on Codeforces mostly fixes the badly transferring tuned settings.

Adaptive round weighting did not help: neutral on AtCoder and with the authors' settings, about -0.6 with our tuned Codeforces settings. Our guess is that in narrow rating divisions (Div. 1) rating error looks like round noise, so the most informative rounds get down weighted, but we did not test this directly.

### Limitations

* Results depend on hyperparameters: with the paper's own 10% tuning split, TopCoder beats Elo-MMR(ρ) on Codeforces.
* Our gains are small (+0.02 to +0.10 points in comparable settings).
* Nearest 500 subsampling breaks exact monotonicity (up to 0.3 rating points in our test).
* No TopCoder, Reddit or TrueSkill comparison.
* Both metrics only score the ordering, so whether σ is calibrated is untested.
