import sys, json, time
from data import load_codeforces
from systems import Codeforces, EloMMR
from evaluate import run
mk = {
 "Codeforces (bucketed)": lambda: Codeforces(beta=65, weight=0.631),
 "Codeforces (exact O(n^2))": lambda: Codeforces(beta=65, weight=0.631, exact=True),
 "Elo-MMR(rho), authors' params": lambda: EloMMR(weight_limit=0.23, sig_limit=54, rho=0.04),
 "Elo-MMchi, authors' params": lambda: EloMMR(weight_limit=0.16, sig_limit=74, gaussian=True),
}
warm = load_codeforces(max_contests=3)
for f in mk.values(): run(f(), warm)          # JIT warm-up, not timed
ds = load_codeforces(); n_train = len(ds) // 10
print(ds.summary(), flush=True)
out = {}
for name, f in mk.items():
    r = run(f(), ds, eval_from=n_train)
    out[name] = {"seconds": r["seconds"], "pair_inv": r["pair_inv"], "rank_dev": r["rank_dev"]}
    print(f"{name:32s} time={r['seconds']:8.1f}s  pair={r['pair_inv']:.3f}  dev={r['rank_dev']:.3f}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
