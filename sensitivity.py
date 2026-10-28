import json
from data import load_codeforces
from systems import EloMMR
from evaluate import run
ds = load_codeforces(); n=len(ds)//10
cfgs = [("authors mmr", dict(weight_limit=0.23, sig_limit=54, rho=0.04)),
        ("authors mmr + drift10", dict(weight_limit=0.23, sig_limit=54, rho=0.04, drift_per_day=10)),
        ("authors mmr + drift3", dict(weight_limit=0.23, sig_limit=54, rho=0.04, drift_per_day=3)),
        ("authors mmx", dict(weight_limit=0.16, sig_limit=74, gaussian=True)),
        ("authors mmx + drift10", dict(weight_limit=0.16, sig_limit=74, gaussian=True, drift_per_day=10)),
        ("default mmr rho1", dict(weight_limit=0.2, sig_limit=80, rho=1.0)),
        ("authors mmr + adapt0.5", dict(weight_limit=0.23, sig_limit=54, rho=0.04, adaptive_alpha=0.5)),
        ("tuned + adapt0.5", dict(weight_limit=0.1, sig_limit=40, rho=0.04, adaptive_alpha=0.5))]
out={}
for name,kw in cfgs:
    r=run(EloMMR(**kw), ds, eval_from=n); out[name]=dict(kw=kw, pair=r['pair_inv'], dev=r['rank_dev'])
    print(name, round(r['pair_inv'],3), round(r['rank_dev'],3), flush=True)
json.dump(out, open('results/codeforces_sensitivity.json','w'), indent=1)
