import json, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def table(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    for r in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, (32, 10, 10))))


# Main results: paper protocol (tune on first 10%, test on last 90%).
for ds in ["codeforces", "atcoder", "synthetic"]:
    d = load(f"{ds}.json")
    if d:
        rows = [("system", "pair inv", "rank dev")]
        rows += [(k, f"{v['pair_inv']:.3f}", f"{v['rank_dev']:.3f}")
                 for k, v in sorted(d["systems"].items(), key=lambda kv: kv[1]["pair_inv"])]
        table(f"{ds} (paper protocol, last 90% of rounds)", rows)

# Paper vs ours.
cf, sens = load("codeforces.json"), load("codeforces_sensitivity.json")
syn, rt = load("replication_checks_synthetic.json"), load("runtime_codeforces_exact.json")
rows = [("claim", "paper", "ours")]
if sens: rows.append(("codeforces acc, authors' params", "78.6", f"{sens['authors mmr']['pair']:.2f}"))
if syn: rows.append(("synthetic acc, all play", "81.7", f"{syn['sd, all players every round']['Elo-MMR(rho)']['pair_inv']:.2f}"))
if rt:
    cfe, mmr = rt["Codeforces (exact O(n^2))"]["seconds"], rt["Elo-MMR(rho), authors' params"]["seconds"]
    rows.append(("speedup vs exact codeforces", "6x", f"{cfe / mmr:.0f}x ({mmr:.0f}s vs {cfe:.0f}s)"))
table("replication", rows)

# Bootstrap 95% confidence intervals.
for ds in ["codeforces", "atcoder"]:
    d = load(f"replication_checks_{ds}.json")
    if d:
        print(f"\n{ds}: differences with 95% CI (bootstrap over rounds)")
        for k, v in d["bootstrap"].items():
            a, b = k.split("  minus  ")
            print(f"  {a} vs {b}: {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
