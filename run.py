import argparse, runpy, sys, warnings
warnings.filterwarnings("ignore")


def call(script, *args):
    #run one of the scripts as if it was called from the shell
    sys.argv = [script, *[str(x) for x in args]]
    runpy.run_path(script, run_name="__main__")


# Parse command line flags. Every flag is one step of the pipeline, so they can be chained.
ap = argparse.ArgumentParser()
ap.add_argument("--download", choices=["codeforces", "atcoder"])
ap.add_argument("--dataset", nargs="+", choices=["codeforces", "atcoder", "synthetic"])
ap.add_argument("--max-contests", type=int)
ap.add_argument("--checks", nargs="*", choices=["codeforces", "atcoder", "synthetic", "seeds"])
ap.add_argument("--runtime", action="store_true")
ap.add_argument("--sensitivity", action="store_true")
ap.add_argument("--figures", action="store_true")
ap.add_argument("--show", action="store_true")
a = ap.parse_args()
no_args = len(sys.argv) == 1

if a.download:
    call(f"download_{a.download}.py")
for d in a.dataset or []:
    extra = ["--max-contests", a.max_contests] if a.max_contests else []
    call("experiment.py", d, *extra)
if a.checks is not None:
    call("checks.py", *a.checks)
if a.runtime:
    call("benchmark.py", "results/runtime_codeforces_exact.json")
if a.sensitivity:
    call("sensitivity.py")
if a.figures:
    call("theory_figures.py")
    call("result_figures.py")
if a.show:
    call("show.py")
if no_args:
    ap.print_help()
