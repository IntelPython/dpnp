"""Compare two result directories written by bench_gemm.py.

python compare.py results/base results/new [--min-ratio 0.0]
"""

import argparse
import glob
import json
import os


def load(d):
    out = {}
    for fn in glob.glob(os.path.join(d, "*.json")):
        for r in json.load(open(fn)):
            out[(r["api"], r["dtype"], r["name"])] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("new")
    p.add_argument("--filter", default="")
    args = p.parse_args()
    a, b = load(args.base), load(args.new)
    keys = sorted(set(a) & set(b), key=lambda k: (k[1], k[2]))
    print(
        f"{'dtype':5s} {'case':32s} {'base ms':>10s} {'new ms':>10s} "
        f"{'speedup':>8s}"
    )
    for k in keys:
        if args.filter not in k[2]:
            continue
        ra, rb = a[k], b[k]
        sp = ra["median_ms"] / rb["median_ms"]
        flag = "" if rb["ok"] else "  BAD"
        mark = " <<" if sp < 0.9 else (" >>" if sp > 1.2 else "")
        print(
            f"{k[1]:5s} {k[2]:32s} {ra['median_ms']:10.3f} "
            f"{rb['median_ms']:10.3f} {sp:8.2f}{mark}{flag}"
        )


if __name__ == "__main__":
    main()
