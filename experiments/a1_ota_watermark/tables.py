"""A1 tables with one decision rule per method and Wilson 95 % intervals.

usage: python tables.py <results dir> [<results dir> ...] [--rule-map ours=crc@4,...] [--metric fa1|raw]

Reads every ``eval_*.json`` (simulated) and ``real_*.json`` (real room) below the given directories and prints one
markdown row per (file kind, method, target, condition): message success at <= 1 % false alarm (``fa1``, two-fold
thresholds on the unmarked twins) or the raw success with the CRC / decoder decision (``raw``), the number of
recordings pooled over seeds, the Wilson interval, and the raw false-alarm count on the unmarked twins.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

RULE = {"ours": "crc@1", "audioseal": "top", "wavmark": "top", "wavmark_soft": "top", "dear": "top"}


def rule_for(method, rule_map):
    base = "ours" if method.startswith("ours") else method
    return rule_map.get(method, rule_map.get(base, "top"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--rule-map", default="", help="e.g. ours=crc@1,audioseal=oracle")
    ap.add_argument("--metric", default="fa1", choices=["fa1", "raw"])
    a = ap.parse_args()
    rule_map = dict(RULE)
    for kv in filter(None, a.rule_map.split(",")):
        k, v = kv.split("="); rule_map[k] = v
    cells = defaultdict(lambda: [0.0, 0, 0, 0])            # successes, recordings, false alarms, unmarked
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            name = os.path.basename(f)
            if not (name.startswith("eval_") or name.startswith("real_")):
                continue
            res = json.load(open(f))
            kind = name.rsplit("_seed", 1)[0] if name.startswith("eval_") else name[:-5]
            for r in res.get("rows", []):
                if r["rule"] != rule_for(r["method"], rule_map):
                    continue
                pr = r.get("per_rec", {})
                n = r.get("n", len(pr.get("correct", [])))
                if a.metric == "fa1":
                    succ = r["fa1"]["success"] * n
                else:
                    succ = r["raw_success"] * n
                c = cells[(kind, r["method"], r["target"], r["condition"])]
                c[0] += succ; c[1] += n
                c[2] += round(r["raw_false_alarm"] * n); c[3] += n
    print(f"| file | method | target | condition | success ({a.metric}) | 95 % CI | n | raw FA |")
    print("|---|---|---|---|---|---|---|---|")
    for (kind, m, t, cond), (s, n, fa, nu) in sorted(cells.items()):
        k = round(s)
        lo, hi = wilson(k, n)
        print(f"| {kind} | {m} | {t} | {cond} | {s / n:.3f} | [{lo:.2f}, {hi:.2f}] | {n} | {fa}/{nu} |")


if __name__ == "__main__":
    main()
