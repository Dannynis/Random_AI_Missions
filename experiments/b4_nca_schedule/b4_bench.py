"""B4: re-seed schedules for the persistent self-classifying NCA, at 5 seeds.

Ported from the Random-Ideas daily runs (PR #44 persistent pool, PR #58 re-seed annealing).
The model, training recipe and evaluation are unchanged (``nca_classifier.py``,
``nca_persist.py``, ``nca_persist_bench.py`` copied verbatim, plus a step-switch option in
``reseed_count``). Only the arms and the number of seeds differ.

    python b4_bench.py worker W N     # train + evaluate every N-th (arm, seed), starting at W
    python b4_bench.py collect        # merge per-run JSONs into $RUN_DIR/out/results.json

Env: RUN_DIR (ckpt/ and out/), LAB_DATA (MNIST root containing MNIST/raw),
NCA_ITERS (default 4000), NCA_WORKERS (default 4), NCA_SEEDS (default 10),
NCA_TINY=1 (50 iterations, 100 test digits, 2 seeds: pipeline check only).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import nca_persist_bench as npb
from nca_classifier import load_mnist, n_params
from nca_persist import (PersistConfig, PersistentNCA, damage_curve, reseed_count,
                         step_curve, swap_curve, train_persistent)

TINY = os.environ.get("NCA_TINY") == "1"
RUN_DIR = Path(os.environ["RUN_DIR"])
CKPT = RUN_DIR / "ckpt"
OUT = RUN_DIR / "out"
DATA = Path(os.environ.get("LAB_DATA", "data")) / "MNIST" / "raw"
ITERS = 50 if TINY else int(os.environ.get("NCA_ITERS", 4000))
N_SEEDS = 2 if TINY else int(os.environ.get("NCA_SEEDS", 10))
N_TEST = 100 if TINY else 1000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POOL = dict(scheme="pool", reseed=2, reseed_mode="random", overflow=1.0, overflow_bound=5.0)
DET = dict(fire_rate=1.0)
ARMS = {
    "pool_det": {**POOL},                                                # constant 2 (PR #44)
    "const5": {**POOL, "reseed": 5},                                     # constant 5
    "const8": {**POOL, "reseed": 8},                                     # constant 8
    "anneal": {**POOL, "reseed": 8, "reseed_end": 2},                    # linear 8 -> 2 (PR #58)
    "anneal_rev": {**POOL, "reseed": 2, "reseed_end": 8},                # linear 2 -> 8 (PR #58, 1 seed)
    "switch_rev": {**POOL, "reseed": 2, "reseed_end": 8, "reseed_switch_frac": 0.5},  # step 2 -> 8 at 50 %
}
STEP_GRID = (5, 10, 20, 30, 40, 60, 80, 120, 200, 300, 500, 1000, 2000)
ANYTIME = (40, 60, 80, 120, 200, 300, 500, 1000)
SCENES = ("occlude8", "pair_gap2")
DMG_STEPS = (100, 101, 105, 110, 120, 140, 200, 300)
SWAP_STEPS = (101, 105, 110, 120, 140, 160, 200, 300)


def fresh_digits(arm: str) -> int:
    cfg = PersistConfig(iters=ITERS, **ARMS[arm])
    return cfg.pool_size + sum(reseed_count(cfg, it) for it in range(ITERS))


def stem(arm: str, seed: int) -> Path:
    return CKPT / f"{arm}_s{seed}"


def run_one(spec):
    """Train (if no checkpoint) and evaluate (if no JSON) one (arm, seed). Resumable per run."""
    arm, seed = spec
    torch.set_num_threads(1)
    js = stem(arm, seed).with_suffix(".json")
    if js.exists():
        return arm, seed, "cached"
    xtr, ytr, xte, yte = load_mnist(DATA)
    pt = stem(arm, seed).with_suffix(".pt")
    if not pt.exists():
        torch.manual_seed(seed)
        model = PersistentNCA(**DET).to(DEV)
        log = train_persistent(model, xtr, ytr, PersistConfig(iters=ITERS, seed=seed, **ARMS[arm]))
        torch.save({"state": model.state_dict(), "seconds": log.seconds, "losses": log.losses,
                    "pool_age": log.pool_age, "iters": ITERS}, str(pt) + ".tmp")
        os.replace(str(pt) + ".tmp", pt)
    model = PersistentNCA(**DET).to(DEV)
    blob = torch.load(pt, map_location=DEV)
    model.load_state_dict(blob["state"])
    model.eval()
    sets, swap = npb.build_test_sets(xte, yte, n=N_TEST, n_pair=N_TEST // 2)
    sets = {k: tuple(t.to(DEV) for t in v) for k, v in sets.items()}
    swap = tuple(t.to(DEV) for t in swap)
    clean_x, clean_y = sets["clean28"][0], sets["clean28"][2][:, 0]
    t0 = time.time()
    r = {"arm": arm, "seed": seed, "train_min": blob["seconds"] / 60, "final_loss": blob["losses"][-1]}
    if blob.get("pool_age"):
        r["pool_age_final_mean"], r["pool_age_final_max"] = blob["pool_age"][-1]
    curve = step_curve(model, clean_x, clean_y, STEP_GRID)
    r["step_curve"] = {str(t): list(v) for t, v in curve.items()}
    r["anytime"] = float(np.mean([curve[t][0] for t in ANYTIME]))
    r["peak"] = float(max(curve[t][0] for t in STEP_GRID))
    r["scenes"] = npb.scene_scores(model, {k: sets[k] for k in SCENES})
    n_probe = min(500, N_TEST)
    dc = damage_curve(model, clean_x[:n_probe], clean_y[:n_probe], 100, 8, DMG_STEPS)
    r["damage"] = {str(t): list(v) for t, v in dc.items()}
    sc = swap_curve(model, *swap, 100, SWAP_STEPS)
    r["swap"] = {str(t): list(v) for t, v in sc.items()}
    r["eval_s"] = time.time() - t0
    js.write_text(json.dumps(r))
    print(arm, seed, f"anytime {r['anytime']:.3f} peak {r['peak']:.3f}",
          {t: round(curve[t][0], 3) for t in (40, 1000, 2000)}, flush=True)
    return arm, seed, "done"


def _ms(v):
    return [float(np.mean(v)), float(np.std(v))]


def collect() -> dict:
    res = {"config": {"iters": ITERS, "tiny": TINY, "n_test": N_TEST, "seeds": N_SEEDS,
                      "step_grid": list(STEP_GRID), "anytime_steps": list(ANYTIME),
                      "params": n_params(PersistentNCA(**DET)),
                      "arms": {a: {"train": ARMS[a], "fresh_digits": fresh_digits(a)} for a in ARMS}},
           "arms": {}, "paired_vs_pool_det": {}}
    per = {}
    for arm in ARMS:
        rs = [json.loads(stem(arm, s).with_suffix(".json").read_text()) for s in range(N_SEEDS)
              if stem(arm, s).with_suffix(".json").exists()]
        if not rs:
            continue
        per[arm] = {r["seed"]: r for r in rs}
        res["arms"][arm] = {
            "n": len(rs),
            "anytime": _ms([r["anytime"] for r in rs]), "peak": _ms([r["peak"] for r in rs]),
            **{f"acc@{t}": _ms([r["step_curve"][str(t)][0] for r in rs]) for t in (40, 200, 1000, 2000)},
            "damage@200": _ms([r["damage"]["200"][0] for r in rs]),
            "swap@300": _ms([r["swap"]["300"][0] for r in rs]),
            "pool_age_final": _ms([r.get("pool_age_final_mean", 0) for r in rs]),
            "per_seed": rs,
        }
    base = per.get("pool_det", {})
    for arm, rs in per.items():
        if arm == "pool_det":
            continue
        d = {}
        for key, get in [("anytime", lambda r: r["anytime"]), ("peak", lambda r: r["peak"]),
                         ("acc@40", lambda r: r["step_curve"]["40"][0]),
                         ("acc@2000", lambda r: r["step_curve"]["2000"][0])]:
            diffs = [get(r) - get(base[s]) for s, r in rs.items() if s in base]
            d[key] = {"diffs": diffs, "mean": float(np.mean(diffs)) if diffs else None,
                      "n_pos": int(sum(x > 0 for x in diffs))}
        res["paired_vs_pool_det"][arm] = d
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    summary = {a: {k: v for k, v in s.items() if k != "per_seed"} for a, s in res["arms"].items()}
    print(json.dumps(summary, indent=1))
    return res


if __name__ == "__main__":
    CKPT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "worker":
        # worker W of N: seed-major order so every arm gets its first seeds early.
        # (Plain processes started by run.sh; multiprocessing pools need /dev/shm, which jobs may lack.)
        w, n = int(sys.argv[2]), int(sys.argv[3])
        specs = [(a, s) for s in range(N_SEEDS) for a in ARMS][w::n]
        for spec in specs:
            arm, seed, st = run_one(spec)
            print(f"[{time.strftime('%H:%M:%S')}] worker {w}: {arm} s{seed}: {st}", flush=True)
    else:
        collect()
