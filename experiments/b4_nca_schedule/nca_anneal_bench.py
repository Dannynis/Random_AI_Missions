"""Annealed re-seed rate for the persistent NCA classifier (2026-11-23).

PR #44 (``nca_persist_bench.py``) found two levers that pull against each other in
sample-pool training: long state ages (persistence; 2 re-seeds / batch) and fresh-digit
throughput (peak accuracy; 8 re-seeds / batch). This bench tests its next steps 1 + 2:
anneal the re-seed count from 8 to 2 during training, on the deterministic-update arm
``pool_det`` (PR #44's best, 1 seed), with 3 seeds, plus an equal-fresh-data control.

    python nca_anneal_bench.py train ARM:SEED [ARM:SEED ...]   # -> $NCA_CKPT/anneal_*.pt
    python nca_anneal_bench.py eval ARM:SEED [ARM:SEED ...]    # -> $NCA_CKPT/anneal_*.json
    python nca_anneal_bench.py collect                         # -> nca_anneal_results.json

Arms (same 7.7k-param rule, fire rate 1.0, pool 1024, T ~ U[16,32] from stored state,
overflow penalty mean(relu(|x|-5)^2), 4000 iterations x batch 32, Adam 2e-3 cosine):

    pool_det      constant 2 re-seeds / batch  (PR #44's pool_det; ~8k fresh digits)  [baseline]
    anneal        re-seeds linear 8 -> 2 over training (~20k fresh digits)             [new]
    const5        constant 5 re-seeds / batch (same ~20k fresh digits as ``anneal``)   [control]
    anneal_rev    re-seeds linear 2 -> 8 (same fresh digits, schedule reversed)        [control]

Evaluation is a reduced version of PR #44's (scenes cut to the cheap ones): step curve to
1000 steps on 1000 test digits (+ step 2000), damage and swap probes on 500 digits,
occlusion and gap-2 pairs at readout 40 / 200.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nca_classifier import load_mnist, n_params
from nca_persist import (PersistConfig, PersistentNCA, damage_curve, reseed_count,
                         step_curve, swap_curve, train_persistent)
from nca_persist_bench import build_test_sets, scene_scores, summarise

CKPT = Path(os.environ.get("NCA_CKPT", "nca_ckpt"))
ITERS = int(os.environ.get("NCA_ITERS", 4000))
OUT = Path("nca_anneal_results.json")

POOL = dict(scheme="pool", reseed=2, reseed_mode="random", overflow=1.0, overflow_bound=5.0)
DET = dict(fire_rate=1.0)
# arm -> (model kwargs, config kwargs, seeds)
ARMS = {
    "pool_det": (DET, POOL, (0, 1, 2)),
    "anneal": (DET, {**POOL, "reseed": 8, "reseed_end": 2}, (0, 1, 2)),
    "const5": (DET, {**POOL, "reseed": 5}, (0,)),
    "anneal_rev": (DET, {**POOL, "reseed": 2, "reseed_end": 8}, (0,)),
}
STEP_GRID = (5, 10, 20, 30, 40, 60, 80, 120, 200, 300, 500, 1000, 2000)
ANYTIME = (40, 60, 80, 120, 200, 300, 500, 1000)   # same anytime score as PR #44
SCENES = ("occlude8", "pair_gap2")
DMG_STEPS = (100, 101, 105, 110, 120, 140, 200, 300)
SWAP_STEPS = (101, 105, 110, 120, 140, 160, 200, 300)


def fresh_digits(arm: str) -> int:
    """Fresh digits seen in training: initial pool + sum of re-seeds over iterations."""
    cfg = PersistConfig(iters=ITERS, **ARMS[arm][1])
    return cfg.pool_size + sum(reseed_count(cfg, it) for it in range(ITERS))


def stem(arm: str, seed: int) -> Path:
    return CKPT / f"anneal_{arm}_s{seed}"


def make(arm: str) -> PersistentNCA:
    return PersistentNCA(**ARMS[arm][0])


def train_one(arm: str, seed: int, xtr, ytr):
    p = stem(arm, seed).with_suffix(".pt")
    if p.exists():
        return
    CKPT.mkdir(exist_ok=True)
    torch.manual_seed(seed)
    model = make(arm)
    log = train_persistent(model, xtr, ytr, PersistConfig(iters=ITERS, seed=seed, **ARMS[arm][1]))
    torch.save({"state": model.state_dict(), "seconds": log.seconds, "losses": log.losses,
                "pool_age": log.pool_age, "iters": ITERS}, p)
    print(f"{arm} s{seed}: {log.seconds / 60:.1f} min, final loss {log.losses[-1]:.3f}",
          flush=True)


def load(arm: str, seed: int):
    model = make(arm)
    blob = torch.load(stem(arm, seed).with_suffix(".pt"))
    model.load_state_dict(blob["state"])
    return model.eval(), blob


def eval_one(arm: str, seed: int, xte, yte) -> dict:
    """Evaluate one checkpoint; writes and returns a per-checkpoint result dict."""
    sets, swap = build_test_sets(xte, yte)
    clean_x, clean_y = sets["clean28"][0], sets["clean28"][2][:, 0]
    model, blob = load(arm, seed)
    t0 = time.time()
    r = {"seed": seed, "train_min": blob["seconds"] / 60, "final_loss": blob["losses"][-1],
         "loss_curve": blob["losses"][::4]}
    if blob.get("pool_age"):
        r["pool_age_final_mean"], r["pool_age_final_max"] = blob["pool_age"][-1]
        r["pool_age_curve"] = [a for a, _ in blob["pool_age"][::4]]
    curve = step_curve(model, clean_x, clean_y, STEP_GRID)
    r["step_curve"] = {str(t): list(v) for t, v in curve.items()}
    r["anytime"] = float(np.mean([curve[t][0] for t in ANYTIME]))
    r["peak"] = float(max(curve[t][0] for t in STEP_GRID))
    r["scenes"] = scene_scores(model, {k: sets[k] for k in SCENES})
    dc = damage_curve(model, clean_x[:500], clean_y[:500], 100, 8, DMG_STEPS)
    r["damage"] = {str(t): list(v) for t, v in dc.items()}
    sc = swap_curve(model, *swap, 100, SWAP_STEPS)
    r["swap"] = {str(t): list(v) for t, v in sc.items()}
    r["eval_s"] = time.time() - t0
    stem(arm, seed).with_suffix(".json").write_text(json.dumps(r))
    print(arm, seed, f"anytime {r['anytime']:.3f} peak {r['peak']:.3f}",
          {t: round(curve[t][0], 3) for t in (40, 200, 1000, 2000)}, f"{r['eval_s']:.0f}s",
          flush=True)
    return r


def collect() -> dict:
    """Merge per-checkpoint JSONs into ``nca_anneal_results.json`` (with paired deltas)."""
    res = {"config": {"iters": ITERS, "step_grid": list(STEP_GRID), "anytime_steps": list(ANYTIME),
                      "scenes": list(SCENES), "n_single": 1000, "n_probe": 500,
                      "damage": "8x8 state patch zeroed at step 100 (cells stay alive)",
                      "swap": "image replaced by a different-label digit at step 100",
                      "arms": {a: {"model": ARMS[a][0], "train": ARMS[a][1],
                                   "fresh_digits": fresh_digits(a)} for a in ARMS}},
           "arms": {}}
    for arm, (_, _, seeds) in ARMS.items():
        per_seed = [json.loads(stem(arm, s).with_suffix(".json").read_text())
                    for s in seeds if stem(arm, s).with_suffix(".json").exists()]
        if not per_seed:
            continue
        summ = summarise(per_seed)
        summ["peak"] = [float(np.mean([r["peak"] for r in per_seed])),
                        float(np.std([r["peak"] for r in per_seed]))]
        for r in per_seed:          # keep the JSON compact
            r.pop("loss_curve", None)
        res["arms"][arm] = {"params": n_params(make(arm)), "seeds": len(per_seed),
                            "per_seed": per_seed, "summary": summ}
    # paired (same seed) differences vs the baseline
    base = {r["seed"]: r for r in res["arms"].get("pool_det", {}).get("per_seed", [])}
    res["paired_vs_pool_det"] = {}
    for arm, v in res["arms"].items():
        if arm == "pool_det":
            continue
        d = {}
        for key in ("anytime", "peak"):
            diffs = [r[key] - base[r["seed"]][key] for r in v["per_seed"] if r["seed"] in base]
            d[key] = {"diffs": diffs, "mean": float(np.mean(diffs)) if diffs else None}
        for t in ("40", "1000", "2000"):
            diffs = [r["step_curve"][t][0] - base[r["seed"]]["step_curve"][t][0]
                     for r in v["per_seed"] if r["seed"] in base]
            d[f"acc@{t}"] = {"diffs": diffs, "mean": float(np.mean(diffs)) if diffs else None}
        res["paired_vs_pool_det"][arm] = d
    OUT.write_text(_compact_json(res))
    return res


def _compact_json(obj) -> str:
    """Indented JSON with every innermost list of scalars kept on one line (short file)."""
    import re
    txt = json.dumps(obj, indent=1)
    return re.sub(r"\[[^\[\]{}]*\]", lambda m: re.sub(r"\s+", " ", m.group(0)), txt)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("NCA_THREADS", 1)))
    if sys.argv[1] == "collect":
        collect()
        sys.exit()
    xtr, ytr, xte, yte = load_mnist()
    for spec in sys.argv[2:]:
        arm, seed = spec.split(":")
        if sys.argv[1] == "train":
            train_one(arm, int(seed), xtr, ytr)
        else:
            eval_one(arm, int(seed), xte, yte)
