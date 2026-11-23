"""Train and evaluate persistent-NCA training schemes (see ``nca_persist.py``).

    python nca_persist_bench.py train ARM:SEED [ARM:SEED ...]   # checkpoints -> $NCA_CKPT
    python nca_persist_bench.py eval                             # -> nca_persist_results.json

Arms (all the same 7.7k-param rule as the 2026-10-23 ``nca``; 4000 iterations, batch 32,
Adam 2e-3 cosine, clip 1, +-2 px shift):

    fixed         10-23 recipe (T ~ U[24,40] from zero state), fire 0.5      [baseline]
    pool          sample pool (1024), T ~ U[16,32] from stored state, 2/32 random
                  re-seeds per batch, overflow penalty mean(relu(|x|-5)^2), fire 0.5
    pool_noovf    pool without the overflow penalty
    pool_r8       pool with 8/32 random re-seeds per batch (4x the fresh digits)
    pool_oldest   first pool variant: 8/32 *oldest* re-seeded, no overflow penalty
                  (caps state age at ~120 steps)
    pool_det      pool, deterministic updates (fire 1.0)
    pool_dil1     pool, alive mask dilated 1 px (relay cells)
    warmup        no pool; W ~ U[0,120] no-grad warm-up steps, then T ~ U[24,40]

Checkpoints are not committed (default ``nca_ckpt/``, gitignored).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nca_classifier import (ALIVE_THR, canvas_pair, canvas_single, digit_accuracy, load_mnist,
                            n_params, occlude, vote)
from nca_persist import (PersistConfig, PersistentNCA, damage_curve, step_curve, swap_curve,
                         train_persistent)

CKPT = Path(os.environ.get("NCA_CKPT", "nca_ckpt"))
ITERS = int(os.environ.get("NCA_ITERS", 4000))

# arm -> (model kwargs, config kwargs, seeds)
POOL = dict(scheme="pool", reseed=2, reseed_mode="random", overflow=1.0, overflow_bound=5.0)
ARMS = {
    "fixed": (dict(), dict(scheme="fixed"), (0, 1)),
    "pool": (dict(), POOL, (0, 1)),
    "pool_noovf": (dict(), {**POOL, "overflow": 0.0}, (0,)),
    "pool_r8": (dict(), {**POOL, "reseed": 8}, (0,)),
    "pool_oldest": (dict(), dict(scheme="pool", reseed=8, reseed_mode="oldest"), (1,)),
    "pool_det": (dict(fire_rate=1.0), POOL, (0,)),
    "pool_dil1": (dict(dilate=1), POOL, (0,)),
    "warmup": (dict(), dict(scheme="warmup"), (0,)),
}
STEP_GRID = (5, 10, 20, 30, 40, 60, 80, 120, 200, 300, 500, 1000)
ANYTIME = (40, 60, 80, 120, 200, 300, 500, 1000)   # steps averaged into the anytime score
READOUTS = (40, 200)                                 # readout steps for the scene tests
PAIR_GAPS = (0, 2, 8)


def ckpt_path(arm: str, seed: int) -> Path:
    return CKPT / f"persist_{arm}_s{seed}.pt"


def make(arm: str) -> PersistentNCA:
    return PersistentNCA(**ARMS[arm][0])


def train_one(arm: str, seed: int, xtr, ytr):
    p = ckpt_path(arm, seed)
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
    blob = torch.load(ckpt_path(arm, seed))
    model.load_state_dict(blob["state"])
    return model.eval(), blob


def build_test_sets(xte, yte, n: int = 1000, n_pair: int = 500, seed: int = 0):
    """Fixed evaluation scenes (subset of the 10-23 protocol, same generators)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xte))
    a = idx[:n]
    sets = {"clean28": (xte[a], (xte[a] > ALIVE_THR).unsqueeze(1), yte[a][:, None])}
    c, m = canvas_single(xte[a], 84, rng)
    sets["canvas84"] = (c, m, yte[a][:, None])
    o = occlude(xte[a], 8, rng)
    sets["occlude8"] = (o, (o > ALIVE_THR).unsqueeze(1), yte[a][:, None])
    pa, pb = idx[n:n + 3 * n_pair], idx[n + 3 * n_pair:n + 6 * n_pair]
    keep = (yte[pa] != yte[pb]).nonzero().squeeze(1)[:n_pair]
    pa, pb = pa[keep.numpy()], pb[keep.numpy()]
    for g in PAIR_GAPS:
        c, m = canvas_pair(xte[pa], xte[pb], g, rng)
        sets[f"pair_gap{g}"] = (c, m, torch.stack([yte[pa], yte[pb]], 1))
    # swap probe: digit A then a different-label digit B
    sb = idx[n + 6 * n_pair:n + 6 * n_pair + 2 * n]
    keep = (yte[a] != yte[sb[:n]]).nonzero().squeeze(1)[:500].numpy()
    swap = (xte[a][keep], xte[sb[:n]][keep], yte[sb[:n]][keep])
    return sets, swap


@torch.no_grad()
def scene_scores(model, sets) -> dict:
    """Accuracy (and both-correct for pairs) on every scene at each readout step."""
    out = {}
    for name, (imgs, masks, labels) in sets.items():
        torch.manual_seed(123)
        _, recs = model.rollout(imgs, max(READOUTS), record=READOUTS)
        for t in READOUTS:
            pred, agree = vote(recs[t], masks)
            r = {"acc": digit_accuracy(pred, labels), "agreement": float(agree.mean())}
            if labels.shape[1] == 2:
                r["both_correct"] = float((pred == labels).all(1).float().mean())
            out[f"{name}@{t}"] = r
    return out


def evaluate_all(xte, yte) -> dict:
    sets, swap = build_test_sets(xte, yte)
    clean_x, clean_y = sets["clean28"][0], sets["clean28"][2][:, 0]
    res = {"config": {"iters": ITERS, "n_single": len(clean_x),
                      "n_pair": len(sets["pair_gap0"][0]), "n_swap": len(swap[0]),
                      "step_grid": list(STEP_GRID), "anytime_steps": list(ANYTIME),
                      "readouts": list(READOUTS), "pair_gaps": list(PAIR_GAPS),
                      "damage": "8x8 state patch zeroed at step 100 (cells stay alive)",
                      "swap": "image replaced by a different-label digit at step 100",
                      "arms": {a: {"model": ARMS[a][0], "train": ARMS[a][1]} for a in ARMS}},
           "arms": {}}
    dmg_steps = (100, 101, 105, 110, 120, 140, 200, 300)
    swap_steps = (101, 105, 110, 120, 140, 160, 200, 300)
    for arm, (_, _, seeds) in ARMS.items():
        per_seed = []
        for s in seeds:
            if not ckpt_path(arm, s).exists():
                continue
            model, blob = load(arm, s)
            t0 = time.time()
            r = {"train_min": blob["seconds"] / 60, "final_loss": blob["losses"][-1]}
            if blob.get("pool_age"):
                r["pool_age_final_mean"], r["pool_age_final_max"] = blob["pool_age"][-1]
            curve = step_curve(model, clean_x, clean_y, STEP_GRID)
            r["step_curve"] = {str(t): list(v) for t, v in curve.items()}
            r["anytime"] = float(np.mean([curve[t][0] for t in ANYTIME]))
            r["scenes"] = scene_scores(model, sets)
            dc = damage_curve(model, clean_x[:500], clean_y[:500], 100, 8, dmg_steps)
            r["damage"] = {str(t): list(v) for t, v in dc.items()}
            sc = swap_curve(model, *swap, 100, swap_steps)
            r["swap"] = {str(t): list(v) for t, v in sc.items()}
            per_seed.append(r)
            print(arm, s, f"anytime {r['anytime']:.3f}",
                  {t: round(curve[t][0], 3) for t in (40, 200, 1000)},
                  f"{time.time() - t0:.0f}s", flush=True)
        if not per_seed:
            continue
        res["arms"][arm] = {"params": n_params(make(arm)), "seeds": len(per_seed),
                            "per_seed": per_seed, "summary": summarise(per_seed)}
    return res


def _ms(vals):
    return [float(np.mean(vals)), float(np.std(vals))]


def summarise(per_seed: list) -> dict:
    """Mean/std over seeds of every scalar and curve point."""
    s = {k: _ms([r[k] for r in per_seed]) for k in ("train_min", "final_loss", "anytime")}
    for key in ("step_curve", "damage", "swap"):
        s[key] = {t: [_ms([r[key][t][j] for r in per_seed]) for j in range(len(v))]
                  for t, v in per_seed[0][key].items()}
    s["scenes"] = {name: {m: _ms([r["scenes"][name][m] for r in per_seed]) for m in v}
                   for name, v in per_seed[0]["scenes"].items()}
    return s


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("NCA_THREADS", 4)))
    xtr, ytr, xte, yte = load_mnist()
    if sys.argv[1] == "train":
        for spec in sys.argv[2:]:
            arm, seed = spec.split(":")
            train_one(arm, int(seed), xtr, ytr)
    else:
        res = evaluate_all(xte, yte)
        Path("nca_persist_results.json").write_text(json.dumps(res, indent=1))
