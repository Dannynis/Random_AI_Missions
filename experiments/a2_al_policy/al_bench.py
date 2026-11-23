"""A2 driver: baselines and the learned query policy on frozen DINOv2 features.

    python al_bench.py baselines <feat>        # every strategy x seed on every eval task
    python al_bench.py policy <feat>           # meta-train the policy (GRPO), then evaluate it
    python al_bench.py policy2 <feat>          # v2: cluster features + behaviour cloning from TypiClust + GRPO
    python al_bench.py collect                 # merge per-run JSONs -> $RUN_DIR/out/summary.json

Eval tasks (all on frozen features; pool = train split, test = test split):
    cifar10        10 classes, 50k pool
    cifar100       100 classes, 50k pool
    c100_heldout_t 5 fixed 10-way tasks from CIFAR-100 classes 50-99 (never used to train the policy)
The policy is meta-trained only on random 10-way tasks from CIFAR-100 classes 0-49.

Env: RUN_DIR, LAB_DATA (features in $LAB_DATA/features), AL_TINY=1 for a pipeline check,
AL_SEEDS (default 10), AL_ITERS (policy iterations, default 150).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import al_lib as L

TINY = os.environ.get("AL_TINY") == "1"
RUN_DIR = Path(os.environ["RUN_DIR"])
CKPT, OUT = RUN_DIR / "ckpt", RUN_DIR / "out"
FEAT = Path(os.environ["LAB_DATA"]) / "features"
SEEDS = range(2 if TINY else int(os.environ.get("AL_SEEDS", 10)))
BUDGETS = [10, 20, 40, 60] if TINY else [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
TRAIN_BUDGETS = [10, 20, 40] if TINY else [10, 20, 40, 80, 160]
HELDOUT = [list(range(50 + 10 * i, 60 + 10 * i)) for i in range(5)]   # disjoint 10-way tasks


def load_feats(feat: str, ds: str):
    name = f"{feat}_{ds}{'_tiny' if TINY else ''}.pt"
    b = torch.load(FEAT / name)
    X = F.normalize(b["train"].float(), dim=1).to(L.DEV)
    Xt = F.normalize(b["test"].float(), dim=1).to(L.DEV)
    return X, b["ytr"].to(L.DEV), Xt, b["yte"].to(L.DEV)


def subset_task(X, y, Xt, yt, classes, seed=0):
    cl = torch.tensor(classes, device=y.device)
    m, mt = torch.isin(y, cl), torch.isin(yt, cl)
    remap = torch.full((int(y.max()) + 1,), -1, device=y.device, dtype=torch.long)
    remap[cl] = torch.arange(len(cl), device=y.device)
    return L.Task(X[m], remap[y[m]], Xt[mt], remap[yt[mt]], len(classes)).prepare(seed=seed)


_FEAT_CACHE: dict = {}
POOL_FRAC = 0.8


def _feats(feat, ds):
    if (feat, ds) not in _FEAT_CACHE:
        _FEAT_CACHE[(feat, ds)] = load_feats(feat, ds)
    return _FEAT_CACHE[(feat, ds)]


def _subsample(X, y, seed):
    """Each seed sees a different random 80 % of the pool, so deterministic rules have seed variance."""
    g = torch.Generator().manual_seed(10_000 + seed)
    keep = torch.randperm(len(X), generator=g)[: int(POOL_FRAC * len(X))].sort().values.to(X.device)
    return X[keep], y[keep]


def eval_tasks(feat: str, seed: int):
    tasks = {}
    X, y, Xt, yt = _feats(feat, "cifar10")
    Xs, ys = _subsample(X, y, seed)
    tasks["cifar10"] = L.Task(Xs, ys, Xt, yt, 10).prepare(seed=seed)
    X, y, Xt, yt = _feats(feat, "cifar100")
    Xs, ys = _subsample(X, y, seed)
    tasks["cifar100"] = L.Task(Xs, ys, Xt, yt, 100).prepare(seed=seed)
    if not TINY:
        for i, cl in enumerate(HELDOUT):
            tasks[f"c100_heldout_{i}"] = subset_task(Xs, ys, Xt, yt, cl, seed=seed)
    return tasks


def run_and_save(task_name, task, method, strategy, seed, feat, init_random=0):
    p = CKPT / feat / f"{task_name}__{method}__s{seed}.json"
    if p.exists():
        return json.loads(p.read_text())
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(seed)
    lin, lp = L.run_episode(task, strategy, BUDGETS, seed, init_random=init_random)
    r = {"task": task_name, "method": method, "seed": seed, "feat": feat, "budgets": BUDGETS,
         "linear": lin, "lp": lp, "auc_linear": L.auc(lin, BUDGETS), "auc_lp": L.auc(lp, BUDGETS),
         "seconds": time.time() - t0}
    p.write_text(json.dumps(r))
    print(f"{task_name:16s} {method:12s} s{seed} auc lin {r['auc_linear']:.4f} lp {r['auc_lp']:.4f} "
          f"@1000 {lin[-1]:.3f}/{lp[-1]:.3f} {r['seconds']:.0f}s", flush=True)
    return r


def baselines(feat: str):
    for seed in SEEDS:
        tasks = eval_tasks(feat, seed)
        for tn, task in tasks.items():
            for m, fn in L.BASELINES.items():
                run_and_save(tn, task, m, fn, seed, feat)
        collect()


def train_policy(feat: str):
    """GRPO: groups of G sampled episodes on the same (task, seed); advantage = z-scored AUC."""
    pt = CKPT / feat / "policy.pt"
    pt.parent.mkdir(parents=True, exist_ok=True)
    X, y, Xt, yt = _feats(feat, "cifar100")
    pool_classes = list(range(50))
    iters = 3 if TINY else int(os.environ.get("AL_ITERS", 150))
    G = 4 if TINY else 8
    policy = L.Policy().to(L.DEV)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    start, log = 0, []
    if pt.exists():
        blob = torch.load(pt)
        policy.load_state_dict(blob["policy"]); opt.load_state_dict(blob["opt"])
        start, log = blob["iter"], blob["log"]
    if start >= iters:
        return policy
    rng = np.random.default_rng(1234)
    for _ in range(start):                       # keep the task stream identical on resume
        rng.choice(pool_classes, 10, replace=False); rng.integers(1 << 30)
    task_cache: dict = {}
    for it in range(start, iters):
        t0 = time.time()
        cl = tuple(sorted(rng.choice(pool_classes, 10, replace=False).tolist()))
        seed = int(rng.integers(1 << 30))
        if cl not in task_cache:
            if len(task_cache) > 30:
                task_cache.pop(next(iter(task_cache)))
            task_cache[cl] = subset_task(X, y, Xt, yt, list(cl), seed=seed % 1000)
        task = task_cache[cl]
        rewards, recs = [], []
        for g in range(G):
            rec: list = []
            q = L.q_policy_factory(policy, greedy=False, record=rec)
            lin, _ = L.run_episode(task, q, TRAIN_BUDGETS, seed * 31 + g, eval_lp=False)
            rewards.append(L.auc(lin, TRAIN_BUDGETS)); recs.append(rec)
        r = torch.tensor(rewards)
        adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = 0.0
        for a, rec in zip(adv, recs):
            lp = sum(L.pl_logprob(policy, f, s) for f, s in rec)
            loss = loss - a.item() * lp / G
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        log.append({"iter": it, "reward_mean": float(r.mean()), "reward_std": float(r.std()),
                    "seconds": time.time() - t0})
        print(f"iter {it} reward {r.mean():.4f} +- {r.std():.4f} {time.time() - t0:.1f}s", flush=True)
        if (it + 1) % 10 == 0 or it + 1 == iters:
            torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(), "iter": it + 1, "log": log},
                       str(pt) + ".tmp")
            os.replace(str(pt) + ".tmp", pt)
    return policy


def policy_eval(feat: str):
    policy = train_policy(feat).eval()
    q = L.q_policy_factory(policy, greedy=True)
    for seed in SEEDS:
        tasks = eval_tasks(feat, seed)
        for tn, task in tasks.items():
            run_and_save(tn, task, "policy", q, seed, feat)
        collect()


def _task_stream(X, y, Xt, yt, rng, cache):
    cl = tuple(sorted(rng.choice(list(range(50)), 10, replace=False).tolist()))
    seed = int(rng.integers(1 << 30))
    if cl not in cache:
        if len(cache) > 30:
            cache.pop(next(iter(cache)))
        cache[cl] = subset_task(X, y, Xt, yt, list(cl), seed=seed % 1000)
    return cache[cl], seed


def train_policy2(feat: str):
    """v2: TypiClust-aware features, behaviour cloning from TypiClust, then GRPO fine-tuning.
    Returns (bc_policy, rl_policy); both checkpoints are resumable."""
    d = CKPT / feat
    d.mkdir(parents=True, exist_ok=True)
    X, y, Xt, yt = _feats(feat, "cifar100")
    nf = len(L.POLICY_FEATURES_V2)
    bc_iters = 4 if TINY else int(os.environ.get("AL_BC_ITERS", 100))
    rl_iters = 3 if TINY else int(os.environ.get("AL_ITERS", 200))
    G = 4 if TINY else 8
    cache: dict = {}
    # ---- behaviour cloning
    bc_path = d / "policy2_bc.pt"
    policy = L.Policy(nf).to(L.DEV)
    if bc_path.exists():
        policy.load_state_dict(torch.load(bc_path)["policy"])
        bc_log = torch.load(bc_path)["log"]
    else:
        opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
        rng = np.random.default_rng(99)
        bc_log = []
        for it in range(bc_iters):
            task, seed = _task_stream(X, y, Xt, yt, rng, cache)
            rec: list = []
            q = L.q_policy_factory(policy, v2=True, bc_target=True, record=rec)
            L.run_episode(task, q, TRAIN_BUDGETS, seed, eval_lp=False)
            for ep in range(3):
                loss = -sum(L.pl_logprob(policy, f, sel) / max(1, len(sel)) for f, sel in rec) / len(rec)
                opt.zero_grad(); loss.backward(); opt.step()
            bc_log.append({"iter": it, "nll_per_pick": float(loss)})
            print(f"bc {it} nll/pick {float(loss):.3f}", flush=True)
        torch.save({"policy": policy.state_dict(), "log": bc_log}, bc_path)
    bc_policy = L.Policy(nf).to(L.DEV)
    bc_policy.load_state_dict(policy.state_dict())
    # ---- GRPO fine-tuning from the BC policy
    pt = d / "policy2.pt"
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    start, log = 0, []
    if pt.exists():
        blob = torch.load(pt)
        policy.load_state_dict(blob["policy"]); opt.load_state_dict(blob["opt"])
        start, log = blob["iter"], blob["log"]
    rng = np.random.default_rng(1234)
    for _ in range(start):
        rng.choice(list(range(50)), 10, replace=False); rng.integers(1 << 30)
    for it in range(start, rl_iters):
        t0 = time.time()
        task, seed = _task_stream(X, y, Xt, yt, rng, cache)
        rewards, recs = [], []
        for g in range(G):
            rec: list = []
            q = L.q_policy_factory(policy, greedy=False, record=rec, v2=True)
            lin, _ = L.run_episode(task, q, TRAIN_BUDGETS, seed * 31 + g, eval_lp=False)
            rewards.append(L.auc(lin, TRAIN_BUDGETS)); recs.append(rec)
        r = torch.tensor(rewards)
        adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = 0.0
        for a, rec in zip(adv, recs):
            loss = loss - a.item() * sum(L.pl_logprob(policy, f, s) for f, s in rec) / G
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        log.append({"iter": it, "reward_mean": float(r.mean()), "reward_std": float(r.std()),
                    "seconds": time.time() - t0})
        print(f"rl {it} reward {r.mean():.4f} +- {r.std():.4f} {time.time() - t0:.1f}s", flush=True)
        if (it + 1) % 10 == 0 or it + 1 == rl_iters:
            torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(), "iter": it + 1, "log": log,
                        "bc_log": bc_log}, str(pt) + ".tmp")
            os.replace(str(pt) + ".tmp", pt)
    return bc_policy.eval(), policy.eval()


def policy2_eval(feat: str):
    bc, rl = train_policy2(feat)
    qs = {"policy2_bc": L.q_policy_factory(bc, greedy=True, v2=True),
          "policy2": L.q_policy_factory(rl, greedy=True, v2=True)}
    for seed in SEEDS:
        tasks = eval_tasks(feat, seed)
        for tn, task in tasks.items():
            for m, q in qs.items():
                run_and_save(tn, task, m, q, seed, feat)
        collect()


def collect():
    rows = [json.loads(p.read_text()) for p in CKPT.glob("*/*__*__s*.json")]
    out = {"budgets": BUDGETS, "tiny": TINY, "n_runs": len(rows), "cells": {}}
    by = {}
    for r in rows:
        by.setdefault((r["feat"], r["task"], r["method"]), []).append(r)
    for (feat, task, m), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        out["cells"][f"{feat}/{task}/{m}"] = {
            "n": len(rs), "seeds": [r["seed"] for r in rs],
            "auc_linear": [float(np.mean([r["auc_linear"] for r in rs])), float(np.std([r["auc_linear"] for r in rs]))],
            "auc_lp": [float(np.mean([r["auc_lp"] for r in rs])), float(np.std([r["auc_lp"] for r in rs]))],
            "linear_curve": np.mean([r["linear"] for r in rs], 0).round(4).tolist(),
            "lp_curve": np.mean([r["lp"] for r in rs], 0).round(4).tolist(),
            "per_seed_auc_linear": [round(r["auc_linear"], 5) for r in rs],
            "per_seed_auc_lp": [round(r["auc_lp"], 5) for r in rs],
        }
    pol = list(CKPT.glob("*/policy.pt")) + list(CKPT.glob("*/policy2.pt"))
    out["policy_train"] = {}
    for p in pol:
        b = torch.load(p, map_location="cpu")
        out["policy_train"][f"{p.parent.name}/{p.stem}"] = b["log"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "baselines":
        baselines(sys.argv[2])
    elif mode == "policy":
        policy_eval(sys.argv[2])
    elif mode == "policy2":
        policy2_eval(sys.argv[2])
    collect()
