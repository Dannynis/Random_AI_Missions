"""B7 step 1: does typicality lose its edge when it is a poor prior?

A2 found that on clean CIFAR with frozen DINOv2 features, TypiClust is the best
cold-start active-learning rule and no learned policy beats it. The natural
question (A2's report, hypothesis B7): on pools where "typical" points are a bad
thing to label -- long-tailed class frequencies (typical points over-represent
head classes) and heavy label noise (typical points can be confidently mislabeled)
-- does TypiClust's advantage shrink, so that another rule (or, later, a learned
policy) has headroom?

This driver reuses the A2 library and its cached DINOv2 features. It runs the same
baseline rules on three hard eval regimes and compares every rule to TypiClust on
paired seeds. No policy is trained here; this is the precondition test that decides
whether the full B7 policy build is worth strong-GPU time.

    python b7_bench.py baselines <feat>   # every rule x seed on every hard task
    python b7_bench.py collect            # merge per-run JSONs -> $RUN_DIR/out/summary.json

Env: RUN_DIR, LAB_DATA (features in $LAB_DATA/features), AL_TINY=1 for a smoke test,
AL_SEEDS (default 10).
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

# Reuse the validated A2 library.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "a2_al_policy"))
import al_lib as L  # noqa: E402

TINY = os.environ.get("AL_TINY") == "1"
RUN_DIR = Path(os.environ["RUN_DIR"])
CKPT, OUT = RUN_DIR / "ckpt", RUN_DIR / "out"
FEAT = Path(os.environ["LAB_DATA"]) / "features"
SEEDS = range(2 if TINY else int(os.environ.get("AL_SEEDS", 10)))
BUDGETS = [10, 20, 40, 60] if TINY else [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
POOL_FRAC = 0.8


def load_feats(feat: str, ds: str):
    name = f"{feat}_{ds}{'_tiny' if TINY else ''}.pt"
    b = torch.load(FEAT / name)
    X = F.normalize(b["train"].float(), dim=1).to(L.DEV)
    Xt = F.normalize(b["test"].float(), dim=1).to(L.DEV)
    return X, b["ytr"].to(L.DEV), Xt, b["yte"].to(L.DEV)


def subsample(X, y, seed, frac=POOL_FRAC):
    g = torch.Generator().manual_seed(10_000 + seed)
    keep = torch.randperm(len(X), generator=g)[: int(frac * len(X))].sort().values.to(X.device)
    return X[keep], y[keep]


def longtail_pool(X, y, n_cls, seed, rho=100.0):
    """Exponential class-imbalance subsample of the pool (largest/smallest = rho).

    Class c keeps n_c = n_max * rho**(-c'/(C-1)) points, where c' is the class rank
    under a fixed per-seed shuffle of class order (so which classes are head/tail
    varies by seed). Test set is left balanced.
    """
    g = np.random.default_rng(20_000 + seed)
    order = g.permutation(n_cls)
    rank = np.empty(n_cls, dtype=int)
    rank[order] = np.arange(n_cls)
    keep_idx = []
    for c in range(n_cls):
        idx = (y == c).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(idx) == 0:
            continue
        n_max = len(idx)
        n_c = max(1, int(round(n_max * rho ** (-rank[c] / max(1, n_cls - 1)))))
        sel = g.choice(idx, size=min(n_c, len(idx)), replace=False)
        keep_idx.append(sel)
    keep = torch.tensor(np.concatenate(keep_idx), device=X.device).sort().values
    return X[keep], y[keep]


def add_label_noise(y, n_cls, seed, rate, mode="sym"):
    """Return a copy of pool labels with `rate` fraction corrupted.

    sym:  flipped to a uniformly random *different* class.
    asym: flipped to a fixed confusable class (c -> (c+1) % n_cls).
    """
    g = np.random.default_rng(30_000 + seed)
    yn = y.clone().cpu().numpy()
    n = len(yn)
    flip = g.random(n) < rate
    idx = np.nonzero(flip)[0]
    if mode == "sym":
        for i in idx:
            c = yn[i]
            nc = g.integers(0, n_cls - 1)
            yn[i] = nc if nc < c else nc + 1
    else:  # asym
        yn[idx] = (yn[idx] + 1) % n_cls
    return torch.tensor(yn, device=y.device), float(flip.mean())


def hard_tasks(feat: str, seed: int):
    tasks, meta = {}, {}
    # Long-tailed CIFAR-100 (balanced test).
    X, y, Xt, yt = load_feats(feat, "cifar100")
    Xs, ys = subsample(X, y, seed)
    Xlt, ylt = longtail_pool(Xs, ys, 100, seed, rho=100.0)
    tasks["c100_longtail"] = L.Task(Xlt, ylt, Xt, yt, 100).prepare(seed=seed)
    meta["c100_longtail"] = {"pool": int(len(ylt)), "rho": 100.0,
                             "min_class": int(torch.bincount(ylt, minlength=100).min()),
                             "max_class": int(torch.bincount(ylt, minlength=100).max())}
    # Label-noise CIFAR-10 (clean test).
    X, y, Xt, yt = load_feats(feat, "cifar10")
    Xs, ys = subsample(X, y, seed)
    for mode, rate in (("sym", 0.4), ("asym", 0.4)):
        yn, got = add_label_noise(ys, 10, seed, rate, mode)
        tasks[f"c10_{mode}{int(rate*100)}"] = L.Task(Xs, yn, Xt, yt, 10).prepare(seed=seed)
        meta[f"c10_{mode}{int(rate*100)}"] = {"pool": int(len(ys)), "noise_rate": got, "mode": mode}
    return tasks, meta


def run_and_save(task_name, task, method, strategy, seed, feat, meta):
    p = CKPT / feat / f"{task_name}__{method}__s{seed}.json"
    if p.exists():
        return json.loads(p.read_text())
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(seed)
    lin, lp = L.run_episode(task, strategy, BUDGETS, seed)
    r = {"task": task_name, "method": method, "seed": seed, "feat": feat, "budgets": BUDGETS,
         "linear": lin, "lp": lp, "auc_linear": L.auc(lin, BUDGETS), "auc_lp": L.auc(lp, BUDGETS),
         "meta": meta.get(task_name, {}), "seconds": time.time() - t0}
    p.write_text(json.dumps(r))
    print(f"{task_name:16s} {method:10s} s{seed} auc lin {r['auc_linear']:.4f} lp {r['auc_lp']:.4f} "
          f"@1000 {lin[-1]:.3f}/{lp[-1]:.3f} {r['seconds']:.0f}s", flush=True)
    return r


def baselines(feat: str):
    for seed in SEEDS:
        tasks, meta = hard_tasks(feat, seed)
        for tn, task in tasks.items():
            for m, fn in L.BASELINES.items():
                run_and_save(tn, task, m, fn, seed, feat, meta)
        collect()


def collect():
    rows = []
    for p in sorted((CKPT).rglob("*.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        print("no runs yet"); return
    # aggregate per (task, method): mean/std AUC (linear and lp) over seeds
    agg = {}
    for r in rows:
        for rd in ("linear", "lp"):
            agg.setdefault((r["task"], r["method"], rd), {})[r["seed"]] = r[f"auc_{rd}"]
    summ = {"n_runs": len(rows), "methods": sorted({r["method"] for r in rows}),
            "tasks": sorted({r["task"] for r in rows}), "agg": {}, "vs_typiclust": {}}
    for (t, m, rd), bys in agg.items():
        vals = np.array(list(bys.values()))
        summ["agg"][f"{t}|{m}|{rd}"] = {"mean": float(vals.mean()), "std": float(vals.std()),
                                        "n": int(len(vals))}
    # paired contrast: each rule - typiclust, per task, readout = lp (semi-sup) and linear
    for t in summ["tasks"]:
        for rd in ("linear", "lp"):
            ty = agg.get((t, "typiclust", rd), {})
            for m in summ["methods"]:
                if m == "typiclust":
                    continue
                mm = agg.get((t, m, rd), {})
                seeds = sorted(set(ty) & set(mm))
                if not seeds:
                    continue
                d = np.array([mm[s] - ty[s] for s in seeds])
                summ["vs_typiclust"][f"{t}|{m}|{rd}"] = {
                    "mean": float(d.mean()), "n": len(d), "n_pos": int((d > 0).sum()),
                    "win_frac": float((d > 0).mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    # verdict: any rule beats typiclust by >=0.01 mean on >=80% seeds, on any task/readout
    head = [(k, v) for k, v in summ["vs_typiclust"].items()
            if v["mean"] >= 0.01 and v["win_frac"] >= 0.8]
    summ["headroom"] = sorted(head, key=lambda kv: -kv[1]["mean"])
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"collected {len(rows)} runs; headroom cells (rule beats TypiClust >=0.01 on >=80% seeds): {len(head)}")
    for k, v in summ["headroom"][:10]:
        print(f"  {k}: +{v['mean']:.4f} ({v['n_pos']}/{v['n']})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baselines"
    if cmd == "collect":
        collect()
    else:
        baselines(sys.argv[2] if len(sys.argv) > 2 else "dinov2-base")
