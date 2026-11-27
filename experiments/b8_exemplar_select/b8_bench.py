#!/usr/bin/env python
"""B8: which examples to write into a nearest-neighbour memory classifier, on foundation features.

Lifts Random-Ideas PR #62 (2026-11-27-memory-exemplar-select: k-center writes beat random by +0.062 at
20/class on 47-way class-incremental EMNIST with a small frozen CNN) to frozen DINOv2 features on
class-incremental CIFAR-100 / CIFAR-10, and adds the baseline that foundation features make obvious:
a class-mean prototype (NCM), which needs no selection at all.

Protocol (per seed): random class order, classes arrive in stages (CIFAR-100: 10 x 10, CIFAR-10: 5 x 2).
When a class arrives, the selector sees that class's training features and writes B of them to memory;
memory is never rewritten. After each stage the readout classifies the test images of all seen classes.
Readout (as in PR #62): global top-3 cosine neighbours, label scores sum exp(cos/0.1).
Selectors: random, herding (iCaRL), kcenter (spherical k-means, k-means++ init, 15 iters, store nearest
real point to each centroid), all (whole class; upper bound). Prototype baselines (no selection):
ncm (1 class mean, nearest mean), and kmeans_proto (B spherical k-means centroids, not real points, same
top-3 readout) to separate "coverage" from "storing real examples".

Outputs: ckpt/<dataset>_<feat>_s<seed>.json per seed (resumable), out/summary.json from `collect`.
Paths from env only (RUN_DIR, LAB_DATA).
"""
import os, sys, json, glob, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

RUN_DIR = Path(os.environ.get("RUN_DIR", "./_run")); CKPT = RUN_DIR / "ckpt"; OUT = RUN_DIR / "out"
CKPT.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
FEAT = Path(os.environ.get("LAB_DATA", ".")) / "features"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TINY = os.environ.get("B8_TINY", "0") == "1"
BUDGETS = [1, 5, 20] if TINY else [1, 2, 5, 10, 20, 40]
SEEDS = [0, 1] if TINY else list(range(10))
SELECTORS = ["random", "herding", "kcenter", "kmeans_proto"]
STAGES = {"cifar100": 10, "cifar10": 5}
TAU, TOPK = 0.1, 3


def load(feat, ds):
    d = torch.load(FEAT / f"{feat}_{ds}.pt", map_location="cpu")
    xtr = F.normalize(d["train"].float(), dim=1); xte = F.normalize(d["test"].float(), dim=1)
    ytr = torch.as_tensor(d["ytr"]).long(); yte = torch.as_tensor(d["yte"]).long()
    if TINY:
        keep = torch.cat([torch.nonzero(ytr == c).flatten()[:100] for c in range(int(ytr.max()) + 1)])
        xtr, ytr = xtr[keep], ytr[keep]
    return xtr.to(DEV), ytr.to(DEV), xte.to(DEV), yte.to(DEV)


def spherical_kmeans(x, k, g, iters=15):
    n = x.shape[0]; k = min(k, n)
    idx = [int(torch.randint(n, (1,), generator=g))]
    d2 = (1 - x @ x[idx[0]]).clamp_min(0)
    for _ in range(1, k):  # k-means++ seeding on cosine distance
        p = d2 / d2.sum() if d2.sum() > 0 else torch.full_like(d2, 1 / n)
        j = int(torch.multinomial(p.cpu(), 1, generator=g)); idx.append(j)
        d2 = torch.minimum(d2, (1 - x @ x[j]).clamp_min(0))
    c = x[idx].clone()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        for j in range(k):
            m = a == j
            if m.any(): c[j] = F.normalize(x[m].mean(0), dim=0)
    return c


def select(name, x, b, g):
    n = x.shape[0]
    if name == "random":
        return x[torch.randperm(n, generator=g)[:b].to(x.device)]
    if name == "herding":
        mu = x.mean(0); chosen = []; acc = torch.zeros_like(mu)
        for t in range(1, min(b, n) + 1):
            score = -((mu - (acc + x) / t) ** 2).sum(1)
            if chosen: score[chosen] = -1e9
            j = int(score.argmax()); chosen.append(j); acc = acc + x[j]
        return x[chosen]
    c = spherical_kmeans(x, b, g)
    if name == "kmeans_proto":
        return c
    if name == "kcenter":  # nearest distinct real point to each centroid
        sim = x @ c.T; chosen = []
        for j in range(c.shape[0]):
            s = sim[:, j].clone()
            if chosen: s[chosen] = -1e9
            chosen.append(int(s.argmax()))
        return x[chosen]
    raise ValueError(name)


@torch.no_grad()
def knn_acc(mem_x, mem_y, xq, yq, ncls):
    k = min(TOPK, mem_x.shape[0]); correct = 0
    for i in range(0, xq.shape[0], 2048):
        s = xq[i:i + 2048] @ mem_x.T
        v, j = s.topk(k, dim=1)
        sc = torch.zeros(v.shape[0], ncls, device=xq.device)
        sc.scatter_add_(1, mem_y[j], torch.exp(v / TAU))
        correct += (sc.argmax(1) == yq[i:i + 2048]).sum().item()
    return correct / xq.shape[0]


@torch.no_grad()
def ncm_acc(means, mean_y, xq, yq):
    pred = mean_y[(xq @ F.normalize(means, dim=1).T).argmax(1)]
    return (pred == yq).float().mean().item()


@torch.no_grad()
def run_seed(ds, feat, seed):
    xtr, ytr, xte, yte = load(feat, ds)
    ncls = int(ytr.max()) + 1; nst = STAGES[ds]; per = ncls // nst
    order = torch.randperm(ncls, generator=torch.Generator().manual_seed(1000 + seed)).tolist()
    stages = [order[i * per:(i + 1) * per] for i in range(nst)]
    res = {}  # method|budget -> list of per-stage acc
    arms = [(s, b) for s in SELECTORS for b in BUDGETS] + [("all", 0), ("ncm", 0)]
    for sel, b in arms:
        g = torch.Generator().manual_seed(seed * 7919 + b)
        mx, my, seen, accs = [], [], [], []
        for st in stages:
            for c in st:
                xc = xtr[ytr == c]
                if sel == "all": m = xc
                elif sel == "ncm": m = xc.mean(0, keepdim=True)
                else: m = select(sel, xc, b, g)
                mx.append(m); my.append(torch.full((m.shape[0],), c, device=DEV, dtype=torch.long))
            seen += st
            q = torch.isin(yte, torch.tensor(seen, device=DEV))
            MX, MY = torch.cat(mx), torch.cat(my)
            a = ncm_acc(MX, MY, xte[q], yte[q]) if sel == "ncm" else knn_acc(MX, MY, xte[q], yte[q], ncls)
            accs.append(a)
        res[f"{sel}|{b}"] = accs
    return {"dataset": ds, "feat": feat, "seed": seed, "acc": res}


def worker(ds, feat):
    for s in SEEDS:
        f = CKPT / f"{ds}_{feat}_s{s}.json"
        if f.exists(): continue
        t0 = time.time(); r = run_seed(ds, feat, s); r["sec"] = round(time.time() - t0, 1)
        json.dump(r, open(f, "w"))
        fin = {k: round(v[-1], 4) for k, v in r["acc"].items() if k.endswith("|20") or k.endswith("|0")}
        print(f"{ds} {feat} s{s} final {fin} ({r['sec']}s)", flush=True)


def collect():
    rows = [json.load(open(f)) for f in sorted(glob.glob(str(CKPT / "*.json")))]
    groups = {}
    for r in rows: groups.setdefault((r["dataset"], r["feat"]), []).append(r)
    summ = {}
    for (ds, feat), lst in groups.items():
        keys = lst[0]["acc"].keys(); g = {}
        for k in keys:
            fin = np.array([r["acc"][k][-1] for r in lst]); avg = np.array([np.mean(r["acc"][k]) for r in lst])
            g[k] = {"final_mean": float(fin.mean()), "final_std": float(fin.std()),
                    "avg_inc_mean": float(avg.mean()), "final": fin.tolist()}
        # pre-stated tests (README): H1 kcenter vs random at 20/class; H2 best exemplar kNN at 20 vs NCM
        def paired(a, b):
            d = np.array(g[a]["final"]) - np.array(g[b]["final"]); return float(d.mean()), int((d > 0).sum())
        tests = {}
        if "kcenter|20" in g:
            m, w = paired("kcenter|20", "random|20"); tests["H1_kcenter_vs_random_20"] = {"gain": m, "wins": w, "n": len(lst)}
            best = max(["random|20", "herding|20", "kcenter|20"], key=lambda k: g[k]["final_mean"])
            m, w = paired(best, "ncm|0"); tests["H2_best_exemplar20_vs_ncm"] = {"arm": best, "gain": m, "wins": w, "n": len(lst)}
        summ[f"{ds}|{feat}"] = {"arms": g, "tests": tests, "n_seeds": len(lst)}
    json.dump({"summary": summ, "config": {"budgets": BUDGETS, "seeds": SEEDS, "tiny": TINY, "tau": TAU,
               "topk": TOPK, "stages": STAGES}}, open(OUT / "summary.json", "w"), indent=1)
    for k, v in summ.items(): print(k, json.dumps(v["tests"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "worker": worker(sys.argv[2], sys.argv[3])
    elif cmd == "collect": collect()
