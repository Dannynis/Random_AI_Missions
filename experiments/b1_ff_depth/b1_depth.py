#!/usr/bin/env python
"""B1: does a layer-local learner compose features with depth on a hard dataset?

Decomposition (lifting the MNIST study in Random-Ideas PR #46 / #60 to CIFAR):
  bp        - end-to-end backprop (neither local nor goodness): the baseline.
  local_ce  - greedy layer-local cross-entropy (LOCAL, standard loss): each conv
              block is trained with its own detached input and its own linear CE
              head; earlier blocks frozen. Tests locality alone.
  ff        - greedy layer-local GOODNESS with the pairwise SymBa loss (LOCAL +
              goodness loss): supervised Forward-Forward, label written into
              extra input channels; per block g = mean(act^2), loss
              softplus(-4*(g_pos - g_neg)); readout by a neutral-overlay linear
              probe. Tests locality + the goodness objective together.

For every rule we record each block's linear-probe test accuracy (the depth
curve) so we can ask whether depth composes (deepest block >> first block) or
plateaus as it did on MNIST.

Resumable: per (rule, depth, seed) cell -> ckpt/<cell>.pt holds model+optim+epoch
state (epoch-level checkpoint, survives preemption); the finished cell's metrics
go to ckpt/<cell>.json and are skipped on resume. `collect` aggregates to
out/summary.json.

Paths come from env only (RUN_DIR, LAB_DATA, HF_HOME); nothing host-specific.
"""
import os, sys, json, time, math, signal, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T

RUN_DIR = os.environ.get("RUN_DIR", "./_run")
CKPT = os.path.join(RUN_DIR, "ckpt"); OUT = os.path.join(RUN_DIR, "out")
DATA = os.environ.get("LAB_DATA", os.path.expanduser("~/lab_data"))
os.makedirs(CKPT, exist_ok=True); os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TINY = os.environ.get("B1_TINY", "0") == "1"

# preemption flag: job.sh forwards SIGUSR1; we finish the epoch and checkpoint.
_PREEMPT = {"hit": False}
def _on_sig(*a): _PREEMPT["hit"] = True
for _sig in ("SIGUSR1", "SIGTERM"):  # SIGUSR1 absent on Windows; present on the Linux cluster
    try: signal.signal(getattr(signal, _sig), _on_sig)
    except Exception: pass

# ----- config -----
EPOCHS = 3 if TINY else 40          # per cell (bp: total; local/ff: per block)
BATCH = 128
DEPTHS = [2, 4] if TINY else [4, 8, 12]
SEEDS = [0, 1] if TINY else [0, 1, 2]
RULES = ["bp", "local_ce", "ff"]
# channel width per block index (downsample where width increases)
WIDTHS = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]

def width(i): return WIDTHS[min(i, len(WIDTHS) - 1)]

def loaders(dataset, seed):
    g = torch.Generator().manual_seed(seed)
    mean = (0.4914, 0.4822, 0.4465); std = (0.247, 0.243, 0.261)
    tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(mean, std)])
    te_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    DS = tv.datasets.CIFAR100 if dataset == "cifar100" else tv.datasets.CIFAR10
    root = os.path.join(DATA, "cifar")
    tr = DS(root, train=True, download=True, transform=tr_tf)
    te = DS(root, train=False, download=True, transform=te_tf)
    if TINY:
        tr = torch.utils.data.Subset(tr, range(2000))
        te = torch.utils.data.Subset(te, range(1000))
    nw = 4
    trl = torch.utils.data.DataLoader(tr, BATCH, shuffle=True, num_workers=nw,
                                      drop_last=True, generator=g, persistent_workers=nw > 0)
    tel = torch.utils.data.DataLoader(te, 256, shuffle=False, num_workers=nw,
                                      persistent_workers=nw > 0)
    ncls = 100 if dataset == "cifar100" else 10
    return trl, tel, ncls

class Block(nn.Module):
    """One conv block; downsamples (stride 2) when the width increases.
    BN is disabled for Forward-Forward: BN would pin mean(act^2) and destroy the
    goodness signal, so FF relies on the between-block RMS norm instead."""
    def __init__(self, c_in, c_out, down, use_bn=True):
        super().__init__()
        s = 2 if down else 1
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=s, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(c_out) if use_bn else nn.Identity()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

def make_blocks(depth, c_in, use_bn=True):
    blocks = nn.ModuleList(); prev = c_in
    for i in range(depth):
        w = width(i); down = (w != prev)
        blocks.append(Block(prev, w, down and i > 0, use_bn))
        prev = w
    return blocks, prev

def gap(x): return x.mean(dim=(2, 3))

# ===================== backprop baseline =====================
class BPNet(nn.Module):
    def __init__(self, depth, ncls, c_in=3):
        super().__init__()
        self.blocks, cout = make_blocks(depth, c_in)
        self.head = nn.Linear(cout, ncls)
        # detached per-block probes for the depth curve (do not affect bp grad)
        self.probes = nn.ModuleList([nn.Linear(width(i), ncls) for i in range(depth)])
    def forward(self, x):
        feats = []
        for b in self.blocks:
            x = b(x); feats.append(gap(x))
        return self.head(feats[-1]), feats

def run_bp(depth, ncls, trl, tel, state):
    net = BPNet(depth, ncls).to(DEV)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    start = 0
    if state is not None:
        net.load_state_dict(state["net"]); opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"]); start = state["epoch"]
    for ep in range(start, EPOCHS):
        net.train()
        for x, y in trl:
            x, y = x.to(DEV), y.to(DEV)
            out, feats = net(x)
            loss = F.cross_entropy(out, y)
            # train detached probes alongside (no effect on main grad)
            for p, f in zip(net.probes, feats):
                loss = loss + F.cross_entropy(p(f.detach()), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if _PREEMPT["hit"]:
            return None, {"net": net.state_dict(), "opt": opt.state_dict(),
                          "sched": sched.state_dict(), "epoch": ep + 1}
    # eval: head + each probe
    per_block = eval_probes(net, tel, depth, head=net.head)
    return per_block, None

# ===================== greedy local (CE or FF) =====================
def overlay(x, y, ncls):
    # append one-hot label channels (broadcast over spatial); y may be None -> neutral
    b, _, h, w = x.shape
    lab = torch.zeros(b, ncls, h, w, device=x.device)
    if y is not None:
        lab[torch.arange(b), y] = 1.0
    return torch.cat([x, lab], dim=1)

def rms_norm(z):  # per-sample RMS normalization of a feature map
    return z / (z.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-6)

def run_local(rule, depth, ncls, trl, tel, state):
    """Greedy block-wise training.
      local_ce: each block trained with a detached input and its own linear CE
                head (locality, standard loss); head is also the readout.
      ff:       label written into ncls extra input channels; per block goodness
                g=mean(act^2), pairwise SymBa loss softplus(-4(g+ - g-)); blocks
                trained by goodness only, then neutral-overlay linear probes are
                fit on the frozen blocks for readout (matches eval)."""
    c_extra = ncls if rule == "ff" else 0
    blocks, _ = make_blocks(depth, 3 + c_extra, use_bn=(rule != "ff"))
    heads = nn.ModuleList([nn.Linear(width(i), ncls) for i in range(depth)])
    blocks, heads = blocks.to(DEV), heads.to(DEV)
    cur_block = 0; start_ep = 0; phase = "blocks"
    if state is not None:
        blocks.load_state_dict(state["blocks"]); heads.load_state_dict(state["heads"])
        cur_block = state["cur_block"]; start_ep = state["epoch"]; phase = state.get("phase", "blocks")

    def save(bi, ep, ph, opt=None, sched=None):
        st = {"blocks": blocks.state_dict(), "heads": heads.state_dict(),
              "cur_block": bi, "epoch": ep, "phase": ph}
        if opt is not None: st["opt"] = opt.state_dict(); st["sched"] = sched.state_dict()
        return st

    if phase == "blocks":
        for bi in range(cur_block, depth):
            for j in range(bi):
                for p in blocks[j].parameters(): p.requires_grad_(False)
            tgt = list(blocks[bi].parameters())
            if rule == "local_ce": tgt += list(heads[bi].parameters())
            opt = torch.optim.SGD(tgt, lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            s0 = start_ep if bi == cur_block else 0
            if state is not None and bi == cur_block and "opt" in state:
                opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
            for ep in range(s0, EPOCHS):
                blocks.train(); heads.train()
                for x, y in trl:
                    x, y = x.to(DEV), y.to(DEV)
                    with torch.no_grad():
                        if rule == "ff":
                            yneg = (y + torch.randint(1, ncls, y.shape, device=DEV)) % ncls
                            hpos = overlay(x, y, ncls); hneg = overlay(x, yneg, ncls)
                            for j in range(bi):
                                hpos = rms_norm(blocks[j](hpos)); hneg = rms_norm(blocks[j](hneg))
                        else:
                            h = x
                            for j in range(bi): h = blocks[j](h).detach()
                    if rule == "ff":
                        gp = blocks[bi](hpos).pow(2).mean(dim=(1, 2, 3))
                        gn = blocks[bi](hneg).pow(2).mean(dim=(1, 2, 3))
                        loss = F.softplus(-4.0 * (gp - gn)).mean()
                    else:
                        loss = F.cross_entropy(heads[bi](gap(blocks[bi](h))), y)
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                if _PREEMPT["hit"]:
                    return None, save(bi, ep + 1, "blocks", opt, sched)
            start_ep = 0
        cur_block = 0

    # readout: for ff, fit neutral-overlay probes on frozen blocks
    if rule == "ff":
        for p in blocks.parameters(): p.requires_grad_(False)
        opt = torch.optim.Adam(heads.parameters(), lr=1e-3, weight_decay=1e-4)
        pe = 1 if TINY else 8
        for ep in range(start_ep if phase == "probe" else 0, pe):
            heads.train()
            for x, y in trl:
                x, y = x.to(DEV), y.to(DEV)
                with torch.no_grad():
                    hh = overlay(x, None, ncls); feats = []
                    for j in range(depth):
                        hh = blocks[j](hh); feats.append(gap(hh)); hh = rms_norm(hh)
                loss = sum(F.cross_entropy(heads[j](feats[j]), y) for j in range(depth))
                opt.zero_grad(); loss.backward(); opt.step()
            if _PREEMPT["hit"]:
                return None, save(0, ep + 1, "probe")
    per_block = eval_local(blocks, heads, tel, depth, rule, ncls)
    return per_block, None

# ===================== evaluation =====================
@torch.no_grad()
def eval_probes(net, tel, depth, head):
    net.eval()
    correct = np.zeros(depth); chead = 0; n = 0
    for x, y in tel:
        x, y = x.to(DEV), y.to(DEV)
        out, feats = net(x)
        chead += (out.argmax(1) == y).sum().item()
        for i, f in enumerate(feats):
            correct[i] += (net.probes[i](f).argmax(1) == y).sum().item()
        n += y.numel()
    return {"block_acc": (correct / n).tolist(), "head_acc": chead / n, "n": n}

@torch.no_grad()
def eval_local(blocks, heads, tel, depth, rule, ncls):
    blocks.eval(); heads.eval()
    correct = np.zeros(depth); n = 0
    def rms_norm(z): return z / (z.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-6)
    for x, y in tel:
        x, y = x.to(DEV), y.to(DEV)
        if rule == "ff":
            b, _, h, w = x.shape
            lab = torch.zeros(b, ncls, h, w, device=DEV)  # neutral overlay
            hh = torch.cat([x, lab], dim=1)
            for i in range(depth):
                hh = blocks[i](hh)
                correct[i] += (heads[i](gap(hh)).argmax(1) == y).sum().item()
                hh = rms_norm(hh)
        else:
            hh = x
            for i in range(depth):
                hh = blocks[i](hh)
                correct[i] += (heads[i](gap(hh)).argmax(1) == y).sum().item()
        n += y.numel()
    ba = (correct / n).tolist()
    return {"block_acc": ba, "head_acc": max(ba), "n": n}

# ===================== driver =====================
def cell_name(rule, dataset, depth, seed):
    return f"{rule}_{dataset}_d{depth}_s{seed}"

def worker(widx, nworkers):
    datasets = ["cifar10"] if TINY else ["cifar10", "cifar100"]
    cells = [(r, ds, d, s) for ds in datasets for d in DEPTHS for s in SEEDS for r in RULES]
    for k, (rule, ds, depth, seed) in enumerate(cells):
        if k % nworkers != widx: continue
        done = os.path.join(CKPT, cell_name(rule, ds, depth, seed) + ".json")
        if os.path.exists(done): continue
        torch.manual_seed(seed); np.random.seed(seed)
        trl, tel, ncls = loaders(ds, seed)
        ptf = os.path.join(CKPT, cell_name(rule, ds, depth, seed) + ".pt")
        state = torch.load(ptf, map_location=DEV) if os.path.exists(ptf) else None
        t0 = time.time()
        if rule == "bp":
            res, resume = run_bp(depth, ncls, trl, tel, state)
        else:
            res, resume = run_local(rule, depth, ncls, trl, tel, state)
        if resume is not None:  # preempted: save and exit this cell
            torch.save(resume, ptf)
            print(f"[w{widx}] preempted {cell_name(rule,ds,depth,seed)} -> checkpoint", flush=True)
            return
        res.update(rule=rule, dataset=ds, depth=depth, seed=seed, sec=round(time.time() - t0, 1))
        json.dump(res, open(done, "w"))
        if os.path.exists(ptf): os.remove(ptf)
        print(f"[w{widx}] {cell_name(rule,ds,depth,seed)} head={res['head_acc']:.4f} "
              f"blocks={[round(a,3) for a in res['block_acc']]} ({res['sec']}s)", flush=True)

def collect():
    cells = {}
    for f in glob.glob(os.path.join(CKPT, "*.json")):
        d = json.load(open(f)); cells[os.path.basename(f)[:-5]] = d
    # aggregate over seeds: per (rule, dataset, depth)
    agg = {}
    import collections
    groups = collections.defaultdict(list)
    for name, d in cells.items():
        groups[(d["rule"], d["dataset"], d["depth"])].append(d)
    for (rule, ds, depth), lst in groups.items():
        head = np.array([x["head_acc"] for x in lst])
        # block curve: pad to depth
        bc = np.array([x["block_acc"] for x in lst])
        first = bc[:, 0]; best = bc.max(axis=1)
        agg[f"{rule}|{ds}|d{depth}"] = {
            "head_acc_mean": float(head.mean()), "head_acc_std": float(head.std()),
            "first_block_mean": float(first.mean()), "best_block_mean": float(best.mean()),
            "depth_gain_mean": float((best - first).mean()), "n_seeds": len(lst),
            "block_acc_mean": bc.mean(axis=0).tolist()}
    out = {"cells": cells, "agg": agg, "config": {"epochs": EPOCHS, "depths": DEPTHS,
           "seeds": SEEDS, "rules": RULES, "tiny": TINY}}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("collected", len(cells), "cells ->", os.path.join(OUT, "summary.json"))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "worker"
    if cmd == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "collect":
        collect()
