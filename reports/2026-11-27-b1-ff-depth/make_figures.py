"""Figures for the B1 report. Reads results/b1_ff_depth/1303586/summary.json."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "../../results/b1_ff_depth/1303586/summary.json")))
cells = S["cells"].values()
RULES = [("bp", "backprop", "#1f4e79"), ("local_ce", "greedy local CE", "#d9822b"), ("ff", "FF goodness (pairwise)", "#7a9a3a")]
DS = [("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100")]

def per_seed(rule, ds, depth, key):
    return np.array([key(c) for c in cells if c["rule"] == rule and c["dataset"] == ds and c["depth"] == depth])

# fig1: final accuracy vs depth
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (ds, dname) in zip(axs, DS):
    for r, lab, col in RULES:
        ds_ = [4, 8, 12]
        m = [per_seed(r, ds, d, lambda c: c["block_acc"][-1]).mean() for d in ds_]
        sd = [per_seed(r, ds, d, lambda c: c["block_acc"][-1]).std() for d in ds_]
        ax.errorbar(ds_, m, yerr=sd, marker="o", color=col, label=lab, capsize=3)
    ax.set_title(dname); ax.set_xlabel("depth (conv blocks)"); ax.set_xticks([4, 8, 12])
    ax.set_ylabel("test accuracy (last block)"); ax.grid(alpha=.3)
axs[0].legend(fontsize=8)
fig.suptitle("Final accuracy vs depth, 3 seeds, 40 epochs per block (local) / per network (bp)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_acc_vs_depth.png"), dpi=130)

# fig2: per-block probe curves at depth 12
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (ds, dname) in zip(axs, DS):
    for r, lab, col in RULES:
        bc = np.array([c["block_acc"] for c in cells if c["rule"] == r and c["dataset"] == ds and c["depth"] == 12])
        x = np.arange(1, 13)
        ax.plot(x, bc.mean(0), marker=".", color=col, label=lab)
        ax.fill_between(x, bc.min(0), bc.max(0), color=col, alpha=.2)
    ax.set_title(f"{dname}, depth 12"); ax.set_xlabel("block"); ax.set_ylabel("linear-probe test accuracy")
    ax.grid(alpha=.3)
axs[0].legend(fontsize=8)
fig.suptitle("Depth curve: where does each rule stop composing?")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_block_curves.png"), dpi=130)

# table numbers
for ds, _ in DS:
    for d in [4, 8, 12]:
        bp = per_seed("bp", ds, d, lambda c: c["block_acc"][-1])
        for r in ["local_ce", "ff"]:
            v = per_seed(r, ds, d, lambda c: c["block_acc"][-1]); b = per_seed(r, ds, d, lambda c: max(c["block_acc"]))
            g = per_seed(r, ds, d, lambda c: max(c["block_acc"]) - c["block_acc"][0])
            print(ds, d, r, f"last {v.mean():.3f} best {b.mean():.3f} bp {bp.mean():.3f} gap(best) {bp.mean()-b.mean():+.3f} gain {g.min():.3f}..{g.max():.3f}")
for ds, _ in DS:
    for r in ["bp", "local_ce", "ff"]:
        bc = np.array([c["block_acc"] for c in cells if c["rule"] == r and c["dataset"] == ds and c["depth"] == 12]).mean(0)
        print(ds, r, "d12 peak block", int(bc.argmax()) + 1, "b6", round(bc[5], 3), "b12", round(bc[11], 3))
