#!/usr/bin/env python
"""B7 figures: how much AUC headroom each existing rule opens over TypiClust on
three hard active-learning pools (long-tailed CIFAR-100, symmetric- and
asymmetric-40% label noise on CIFAR-10). Reads the collected summary.json.

usage: make_figures.py <summary.json> <out_dir>
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summ = json.load(open(sys.argv[1]))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)

TASKS = [("c100_longtail", "long-tailed CIFAR-100 (imb 100)"),
         ("c10_sym40", "CIFAR-10, 40% symmetric noise"),
         ("c10_asym40", "CIFAR-10, 40% asymmetric noise")]
RULES = ["random", "entropy", "margin", "coreset", "typiclust", "probcover", "badge"]
READ = ["lp", "linear"]
agg = summ["agg"]
vs = summ["vs_typiclust"]

# ---- Fig 1: mean AUC per rule, grouped by task (lp readout), TypiClust highlighted ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
for ax, (tk, title) in zip(axes, TASKS):
    means = [agg[f"{tk}|{r}|lp"]["mean"] for r in RULES]
    stds = [agg[f"{tk}|{r}|lp"]["std"] for r in RULES]
    colors = ["#c44e52" if r == "typiclust" else "#4c72b0" for r in RULES]
    # best non-typiclust rule in green
    order = np.argsort(means)
    best = RULES[order[-1]]
    colors = ["#c44e52" if r == "typiclust" else ("#55a868" if r == best else "#4c72b0")
              for r in RULES]
    ax.bar(range(len(RULES)), means, yerr=stds, color=colors, capsize=2)
    ax.set_xticks(range(len(RULES)))
    ax.set_xticklabels(RULES, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("label-efficiency AUC (lp)" if tk == TASKS[0][0] else "")
    ax.grid(axis="y", alpha=0.3)
axes[0].text(0.02, 0.97, "red = TypiClust   green = best rule",
             transform=axes[0].transAxes, va="top", fontsize=8)
fig.suptitle("B7: TypiClust is not the best rule when it is a poor prior", fontsize=11)
fig.tight_layout()
fig.savefig(f"{out}/fig1_auc_by_task.png", dpi=130)
plt.close(fig)

# ---- Fig 2: headroom (best rule minus TypiClust) with win fraction, per (task, readout) ----
fig, ax = plt.subplots(figsize=(9, 4.6))
labels, gaps, wins, cols = [], [], [], []
for tk, title in TASKS:
    for rd in READ:
        # best positive contrast vs typiclust on this cell
        cands = [(r, vs.get(f"{tk}|{r}|{rd}")) for r in RULES if r != "typiclust"]
        cands = [(r, c) for r, c in cands if c is not None]
        r, c = max(cands, key=lambda x: x[1]["mean"])
        labels.append(f"{tk}\n{rd}: {r}")
        gaps.append(c["mean"])
        wins.append(c["win_frac"])
        cols.append("#55a868" if (c["mean"] >= 0.01 and c["win_frac"] >= 0.8) else "#bbbbbb")
y = np.arange(len(labels))
ax.barh(y, gaps, color=cols)
ax.axvline(0.01, color="k", ls="--", lw=1)
ax.text(0.0105, len(labels) - 0.5, "pass bar 0.01", fontsize=8, rotation=90, va="top")
for i, (g, w) in enumerate(zip(gaps, wins)):
    ax.text(g + 0.001 if g >= 0 else g - 0.001, i, f"{w:.0%} seeds",
            va="center", ha="left" if g >= 0 else "right", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("best rule's AUC gain over TypiClust (paired, 10 seeds)")
ax.set_title("B7 headroom: green = clears the pre-stated bar (>=0.01 on >=8/10 seeds)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{out}/fig2_headroom.png", dpi=130)
plt.close(fig)
print("wrote", out)
