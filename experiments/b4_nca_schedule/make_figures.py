"""Figures for B4 from results.json:  python make_figures.py <results.json> <out_dir>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
arms = list(res["arms"])
steps = res["config"]["step_grid"]

fig, ax = plt.subplots(figsize=(7, 4))
for a in arms:
    curves = np.array([[r["step_curve"][str(t)][0] for t in steps] for r in res["arms"][a]["per_seed"]])
    m, s = curves.mean(0), curves.std(0)
    ax.plot(steps, m, label=f"{a} (n={len(curves)})")
    ax.fill_between(steps, m - s, m + s, alpha=0.15)
ax.set_xscale("log"); ax.set_xlabel("readout step"); ax.set_ylabel("test accuracy")
ax.set_title("Accuracy vs readout step (mean ± std over seeds)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out / "fig1_step_curve.png", dpi=130)

keys = ["anytime", "acc@40", "acc@2000", "damage@200", "swap@300"]
fig, axs = plt.subplots(1, len(keys), figsize=(3 * len(keys), 3.2), sharey=False)
for ax, k in zip(axs, keys):
    m = [res["arms"][a][k][0] for a in arms]; s = [res["arms"][a][k][1] for a in arms]
    ax.bar(range(len(arms)), m, yerr=s, color=[f"C{i}" for i in range(len(arms))])
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=60, fontsize=7); ax.set_title(k)
    lo = min(mm - ss for mm, ss in zip(m, s)); ax.set_ylim(max(0, lo - 0.05), None)
fig.tight_layout(); fig.savefig(out / "fig2_headline.png", dpi=130)

pv = res["paired_vs_pool_det"]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, k in zip(axs, ["anytime", "peak", "acc@40", "acc@2000"]):
    for i, a in enumerate(pv):
        d = pv[a][k]["diffs"]
        ax.scatter([i] * len(d), d, s=12, color=f"C{arms.index(a)}")
        ax.plot([i - 0.3, i + 0.3], [np.mean(d)] * 2, color="k")
    ax.axhline(0, color="gray", lw=0.8); ax.set_xticks(range(len(pv)))
    ax.set_xticklabels(list(pv), rotation=60, fontsize=7); ax.set_title(f"Δ {k} vs pool_det (paired)")
fig.tight_layout(); fig.savefig(out / "fig3_paired.png", dpi=130)
print("figures in", out)
