"""Figures for the B8 report. Usage: python make_figures.py <job id>  (reads results/b8_exemplar_select/<id>/summary.json)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, f"../../results/b8_exemplar_select/{sys.argv[1]}/summary.json")))
SUM, B = S["summary"], S["config"]["budgets"]
FEATS = [("dinov2-small", "ViT-S/14"), ("dinov2-base", "ViT-B/14"), ("dinov2-large", "ViT-L/14")]
SEL = [("random", "#888888"), ("herding", "#d9822b"), ("kcenter", "#1f4e79"), ("kmeans_proto", "#7a9a3a")]
for ds, dname in [("cifar100", "CIFAR-100, 10 stages"), ("cifar10", "CIFAR-10, 5 stages")]:
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, (f, fname) in zip(axs, FEATS):
        k = f"{ds}|{f}"
        if k not in SUM: ax.set_visible(False); continue
        A = SUM[k]["arms"]
        for s, col in SEL:
            m = [A[f"{s}|{b}"]["final_mean"] for b in B]; sd = [A[f"{s}|{b}"]["final_std"] for b in B]
            ax.errorbar(B, m, yerr=sd, marker="o", ms=3, color=col, label=s, capsize=2)
        ax.axhline(A["ncm|0"]["final_mean"], color="crimson", ls="--", label="NCM (1 mean/class)")
        ax.axhline(A["all|0"]["final_mean"], color="black", ls=":", label="all training images")
        ax.set_xscale("log"); ax.set_xticks(B); ax.set_xticklabels(B); ax.set_title(fname)
        ax.set_xlabel("stored per class"); ax.set_ylabel("final accuracy"); ax.grid(alpha=.3)
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{dname}: final accuracy vs memory budget (10 seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"fig_{ds}_budget.png"), dpi=130)
for k, v in SUM.items():
    A = v["arms"]
    print(k, v["n_seeds"], json.dumps(v["tests"]))
    print("   ", {a: round(A[a]["final_mean"], 4) for a in ["random|20", "herding|20", "kcenter|20", "kmeans_proto|20",
          "random|1", "kcenter|1", "kmeans_proto|1", "all|0", "ncm|0"]})
