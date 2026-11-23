"""Figures + paired table for A2 from one or more summary.json files.

    python make_figures.py <out_dir> <summary.json> [<summary.json> ...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
cells, budgets, ptrain = {}, None, {}
for p in sys.argv[2:]:
    s = json.loads(Path(p).read_text())
    cells.update(s["cells"]); budgets = s["budgets"]; ptrain.update(s.get("policy_train", {}))

feats = sorted({k.split("/")[0] for k in cells})
lines = []
for feat in feats:
    tasks = sorted({k.split("/")[1] for k in cells if k.startswith(feat + "/")})
    groups = {"cifar10": ["cifar10"], "cifar100": ["cifar100"],
              "c100_heldout": [t for t in tasks if t.startswith("c100_heldout")]}
    for learner in ("linear", "lp"):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (g, ts) in zip(axs, groups.items()):
            if not ts:
                continue
            methods = sorted({k.split("/")[2] for k in cells if k.split("/")[1] in ts and k.startswith(feat + "/")})
            # per method: average curve over tasks in the group; per-seed AUC averaged over tasks
            auc = {}
            for m in methods:
                cs = [cells[f"{feat}/{t}/{m}"] for t in ts if f"{feat}/{t}/{m}" in cells]
                curve = np.mean([c[f"{learner}_curve"] for c in cs], 0)
                ax.plot(budgets, curve, label=m, lw=2.5 if m == "policy" else 1.2)
                n = min(len(c[f"per_seed_auc_{learner}"]) for c in cs)
                auc[m] = np.mean([c[f"per_seed_auc_{learner}"][:n] for c in cs], 0)
            ax.set_xscale("log"); ax.set_title(f"{feat} {g} ({learner})"); ax.set_xlabel("labels")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
            base = {m: v for m, v in auc.items() if m != "policy"}
            best = max(base, key=lambda m: base[m].mean()) if base else None
            for m, v in sorted(auc.items(), key=lambda kv: -kv[1].mean()):
                row = f"| {feat} | {g} | {learner} | {m} | {v.mean():.4f} ± {v.std():.4f} | {len(v)} |"
                if m == "policy" and best is not None:
                    n = min(len(v), len(base[best]))
                    d = v[:n] - base[best][:n]
                    row += f" vs {best}: {d.mean():+.4f}, wins {int((d > 0).sum())}/{n} |"
                else:
                    row += " |"
                lines.append(row)
        fig.tight_layout(); fig.savefig(out / f"curves_{feat}_{learner}.png", dpi=120)
        plt.close(fig)

for name, log in ptrain.items():
    fig, ax = plt.subplots(figsize=(6, 3))
    r = [e["reward_mean"] for e in log]
    ax.plot(r, alpha=0.4); ax.plot(np.convolve(r, np.ones(10) / 10, mode="valid"))
    ax.set_xlabel("GRPO iteration"); ax.set_ylabel("train reward (AUC)"); ax.set_title(f"policy training {name}")
    fig.tight_layout(); fig.savefig(out / f"policy_train_{name}.png", dpi=120); plt.close(fig)

table = "| features | task | learner | method | AUC (mean ± std over seeds) | seeds | policy vs best baseline |\n|---|---|---|---|---|---|---|\n"
(out / "table.md").write_text(table + "\n".join(lines) + "\n", encoding="utf-8")
print(table + "\n".join(lines))
