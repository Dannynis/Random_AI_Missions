"""Figures and a markdown table for A1 from synced results.

usage: python make_figures.py <results dir holding eval_seed*.json (one or more job folders)> <output dir>
Writes fig_success_fa1.png, fig_training.png, table.md.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def load(root):
    rows, aud, train = [], [], {}
    for f in sorted(glob.glob(os.path.join(root, "**", "eval_seed*.json"), recursive=True)):
        d = json.load(open(f))
        rows += [{**r, "seed": d["seed"]} for r in d["rows"]]
        aud += [{**r, "seed": d["seed"]} for r in d["audibility"]]
    for f in sorted(glob.glob(os.path.join(root, "**", "train_seed*.json"), recursive=True)):
        train[os.path.basename(f)[11:-5]] = json.load(open(f))
    return rows, aud, train


PRIMARY = {"ours": "crc@1", "audioseal": "top", "wavmark": "top"}
CONDS = ["clean", "mp3_64k", "opus_32k", "reverb", "ota_legacy", "ota", "ota_mp3"]


def table(rows, aud, key="fa1"):
    out = {}
    for r in rows:
        if r["rule"] != PRIMARY.get(r["method"]):
            continue
        v = r[key]["success"] if isinstance(r[key], dict) else r[key]
        out.setdefault((r["method"], r["target"], r["condition"]), []).append(v)
    return out


def main():
    root, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    rows, aud, train = load(root)
    if not rows:
        print("no eval results yet"); return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tab = table(rows, aud)
    targets = sorted({k[1] for k in tab}, key=lambda t: (t != "native", t))
    methods = [m for m in ("ours", "audioseal", "wavmark") if any(k[0] == m for k in tab)]
    conds = [c for c in CONDS if any(k[2] == c for k in tab)]
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], targets):
        x = np.arange(len(conds)); w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            mu = [np.mean(tab.get((m, t, c), [np.nan])) for c in conds]
            sd = [np.std(tab.get((m, t, c), [np.nan])) for c in conds]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, mu, w, yerr=sd, label=m, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=40, ha="right"); ax.set_title(t)
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("16-bit message success at <= 1 % false alarm"); axes[0][0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(dst, "fig_success_fa1.png"), dpi=130)
    if train:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for s, h in train.items():
            ax[0].plot([r["step"] for r in h], [r["ber"] for r in h], label=f"seed {s}")
            ax[1].plot([r["step"] for r in h], [r["lsd"] for r in h])
        ax[0].set_ylabel("training BER (one window)"); ax[1].set_ylabel("LSD (dB)")
        for a in ax: a.set_xlabel("step"); a.grid(alpha=0.3)
        ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(dst, "fig_training.png"), dpi=130)
    lines = ["| method | target | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
    for t in targets:
        for m in methods:
            cells = []
            for c in conds:
                v = tab.get((m, t, c))
                cells.append("" if not v else f"{np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)})")
            lines.append(f"| {m} | {t} | " + " | ".join(cells) + " |")
    am = {}
    for r in aud:
        am.setdefault((r["method"], r["target"]), []).append(r)
    lines += ["", "| method | target | PESQ (speech) | SI-SNR dB | LSD dB | gain dB |", "|---|---|---|---|---|---|"]
    for (m, t), v in sorted(am.items()):
        lines.append(f"| {m} | {t} | {np.nanmean([x['pesq_speech'] for x in v]):.2f} | "
                     f"{np.mean([x['si_snr_db'] for x in v]):.1f} | {np.mean([x['lsd_db'] for x in v]):.2f} | "
                     f"{np.mean([x['gain_db'] for x in v]):.1f} |")
    open(os.path.join(dst, "table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
