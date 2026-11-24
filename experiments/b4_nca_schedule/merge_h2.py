"""Merge B4 re-seed-schedule runs across seed batches and settle H2.

H2: for the persistent self-classifying NCA, the benefit of a rising re-seed
schedule comes from *when* fresh digits arrive, not how many. Test: does
`anneal_rev` (2->8) beat `const5` (equal fresh-digit count) on anytime accuracy
in a paired per-seed test? Pre-stated criterion (scaled from 8/10 to the pooled
seed count): win fraction >= 0.8 of paired seeds.

Also reports the ramp end-points `anneal_rev12` (2->12) and `anneal_rev1` (1->8)
against const5 and pool_det (20 seeds, 10-29).

Usage: python merge_h2.py <results_dir> [out.json]
  results_dir holds <jobid>/results.json folders.
"""
import json, sys, glob, os
import numpy as np


def load_runs(results_dir):
    """Return {arm: {seed: metrics_dict}} pooled over all job folders."""
    runs = {}
    for rj in sorted(glob.glob(os.path.join(results_dir, "*", "results.json"))):
        d = json.load(open(rj))
        for arm, a in d.get("arms", {}).items():
            for ps in a.get("per_seed", []):
                seed = ps["seed"]
                runs.setdefault(arm, {})[seed] = ps
    return runs


def metric_val(ps, metric):
    """Scalar metric for one per-seed record. acc@<step> read from step_curve."""
    if metric.startswith("acc@"):
        step = metric.split("@", 1)[1]
        return ps["step_curve"][step][0]
    return ps[metric]


def paired(runs, arm_a, arm_b, metric="anytime"):
    """Paired diff arm_a - arm_b over seeds present in both."""
    sa, sb = runs.get(arm_a, {}), runs.get(arm_b, {})
    seeds = sorted(set(sa) & set(sb))
    diffs = np.array([metric_val(sa[s], metric) - metric_val(sb[s], metric) for s in seeds])
    return seeds, diffs


def boot_ci(diffs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(runs, arm_a, arm_b, metric="anytime"):
    seeds, diffs = paired(runs, arm_a, arm_b, metric)
    n = len(diffs)
    n_pos = int((diffs > 0).sum())
    lo, hi = boot_ci(diffs) if n > 1 else (float("nan"), float("nan"))
    return {
        "contrast": f"{arm_a} - {arm_b}",
        "metric": metric,
        "n_seeds": n,
        "mean": float(diffs.mean()),
        "se": float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "ci95": [lo, hi],
        "n_pos": n_pos,
        "win_frac": n_pos / n if n else float("nan"),
        "seeds": seeds,
    }


def arm_means(runs, arm, metrics=("anytime", "peak", "acc@40", "acc@2000")):
    d = runs.get(arm, {})
    out = {"n_seeds": len(d)}
    for m in metrics:
        vals = np.array([metric_val(d[s], m) for s in d])
        out[m] = [float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0]
    return out


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/b4_nca_schedule"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    runs = load_runs(results_dir)

    arms = sorted(runs)
    summary = {"arms_present": {a: sorted(runs[a]) for a in arms},
               "arm_means": {a: arm_means(runs, a) for a in arms},
               "contrasts": {}}

    # H2 primary: anneal_rev vs const5 (equal fresh data), and vs pool_det baseline.
    for a, b in [("anneal_rev", "const5"), ("anneal_rev", "pool_det"),
                 ("anneal_rev12", "const5"), ("anneal_rev1", "const5"),
                 ("anneal_rev12", "pool_det"), ("anneal_rev1", "pool_det"),
                 ("const5", "pool_det")]:
        if a in runs and b in runs:
            for m in ("anytime", "peak", "acc@2000"):
                r = report(runs, a, b, m)
                summary["contrasts"][f"{a}-{b}:{m}"] = r

    # H2 verdict on the primary metric.
    key = "anneal_rev-const5:anytime"
    if key in summary["contrasts"]:
        r = summary["contrasts"][key]
        passed = r["win_frac"] >= 0.8 and r["ci95"][0] > 0
        summary["H2_verdict"] = {
            "criterion": "anneal_rev beats const5 on anytime in >=80% paired seeds and 95% CI above 0",
            "win_frac": r["win_frac"], "n_pos": r["n_pos"], "n_seeds": r["n_seeds"],
            "mean": r["mean"], "ci95": r["ci95"],
            "verdict": "PASS" if passed else "FAIL",
        }

    txt = json.dumps(summary, indent=1)
    print(txt[:4000])
    if out_path:
        json.dump(summary, open(out_path, "w"), indent=1)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
