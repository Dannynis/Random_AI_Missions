"""A3 driver: baselines, generator training (LoRA on an open LLM), in-context baseline, evaluation, report.

    python a3/main.py [--tiny] [--arms a,b,c]

Every arm writes $RUN_DIR/out/arm_<name>.json when finished; finished arms are skipped on restart and an
unfinished training arm resumes from $RUN_DIR/ckpt/<name>.pt. The report (summary.json, figures) is rebuilt
after every arm, so partial results are always available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import EXTRA_CC18, SUITE, SUITE18, all_episodes  # noqa: E402
from model import Generator, mlp_logits, ridge_head, ridge_loo_acc  # noqa: E402
from prior import sample_scm_task  # noqa: E402

RUN = os.environ.get("RUN_DIR", "./run_local")
if "--continue" in sys.argv:          # continue an earlier run (sibling run id) after a time-limit stop
    RUN = os.path.join(os.path.dirname(RUN.rstrip("/")), sys.argv[sys.argv.index("--continue") + 1])
OUT, CKPT = os.path.join(RUN, "out"), os.path.join(RUN, "ckpt")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS_TRAIN, KP_TRAIN = [1, 2, 5, 10, 20], [0.15, 0.2, 0.25, 0.2, 0.2]
LLMS = {"0.5b": "Qwen/Qwen2.5-0.5B", "1.5b": "Qwen/Qwen2.5-1.5B", "7b": "Qwen/Qwen2.5-7B"}
LABELS = "ABCDEFGHIJ"
STOP = {"flag": False}


def _on_signal(sig, frame):
    STOP["flag"] = True
    print(f"signal {sig}: will checkpoint and stop", flush=True)


signal.signal(signal.SIGUSR1, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_json(obj, name):
    p = os.path.join(OUT, name)
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(p + ".tmp", p)


# ======================================================================================= training data
class TaskStream(torch.utils.data.IterableDataset):
    def __init__(self, seed, B, q, salt, d_max=40, c_max=10):
        self.seed, self.B, self.q, self.salt, self.d_max, self.c_max = seed, B, q, salt, d_max, c_max

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, self.salt, wi.id if wi else 0])
        B, q = self.B, self.q
        while True:
            k = int(rng.choice(KS_TRAIN, p=KP_TRAIN))
            d, C = int(rng.integers(2, self.d_max + 1)), int(rng.integers(2, self.c_max + 1))
            ts = [sample_scm_task(rng, d=d, C=C) for _ in range(B)]
            xs = np.zeros((B, C, k, d), np.float32)
            xq = np.zeros((B, C * q, d), np.float32)
            yq = np.zeros((B, C * q), np.int64)
            for b, t in enumerate(ts):
                for c in range(C):
                    pool = np.flatnonzero(t.ytr == c)
                    xs[b, c] = t.Xtr[rng.choice(pool, k, replace=len(pool) < k)]
                    qi = np.flatnonzero(t.yte == c)
                    xq[b, c * q:(c + 1) * q] = t.Xte[rng.choice(qi, q, replace=len(qi) < q)]
                    yq[b, c * q:(c + 1) * q] = c
            yield torch.from_numpy(xs), torch.from_numpy(xq), torch.from_numpy(yq)


def train_arm(name, cfg, tiny):
    path = os.path.join(CKPT, name + ".pt")
    model = Generator(LLMS[cfg["llm"]], lora_r=cfg.get("lora_r", 16), random_init=cfg.get("random_init", False))
    model.to(DEV)
    llm_p, new_p, scaf_p = model.groups()
    lr_llm = 5e-5 if model.full_ft else 1e-4
    opt = torch.optim.AdamW([{"params": llm_p, "lr": lr_llm, "weight_decay": 0.0},
                             {"params": new_p, "lr": 5e-4, "weight_decay": 0.01},
                             {"params": scaf_p, "lr": 3e-3, "weight_decay": 0.0}])
    base_lr = [g["lr"] for g in opt.param_groups]
    steps, warm = cfg["steps"], min(200, max(1, cfg["steps"] // 10))
    step, hist, train_s = 0, [], 0.0
    if os.path.exists(path):
        st = torch.load(path, map_location="cpu", weights_only=False)
        missing = model.load_state_dict(st["model"], strict=False)
        assert not missing.unexpected_keys
        opt.load_state_dict(st["opt"])
        step, hist, train_s = st["step"], st["hist"], st.get("train_s", 0.0)
        log(f"{name}: resumed at step {step}")
    if step < steps:
        torch.manual_seed(cfg["seed"] * 1000 + step)
        dl = torch.utils.data.DataLoader(TaskStream(cfg["seed"], cfg.get("B", 8), 20, salt=step), batch_size=None,
                                         num_workers=0 if tiny else min(6, max(1, (os.cpu_count() or 4) - 2)))
        it = iter(dl)
        model.train()
        t0, last_ck, run_loss = time.time(), time.time(), []

        def ckpt():
            tmp = path + ".tmp"
            torch.save({"model": model.trainable_state(), "opt": opt.state_dict(), "step": step, "hist": hist,
                        "train_s": train_s + time.time() - t0}, tmp)
            os.replace(tmp, path)

        while step < steps:
            xs, xq, yq = (t.to(DEV, non_blocking=True) for t in next(it))
            fr = step / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g, b in zip(opt.param_groups, base_lr):
                g["lr"] = b * fr
            names = [f"f{j + 1}" for j in range(xs.shape[-1])]
            W, _ = model(xs, names)
            loss = F.cross_entropy(mlp_logits(W, xq).flatten(0, 1), yq.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            step += 1
            run_loss.append(loss.item())
            if step % 50 == 0 or step == steps:
                hist.append([step, float(np.mean(run_loss)), time.time() - t0])
                log(f"{name} step {step}/{steps} loss {np.mean(run_loss):.4f} "
                    f"lam {model.log_lam.exp().item():.3f} {(time.time() - t0) / max(1, len(run_loss)):.2f}s/it")
                run_loss = []
            if STOP["flag"]:
                ckpt()
                log("checkpointed on signal, exiting")
                sys.exit(75)
            if time.time() - last_ck > 600 or step == steps:
                ckpt()
                last_ck = time.time()
        train_s += time.time() - t0
    return model, hist, train_s


# ======================================================================================= evaluation
def acc(pred, y):
    return float((pred == y).mean())


@torch.no_grad()
def eval_generator(model, eps, names, anon=True):
    model.eval()
    lam = model.log_lam.exp()
    rows = []
    for e in eps:
        xs = torch.as_tensor(e["xs"], device=DEV)[None]
        xq = torch.as_tensor(e["xq"], device=DEV)[None]
        y = e["yq"]
        d = xs.shape[-1]
        r = dict(table=e["table"], split=e["split"], k=e["k"], draw=e["draw"])
        Ws, fs = model.assemble(xs)                                  # silenced = trained ridge scaffold
        Wg, fg = model(xs, names[e["table"]])
        ps = mlp_logits(Ws, xq).argmax(-1)[0].cpu().numpy()
        pg = mlp_logits(Wg, xq).argmax(-1)[0].cpu().numpy()
        r["scaffold"], r["gen"] = acc(ps, y), acc(pg, y)
        r["scaffold@20"], r["gen@20"] = acc(ps[:20], y[:20]), acc(pg[:20], y[:20])
        use_gen = e["k"] > 1 and ridge_loo_acc(fg, lam).item() > ridge_loo_acc(fs, lam).item()
        pgt = pg if use_gen else ps
        r["gated"], r["gated@20"], r["gate_on"] = acc(pgt, y), acc(pgt[:20], y[:20]), bool(use_gen)
        if anon:
            Wa, _ = model(xs, [f"f{j + 1}" for j in range(d)])
            pa = mlp_logits(Wa, xq).argmax(-1)[0].cpu().numpy()
            r["gen_anon"], r["gen_anon@20"] = acc(pa, y), acc(pa[:20], y[:20])
        rows.append(r)
    return rows


def tabpfn_model_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt")


def run_baselines(eps):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    try:
        from tabpfn import TabPFNClassifier
        tp = TabPFNClassifier(device=DEV, model_path=tabpfn_model_path())
        tp_err = None
    except Exception as ex:                                          # recorded, not fatal
        tp, tp_err = None, f"{type(ex).__name__}: {ex}"[:300]
    rows = []
    t0 = time.time()
    for i, e in enumerate(eps):
        C, k, d = e["xs"].shape
        X, y = e["xs"].reshape(C * k, d), np.repeat(np.arange(C), k)
        xq, yq = e["xq"], e["yq"]
        r = dict(table=e["table"], split=e["split"], k=k, draw=e["draw"])
        preds = {}
        try:
            preds["logreg"] = LogisticRegression(max_iter=2000).fit(X, y).predict(xq)
        except Exception:
            pass
        preds["knn"] = KNeighborsClassifier(min(5, len(y))).fit(X, y).predict(xq)
        cen = e["xs"].mean(1)
        preds["centroid"] = ((xq[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        xs_t = torch.as_tensor(e["xs"])[None].double()
        n_id = min(64, 64)
        W1 = torch.cat([torch.eye(64), -torch.eye(64)])[:, :d].double()
        f = F.relu(torch.einsum("bcki,hi->bckh", xs_t, W1))
        Wr, br = ridge_head(f, torch.tensor(0.1, dtype=torch.float64))
        fq = F.relu(torch.as_tensor(xq).double() @ W1.T)
        preds["ridge_untrained"] = (fq @ Wr[0].T + br[0]).argmax(-1).numpy()
        if tp is not None:
            try:
                preds["tabpfn"] = tp.fit(X, y).predict(xq)
            except Exception as ex:
                tp_err = f"{type(ex).__name__}: {ex}"[:300]
        for m, p in preds.items():
            r[m], r[m + "@20"] = acc(p, yq), acc(p[:20], yq[:20])
        r["majority"] = float(np.bincount(yq).max() / len(yq))
        rows.append(r)
        if i % 200 == 0:
            log(f"baselines {i}/{len(eps)} {time.time() - t0:.0f}s")
    return rows, tp_err


@torch.no_grad()
def run_icl(llm_key, eps, names, max_prefix=7000, nq=20):
    """In-context prompting of the same LLM: the k labelled rows as text, then each query row; the
    prediction is the label letter with the highest next-token logit. Draw 0 only, first nq queries."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    path = LLMS[llm_key]
    tok = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(DEV).eval()
    lab_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][-1] for c in LABELS]
    rows = []
    t0 = time.time()
    sub = [e for e in eps if e["draw"] == 0]
    for i, e in enumerate(sub):
        C, k, d = e["xs"].shape
        r = dict(table=e["table"], split=e["split"], k=k, draw=0)
        order = np.random.default_rng([e["split"], k, 99]).permutation(C * k)
        fmt = lambda v: ", ".join(f"{x:.2f}" for x in v)
        s = (f"Classify each row into one of {', '.join(LABELS[:C])}. Columns: "
             f"{', '.join(str(n)[:24] for n in names[e['table']])}.\n\n")
        for o in order:
            c, j = divmod(int(o), k)
            s += f"{fmt(e['xs'][c, j])} -> {LABELS[c]}\n"
        ids = tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] > max_prefix:
            r["icl_skipped"] = True
            rows.append(r)
            continue
        pkv = lm(ids, use_cache=True).past_key_values
        legacy = pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv
        qs = [tok(f"{fmt(x)} ->", add_special_tokens=False)["input_ids"] for x in e["xq"][:nq]]
        Q, L = len(qs), max(len(q) for q in qs)
        qid = torch.full((Q, L), tok.pad_token_id or 0, dtype=torch.long)
        qm = torch.zeros(Q, L, dtype=torch.long)
        for a, q in enumerate(qs):
            qid[a, :len(q)] = torch.tensor(q)
            qm[a, :len(q)] = 1
        cache = DynamicCache.from_legacy_cache(tuple((kk.expand(Q, -1, -1, -1).contiguous(),
                                                      vv.expand(Q, -1, -1, -1).contiguous()) for kk, vv in legacy))
        mask = torch.cat([torch.ones(Q, ids.shape[1], dtype=torch.long), qm], 1).to(DEV)
        out = lm(input_ids=qid.to(DEV), attention_mask=mask, past_key_values=cache, use_cache=True)
        last = qm.sum(1) - 1
        lg = out.logits[torch.arange(Q), last.to(DEV)][:, lab_ids[:C]].float()
        p = lg.argmax(-1).cpu().numpy()
        r["icl@20"] = acc(p, e["yq"][:Q])
        r["icl_prefix_tokens"] = int(ids.shape[1])
        rows.append(r)
        del cache, out, pkv
        if i % 50 == 0:
            log(f"icl {i}/{len(sub)} {time.time() - t0:.0f}s prefix {ids.shape[1]}")
    del lm
    torch.cuda.empty_cache()
    return rows


# ======================================================================================= report
def table_means(rows, key, tables):
    m = {}
    for t in tables:
        v = [r[key] for r in rows if r["table"] == t and key in r]
        if v:
            m[t] = float(np.mean(v))
    return m


def boot_diff(a, b, n=5000, seed=0):
    """Paired bootstrap over tables of mean(a - b). a, b: dict table -> score."""
    ts = sorted(set(a) & set(b))
    if len(ts) < 2:
        return None
    d = np.array([a[t] - b[t] for t in ts])
    r = np.random.default_rng(seed)
    bs = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return dict(mean=float(d.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                n_tables=len(ts), wins=int((d > 0).sum()))


def report():
    arms = {f[4:-5]: json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT))
            if f.startswith("arm_") and f.endswith(".json")}
    rows_by = {}                                        # method label -> list of row dicts (all seeds pooled)
    seeds = {}                                          # method label -> list of per-seed row lists
    for name, a in arms.items():
        if name == "baselines":
            for m in ("logreg", "knn", "centroid", "tabpfn", "ridge_untrained", "majority"):
                rows_by[m] = a["rows"]
                for suf in ("", "@20"):
                    pass
        elif name.startswith("icl"):
            rows_by[name] = a["rows"]
        elif name.startswith("gen"):
            fam = name.rsplit("-s", 1)[0]
            seeds.setdefault(fam, []).append(a["rows"])
    tables = [t for t in SUITE]
    summ = {"methods": {}, "comparisons": {}}

    def score(rows_list, key, tabs):
        per_seed = [table_means(r, key, tabs) for r in rows_list]
        per_seed = [p for p in per_seed if p]
        if not per_seed:
            return None, None
        common = set.intersection(*[set(p) for p in per_seed])
        mean = {t: float(np.mean([p[t] for p in per_seed])) for t in common}
        overall = [float(np.mean(list(p.values()))) for p in per_seed]
        return mean, overall

    entries = {}
    for m in ("logreg", "knn", "centroid", "tabpfn", "ridge_untrained", "majority"):
        if m in rows_by:
            entries[m] = ([rows_by[m]], m)
    for fam, rl in seeds.items():
        for key in ("gen", "gated", "scaffold", "gen_anon"):
            entries[f"{fam}:{key}"] = (rl, key)
    for grp, tabs in (("all", list(SUITE)), ("suite18", list(SUITE18)), ("extra", list(EXTRA_CC18))):
        for lab, (rl, key) in entries.items():
            mean, overall = score(rl, key, tabs)
            if mean:
                summ["methods"].setdefault(lab, {})[grp] = dict(
                    mean=float(np.mean(list(mean.values()))), seed_std=float(np.std(overall)) if len(overall) > 1 else 0.0,
                    n_seeds=len(rl), n_tables=len(mean), per_table=mean)
        per_k = {}
    for lab, (rl, key) in entries.items():
        pk = {}
        for k in (1, 2, 5, 10, 20):
            v = [np.mean([r[key] for r in rows if r["k"] == k and r["table"] == t and key in r])
                 for rows in rl for t in tables if any(r["k"] == k and r["table"] == t and key in r for r in rows)]
            if v:
                pk[k] = float(np.mean(v))
        summ["methods"].setdefault(lab, {})["per_k"] = pk
    # comparisons (pre-registered: 7B generator vs its scaffold and vs in-context 7B, all 23 tables)
    for fam, rl in seeds.items():
        g, _ = score(rl, "gen", tables)
        gt, _ = score(rl, "gated", tables)
        s, _ = score(rl, "scaffold", tables)
        for lab, x in (("gen", g), ("gated", gt)):
            if x and s:
                summ["comparisons"][f"{fam}:{lab} - scaffold"] = boot_diff(x, s)
            for b in ("logreg", "tabpfn"):
                if x and b in rows_by:
                    bm = table_means(rows_by[b], b, tables)
                    summ["comparisons"][f"{fam}:{lab} - {b}"] = boot_diff(x, bm)
        for icl in [n for n in rows_by if n.startswith("icl")]:
            ic = table_means(rows_by[icl], "icl@20", tables)
            g20, _ = score([[r for r in rows if r["draw"] == 0] for rows in rl], "gen@20", tables)
            if g20 and ic:
                summ["comparisons"][f"{fam}:gen@20 - {icl}"] = boot_diff(g20, ic)
                summ["methods"].setdefault(icl, {})["all"] = dict(mean=float(np.mean(list(ic.values()))),
                                                                  n_tables=len(ic), per_table=ic)
                summ["methods"].setdefault(f"{fam}:gen@20(draw0)", {})["all"] = dict(
                    mean=float(np.mean([g20[t] for t in ic if t in g20])), n_tables=len(ic))
    main = "gen-7b"
    c1, c2 = summ["comparisons"].get(f"{main}:gen - scaffold"), summ["comparisons"].get(f"{main}:gen@20 - icl-7b")
    if c1 and c2 and len(seeds.get(main, [])) >= 3:
        summ["verdict"] = "PASS" if (c1["lo"] > 0 and c2["lo"] > 0) else "FAIL"
    else:
        summ["verdict"] = "incomplete"
    summ["arms_done"] = sorted(arms)
    summ["train"] = {n: dict(steps=a.get("steps"), train_s=a.get("train_s"), hist=a.get("hist")[-1:] if a.get("hist") else None)
                     for n, a in arms.items() if n.startswith("gen")}
    save_json(summ, "summary.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lab, v in summ["methods"].items():
            pk = v.get("per_k")
            if pk and (":" not in lab or lab.endswith(":gen") or lab.endswith(":scaffold")):
                ks = sorted(pk, key=int)
                ax.plot([int(k) for k in ks], [pk[k] for k in ks], marker="o", label=lab)
        ax.set_xscale("log"); ax.set_xlabel("labelled rows per class (k)"); ax.set_ylabel("query accuracy (mean over tables)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_accuracy_vs_k.png"), dpi=120)
        plt.close(fig)
    except Exception as ex:
        log("figure failed", ex)
    return summ


# ======================================================================================= main
def arm_plan(tiny):
    if tiny:
        return [("baselines", {}), ("gen-0.5b-s0", dict(llm="0.5b", seed=0, steps=30, B=4)),
                ("gen-0.5brand-s0", dict(llm="0.5b", seed=0, steps=10, B=4, random_init=True)),
                ("icl-0.5b", dict(llm="0.5b"))]
    steps = 6000
    plan = [("baselines", {})]
    plan += [(f"gen-0.5b-s{s}", dict(llm="0.5b", seed=s, steps=steps)) for s in range(3)]
    plan += [("gen-7b-s0", dict(llm="7b", seed=0, steps=steps)), ("icl-7b", dict(llm="7b"))]
    plan += [(f"gen-7b-s{s}", dict(llm="7b", seed=s, steps=steps)) for s in (1, 2)]
    plan += [(f"gen-1.5b-s{s}", dict(llm="1.5b", seed=s, steps=steps)) for s in range(3)]
    plan += [("gen-0.5brand-s0", dict(llm="0.5b", seed=0, steps=steps, random_init=True))]
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--arms", default="")
    ap.add_argument("--continue", dest="cont", default="", help="run id of an earlier run to continue in place")
    ap.add_argument("--resume-test", action="store_true", help="train 3 steps, then resume to 5 (validation)")
    a = ap.parse_args()
    if a.resume_test:
        cfg = dict(llm="0.5b", seed=0, steps=3, B=2)
        train_arm("rt", cfg, True)
        cfg["steps"] = 5
        _, hist, _ = train_arm("rt", cfg, True)
        log("resume test hist", hist)
        save_json(dict(hist=hist, ok=bool(hist and hist[-1][0] == 5)), "resume_test.json")
        return
    tables = ["iris", "banknote", "cmc"] if a.tiny else None
    eps, names = all_episodes(tables, draws=1 if a.tiny else 4)
    log(f"{len(eps)} episodes over {len(names)} tables on {DEV}")
    plan = arm_plan(a.tiny)
    if a.arms:
        want = a.arms.split(",")
        plan = [p for p in plan if p[0] in want]
    for name, cfg in plan:
        if os.path.exists(os.path.join(OUT, f"arm_{name}.json")):
            log(f"{name}: done, skipping")
            continue
        t0 = time.time()
        log(f"=== arm {name} {cfg}")
        if name == "baselines":
            rows, err = run_baselines(eps)
            res = dict(rows=rows, tabpfn_error=err)
        elif name.startswith("icl"):
            res = dict(rows=run_icl(cfg["llm"], eps, names))
        else:
            model, hist, train_s = train_arm(name, cfg, a.tiny)
            res = dict(rows=eval_generator(model, eps, names, anon=not cfg.get("random_init", False)),
                       hist=hist, train_s=train_s, steps=cfg["steps"], cfg=cfg)
            del model
            torch.cuda.empty_cache()
        res["wall_s"] = time.time() - t0
        save_json(res, f"arm_{name}.json")
        s = report()
        log(f"{name} finished in {res['wall_s']:.0f}s; verdict so far: {s['verdict']}")
        for lab, v in s["methods"].items():
            if "all" in v:
                log(f"   {lab:28s} {v['all']['mean']:.4f}")
        ck = os.path.join(CKPT, name + ".pt")
        if os.path.exists(ck):
            os.remove(ck)                               # disk budget: drop checkpoints of finished arms
    report()
    log("all arms done")


if __name__ == "__main__":
    main()
