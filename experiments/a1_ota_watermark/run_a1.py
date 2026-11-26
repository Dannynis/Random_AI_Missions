"""A1 driver: train the scaled watermark (checkpoint / resume) and run the matched-audibility
benchmark against AudioSeal and WavMark.

usage:
  python run_a1.py train --seed 0 --steps 40000
  python run_a1.py eval  --seed 0            (needs the trained seed; evaluates ours + baselines)
  python run_a1.py summary                   (aggregates every eval_seed*.json in $RUN_DIR/out)
Paths: $RUN_DIR/ckpt (checkpoints), $RUN_DIR/out (metrics), audio from $A1_DATA / $HF_HOME.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "legacy"))
import wmx  # noqa: E402
import wm_coding as wc  # noqa: E402
import wm_crc as wcr  # noqa: E402
import wm_sync as ws  # noqa: E402

RUN = os.environ.get("RUN_DIR", os.path.join(HERE, "_run"))
CK, OUT = os.path.join(RUN, "ckpt"), os.path.join(RUN, "out")
os.makedirs(CK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def save_atomic(obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)


def dump(obj, path):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)


# ======================================================================= training
def host_batch(corp, rng, B, length):
    nsp = B // 2
    x = np.concatenate([corp[0].segments(rng, nsp, length), corp[1].segments(rng, B - nsp, length)])
    x = torch.from_numpy(x).to(DEV)
    x = wmx.normalize(x, torch.from_numpy(rng.uniform(-6, 6, B)).float().to(DEV))
    return x * torch.from_numpy(rng.choice([-1.0, 1.0], (B, 1))).float().to(DEV)


def window_mean(feats, s):
    """feats (B, T, D), s (B,) start frames -> mean over [s, s + FPC) clipped."""
    B, T, D = feats.shape
    cs = torch.cat([feats.new_zeros(B, 1, D), feats.cumsum(1)], 1)
    s0 = s.clamp(0, T); e = (s + wmx.FPC).clamp(0, T)
    g = lambda i: cs.gather(1, i[:, None, None].expand(-1, 1, D))[:, 0]
    return (g(e) - g(s0)) / (e - s0).clamp(min=1)[:, None].float()


def train(a):
    path = os.path.join(CK, f"seed{a.seed}.pt")
    final = os.path.join(CK, f"seed{a.seed}_final.pt")
    if os.path.exists(final):
        say("seed", a.seed, "already trained"); return
    torch.manual_seed(a.seed)
    enc, dec = wmx.Encoder(ch=a.ch, nblk=a.nblk).to(DEV), wmx.Decoder().to(DEV)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    step, hist = 0, []
    if os.path.exists(path):
        st = torch.load(path, map_location=DEV, weights_only=False)
        enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); step, hist = st["step"], st["hist"]
        say("resumed seed", a.seed, "at step", step)
    elif a.init:                                  # fine-tune: start from a trained model (fresh optimiser and schedule)
        st = torch.load(os.path.join(a.init, f"seed{a.seed}_final.pt"), map_location=DEV, weights_only=False)
        enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
        say("initialised seed", a.seed, "from", os.path.basename(os.path.dirname(a.init.rstrip("/"))) or a.init)
    say("params enc", sum(p.numel() for p in enc.parameters()), "dec", sum(p.numel() for p in dec.parameters()))
    corp = (wmx.Corpus("speech_train"), wmx.Corpus("music_train"))
    chan = wmx.Channel(DEV, "train")
    S = 3
    # the codec stage only codes the decoded middle chunk plus a margin wider than the decoder's receptive field
    span = (wmx.N - 6000, 2 * wmx.N + wmx.MAX_OFFSET + 6000)
    run = {"loss": 0.0, "ber": 0.0, "lsd": 0.0, "n": 0}
    t1 = time.time(); t_data = 0.0
    while step < a.steps:
        step += 1
        rng = np.random.default_rng([a.seed, step])
        g = torch.Generator(device=DEV).manual_seed(a.seed * 10 ** 7 + step)
        td = time.time()
        host = host_batch(corp, rng, a.batch, S * wmx.N)
        torch.cuda.synchronize() if DEV.type == "cuda" else None
        t_data += time.time() - td
        bits = torch.randint(0, 2, (a.batch, S, wmx.K), generator=g, device=DEV)
        frac = step / a.steps
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            w = wmx.embed_chunks(enc, host, bits, a.whr).float()
            y, off = chan.train(host + w, g, strength=min(1.0, frac / 0.3), mode=a.train_channel,
                                opus_p=a.opus_p, opus_kbps=a.opus_kbps, codec_span=span)
            feats = dec.frames(y.float())
            jit = torch.randint(-3, 4, (a.batch,), generator=g, device=DEV)
            s = torch.round((wmx.N + off) / wmx.HOP).long() + jit
            logits = dec.head(window_mean(feats.float(), s)).float()
        loss = F.binary_cross_entropy_with_logits(logits, bits[:, 1].float())
        d = wmx.lsd(host, w).mean()
        a_ = min(1.0, max(0.0, (frac - 0.25) / 0.5))
        tgt = a.lsd_start + a_ * (a.lsd_target - a.lsd_start)
        tot = loss + float(frac >= 0.25) * a.lsd_weight * F.relu(d - tgt)
        opt.zero_grad(set_to_none=True); tot.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run["loss"] += loss.item(); run["lsd"] += d.item(); run["n"] += 1
        run["ber"] += ((logits > 0).long() != bits[:, 1]).float().mean().item()
        if step % 100 == 0 or step == a.steps:
            n = run["n"]
            hist.append({"step": step, "loss": run["loss"] / n, "ber": run["ber"] / n, "lsd": run["lsd"] / n,
                         "lsd_target": tgt, "s_per_step": (time.time() - t1) / n, "data_s": t_data / n})
            say(f"seed {a.seed} step {step}: " + ", ".join(f"{k}={v:.4g}" for k, v in hist[-1].items() if k != "step"))
            run = {k: 0.0 for k in run}; run["n"] = 0; t1 = time.time(); t_data = 0.0
        if step % a.ckpt_every == 0 or step == a.steps:
            save_atomic({"enc": enc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "step": step, "hist": hist, "args": vars(a)}, path)
    save_atomic({"enc": enc.state_dict(), "dec": dec.state_dict(), "hist": hist, "args": vars(a)}, final)
    dump(hist, os.path.join(OUT, f"train_seed{a.seed}.json"))
    os.remove(path)


def load_ours(seed):
    # A1_CKPT (optional, ':'-separated) lets an evaluation-only run read models trained by earlier runs
    dirs = [d for d in os.environ.get("A1_CKPT", "").split(":") if d] + [CK]
    path = next((os.path.join(d, f"seed{seed}_final.pt") for d in dirs if os.path.exists(os.path.join(d, f"seed{seed}_final.pt"))), None)
    if path is None:
        raise FileNotFoundError(f"no trained model for seed {seed}")
    st = torch.load(path, map_location=DEV, weights_only=False)
    ar = st["args"]
    enc, dec = wmx.Encoder(ch=ar["ch"], nblk=ar["nblk"]).to(DEV).eval(), wmx.Decoder().to(DEV).eval()
    enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
    return enc, dec, ar


# ======================================================================= audibility
def _pesq_one(args):
    ref, deg = args
    from pesq import pesq
    try:
        return float(pesq(wmx.SR, ref, deg, "wb"))
    except Exception:
        return float("nan")


def pesq_batch(ref, deg, pool=None):
    # sequential on purpose: process pools need shared memory that batch jobs may not have
    return np.array([_pesq_one(a) for a in zip(ref.astype(np.float64), deg.astype(np.float64))])


def si_snr(ref, deg):
    ref = ref - ref.mean(-1, keepdims=True); deg = deg - deg.mean(-1, keepdims=True)
    a = (deg * ref).sum(-1, keepdims=True) / ((ref ** 2).sum(-1, keepdims=True) + 1e-12)
    t = a * ref; e = deg - t
    return 10 * np.log10((t ** 2).sum(-1) / ((e ** 2).sum(-1) + 1e-12) + 1e-12)


# ======================================================================= methods
def info_for(info, k):
    """Message bits for a k-bit payload: the 16 standard info bits, extended by bits drawn from a generator seeded
    by them (so every method and every run sees the same longer messages for the same streams)."""
    if k <= info.shape[1]:
        return info[:, :k]
    import zlib
    ext = np.random.default_rng(zlib.crc32(np.packbits(info).tobytes())).integers(0, 2, (len(info), k - info.shape[1]))
    return np.concatenate([info, ext], 1)


def make_method(m, seed):
    """Method object from its name: ours, ours_k<bits> (same model, longer message), audioseal, wavmark, dear."""
    if m == "ours" or m.startswith("ours_k"):
        return Ours(seed, int(m[6:]) if m != "ours" else 16)
    if m == "audioseal":
        return AudioSealM()
    if m == "wavmark":
        return WavMarkM()
    if m == "dear":
        from dear import DeARM
        return DeARM(seed)
    raise ValueError(m)


# puncturing patterns of the rate-1/2 K=7 code over its interleaved output pairs: rate 1/2, 2/3, 3/4, 5/6
PUNCTURE = ((1, 1), (1, 1, 1, 0), (1, 1, 1, 0, 0, 1), (1, 1, 1, 0, 0, 1, 1, 0, 0, 1))


def make_layout(k_info, seed, r=8):
    """CRC layout for k_info info bits: the lowest-rate puncturing of the convolutional code whose kept bits fit the
    128 slots of a cycle (kept bits are then repeated cyclically, as without puncturing). k_info <= 50 is the unpunctured
    code of the original layout."""
    lay = wcr.CRCLayout(k_info=min(k_info, 16), r=r, seed=seed)
    if k_info == 16:
        return lay
    k_msg = k_info + r
    for pat in PUNCTURE:
        n_code = 2 * (k_msg + 6)
        kept = np.where(np.resize(np.array(pat, bool), n_code))[0]
        if len(kept) <= lay.n_data:
            break
    else:
        raise ValueError(f"{k_info} info bits do not fit {lay.n_data} slots")
    code = wc.ConvCode(k_msg, n_code)
    code.n, code.idx = lay.n_data, kept[np.arange(lay.n_data) % len(kept)]
    lay.k_info, lay.k_msg, lay.code = k_info, k_msg, code
    return lay


class Ours:
    name = "ours"

    def __init__(self, seed, k_info=16):
        self.enc, self.dec, self.ar = load_ours(seed)
        self.seed = seed
        self.k_info = k_info
        if k_info != 16:
            self.name = f"ours_k{k_info}"
        # k_info info bits + CRC-8, convolutional code in the same 128 slots of the 8 s cycle (punctured when needed)
        self.lay = make_layout(k_info, seed)
        self.cal = None

    def calibrate(self, corp, chan):
        """Per-slot LLR calibration on training audio through the training channel (true timing)."""
        zs, bs = [], []
        for i in range(16 if not TINY else 2):
            rng = np.random.default_rng([777, i]); g = torch.Generator(device=DEV).manual_seed(777 + i)
            host = host_batch(corp, rng, 32, 3 * wmx.N)
            bits = torch.randint(0, 2, (32, 3, wmx.K), generator=g, device=DEV)
            with torch.no_grad():
                w = wmx.embed_chunks(self.enc, host, bits, self.ar["whr"])
                y, off = chan.train(host + w, g, 1.0)
                z = self.dec.head(window_mean(self.dec.frames(y), torch.round((wmx.N + off) / wmx.HOP).long()))
            zs.append(z.cpu().numpy()); bs.append(bits[:, 1].cpu().numpy())
        self.cal = wc.fit_llr(np.concatenate(zs), np.concatenate(bs), "ours")
        return self.cal.nll

    def residual(self, host, g0, info, phi0):
        """host (B, L) cpu; marks S full chunks from g0. Returns residual (B, L) at native WHR."""
        B, L = host.shape
        S = (L - wmx.N) // wmx.N
        cyc = self.lay.chunk_bits(self.lay.frame(info_for(info, self.k_info)))   # (B, 4, K)
        sb = cyc[np.arange(B)[:, None], (np.arange(S)[None] + phi0[:, None]) % 4]
        r = torch.zeros_like(host)
        with torch.no_grad():
            for b0 in range(0, B, 16):
                hb = torch.stack([host[b, g0[b]: g0[b] + S * wmx.N] for b in range(b0, min(B, b0 + 16))]).to(DEV)
                w = wmx.embed_chunks(self.enc, hb, torch.from_numpy(sb[b0:b0 + 16]).to(DEV), self.ar["whr"]).cpu()
                for i, b in enumerate(range(b0, min(B, b0 + 16))):
                    r[b, g0[b]: g0[b] + S * wmx.N] = w[i]
        return r

    class _Head:
        def __init__(self, dec):
            self.dec = dec

        def head(self, x):
            with torch.no_grad():
                return self.dec.head(x.to(DEV)).cpu()

    def decode(self, y, truth):
        with torch.no_grad():
            feats = torch.cat([self.dec.frames(y[i:i + 16].to(DEV)).float().cpu() for i in range(0, len(y), 16)])
        hd = self._Head(self.dec)
        c = wcr.candidates("joint", self.lay, hd, self.cal, feats, truth, partial=True, n_list=4)
        res = {}
        for rule, lst in (("top", 1), ("crc@1", 1), ("crc@4", 4)):
            info, out, nm = wcr.decide(c, self.lay, rule, lst)
            res[rule] = (info, out, nm)
        return res

    def slot_stats(self, y, truth, info):
        """Physical-layer information rate at the true timing, over the full chunks of each window: slot bit error,
        and the soft-information rate (bits/s) with the training-channel LLR calibration and with a calibration refit on
        this condition (two folds over recordings); the binary-symmetric-channel rate 1 - H(slot error) for reference."""
        with torch.no_grad():
            feats = torch.cat([self.dec.frames(y[i:i + 16].to(DEV)).float().cpu() for i in range(0, len(y), 16)])
        B, T, _ = feats.shape
        f0 = np.round(np.asarray(truth.f0)).astype(int)
        J = int(T // wmx.FPC) + 1
        z, frac = ws.window_logits(self._Head(self.dec), feats, f0[:, None] + np.arange(J)[None] * wmx.FPC)
        bits = self.lay.chunk_bits(self.lay.frame(info_for(info, self.k_info)))
        bits = bits[np.arange(B)[:, None], (np.asarray(truth.phase)[:, None] + np.arange(J)[None]) % 4]
        full = frac >= 0.999
        zf, bf = z[full], bits[full]
        rec = np.repeat(np.arange(B)[:, None], J, 1)[full]
        sps = wmx.K * wmx.SR / wmx.N                                    # slots per second (16)
        cap_cal = wc.capacity_bits(wc.evidence(self.cal(zf), bf)) * sps
        ev = np.zeros_like(zf)
        for fold in (0, 1):                                             # refit on half the recordings, score the other
            tr, te = rec % 2 != fold, rec % 2 == fold
            ev[te] = wc.evidence(wc.fit_llr(zf[tr], bf[tr])(zf[te]), bf[te])
        p = float(((self.cal(zf) > 0) != bf).mean())
        h = 0.0 if p in (0.0, 1.0) else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return {"slot_ber": p, "n_chunks": int(full.sum()), "cap_cal_bps": cap_cal,
                "cap_refit_bps": wc.capacity_bits(ev) * sps, "cap_bsc_bps": (1 - h) * sps,
                "payload_bps": self.k_info / (4 * wmx.N / wmx.SR)}


class AudioSealM:
    name = "audioseal"

    def __init__(self):
        from audioseal import AudioSeal
        self.gen = AudioSeal.load_generator("audioseal_wm_16bits").to(DEV).eval()
        self.det = AudioSeal.load_detector("audioseal_detector_16bits").to(DEV).eval()

    def residual(self, host, g0, info, phi0):
        out = []
        with torch.no_grad():
            for i in range(0, len(host), 8):
                h = host[i:i + 8].to(DEV)[:, None]
                m = torch.from_numpy(info[i:i + 8]).to(DEV)
                out.append(self.gen.get_watermark(h, wmx.SR, message=m)[:, 0].cpu())
        return torch.cat(out)

    PERIOD = 320          # the public 16-bit decoder only reads the message on this sample grid

    def _raw(self, x):
        if self.det.normalizer is not None:
            x = self.det.normalizer.loudness_normalization(x)
        return self.det.detector(x[:, None]).float()

    def _shift_conf(self, yb, shifts, probe):
        crops = torch.stack([yb[:, s:s + probe] for s in shifts], 1)          # (b, S, probe)
        b, S, _ = crops.shape
        conf = []
        for j in range(0, b * S, 16):
            raw = self._raw(crops.reshape(b * S, probe)[j:j + 16])
            conf.append(raw[:, 2:].mean(-1).abs().mean(-1))
        return torch.cat(conf).reshape(b, S)

    def decode(self, y, truth):
        """Sync search (added for fairness): pick the grid shift with the most confident message
        logits on a 2 s probe (4-sample steps, then 1-sample refinement), then decode the whole
        recording from that shift. Detection score = mean frame watermark probability."""
        sc, ms = [], []
        n = y.shape[-1] - self.PERIOD
        probe = n            # whole recording (a 2 s probe picked the wrong grid shift too often)
        with torch.no_grad():
            for i in range(0, len(y), 8):
                yb = y[i:i + 8].to(DEV)
                coarse = list(range(0, self.PERIOD, 4))
                c = self._shift_conf(yb, coarse, probe)
                s0 = torch.tensor(coarse, device=DEV)[c.argmax(1)]
                fine = list(range(-2, 3))
                cand = torch.stack([(s0 + d) % self.PERIOD for d in fine], 1)       # (b, 7)
                cf = torch.stack([self._shift_conf(yb[k:k + 1], cand[k].tolist(), probe)[0] for k in range(len(yb))])
                best = cand[torch.arange(len(yb)), cf.argmax(1)]
                crop = torch.stack([yb[k, best[k]:best[k] + n] for k in range(len(yb))])
                raw = self._raw(crop)
                det = torch.softmax(raw[:, :2], 1)[:, 1].mean(-1)
                sc.append(det.cpu()); ms.append((raw[:, 2:].mean(-1) > 0).long().cpu())
        sc, ms = torch.cat(sc).numpy(), torch.cat(ms).numpy()
        out = np.ones(len(sc), bool)
        return {"top": (ms, out, sc)}


class WavMarkM:
    name = "wavmark"

    def __init__(self):
        import wavmark
        self.wm = wavmark
        self.model = wavmark.load_model().to(DEV).eval()

    def residual(self, host, g0, info, phi0):
        out = []
        for i in range(len(host)):
            x = host[i].numpy().astype(np.float32)
            y, _ = self.wm.encode_watermark(self.model, x, info[i].astype(np.int64), show_progress=False)
            out.append(torch.from_numpy(np.asarray(y, np.float32) - x))
        return torch.stack(out)

    def decode(self, y, truth):
        B = len(y)
        ms = np.zeros((B, 16), np.int64); out = np.zeros(B, bool); sc = np.zeros(B)
        for i in range(B):
            p, inf = self.wm.decode_watermark(self.model, y[i].numpy().astype(np.float32), decode_batch_size=70,
                                              show_progress=False)
            n = len(inf["results"])
            sc[i] = n
            if p is not None:
                ms[i] = p; out[i] = True
        return {"top": (ms, out, sc)}


# ======================================================================= benchmark
TINY = False
S_CHUNKS = 6
REC_S = 8


def make_streams(seed, n_rec):
    """Host streams of (S_CHUNKS + 1) chunks, half speech / half music, test split."""
    rng = np.random.default_rng(seed + 555)
    L = (S_CHUNKS + 1) * wmx.N
    ns = n_rec // 2
    x = np.concatenate([wmx.Corpus("speech_test").segments(rng, ns, L, 3e-3),
                        wmx.Corpus("music_test").segments(rng, n_rec - ns, L, 3e-3)])
    host = wmx.normalize(torch.from_numpy(x), torch.from_numpy(rng.uniform(-6, 6, n_rec)).float())
    src = np.array([0] * ns + [1] * (n_rec - ns))
    g0 = rng.integers(0, wmx.N // 2, n_rec)
    phi0 = rng.integers(0, 4, n_rec)
    n = REC_S * wmx.SR
    room = S_CHUNKS * wmx.N - n - 2 * wmx.MAX_OFFSET - 3000
    t0 = g0 + wmx.MAX_OFFSET + 2500 + rng.integers(0, room, n_rec)
    info = rng.integers(0, 2, (n_rec, 16))
    return host, src, g0, phi0, t0, info


def truth_for(g0, phi0, t0, off, speed):
    j = np.arange(S_CHUNKS)[None]
    pos = (g0[:, None] + j * wmx.N) / speed[:, None] + off[:, None] - t0[:, None]
    pos = np.where(pos >= -wmx.HOP / 2, pos, np.inf)
    jj = pos.argmin(1)
    return ws.Truth(f0=pos[np.arange(len(g0)), jj] / wmx.HOP, phase=(jj + phi0) % 4)


def two_fold(correct, out_p, sc_p, out_n, sc_n, fa):
    # interleaved halves: the streams are ordered speech first, music second, so a first/second split
    # would calibrate the threshold on one source type and apply it to the other
    B = len(correct); half = np.arange(B) % 2 == 0
    pts = []
    for cal, ev in ((half, ~half), (~half, half)):
        thr = wcr.threshold_for(sc_n[cal], out_n[cal], fa) if fa < 1 else -np.inf
        pts.append(wcr.operating_point(correct[ev], out_p[ev], sc_p[ev], out_n[ev], sc_n[ev], thr))
    return {k: float(np.mean([getattr(p, k) for p in pts])) for k in ("success", "wrong", "none", "false_alarm")}


def auc(pos, neg):
    allv = np.concatenate([pos, neg]); allv = np.where(np.isfinite(allv), allv, -1e9)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    r = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort").astype(float) + 1
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r); r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibrate_alpha(r, host, g0, target, pool, nsp):
    """Gain (dB) on residual r so that mean wideband PESQ over the first REC_S s of the
    marked span of the speech calibration streams equals ``target``."""
    n = REC_S * wmx.SR
    ref = np.stack([host[b, g0[b]:g0[b] + n].numpy() for b in range(nsp)])
    rr = np.stack([r[b, g0[b]:g0[b] + n].numpy() for b in range(nsp)])
    lo, hi = -40.0, 30.0
    for _ in range(9 if not TINY else 4):
        mid = (lo + hi) / 2
        p = np.nanmean(pesq_batch(ref, ref + rr * 10 ** (mid / 20), pool))
        if p > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calibrate_snr(r, host, g0, target):
    """Gain (dB) on residual r so that the mean host-to-watermark SNR (dB) over the first REC_S s of the marked span
    of all calibration streams (speech and music) equals ``target`` (energy-matched comparison)."""
    n = REC_S * wmx.SR
    snr = [10 * np.log10(float((host[b, g0[b]:g0[b] + n] ** 2).sum()) / (float((r[b, g0[b]:g0[b] + n] ** 2).sum()) + 1e-12))
           for b in range(len(host))]
    return float(np.mean(snr) - target)


def target_gain(tname, rc, chost, cg0, pool, nsp):
    if tname == "native":
        return 0.0
    if tname.startswith("pesq"):
        return calibrate_alpha(rc, chost, cg0, float(tname[4:]), pool, nsp)
    if tname.startswith("snr"):
        return calibrate_snr(rc, chost, cg0, float(tname[3:]))
    raise ValueError(tname)


FIXED_COND_INDEX = {"ota_opus12": 50}


def evaluate(a):
    global TINY
    TINY = a.tiny
    outp = os.path.join(OUT, f"eval{a.tag}_seed{a.seed}.json")
    res = json.load(open(outp)) if os.path.exists(outp) else {"seed": a.seed, "rows": [], "audibility": [], "done": []}
    pool = None
    n_rec = 16 if TINY else a.n_rec
    host, src, g0, phi0, t0, info = make_streams(a.seed, n_rec)
    chost, _, cg0, cphi, _, cinfo = make_streams(10_000 + a.seed, 16 if TINY else 48)   # calibration set
    nsp_cal = len(chost) // 2
    chan = wmx.Channel(DEV, "test")
    corp = (wmx.Corpus("speech_train"), wmx.Corpus("music_train"))
    methods = []
    for m in a.methods:
        try:
            o = make_method(m, a.seed)
            if isinstance(o, Ours):
                nll = o.calibrate(corp, wmx.Channel(DEV, "train")); res.setdefault("cal_nll_bits", nll)
            methods.append(o)
        except Exception as e:
            say("method", m, "failed to load:", repr(e)); res.setdefault("errors", []).append(f"{m}: {e!r}")
    n = REC_S * wmx.SR
    conds = a.conditions or list(wmx.Channel.TESTS)
    # channel seed per condition: its position in the list (as before), except conditions added later,
    # which get a fixed index so their draws do not depend on which other conditions are evaluated
    cseed = {c: a.seed * 100 + FIXED_COND_INDEX.get(c, ci) for ci, c in enumerate(conds)}
    # unmarked twins: one channel pass per condition, shared by every method
    unmarked = {}
    for c in conds:
        y, off, sp = chan.test(host.to(DEV), c, cseed[c])
        unmarked[c] = torch.stack([y[b, t0[b]:t0[b] + n] for b in range(n_rec)]).cpu()
    say("unmarked passes done")
    neg_cache = {}
    for m in methods:
        r = m.residual(host, g0, info, phi0)
        rc = m.residual(chost, cg0, cinfo, cphi)
        targets = ([("native", 0.0)] if not a.no_native else []) + [(f"pesq{t:g}", None) for t in a.pesq_targets] \
            + [(f"snr{t:g}", None) for t in a.snr_targets]
        for tname, gdb in targets:
            key = f"{m.name}/{tname}"
            if key in res["done"]:
                continue
            if gdb is None:
                gdb = target_gain(tname, rc, chost, cg0, pool, nsp_cal)
            xm = host + r * 10 ** (gdb / 20)
            ref = np.stack([host[b, g0[b]:g0[b] + n].numpy() for b in range(n_rec)])
            deg = np.stack([xm[b, g0[b]:g0[b] + n].numpy() for b in range(n_rec)])
            pq = pesq_batch(ref[src == 0], deg[src == 0], pool)
            with torch.no_grad():
                ls = wmx.lsd(torch.from_numpy(ref), torch.from_numpy(deg - ref)).numpy()
            aud = {"method": m.name, "target": tname, "gain_db": gdb, "gain_at_bound": bool(gdb > 29 or gdb < -39), "pesq_speech": float(np.nanmean(pq)),
                   "si_snr_db": float(si_snr(ref, deg).mean()), "si_snr_speech": float(si_snr(ref, deg)[src == 0].mean()),
                   "si_snr_music": float(si_snr(ref, deg)[src == 1].mean()), "lsd_db": float(ls.mean())}
            res["audibility"].append(aud)
            say("audibility", aud)
            for c in conds:
                t1 = time.time()
                y, off, sp = chan.test(xm.to(DEV), c, cseed[c])
                ym = torch.stack([y[b, t0[b]:t0[b] + n] for b in range(n_rec)]).cpu()
                truth = truth_for(g0, phi0, t0, off.cpu().numpy().astype(np.float64), sp.cpu().numpy().astype(np.float64))
                dp = m.decode(ym, truth)
                if (m.name, c) not in neg_cache:            # unmarked twins do not depend on the target
                    neg_cache[(m.name, c)] = m.decode(unmarked[c], truth)
                dn = neg_cache[(m.name, c)]
                k = getattr(m, "k_info", 16); im = info_for(info, k)
                if hasattr(m, "slot_stats"):
                    ss = m.slot_stats(ym, truth, info)
                    res.setdefault("slots", []).append({"method": m.name, "target": tname, "condition": c, **ss})
                    say(f"{key} {c} slots:", {x: round(v, 4) for x, v in ss.items()})
                for rule in dp:
                    ip, op, sp_ = dp[rule]; _, on, sn = dn[rule]
                    correct = (ip[:, :k] == im).all(1)
                    row = {"method": m.name, "target": tname, "condition": c, "rule": rule, "k_info": k,
                           "ber": float((ip[:, :k] != im).mean()),
                           "raw_success": float((correct & op).mean()), "raw_false_alarm": float(on.mean()),
                           "success_speech": float((correct & op)[src == 0].mean()),
                           "success_music": float((correct & op)[src == 1].mean()),
                           "auc": auc(np.where(op, sp_, -1e9), np.where(on, sn, -1e9)),
                           "fa1": two_fold(correct, op, sp_, on, sn, 0.01),
                           "fa0": two_fold(correct, op, sp_, on, sn, 0.0),
                           "per_rec": {"correct": correct.astype(int).tolist(), "out_pos": op.astype(int).tolist(),
                                       "score_pos": np.round(np.nan_to_num(sp_, neginf=-1e9), 4).tolist(),
                                       "out_neg": on.astype(int).tolist(),
                                       "score_neg": np.round(np.nan_to_num(sn, neginf=-1e9), 4).tolist()},
                           "sec": time.time() - t1}
                    res["rows"].append(row)
                say(f"{key} {c}: " + ", ".join(f"{x['rule']} raw={x['raw_success']:.3f} fa1={x['fa1']['success']:.3f} "
                                               f"FA={x['raw_false_alarm']:.3f} ber={x['ber']:.3f}"
                                               for x in res["rows"][-len(dp):]))
            res["done"].append(key)
            dump(res, outp)
    res["complete"] = True
    dump(res, outp)


def summary(a):
    import glob
    rows, aud = [], []
    for f in sorted(glob.glob(os.path.join(OUT, "eval*_seed*.json"))):
        d = json.load(open(f))
        for r in d["rows"]:
            rows.append({**r, "seed": d["seed"]})
        for r in d["audibility"]:
            aud.append({**r, "seed": d["seed"]})
    if not rows:
        return
    tab = {}
    for r in rows:
        best = r["rule"] in ("crc@1",) if r["method"] == "ours" else r["rule"] == "top"
        if not best:
            continue
        k = (r["method"], r["target"], r["condition"])
        tab.setdefault(k, []).append(r["fa1"]["success"])
    out = {"success_at_fa1": {"/".join(k): {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)}
                              for k, v in sorted(tab.items())}}
    am = {}
    for r in aud:
        am.setdefault((r["method"], r["target"]), []).append(r)
    out["audibility"] = {"/".join(k): {q: float(np.mean([x[q] for x in v])) for q in ("pesq_speech", "si_snr_db", "lsd_db", "gain_db")}
                         for k, v in am.items()}
    dump(out, os.path.join(OUT, "summary.json"))
    for k, v in out["success_at_fa1"].items():
        print(f"{k:40s} {v['mean']:.3f} +- {v['std']:.3f} (n={v['n_seeds']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "summary"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--nblk", type=int, default=6)
    ap.add_argument("--whr", type=float, default=-20.0)
    ap.add_argument("--lsd_start", type=float, default=8.0)
    ap.add_argument("--lsd_target", type=float, default=1.0)
    ap.add_argument("--lsd_weight", type=float, default=1.0)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--init", default="", help="directory with seed<N>_final.pt to fine-tune from")
    ap.add_argument("--n_rec", type=int, default=300)
    ap.add_argument("--methods", nargs="*", default=["ours", "audioseal", "wavmark", "dear"])
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--pesq_targets", type=float, nargs="*", default=[4.0, 4.3])
    ap.add_argument("--snr_targets", type=float, nargs="*", default=[],
                    help="energy-matched targets: host-to-watermark SNR in dB over speech and music")
    ap.add_argument("--no_native", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--train_channel", default="full", choices=["full", "legacy"])
    ap.add_argument("--opus_p", type=float, default=0.0,
                    help="probability of the Opus stage after the training channel (0 = off, the original channel)")
    ap.add_argument("--opus_kbps", type=float, nargs=2, default=[8.0, 24.0], help="Opus bit-rate range, kbit/s")
    ap.add_argument("--tag", default="", help="suffix for the eval output file, e.g. _stress")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        evaluate(a)
    else:
        summary(a)


if __name__ == "__main__":
    main()
