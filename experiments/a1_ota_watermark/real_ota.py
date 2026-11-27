"""A1 real-world check: play marked audio through a real loudspeaker, record it with a real microphone,
decode with the same receivers and thresholds as the simulated benchmark.

usage:
  python real_ota.py make  --tag pilot --n 6      (GPU side: builds playlist.wav + manifest in $A1_REAL/<tag>)
  python real_play.py <dir>                        (recording side: plays playlist.wav, writes recording.wav)
  python real_ota.py score --tag pilot             (GPU side: aligns recording.wav, decodes, writes metrics)

Paths: $A1_REAL (default $HF_HOME/a1_real) holds playlists and recordings (audio is never committed);
metrics go to $RUN_DIR/out/real_<tag>.json. Trained models are read from A1_CKPT as in run_a1.py.

Each item in the playlist is one 14 s host stream (marked by one method at one matched audibility, or unmarked),
played once; items are shuffled so slow changes in the room affect every method alike. The recording is aligned
to the playlist by cross-correlation only to place the 8 s analysis window and to compute the bookkeeping
truth; every receiver still decodes blind (unknown chunk timing and cycle phase).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import run_a1 as R
import wmx

GAP_S, LEAD_S = 0.5, 3.0
DEFAULT_METHODS = ["ours", "audioseal", "wavmark", "dear"]


def real_dir(tag):
    base = os.environ.get("A1_REAL") or os.path.join(os.environ.get("HF_HOME", R.HERE), "a1_real")
    d = os.path.join(base, tag); os.makedirs(d, exist_ok=True)
    return d


def load_methods(names, seed, need_cal):
    out = []
    for m in names:
        o = R.make_method(m, seed)
        if need_cal and isinstance(o, R.Ours):
            o.calibrate((wmx.Corpus("speech_train"), wmx.Corpus("music_train")), wmx.Channel(R.DEV, "train"))
        out.append(o)
    return out


_NC = {}


def neural_codec(y, p):
    """Re-synthesis by a neural audio codec after the room (removal attack): ``encodec_<kbps>`` (Meta EnCodec 24 kHz at
    1.5 / 3 / 6 / 12 kbps; audio resampled 16 -> 24 -> 16 kHz) or ``dac`` (Descript DAC 16 kHz, all codebooks)."""
    import torchaudio.functional as AF
    if p not in _NC:
        if p == "dac":
            from transformers import DacModel
            _NC[p] = DacModel.from_pretrained("descript/dac_16khz").to(R.DEV).eval()
        else:
            from encodec import EncodecModel
            mdl = EncodecModel.encodec_model_24khz().to(R.DEV).eval()
            mdl.set_target_bandwidth(float(p.split("_")[1]))
            _NC[p] = mdl
    mdl, out = _NC[p], []
    with torch.no_grad():
        for i in range(0, len(y), 8):
            x = y[i:i + 8].to(R.DEV)
            if p == "dac":
                z = mdl(x[:, None]).audio_values
                z = z[:, 0] if z.dim() == 3 else z
            else:
                x24 = AF.resample(x, wmx.SR, 24000)
                z = AF.resample(mdl(x24[:, None])[:, 0], 24000, wmx.SR)
            out.append(z[:, :x.shape[-1]].float().cpu())
    return torch.cat(out)


def adapt_ours(m, a, sname):
    """Receiver-side room adaptation for ours: refit the per-slot LLR calibration (``cal``), or also fine-tune the
    decoder head (``head``), on the marked windows of another playlist recorded in the same room with the same microphone
    (``--adapt TAG[:recording]``), using its known messages and timing. The target playlist is never used."""
    tag, _, recname = a.adapt.partition(":")
    d = real_dir(tag)
    man = json.load(open(os.path.join(d, "manifest.json")))
    st = np.load(os.path.join(d, "streams.npz"))
    g0, phi0, t0, info, L = st["g0"], st["phi0"], st["t0"], st["info"], int(st["L"])
    import scipy.io.wavfile as wf
    _, pl = wf.read(os.path.join(d, "playlist.wav")); pl = pl.astype(np.float32) / 32767
    _, rec = wf.read(os.path.join(d, recname or "recording.wav"))
    rec = rec.astype(np.float32) / (32767 if rec.dtype == np.int16 else 1)
    rec = rec.mean(1) if rec.ndim > 1 else rec
    starts, _, _ = align(rec, pl, man["items"], L)
    n, B = R.REC_S * wmx.SR, man["n"]
    zs, bs = [], []
    lay = m.lay
    for tgt in sorted({g["target"] for g in man["gains"] if g["method"] == sname}):
        y = np.zeros((B, n), np.float32); ok = np.zeros(B, bool)
        for it, s0 in zip(man["items"], starts):
            if it["method"] == sname and it["target"] == tgt and s0 >= 0:
                b = it["stream"]; w = rec[s0 + t0[b]: s0 + t0[b] + n]
                if len(w) == n:
                    y[b] = w; ok[b] = True
        y = wmx.normalize(torch.from_numpy(y[ok]), 0.0)
        tr = R.truth_for(g0[ok], phi0[ok], t0[ok], np.zeros(ok.sum()), np.ones(ok.sum()))
        with torch.no_grad():
            feats = torch.cat([m.dec.frames(y[i:i + 16].to(R.DEV)).float().cpu() for i in range(0, len(y), 16)])
        f0 = np.round(np.asarray(tr.f0)).astype(int); J = feats.shape[1] // wmx.FPC + 1
        st_ = f0[:, None] + np.arange(J)[None] * wmx.FPC
        T = feats.shape[1]
        cs = torch.cat([torch.zeros(len(feats), 1, feats.shape[-1]), feats.cumsum(1)], 1)
        sI = torch.from_numpy(np.clip(st_, 0, T)).long(); eI = torch.from_numpy(np.clip(st_ + wmx.FPC, 0, T)).long()
        gat = lambda idx: cs.gather(1, idx[..., None].expand(-1, -1, feats.shape[-1]))
        mean = (gat(eI) - gat(sI)) / (eI - sI).clamp(min=1).float()[..., None]
        full = ((eI - sI) == wmx.FPC).numpy()
        bits = lay.chunk_bits(lay.frame(R.info_for(info[ok], m.k_info)))
        bits = bits[np.arange(len(y))[:, None], (np.asarray(tr.phase)[:, None] + np.arange(J)[None]) % 4]
        zs.append(mean[torch.from_numpy(full)]); bs.append(torch.from_numpy(bits[full]))
    X, Y = torch.cat(zs), torch.cat(bs).float()
    if a.adapt_mode == "head":
        head = m.dec.head; head.train()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-3)
        for step in range(a.adapt_steps):
            idx = torch.randint(0, len(X), (min(256, len(X)),))
            loss = F.binary_cross_entropy_with_logits(head(X[idx].to(R.DEV)).float(), Y[idx].to(R.DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
    with torch.no_grad():
        z = torch.cat([m.dec.head(X[i:i + 512].to(R.DEV)).float().cpu() for i in range(0, len(X), 512)]).numpy()
    m.cal = R.wc.fit_llr(z, Y.numpy().astype(int), m.name)
    m.name = f"{m.name}_adapt{a.adapt_mode}"
    R.say("adapted", m.name, "on", tag, recname, "chunks", len(X), "cal nll", round(m.cal.nll, 3))


def make(a):
    d = real_dir(a.tag)
    host, src, g0, phi0, t0, info = R.make_streams(a.stream_seed, a.n)
    chost, _, cg0, cphi, _, cinfo = R.make_streams(10_000 + a.seed, 48)
    items, gains = [("unmarked", "-", host.clone())], []
    # targets per method: --items "ours:pesq4 audioseal:snr22 ..." or every method x (pesq targets + snr targets)
    default_t = [f"pesq{t:g}" for t in a.pesq_targets] + [f"snr{t:g}" for t in a.snr_targets]
    spec = {}
    for s in (a.items or []):
        mm, tt = s.split(":"); spec.setdefault(mm, []).append(tt)
    names = list(spec) if spec else a.methods
    for m in load_methods(names, a.seed, need_cal=False):
        r = m.residual(host, g0, info, phi0)
        rc = m.residual(chost, cg0, cinfo, cphi)
        for t in spec.get(m.name, default_t):
            gdb = R.target_gain(t, rc, chost, cg0, None, len(chost) // 2)
            gains.append({"method": m.name, "target": t, "gain_db": gdb})
            items.append((m.name, t, host + r * 10 ** (gdb / 20)))
            R.say("gain", m.name, t, round(gdb, 2))
    L = host.shape[1]
    order = [(k, b) for k in range(len(items)) for b in range(a.n)]
    rng = np.random.default_rng(a.stream_seed + 1)
    order = [order[i] for i in rng.permutation(len(order))]
    step = L + int(GAP_S * wmx.SR)
    lead = int(LEAD_S * wmx.SR)
    pl = np.zeros(lead + len(order) * step + lead, np.float32)
    man = []
    for i, (k, b) in enumerate(order):
        p = lead + i * step
        pl[p:p + L] = items[k][2][b].numpy()
        man.append({"pos": p, "method": items[k][0], "target": items[k][1], "stream": b})
    peak = float(np.abs(pl).max())
    scale = 0.89 / peak                                  # one global gain for everything (keeps every ratio)
    pl *= scale
    import scipy.io.wavfile as wf
    wf.write(os.path.join(d, "playlist.wav"), wmx.SR, (pl * 32767).astype(np.int16))
    np.savez(os.path.join(d, "streams.npz"), src=src, g0=g0, phi0=phi0, t0=t0, info=info, L=L)
    json.dump({"tag": a.tag, "seed": a.seed, "stream_seed": a.stream_seed, "n": a.n, "scale": scale,
               "items": man, "gains": gains, "duration_s": len(pl) / wmx.SR, "rec_s": R.REC_S},
              open(os.path.join(d, "manifest.json"), "w"), indent=1)
    R.say(f"playlist {len(pl) / wmx.SR / 60:.1f} min, {len(man)} items -> {d}")


def _xcorr_lag(a, b):
    """lag of b inside a (a longer), via FFT."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)[:len(a) - len(b) + 1]
    return int(np.argmax(c)), float(c.max() / (np.linalg.norm(b) * np.linalg.norm(a) / np.sqrt(len(a) / len(b)) + 1e-9))


def align(rec, pl, man, L):
    sr = wmx.SR
    # global delay from the first 60 s of playlist content
    seg = pl[man[0]["pos"]: man[0]["pos"] + min(60 * sr, len(pl) - man[0]["pos"])]
    lag0, _ = _xcorr_lag(rec[: man[0]["pos"] + 30 * sr + len(seg)], seg)
    delay = lag0 - man[0]["pos"]
    starts, quality = [], []
    for it in man:
        lo = max(0, it["pos"] + delay - sr)
        w = rec[lo: it["pos"] + delay + L + sr]
        ref = pl[it["pos"]: it["pos"] + L]
        if len(w) < L:
            starts.append(-1); quality.append(0.0); continue
        lag, _ = _xcorr_lag(w, ref)
        s = lo + lag
        seg_r = rec[s:s + L]
        q = float(np.dot(seg_r, ref) / (np.linalg.norm(seg_r) * np.linalg.norm(ref) + 1e-9))
        starts.append(s); quality.append(q)
        delay = s - it["pos"]                           # track slow drift between the two clocks
    return np.array(starts), np.array(quality), delay


def score(a):
    import scipy.io.wavfile as wf
    d = real_dir(a.tag)
    man = json.load(open(os.path.join(d, "manifest.json")))
    st = np.load(os.path.join(d, "streams.npz"))
    src, g0, phi0, t0, info, L = st["src"], st["g0"], st["phi0"], st["t0"], st["info"], int(st["L"])
    sr_pl, pl = wf.read(os.path.join(d, "playlist.wav")); pl = pl.astype(np.float32) / 32767
    sr_r, rec = wf.read(os.path.join(d, a.recording))
    assert sr_pl == sr_r == wmx.SR, (sr_pl, sr_r)
    rec = rec.astype(np.float32) / (32767 if rec.dtype == np.int16 else 1)
    if rec.ndim > 1:
        rec = rec.mean(1)
    items = man["items"]
    assert man.get("rec_s", 8) == R.REC_S, ("set A1_REC_S to the playlist's listening window", man.get("rec_s", 8))
    starts, q, _ = align(rec, pl, items, L)
    res = {"tag": a.tag, "seed": man["seed"], "n_streams": man["n"], "gains": man["gains"],
           "align_corr_median": float(np.median(q)), "align_corr_min": float(q.min()),
           "rec_rms_db": float(20 * np.log10(np.sqrt(np.mean(rec ** 2)) + 1e-12)),
           "rec_clipped_frac": float(np.mean(np.abs(rec) > 0.999)), "rows": []}
    R.say("alignment corr median", res["align_corr_median"], "min", res["align_corr_min"])
    n = R.REC_S * wmx.SR
    B = man["n"]

    def post_fn(y, p):
        """Optional processing of the recorded windows after the room (codec or telephone band), before decoding."""
        if p is None:
            return y
        if p == "phone":
            f = torch.fft.rfftfreq(y.shape[-1], 1 / wmx.SR)[None]
            return wmx.mulaw(wmx.freq_gain(y, ((f >= 300) & (f <= 3400)).float().expand(len(y), -1)))
        if p.startswith("encodec") or p == "dac":
            return neural_codec(y, p)
        kind, br = p.split("_")
        return wmx.codec(y, kind, br)[0].cpu()

    def windows(method, target, p):
        y = np.zeros((B, n), np.float32); ok = np.zeros(B, bool)
        for it, s in zip(items, starts):
            if it["method"] == method and it["target"] == target and s >= 0:
                b = it["stream"]; w = rec[s + t0[b]: s + t0[b] + n]
                if len(w) == n:
                    y[b] = w; ok[b] = True
        y = wmx.normalize(torch.from_numpy(y), 0.0)        # blind level normalisation of each window
        return post_fn(y, p).clamp(-1, 1), ok

    truth = R.truth_for(g0, phi0, t0, np.zeros(B), np.ones(B))
    names = list(dict.fromkeys(g["method"] for g in man["gains"]))
    names = [m for m in names if m in a.methods] if a.methods != DEFAULT_METHODS else names
    objs = load_methods(names, man["seed"], need_cal=True)
    for spec in a.decode_as:                              # e.g. ours=ours_decwavlm: decode ours' items with another decoder
        src_m, dst_m = spec.split("=")
        if src_m in names:
            o = load_methods([dst_m], man["seed"], need_cal=True)[0]
            o.src_name = src_m
            objs.append(o)
    if a.adapt:                                           # adapted copies of ours-family methods, scored alongside
        extra = []
        for o in list(objs):
            if isinstance(o, R.Ours):
                c = load_methods([o.name], man["seed"], need_cal=True)[0] if not hasattr(o, "src_name") else None
                if c is None:
                    continue
                adapt_ours(c, a, o.name)
                extra.append(c)
        objs += extra
    for m in objs:
        sname = getattr(m, "src_name", m.name.split("_adapt")[0])
        for p in ([] if a.skip_plain else [None]) + list(a.post):
            cond = "real" if p is None else f"real+{p}"
            yn, okn = windows("unmarked", "-", p)
            if isinstance(m, R.AudioSealM):             # oracle rule: host sample 0 sits at -t0 in each window
                m.oracle_shift = (-t0.astype(int)) % m.PERIOD
            dn = m.decode(yn, truth)
            for tgt in sorted({g["target"] for g in man["gains"] if g["method"] == sname}):
                yp, okp = windows(sname, tgt, p)
                t1 = time.time()
                dp = m.decode(yp, truth)
                k = getattr(m, "k_info", 16); im = R.info_for(info, k)
                if hasattr(m, "slot_stats"):
                    ss = m.slot_stats(yp[torch.from_numpy(okp)], R.ws.Truth(f0=np.asarray(truth.f0)[okp], phase=np.asarray(truth.phase)[okp]),
                                      info[okp])
                    res.setdefault("slots", []).append({"method": m.name, "target": tgt, "condition": cond, **ss})
                    R.say(f"{m.name}/{tgt} {cond} slots:", {x: round(v, 4) for x, v in ss.items()})
                for rule in dp:
                    ip, op, sp_ = dp[rule]; _, on, sn = dn[rule]
                    correct = (ip[:, :k] == im).all(1)
                    use = okp & okn
                    row = {"method": m.name, "target": tgt, "condition": cond, "rule": rule, "n": int(use.sum()),
                           "k_info": k, "ber": float((ip[use, :k] != im[use]).mean()),
                           "raw_success": float((correct & op)[use].mean()), "raw_false_alarm": float(on[use].mean()),
                           "success_speech": float((correct & op)[use & (src == 0)].mean()),
                           "success_music": float((correct & op)[use & (src == 1)].mean()),
                           "auc": R.auc(np.where(op, sp_, -1e9)[use], np.where(on, sn, -1e9)[use]),
                           "fa1": R.two_fold(correct[use], op[use], sp_[use], on[use], sn[use], 0.01),
                           "fa0": R.two_fold(correct[use], op[use], sp_[use], on[use], sn[use], 0.0),
                           "per_rec": {"correct": correct.astype(int).tolist(), "out_pos": op.astype(int).tolist(),
                                       "score_pos": np.round(np.nan_to_num(sp_, neginf=-1e9), 4).tolist(),
                                       "out_neg": on.astype(int).tolist(),
                                       "score_neg": np.round(np.nan_to_num(sn, neginf=-1e9), 4).tolist()},
                           "sec": time.time() - t1}
                    res["rows"].append(row)
                    R.say(f"{m.name}/{tgt} {cond} {rule}: raw={row['raw_success']:.3f} fa1={row['fa1']['success']:.3f} "
                          f"FA={row['raw_false_alarm']:.3f} ber={row['ber']:.3f} n={row['n']}")
            R.dump(res, os.path.join(R.OUT, f"real_{a.tag}{a.out_suffix}.json"))
    R.dump(res, os.path.join(R.OUT, f"real_{a.tag}{a.out_suffix}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--n", type=int, default=6, help="host streams per (method, target) and unmarked")
    ap.add_argument("--seed", type=int, default=0, help="trained model seed")
    ap.add_argument("--stream_seed", type=int, default=30_000)
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--out_suffix", default="", help="suffix for the metrics file, e.g. _post")
    ap.add_argument("--pesq_targets", type=float, nargs="*", default=[4.0, 4.3])
    ap.add_argument("--snr_targets", type=float, nargs="*", default=[])
    ap.add_argument("--items", nargs="*", default=None, help='per-method targets, e.g. "ours:pesq4 wavmark:snr22"')
    ap.add_argument("--post", nargs="*", default=[],
                    help="also score the recording after a codec, e.g. mp3_64k opus_12k aac_32k phone")
    ap.add_argument("--skip_plain", action="store_true", help="only the --post conditions")
    ap.add_argument("--recording", default="recording.wav")
    ap.add_argument("--adapt", default="", help="TAG[:recording] of another playlist in the same room / mic")
    ap.add_argument("--adapt_mode", default="cal", choices=["cal", "head"])
    ap.add_argument("--adapt_steps", type=int, default=300)
    ap.add_argument("--decode_as", nargs="*", default=[], help='re-decode the items of one method with another, e.g. ours=ours_decwavlm')
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
