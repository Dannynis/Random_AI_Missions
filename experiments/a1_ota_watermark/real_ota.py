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

import run_a1 as R
import wmx

GAP_S, LEAD_S = 0.5, 3.0


def real_dir(tag):
    base = os.environ.get("A1_REAL") or os.path.join(os.environ.get("HF_HOME", R.HERE), "a1_real")
    d = os.path.join(base, tag); os.makedirs(d, exist_ok=True)
    return d


def load_methods(names, seed, need_cal):
    out = []
    for m in names:
        if m == "ours":
            o = R.Ours(seed)
            if need_cal:
                o.calibrate((wmx.Corpus("speech_train"), wmx.Corpus("music_train")), wmx.Channel(R.DEV, "train"))
            out.append(o)
        elif m == "audioseal":
            out.append(R.AudioSealM())
        elif m == "wavmark":
            out.append(R.WavMarkM())
    return out


def make(a):
    d = real_dir(a.tag)
    host, src, g0, phi0, t0, info = R.make_streams(a.stream_seed, a.n)
    chost, _, cg0, cphi, _, cinfo = R.make_streams(10_000 + a.seed, 48)
    items, gains = [("unmarked", "-", host.clone())], []
    for m in load_methods(a.methods, a.seed, need_cal=False):
        r = m.residual(host, g0, info, phi0)
        rc = m.residual(chost, cg0, cinfo, cphi)
        for t in a.pesq_targets:
            gdb = R.calibrate_alpha(rc, chost, cg0, t, None, len(chost) // 2)
            gains.append({"method": m.name, "target": f"pesq{t:g}", "gain_db": gdb})
            items.append((m.name, f"pesq{t:g}", host + r * 10 ** (gdb / 20)))
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
               "items": man, "gains": gains, "duration_s": len(pl) / wmx.SR},
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
    starts, q, _ = align(rec, pl, items, L)
    res = {"tag": a.tag, "seed": man["seed"], "n_streams": man["n"], "gains": man["gains"],
           "align_corr_median": float(np.median(q)), "align_corr_min": float(q.min()),
           "rec_rms_db": float(20 * np.log10(np.sqrt(np.mean(rec ** 2)) + 1e-12)),
           "rec_clipped_frac": float(np.mean(np.abs(rec) > 0.999)), "rows": []}
    R.say("alignment corr median", res["align_corr_median"], "min", res["align_corr_min"])
    n = R.REC_S * wmx.SR
    B = man["n"]

    def windows(method, target):
        y = np.zeros((B, n), np.float32); ok = np.zeros(B, bool)
        for it, s in zip(items, starts):
            if it["method"] == method and it["target"] == target and s >= 0:
                b = it["stream"]; w = rec[s + t0[b]: s + t0[b] + n]
                if len(w) == n:
                    y[b] = w; ok[b] = True
        y = wmx.normalize(torch.from_numpy(y), 0.0)        # blind level normalisation of each window
        return y, ok

    truth = R.truth_for(g0, phi0, t0, np.zeros(B), np.ones(B))
    yn, okn = windows("unmarked", "-")
    for m in load_methods(a.methods, man["seed"], need_cal=True):
        dn = m.decode(yn, truth)
        for tgt in sorted({g["target"] for g in man["gains"] if g["method"] == m.name}):
            yp, okp = windows(m.name, tgt)
            t1 = time.time()
            dp = m.decode(yp, truth)
            for rule in dp:
                ip, op, sp_ = dp[rule]; _, on, sn = dn[rule]
                correct = (ip[:, :16] == info).all(1)
                use = okp & okn
                row = {"method": m.name, "target": tgt, "condition": "real", "rule": rule, "n": int(use.sum()),
                       "ber": float((ip[use, :16] != info[use]).mean()),
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
                R.say(f"{m.name}/{tgt} {rule}: raw={row['raw_success']:.3f} fa1={row['fa1']['success']:.3f} "
                      f"FA={row['raw_false_alarm']:.3f} ber={row['ber']:.3f} n={row['n']}")
    R.dump(res, os.path.join(R.OUT, f"real_{a.tag}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--n", type=int, default=6, help="host streams per (method, target) and unmarked")
    ap.add_argument("--seed", type=int, default=0, help="trained model seed")
    ap.add_argument("--stream_seed", type=int, default=30_000)
    ap.add_argument("--methods", nargs="*", default=["ours", "audioseal", "wavmark"])
    ap.add_argument("--pesq_targets", type=float, nargs="*", default=[4.0, 4.3])
    ap.add_argument("--recording", default="recording.wav")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
