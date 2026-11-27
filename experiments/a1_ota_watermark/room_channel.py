"""Room calibration for A1: estimate the loudspeaker -> room -> microphone channel from a played playlist and design
a transmit-side equaliser for the watermark.

From each real-room set (a directory with playlist.wav, manifest.json and recording*.wav, as written by real_ota.py /
real_play.py), the unmarked items are aligned to the recording and Welch cross-spectra give the channel response
H(f) = S_xy / S_xx; the room noise N(f) is the recording's spectrum in the silent gaps between items. Several sets and
microphones are combined (|H| geometric mean, N mean). Two equalisers G(f) for the watermark residual are written
(dB per rFFT bin of a 512-point frame, clipped to +-6 dB, the speaker / microphone EQ range seen in training, and scaled
to unit mean power over the watermark's spectrum):
  * inv: inverse channel, so the mark arrives with a flat spectrum (third-octave smoothed);
  * wf:  water-filling of the mark's power over frequency for the room-noise-limited SNR |H|^2 S_w / N, S_w being the
         long-term host spectrum (ours' mark follows the host). Host interference is not modelled.
Audibility is re-matched by PESQ when the playlist is made, so G only redistributes the mark over frequency.

usage: python room_channel.py OUT.npz SET_DIR[:recording.wav] [SET_DIR[:recording_phone.wav] ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.io.wavfile as wf
import scipy.signal as ss

SR, NFFT, NLONG = 16000, 512, 4096


def load(path):
    sr, x = wf.read(path)
    assert sr == SR, (path, sr)
    x = x.astype(np.float32) / (32767 if x.dtype == np.int16 else 1)
    return x if x.ndim == 1 else x.mean(1)


def lag(a, b):
    """lag of b inside a (a longer)."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)[:len(a) - len(b) + 1]
    return int(np.argmax(c))


def estimate(d, rec_name):
    man = json.load(open(os.path.join(d, "manifest.json")))
    pl, rec = load(os.path.join(d, "playlist.wav")), load(os.path.join(d, rec_name))
    items = man["items"]
    L = int(np.median(np.diff([it["pos"] for it in items])))           # item step (item + gap)
    p0 = items[0]["pos"]
    seg = pl[p0:p0 + 60 * SR]
    delay = lag(rec[:p0 + 90 * SR], seg) - p0
    acc = {"xy": np.zeros(NLONG // 2 + 1, complex), "xx": np.zeros(NLONG // 2 + 1), "yy": np.zeros(NLONG // 2 + 1)}
    Ngap, ng = np.zeros(NFFT // 2 + 1), 0
    n_item = int(L - 0.5 * SR)
    for it in items:
        lo = max(0, it["pos"] + delay - SR)
        w = rec[lo: it["pos"] + delay + n_item + SR]
        ref = pl[it["pos"]: it["pos"] + n_item]
        if len(w) < n_item + SR:
            continue
        s = lo + lag(w, ref)
        delay = s - it["pos"]
        y = rec[s:s + n_item]
        if it["method"] == "unmarked":
            # long frames (0.25 s) so the room's reverberation tail stays inside a frame; reduced to NFFT bins below
            f4, pxy = ss.csd(ref, y, SR, nperseg=NLONG); _, pxx = ss.welch(ref, SR, nperseg=NLONG)
            _, pyy = ss.welch(y, SR, nperseg=NLONG)
            acc["xy"] += pxy; acc["xx"] += pxx; acc["yy"] += pyy
        gap = rec[s + n_item + int(0.05 * SR): s + n_item + int(0.45 * SR)]  # inside the 0.5 s silent gap
        if len(gap) >= NFFT:
            _, pg = ss.welch(gap, SR, nperseg=NFFT); Ngap += pg; ng += 1
    f = np.fft.rfftfreq(NFFT, 1 / SR); f4 = np.fft.rfftfreq(NLONG, 1 / SR)
    H4 = np.abs(acc["xy"]) / (acc["xx"] + 1e-20)
    coh = np.abs(acc["xy"]) ** 2 / (acc["xx"] * acc["yy"] + 1e-30)
    red = lambda v: np.array([np.mean(v[(f4 >= fc - SR / NFFT / 2) & (f4 < fc + SR / NFFT / 2)]) for fc in f])
    return {"f": f, "H": red(H4), "N": Ngap / max(ng, 1), "Sx": red(acc["xx"]), "coh": red(coh)}


def smooth_octave(f, v, frac=3):
    out = np.empty_like(v)
    for i, fc in enumerate(f):
        lo, hi = fc * 2 ** (-0.5 / frac), fc * 2 ** (0.5 / frac)
        m = (f >= lo) & (f <= hi)
        out[i] = np.exp(np.mean(np.log(v[m] + 1e-20))) if m.any() else v[i]
    return out


def unit_power(g_db, Sw):
    g = 10 ** (g_db / 10)
    return g_db - 10 * np.log10((g * Sw).sum() / Sw.sum())


def design(est):
    f = est[0]["f"]
    H = np.exp(np.mean([np.log(e["H"] + 1e-12) for e in est], 0))
    N = np.mean([e["N"] for e in est], 0)
    Sw = np.mean([e["Sx"] for e in est], 0)                            # mark spectrum ~ host spectrum (shape)
    band = (f >= 60) & (f <= 7800)
    Hs = smooth_octave(f, H)
    inv = np.zeros_like(f); inv[band] = -20 * np.log10(Hs[band] / np.exp(np.mean(np.log(Hs[band]))))
    inv = np.clip(unit_power(np.clip(inv, -6, 6), Sw * band), -6, 6)
    gam = smooth_octave(f, H ** 2 / (N + 1e-20))                         # room-noise-limited SNR per unit mark power
    # water-filling over the mark's spectrum: p_f = max(0, mu - 1 / gam_f), total sum p_f = sum Sw_f
    inv_g = 1 / (gam + 1e-30)
    lo_mu, hi_mu = 0.0, (Sw[band].sum() + inv_g[band].sum()) * 2
    for _ in range(100):
        mu = (lo_mu + hi_mu) / 2
        p = np.maximum(0, mu - inv_g) * band
        lo_mu, hi_mu = (mu, hi_mu) if p.sum() < Sw[band].sum() else (lo_mu, mu)
    wfg = np.zeros_like(f)
    wfg[band] = 10 * np.log10(np.maximum(p[band], 1e-6 * Sw[band].max()) / (Sw[band] + 1e-20))
    wfg[band] -= np.mean(wfg[band])
    wfg = np.clip(unit_power(np.clip(wfg, -6, 6), Sw * band), -6, 6)
    return {"f": f, "H": H, "N": N, "Sw": Sw, "inv_db": inv, "wf_db": wfg}


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    est = []
    for sp in specs:
        d, _, r = sp.partition(":")
        e = estimate(d, r or "recording.wav")
        print(os.path.basename(d.rstrip("/\\")), r or "recording.wav", "coherence 0.3-4 kHz",
              round(float(np.mean(e["coh"][(e["f"] > 300) & (e["f"] < 4000)])), 3), flush=True)
        est.append(e)
    g = design(est)
    np.savez(out, **g)
    f = g["f"]
    for lo, hi in ((60, 250), (250, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 7800)):
        m = (f >= lo) & (f < hi)
        print(f"{lo:5d}-{hi:5d} Hz  |H| {20 * np.log10(np.mean(g['H'][m])):6.1f} dB  N {10 * np.log10(np.mean(g['N'][m])):6.1f} dB"
              f"  inv {np.mean(g['inv_db'][m]):5.1f}  wf {np.mean(g['wf_db'][m]):5.1f}")


if __name__ == "__main__":
    main()
