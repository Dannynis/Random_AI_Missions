"""A1 audibility beyond PESQ, and a listening-test kit.

For every method at its PESQ-matched gains (same calibration as the benchmark) this computes on fresh test streams
(speech and music separately):
  * SQUIM MOS: torchaudio's SQUIM subjective model (predicted listener MOS of the marked audio, clean host as reference);
  * NMR: a PEAQ-style noise-to-mask ratio of the mark against the host's masking threshold (Bark bands, Schroeder
    spreading function, Johnston tonality offset; no absolute threshold, as levels are normalised). Reported as the
    energy-mean NMR in dB over frames and bands, and the fraction of band-frames whose mark exceeds the threshold;
  * STOI of the marked audio against the host, SI-SNR and LSD.
It also writes a listening-test kit (``--kit DIR``): for a few speech and music hosts, the host and every method's
marked version at each target as 16 kHz WAV files (public test audio only), plus ``abx.html``, a self-contained
ABX page (A = host, B = marked, X = one of them at random; the listener picks which).

usage: python audibility.py --seed 0 --n 40 --methods ours audioseal wavmark dear --targets 4.0 4.3 4.5 [--kit DIR]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import run_a1 as R
import wmx

SR = wmx.SR


# ------------------------------------------------------------------ PEAQ-style noise-to-mask ratio
def _bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def nmr(host, mark, n_fft=1024, hop=512):
    """host, mark (L,) float. Returns (energy-mean NMR dB, fraction of band-frames with NMR > 0 dB)."""
    win = np.hanning(n_fft)
    def spec(x):
        fr = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
        return np.abs(np.fft.rfft(fr, axis=-1)) ** 2
    H, W = spec(host), spec(mark)
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    z = _bark(f)
    nb = int(np.ceil(z.max())) + 1
    band = np.minimum(z.astype(int), nb - 1)
    M = np.zeros((nb, len(f))); M[band, np.arange(len(f))] = 1
    Eh, Ew = H @ M.T + 1e-12, W @ M.T + 1e-12                      # (frames, bands)
    zc = np.arange(nb) + 0.5
    dz = zc[None, :] - zc[:, None]                                  # masker i -> maskee j
    sf_db = 15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1 + (dz + 0.474) ** 2)
    spread = Eh @ (10 ** (sf_db / 10))                              # (frames, bands)
    # Johnston tonality from the spectral flatness of the host frame
    gm = np.exp(np.mean(np.log(H + 1e-12), -1)); am = np.mean(H, -1) + 1e-12
    sfm_db = 10 * np.log10(gm / am + 1e-12)
    alpha = np.clip(sfm_db / -60.0, 0, 1)[:, None]
    offset = alpha * (14.5 + zc[None]) + (1 - alpha) * 5.5
    thr = spread * 10 ** (-offset / 10)
    r = Ew / thr
    return float(10 * np.log10(r.mean())), float((r > 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--methods", nargs="*", default=["ours", "audioseal", "wavmark", "dear"])
    ap.add_argument("--targets", type=float, nargs="*", default=[4.0, 4.3, 4.5])
    ap.add_argument("--kit", default="")
    ap.add_argument("--kit_n", type=int, default=6)
    a = ap.parse_args()
    import torchaudio
    from pystoi import stoi
    squim = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model().to(R.DEV).eval()
    host, src, g0, phi0, t0, info = R.make_streams(20_000 + a.seed, a.n)
    chost, _, cg0, cphi, _, cinfo = R.make_streams(10_000 + a.seed, 48)
    n = R.REC_S * SR
    ref = np.stack([host[b, g0[b]:g0[b] + n].numpy() for b in range(a.n)])
    out = {"seed": a.seed, "n": a.n, "rows": []}
    kit = []
    for m in [R.make_method(x, a.seed) for x in a.methods]:
        r = m.residual(host, g0, info, phi0)
        rc = m.residual(chost, cg0, cinfo, cphi)
        for t in a.targets:
            gdb = R.target_gain(f"pesq{t:g}", rc, chost, cg0, None, len(chost) // 2)
            deg = ref + np.stack([r[b, g0[b]:g0[b] + n].numpy() for b in range(a.n)]) * 10 ** (gdb / 20)
            pq = R.pesq_batch(ref[src == 0], deg[src == 0])
            with torch.no_grad():
                mos = torch.cat([squim(torch.from_numpy(deg[i:i + 8]).float().to(R.DEV),
                                       torch.from_numpy(ref[i:i + 8]).float().to(R.DEV)).cpu()
                                 for i in range(0, a.n, 8)]).numpy()
                mos0 = torch.cat([squim(torch.from_numpy(ref[i:i + 8]).float().to(R.DEV),
                                        torch.from_numpy(ref[i:i + 8]).float().to(R.DEV)).cpu()
                                  for i in range(0, a.n, 8)]).numpy()
            nm = np.array([nmr(ref[b], deg[b] - ref[b]) for b in range(a.n)])
            st = np.array([stoi(ref[b], deg[b], SR) for b in range(a.n)])
            row = {"method": m.name, "target": f"pesq{t:g}", "gain_db": gdb}
            for nm_, sel in (("speech", src == 0), ("music", src == 1)):
                row[nm_] = {"pesq": float(np.nanmean(pq)) if nm_ == "speech" else None,
                            "squim_mos": float(mos[sel].mean()), "squim_mos_host": float(mos0[sel].mean()),
                            "nmr_db": float(10 * np.log10(np.mean(10 ** (nm[sel, 0] / 10)))),
                            "nmr_audible_frac": float(nm[sel, 1].mean()), "stoi": float(st[sel].mean()),
                            "si_snr_db": float(R.si_snr(ref[sel], deg[sel]).mean())}
            out["rows"].append(row)
            R.say("audibility+", json.dumps(row))
            if a.kit:
                idx = list(np.where(src == 0)[0][:a.kit_n // 2]) + list(np.where(src == 1)[0][:a.kit_n // 2])
                for b in idx:
                    kit.append((b, m.name, f"pesq{t:g}", deg[b]))
            R.dump(out, os.path.join(R.OUT, f"audibility_seed{a.seed}.json"))
    if a.kit:
        import scipy.io.wavfile as wf
        os.makedirs(a.kit, exist_ok=True)
        hosts = sorted({b for b, *_ in kit})
        for b in hosts:
            wf.write(os.path.join(a.kit, f"host{b:03d}.wav"), SR, (np.clip(ref[b], -1, 1) * 32767).astype(np.int16))
        trials = []
        for b, mname, t, x in kit:
            fn = f"host{b:03d}_{mname}_{t}.wav"
            wf.write(os.path.join(a.kit, fn), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
            trials.append({"host": f"host{b:03d}.wav", "marked": fn, "method": mname, "target": t,
                           "type": "speech" if src[b] == 0 else "music"})
        json.dump(trials, open(os.path.join(a.kit, "trials.json"), "w"), indent=1)
        open(os.path.join(a.kit, "abx.html"), "w").write(ABX_HTML.replace("__TRIALS__", json.dumps(trials)))
        R.say("kit", len(trials), "trials ->", a.kit)


ABX_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Watermark ABX test</title><style>
:root{--bg:#fff;--fg:#111;--mut:#666;--acc:#2a5bd7;--card:#f4f5f7}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#aaa;--acc:#7aa2ff;--card:#1c1d20}}
body{background:var(--bg);color:var(--fg);font:16px system-ui,sans-serif;max-width:640px;margin:0 auto;padding:16px}
.card{background:var(--card);border-radius:10px;padding:16px;margin:12px 0}button{font-size:16px;margin:4px;padding:8px 14px;border-radius:8px;border:1px solid var(--mut);background:var(--bg);color:var(--fg)}
button.pick{border-color:var(--acc)}small{color:var(--mut)}</style></head><body>
<h1>ABX listening test</h1><p><small>Use headphones in a quiet room. A is the original, B the watermarked version, X is one
of them at random. Listen as often as you like, then say whether X is A or B. Results stay in this page; press
"Download results" at the end.</small></p><div id="t" class="card"></div><div class="card"><span id="p"></span>
<button onclick="dl()">Download results</button></div>
<script>
const T=__TRIALS__;const order=T.map((_,i)=>i).sort(()=>Math.random()-0.5);let k=0;const res=[];
function au(f){const a=new Audio(f);a.play();}
function show(){if(k>=order.length){document.getElementById('t').innerHTML='<b>Done.</b> '+res.filter(r=>r.correct).length+' / '+res.length+' correct.';return;}
const t=T[order[k]];const xIsB=Math.random()<0.5;t._x=xIsB;
document.getElementById('t').innerHTML=`<div>Trial ${k+1} of ${order.length}</div>
<button onclick="au('${t.host}')">Play A</button><button onclick="au('${t.marked}')">Play B</button>
<button onclick="au('${xIsB?t.marked:t.host}')">Play X</button><div>X is: <button class="pick" onclick="ans(false)">A</button>
<button class="pick" onclick="ans(true)">B</button></div>`;document.getElementById('p').textContent=`${k} answered`;}
function ans(b){const t=T[order[k]];res.push({...t,x_is_marked:t._x,answer_marked:b,correct:b===t._x});k++;show();}
function dl(){const u=URL.createObjectURL(new Blob([JSON.stringify(res,null,1)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='abx_results.json';a.click();}
show();</script></body></html>"""


if __name__ == "__main__":
    main()
