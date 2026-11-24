"""DeAR baseline (Liu et al., AAAI 2023, "DeAR: A Deep-learning-based Audio Re-recording Resilient
Watermarking", arXiv 2212.02339), reimplemented from the paper; no official code was released.

What follows the paper:
- one-level DWT of each audio segment; the watermark is embedded only in the approximate coefficients
  A_ac (eq. 1-2): A_ac^W = En(A_ac, W) * S + A_ac, strength factor S = 1 by default; the detail
  coefficients are kept and the audio is rebuilt by IDWT (eq. 6). Since the DWT is linear the time-domain
  residual is IDWT(S * En(A_ac, W), 0).
- encoder, decoder and discriminator are fully convolutional 1-D networks; the encoder keeps the feature
  length unchanged, the decoder and discriminator end with strided-convolution downsampling and a mean over
  time; the input (A_ac) is concatenated to the feature map of every layer (skip connection).
- bipolar message W in {-1, 1}, watermark loss L_w = MSE(W, W'), embedding loss L_e = MSE(A_ac, A_ac^W),
  adversarial loss L_d against a discriminator on A_ac^W vs A_ac; L = 150 L_e + 0.01 L_d + 1 L_w; Adam, lr 1e-4.
- distortion layer DAR = GN(BF(ER(.))) on the rebuilt waveform, applied to every training batch:
  ER = convolution with a random room impulse response, BF = high-pass at 1 kHz then low-pass at 4 kHz,
  GN = additive Gaussian noise at an SNR drawn uniformly from 20-25 dB; the decoder reads the DWT approximate
  coefficients of the attacked audio (eq. 12).
- the same message is embedded in every segment of a stream (the paper embeds one fixed bit string per audio).

What the paper does not specify and had to be chosen here:
- sample rate 16 kHz (paper: 44.1 kHz) because the whole benchmark runs at 16 kHz; the 1-4 kHz band-pass is
  kept as stated (it lies inside the 0-4 kHz band of the approximate coefficients at 16 kHz).
- payload 16 bits per segment (paper: 100 bits per audio; its ablation trains 64-225 bits). The 16 bits of
  the benchmark message are carried directly, no channel code or CRC.
- segment length 1 s (16000 samples, 8000 approximate coefficients); the paper gives no segment length.
- Haar wavelet for the DWT (the paper only says "differential DWT").
- layer counts, widths and kernels: encoder = 4 conv blocks (Conv1d k=3, BatchNorm, LeakyReLU, 64 channels,
  input skip at every layer), then the message (16 channels, each bit broadcast over time, as in HiDDeN)
  concatenated with features and input, one more block, 1x1 conv to the residual. Decoder = 4 such blocks,
  3 stride-2 downsampling blocks, 1x1 conv to 16 channels, mean over time. Discriminator = 3 blocks,
  3 stride-2 downsampling blocks, 1x1 conv to 1 channel, mean over time.
- the adversarial terms use the usual non-saturating binary cross-entropy forms of eq. 4 (same optimum).
- the impulse responses come from the benchmark's own simulated training room bank (rir_train) instead of the
  Bell Labs varechoic-chamber recordings (not redistributable in this pipeline); the filters are zero-phase
  Butterworth-shaped magnitude responses (order 4 high-pass, order 8 low-pass) applied in the FFT domain.
- training data: the benchmark's speech + music training split (paper: 12000 FMA music clips); hosts are
  levelled like the other trained model's (RMS 0.05 +- 6 dB, random polarity) and multiplied by AMP = 2 inside
  the model, so the typical host sits near -20 dBFS (mastered music level) where the paper's loss weights
  were tuned; the residual is divided by AMP again on output.
- batch size and number of steps (not given): batch 64 one-second segments; steps set on the command line.
- receiver synchronisation: the paper aligns re-recordings by trying shifts in a small range and keeping the
  shift with the highest bit accuracy (an oracle). Here the alignment is blind: dense decoder outputs are
  computed for the 16 sample phases of the downsampling grid, every segment offset 0..15999 is scored by the
  mean |soft bit| of the soft outputs averaged over all 7 whole segments of the 8 s recording, the best
  candidate offsets are re-decoded exactly (segment crops, as in training) and the most confident one is kept.
  Bits = sign of the segment-averaged soft output; detection score = its mean absolute value.

usage:
  python dear.py train --seed 0 --steps 100000
Checkpoints: $RUN_DIR/ckpt/dear_seed{S}.pt (resumable) and dear_seed{S}_final.pt; evaluation reads them from
the ':'-separated directories in $A1_CKPT, then $RUN_DIR/ckpt.
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
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wmx  # noqa: E402

RUN = os.environ.get("RUN_DIR", os.path.join(HERE, "_run"))
CK, OUT = os.path.join(RUN, "ckpt"), os.path.join(RUN, "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
T0 = time.time()

SEG = 16000              # samples per segment (1 s)
NBITS = 16
AMP = 2.0
DS = 8                   # decoder downsampling in coefficient frames (= 16 samples)


def say(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ Haar DWT (differentiable)
def dwt(x):
    a = (x[..., 0::2] + x[..., 1::2]) / math.sqrt(2)
    d = (x[..., 0::2] - x[..., 1::2]) / math.sqrt(2)
    return a, d


def idwt(a, d):
    x = torch.stack([(a + d) / math.sqrt(2), (a - d) / math.sqrt(2)], -1)
    return x.flatten(-2)


# ------------------------------------------------------------------ networks
def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv1d(cin, cout, 3, stride=stride, padding=1), nn.BatchNorm1d(cout), nn.LeakyReLU(0.2))


class Encoder(nn.Module):
    def __init__(self, ch=64, nb=4):
        super().__init__()
        self.first = block(1, ch)
        self.body = nn.ModuleList([block(ch + 1, ch) for _ in range(nb - 1)])
        self.fuse = block(ch + 1 + NBITS, ch)
        self.out = nn.Conv1d(ch + 1, 1, 1)

    def forward(self, aac, w):
        """aac (B, T), w (B, 16) in {-1, 1} -> residual on the approximate coefficients (B, T)."""
        x = aac[:, None]
        h = self.first(x)
        for b in self.body:
            h = b(torch.cat([h, x], 1))
        m = w[:, :, None].expand(-1, -1, x.shape[-1])
        h = self.fuse(torch.cat([h, x, m], 1))
        return self.out(torch.cat([h, x], 1))[:, 0]


class Decoder(nn.Module):
    def __init__(self, ch=64, nb=4, nd=3):
        super().__init__()
        self.first = block(1, ch)
        self.body = nn.ModuleList([block(ch + 1, ch) for _ in range(nb - 1)])
        self.down = nn.Sequential(*[block(ch, ch, 2) for _ in range(nd)])
        self.out = nn.Conv1d(ch, NBITS, 1)

    def frames(self, aac):
        """(B, T) -> per-frame outputs (B, 16, T / 8); the message estimate is their mean over time."""
        x = aac[:, None]
        h = self.first(x)
        for b in self.body:
            h = b(torch.cat([h, x], 1))
        return self.out(self.down(h))

    def forward(self, aac):
        return self.frames(aac).mean(-1)


class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(block(1, ch), block(ch, ch), block(ch, ch),
                                 block(ch, ch, 2), block(ch, ch, 2), block(ch, ch, 2), nn.Conv1d(ch, 1, 1))

    def forward(self, aac):
        return self.net(aac[:, None]).mean((1, 2))


# ------------------------------------------------------------------ distortion layer (re-recording model)
class DAR:
    def __init__(self, device):
        self.rir = torch.from_numpy(np.load(os.path.join(wmx.data_dir(), "rir_train.npy"))).to(device)
        self.bp = None

    def __call__(self, x, g):
        B, n = x.shape
        dev = x.device
        y = wmx.fft_conv(x, self.rir[torch.randint(0, len(self.rir), (B,), generator=g, device=dev)])   # ER
        if self.bp is None or self.bp.shape[-1] != n // 2 + 1:
            lo, hi = torch.tensor([1000.0], device=dev), torch.tensor([4000.0], device=dev)
            self.bp = wmx.band_response(lo, hi, torch.zeros(1, 1, device=dev), n)                       # BF
        y = wmx.freq_gain(y, self.bp)
        snr = 20 + 5 * torch.rand(B, 1, generator=g, device=dev)                                         # GN
        return wmx.add_noise(y, torch.randn((B, n), generator=g, device=dev), snr)


# ------------------------------------------------------------------ training
def host_batch(corp, rng, B, length):
    nsp = B // 2
    x = np.concatenate([corp[0].segments(rng, nsp, length), corp[1].segments(rng, B - nsp, length)])
    x = torch.from_numpy(x).to(DEV)
    x = wmx.normalize(x, torch.from_numpy(rng.uniform(-6, 6, B)).float().to(DEV))
    return x * torch.from_numpy(rng.choice([-1.0, 1.0], (B, 1))).float().to(DEV)


def save_atomic(obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)


def train(a):
    os.makedirs(CK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    path = os.path.join(CK, f"dear_seed{a.seed}.pt")
    final = os.path.join(CK, f"dear_seed{a.seed}_final.pt")
    if os.path.exists(final):
        say("dear seed", a.seed, "already trained"); return
    torch.manual_seed(a.seed)
    enc, dec, dis = Encoder().to(DEV), Decoder().to(DEV), Discriminator().to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    opt_d = torch.optim.Adam(dis.parameters(), lr=a.lr)
    step, hist = 0, []
    if os.path.exists(path):
        st = torch.load(path, map_location=DEV, weights_only=False)
        for m, k in ((enc, "enc"), (dec, "dec"), (dis, "dis"), (opt, "opt"), (opt_d, "opt_d")):
            m.load_state_dict(st[k])
        step, hist = st["step"], st["hist"]
        say("resumed dear seed", a.seed, "at step", step)
    say("params enc", sum(p.numel() for p in enc.parameters()), "dec", sum(p.numel() for p in dec.parameters()),
        "dis", sum(p.numel() for p in dis.parameters()))
    corp = (wmx.Corpus("speech_train"), wmx.Corpus("music_train"))
    dar = DAR(DEV)
    run = {k: 0.0 for k in ("lw", "le", "ld", "lD", "ber", "snr")}; nrun = 0
    t1 = time.time()
    while step < a.steps:
        step += 1
        rng = np.random.default_rng([a.seed, step, 7])
        g = torch.Generator(device=DEV).manual_seed(a.seed * 10 ** 7 + step)
        A = host_batch(corp, rng, a.batch, SEG) * AMP
        W = 2 * torch.randint(0, 2, (a.batch, NBITS), generator=g, device=DEV).float() - 1
        aac, adc = dwt(A)
        aw = enc(aac, W) * a.strength + aac
        # discriminator step
        lD = F.binary_cross_entropy_with_logits(dis(aac), torch.ones(a.batch, device=DEV)) + \
            F.binary_cross_entropy_with_logits(dis(aw.detach()), torch.zeros(a.batch, device=DEV))
        opt_d.zero_grad(set_to_none=True); lD.backward(); opt_d.step()
        # encoder / decoder step
        att = dar(idwt(aw, adc), g)
        Wp = dec(dwt(att)[0])
        lw = F.mse_loss(Wp, W)
        le = F.mse_loss(aw, aac)
        ld = F.binary_cross_entropy_with_logits(dis(aw), torch.ones(a.batch, device=DEV))
        loss = a.le * le + a.ld * ld + a.lw * lw
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            r = aw - aac
            snr = 10 * torch.log10(aac.pow(2).sum(-1) / (r.pow(2).sum(-1) + 1e-12) + 1e-12).mean()
        for k, v in (("lw", lw), ("le", le), ("ld", ld), ("lD", lD), ("snr", snr),
                     ("ber", ((Wp > 0) != (W > 0)).float().mean())):
            run[k] += v.item()
        nrun += 1
        if step % 100 == 0 or step == a.steps:
            hist.append({"step": step, **{k: v / nrun for k, v in run.items()}, "s_per_step": (time.time() - t1) / nrun})
            say(f"dear seed {a.seed} step {step}: " + ", ".join(f"{k}={v:.4g}" for k, v in hist[-1].items() if k != "step"))
            run = {k: 0.0 for k in run}; nrun = 0; t1 = time.time()
        if step % a.ckpt_every == 0 or step == a.steps:
            save_atomic({"enc": enc.state_dict(), "dec": dec.state_dict(), "dis": dis.state_dict(), "opt": opt.state_dict(),
                         "opt_d": opt_d.state_dict(), "step": step, "hist": hist, "args": vars(a)}, path)
    save_atomic({"enc": enc.state_dict(), "dec": dec.state_dict(), "hist": hist, "args": vars(a)}, final)
    tmp = os.path.join(OUT, f"train_dear_seed{a.seed}.json.tmp")
    json.dump(hist, open(tmp, "w"), indent=1); os.replace(tmp, tmp[:-4])
    os.remove(path)


# ------------------------------------------------------------------ benchmark wrapper
def load_dear(seed):
    dirs = [d for d in os.environ.get("A1_CKPT", "").split(":") if d] + [CK]
    fn = f"dear_seed{seed}_final.pt"
    path = next((os.path.join(d, fn) for d in dirs if os.path.exists(os.path.join(d, fn))), None)
    if path is None:
        raise FileNotFoundError(f"no trained DeAR model for seed {seed}")
    st = torch.load(path, map_location=DEV, weights_only=False)
    enc, dec = Encoder().to(DEV).eval(), Decoder().to(DEV).eval()
    enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
    return enc, dec, st["args"]


class DeARM:
    name = "dear"

    def __init__(self, seed):
        self.enc, self.dec, self.ar = load_dear(seed)
        self.strength = float(self.ar.get("strength", 1.0))

    def residual(self, host, g0, info, phi0):
        """Same 16-bit message in every whole 1 s segment of the stream (grid from sample 0)."""
        B, L = host.shape
        S = L // SEG
        W = torch.from_numpy(2 * np.asarray(info, np.float32) - 1)
        r = torch.zeros_like(host)
        segs = host[:, :S * SEG].reshape(B * S, SEG)
        Ws = W.repeat_interleave(S, 0)
        out = []
        with torch.no_grad():
            for i in range(0, len(segs), 256):
                aac, _ = dwt(segs[i:i + 256].to(DEV) * AMP)
                ra = self.enc(aac, Ws[i:i + 256].to(DEV)) * self.strength
                out.append((idwt(ra, torch.zeros_like(ra)) / AMP).cpu())
        r[:, :S * SEG] = torch.cat(out).reshape(B, S * SEG)
        return r

    def _exact(self, y, offs, nseg):
        """y (n, L) on device, offs (n,) -> segment-averaged soft outputs (n, 16)."""
        crops = torch.stack([y[i, o + k * SEG: o + (k + 1) * SEG] for i, o in enumerate(offs.tolist()) for k in range(nseg)])
        v = torch.cat([self.dec(dwt(crops[j:j + 256] * AMP)[0]) for j in range(0, len(crops), 256)])
        return v.reshape(len(offs), nseg, NBITS).mean(1)

    def decode(self, y, truth):
        B, L = y.shape
        nseg = max(1, (L - SEG + 1) // SEG)                     # whole segments available at every offset < SEG
        nfr = SEG // 2 // DS                                     # decoder frames per segment
        Q = SEG // (2 * DS)                                      # offsets per phase (o = 2*DS*q + s)
        ms = np.zeros((B, NBITS), np.int64); sc = np.zeros(B)
        with torch.no_grad():
            for b0 in range(0, B, 8):
                yb = y[b0:b0 + 8].to(DEV).float()
                n = len(yb)
                conf = torch.zeros(n, 2 * DS, Q, device=DEV)
                for s in range(2 * DS):
                    z = yb[:, s:]
                    z = z[:, :z.shape[1] // (2 * DS) * (2 * DS)]
                    fr = self.dec.frames(dwt(z * AMP)[0])                         # (n, 16, T)
                    cs = torch.cat([fr.new_zeros(n, NBITS, 1), fr.cumsum(-1)], -1)
                    T = fr.shape[-1]
                    v = 0
                    for k in range(nseg):
                        st = torch.arange(Q, device=DEV) + k * nfr
                        en = (st + nfr).clamp(max=T); st = st.clamp(max=T)
                        v = v + (cs[..., en] - cs[..., st]) / nfr
                    conf[:, s] = (v / nseg).abs().mean(1)
                flat = conf.permute(0, 2, 1).reshape(n, -1)                     # offset o = 2*DS*q + s
                top = flat.topk(4, dim=1).indices                               # candidate offsets
                best_v, best_c = None, None
                for j in range(top.shape[1]):
                    offs = top[:, j]
                    v = self._exact(yb, offs, nseg)
                    c = v.abs().mean(1)
                    if best_c is None:
                        best_v, best_c = v, c
                    else:
                        upd = c > best_c
                        best_v = torch.where(upd[:, None], v, best_v); best_c = torch.where(upd, c, best_c)
                ms[b0:b0 + n] = (best_v > 0).long().cpu().numpy()
                sc[b0:b0 + n] = best_c.cpu().numpy()
        return {"top": (ms, np.ones(B, bool), sc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--le", type=float, default=150.0)
    ap.add_argument("--lw", type=float, default=1.0)
    ap.add_argument("--ld", type=float, default=0.01)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()
