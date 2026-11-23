"""A1 core: data, GPU channel simulator, scaled encoder / decoder.

The physical-layer interface is kept identical to the original small system
(2 s chunks at 16 kHz, 32 payload slots per chunk, STFT 512 / hop 256, decoder =
conv trunk -> mean over a 125-frame window -> MLP head), so the original link layer
(rate-1/2 K=7 convolutional code over a 4-chunk cycle, CRC-8, list Viterbi, blind
joint timing search) in ``legacy/`` runs unchanged on top of the new models.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR, N, K = 16000, 32000, 32
N_FFT, HOP = 512, 256
FPC = N // HOP                       # 125 decoder frames per chunk
NF = N_FFT // 2 + 1
MAX_OFFSET = 1600


def data_dir() -> str:
    return os.environ.get("A1_DATA") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "a1_audio")


# ---------------------------------------------------------------- data
class Corpus:
    """Concatenated int16 audio + file index. ``segment`` returns contiguous slices inside one file."""

    def __init__(self, name: str, in_ram: bool = True):
        d = data_dir()
        # random access into a memory map on a network filesystem is slow: load into RAM by default
        self.x = np.load(os.path.join(d, f"{name}.npy"), mmap_mode=None if in_ram else "r")
        idx = json.load(open(os.path.join(d, f"{name}.json")))
        self.start = np.array([i[0] for i in idx], np.int64)
        self.len = np.array([i[1] for i in idx], np.int64)
        self.group = [i[2] for i in idx]

    def segments(self, rng: np.random.Generator, n: int, length: int, min_rms: float = 1e-3) -> np.ndarray:
        ok = np.where(self.len >= length)[0]
        if len(ok) == 0:                       # short files (noise): slice the concatenation across files
            s0 = rng.integers(0, len(self.x) - length, n)
            return np.stack([self.x[s:s + length].astype(np.float32) / 32767 for s in s0])
        out = np.zeros((n, length), np.float32)
        i = 0
        tries = 0
        while i < n:
            f = ok[rng.integers(len(ok))]
            s = self.start[f] + rng.integers(0, self.len[f] - length + 1)
            seg = self.x[s:s + length].astype(np.float32) / 32767
            tries += 1
            if np.sqrt((seg ** 2).mean()) < min_rms and tries < 50 * n:
                continue
            out[i] = seg; i += 1
        return out


def normalize(x: torch.Tensor, rms_db: torch.Tensor | float = 0.0, base: float = 0.05) -> torch.Tensor:
    r = x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
    g = base * 10 ** (torch.as_tensor(rms_db, device=x.device, dtype=x.dtype).reshape(-1, 1) / 20)
    return x / r * g


# ---------------------------------------------------------------- dsp (device generic)
def stft(x, n_fft=N_FFT, hop=HOP):
    return torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft, device=x.device), return_complex=True)


def istft(X, length, n_fft=N_FFT, hop=HOP):
    return torch.istft(X, n_fft, hop, window=torch.hann_window(n_fft, device=X.device), length=length)


def fft_conv(x, h):
    n = x.shape[-1] + h.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(h, n), n)[..., :x.shape[-1]]


def freq_gain(x, gain):
    return torch.fft.irfft(torch.fft.rfft(x) * gain, x.shape[-1])


def band_response(lo, hi, eq_db, n, order_lo=4, order_hi=8):
    f = torch.fft.rfftfreq(n, 1 / SR, device=lo.device)[None]
    r = 1 / torch.sqrt(1 + (lo[:, None] / (f + 1e-3)) ** order_lo) / torch.sqrt(1 + (f / hi[:, None]) ** order_hi)
    u = f / (SR / 2)
    basis = torch.stack([torch.cos(math.pi * (k + 1) * u) for k in range(eq_db.shape[1])], -1)
    return r * 10 ** ((basis * eq_db[:, None, :]).sum(-1) / 20)


def warp(x, speed, offset):
    """y[n] = x[(n - offset) * speed] (linear interpolation), zeros outside."""
    n = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32)[None]
    pos = (n - offset[:, None]) * speed[:, None]
    i0 = pos.floor().long(); frac = pos - i0
    valid = (i0 >= 0) & (i0 < x.shape[-1] - 1)
    i0c = i0.clamp(0, x.shape[-1] - 2)
    return (x.gather(-1, i0c) * (1 - frac) + x.gather(-1, i0c + 1) * frac) * valid


def mulaw(x, bits=8):
    mu = 2 ** bits - 1
    pk = x.abs().amax(-1, keepdim=True).detach() + 1e-9
    u = x / pk
    c = torch.sign(u) * torch.log1p(mu * u.abs()) / math.log1p(mu)
    q = c + (torch.round(c * (mu / 2)) / (mu / 2) - c).detach()
    return torch.sign(q) * ((1 + mu) ** q.abs() - 1) / mu * pk


def soft_clip(x, drive):
    """Loudspeaker / amplifier saturation: tanh at a drive relative to the peak."""
    pk = x.abs().amax(-1, keepdim=True).detach() + 1e-9
    g = drive[:, None] / pk
    return torch.tanh(g * x) / g


def pink(shape, g, device):
    w = torch.randn(shape, generator=g, device=device)
    Wf = torch.fft.rfft(w)
    f = torch.arange(Wf.shape[-1], device=device).clamp(min=1).float()
    return torch.fft.irfft(Wf / f.sqrt(), shape[-1])


def add_noise(y, noise, snr_db):
    py = y.pow(2).mean(-1, keepdim=True).detach()
    pn = noise.pow(2).mean(-1, keepdim=True) + 1e-12
    return y + noise * torch.sqrt(py / pn * 10 ** (-snr_db / 10))


def legacy_rirs(n, seed):
    """The original exponential-tail RIR generator (for continuity with the small-scale results)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        rt60 = 0.15 + 0.55 * torch.rand(1, generator=g).item()
        L = 8000
        t = torch.arange(L) / SR
        tail = torch.randn(L, generator=g) * torch.exp(-6.9 * t / rt60)
        tail[:int(0.003 * SR)] = 0
        h = 0.35 * tail / (tail.abs().max() + 1e-9)
        for _ in range(4):
            i = int(torch.randint(int(0.002 * SR), int(0.03 * SR), (1,), generator=g))
            h[i] += (0.2 + 0.5 * torch.rand(1, generator=g).item()) * (1 if torch.rand(1, generator=g) > 0.5 else -1)
        h[0] = 1.0
        out.append(h / h.norm())
    return torch.stack(out)


# ---------------------------------------------------------------- codecs (test only, not differentiable)
def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


CODEC_ARGS = {"mp3": ["-c:a", "libmp3lame", "-f", "mp3"], "opus": ["-c:a", "libopus", "-f", "ogg"],
              "aac": ["-c:a", "aac", "-f", "adts"]}


def codec(x: torch.Tensor, kind: str, bitrate: str) -> tuple[torch.Tensor, int]:
    """Round trip through ffmpeg; returns (y, measured delay in samples)."""
    exe = _ffmpeg(); n = x.shape[-1]

    def one(row):
        pcm = (np.clip(row, -1, 1) * 32767).astype("<i2").tobytes()
        a = subprocess.run([exe, "-v", "quiet", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                            *CODEC_ARGS[kind], "-b:a", bitrate, "-"], input=pcm, capture_output=True, check=True).stdout
        b = subprocess.run([exe, "-v", "quiet", "-i", "-", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-"],
                           input=a, capture_output=True, check=True).stdout
        y = np.frombuffer(b, "<i2").astype(np.float32) / 32767
        return np.pad(y, (0, max(0, n - len(y))))[:n]

    xc = x.detach().float().cpu().numpy()
    with ThreadPoolExecutor(8) as ex:
        y = np.stack(list(ex.map(one, xc)))
    # delay: cross-correlate the first row
    a, b = xc[0], y[0]
    L = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(b, L) * np.conj(np.fft.rfft(a, L)), L)
    d = int(np.argmax(cc[:4000]))
    return torch.from_numpy(y).to(x.device), d


# ---------------------------------------------------------------- channels
class Channel:
    """Randomised play-and-record channel. ``train`` mode is differentiable and broad; the
    named test conditions are fixed recipes drawn with a seeded generator."""

    def __init__(self, device, split: str = "train"):
        d = data_dir()
        self.dev = device
        self.rir = torch.from_numpy(np.load(os.path.join(d, f"rir_{split}.npy"))).to(device)
        self.rir_legacy = legacy_rirs(400 if split == "train" else 100, seed=1 if split == "train" else 777).to(device)
        self.noise = Corpus(f"noise_{split}")

    def _noise(self, B, n, g):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,), generator=g, device=g.device)))
        seg = torch.from_numpy(self.noise.segments(rng, B, n)).to(self.dev)
        return seg

    def train(self, x, g, strength=1.0, mode="full"):
        """x (B, n) on device. Returns y and the integer sample offset applied (B,).
        ``mode="legacy"`` is the ablation: only the original channel family (exponential-tail RIRs,
        band-pass + EQ, AWGN, mu-law, low-pass, drift), no loudspeaker saturation, image-source rooms,
        microphone response, background sounds or pink noise."""
        if mode == "legacy":
            return self._train_legacy(x, g, strength)
        B, n = x.shape
        dev = x.device
        r = lambda: torch.rand(B, generator=g, device=dev)
        p = lambda q: (r() < q * strength).float()[:, None]
        off = torch.randint(0, MAX_OFFSET + 1, (B,), generator=g, device=dev).float()
        speed = 1 + (0.004 * r() - 0.002) * p(0.5)[:, 0]
        y = warp(x, speed, off)
        # loudspeaker: band limits + EQ + saturation
        m = p(0.6)
        resp = band_response(80 + 320 * r(), 3500 + 4500 * r(), 6 * (2 * torch.rand(B, 4, generator=g, device=dev) - 1), n)
        y = m * freq_gain(y, resp) + (1 - m) * y
        m = p(0.3)
        y = m * soft_clip(y, 0.5 + 2.5 * r()) + (1 - m) * y
        # room
        m = p(0.7)
        bank = self.rir if torch.rand(1, generator=g, device=dev).item() < 0.7 else self.rir_legacy
        y = m * fft_conv(y, bank[torch.randint(0, len(bank), (B,), generator=g, device=dev)]) + (1 - m) * y
        # microphone
        m = p(0.4)
        resp = band_response(50 + 200 * r(), 5000 + 3000 * r(), 3 * (2 * torch.rand(B, 3, generator=g, device=dev) - 1), n)
        y = m * freq_gain(y, resp) + (1 - m) * y
        m = p(0.2)
        f = torch.fft.rfftfreq(n, 1 / SR, device=dev)[None]
        y = m * freq_gain(y, torch.sigmoid((3000 + 5000 * r()[:, None] - f) / 50)) + (1 - m) * y
        y = y * 10 ** ((12 * r()[:, None] - 6) / 20)
        # background sound + sensor noise
        m = p(0.6)
        y = y + m * (add_noise(y, self._noise(B, n, g), 5 + 30 * r()[:, None]) - y)
        m = p(0.6)
        w = torch.where(r()[:, None] < 0.5, torch.randn((B, n), generator=g, device=dev), pink((B, n), g, dev))
        y = y + m * (add_noise(y, w, 10 + 35 * r()[:, None]) - y)
        m = p(0.2)
        y = m * mulaw(y) + (1 - m) * y
        return y, off

    def _train_legacy(self, x, g, strength):
        B, n = x.shape
        dev = x.device
        r = lambda: torch.rand(B, generator=g, device=dev)
        p = lambda q: (r() < q * strength).float()[:, None]
        off = torch.randint(0, MAX_OFFSET + 1, (B,), generator=g, device=dev).float()
        speed = 1 + (0.02 * r() - 0.01) * p(0.5)[:, 0]
        y = warp(x, speed, off)
        m = p(0.5)
        y = m * fft_conv(y, self.rir_legacy[torch.randint(0, len(self.rir_legacy), (B,), generator=g, device=dev)]) + (1 - m) * y
        m = p(0.5)
        resp = band_response(100 + 200 * r(), 4000 + 4000 * r(), 4 * (2 * torch.rand(B, 3, generator=g, device=dev) - 1), n)
        y = m * freq_gain(y, resp) + (1 - m) * y
        m = p(0.3)
        f = torch.fft.rfftfreq(n, 1 / SR, device=dev)[None]
        y = m * freq_gain(y, torch.sigmoid((3000 + 5000 * r()[:, None] - f) / 50)) + (1 - m) * y
        y = y * 10 ** ((6 * r()[:, None] - 3) / 20)
        m = p(0.7)
        y = y + m * (add_noise(y, torch.randn((B, n), generator=g, device=dev), 5 + 35 * r()[:, None]) - y)
        m = p(0.3)
        y = m * mulaw(y) + (1 - m) * y
        return y, off

    TESTS = ("clean", "mp3_64k", "opus_32k", "reverb", "ota_legacy", "ota", "ota_mp3")

    def test(self, x, name: str, seed: int):
        """Fixed test recipe on long streams. Returns (y, offset_samples (B,), speed (B,))."""
        B, n = x.shape
        dev = x.device
        g = torch.Generator(device=dev).manual_seed(10_000 + seed)
        r = lambda: torch.rand(B, generator=g, device=dev)
        off = torch.randint(0, MAX_OFFSET + 1, (B,), generator=g, device=dev).float()
        speed = torch.ones(B, device=dev)
        if name in ("ota", "ota_mp3"):
            speed = 1 + (200e-6 * r() - 100e-6)              # +-100 ppm clock mismatch
        y = warp(x, speed, off)
        extra = 0
        if name == "reverb":
            y = fft_conv(y, self.rir[torch.randint(0, len(self.rir), (B,), generator=g, device=dev)])
        elif name == "ota_legacy":
            y = fft_conv(y, self.rir_legacy[torch.randint(0, len(self.rir_legacy), (B,), generator=g, device=dev)])
            y = freq_gain(y, band_response(150 + 150 * r(), 5000 + 2000 * r(), 3 * (2 * torch.rand(B, 3, generator=g, device=dev) - 1), n))
            y = add_noise(y, torch.randn((B, n), generator=g, device=dev), torch.full((B, 1), 20.0, device=dev))
            y = mulaw(y)
        elif name in ("ota", "ota_mp3"):
            y = freq_gain(y, band_response(100 + 250 * r(), 5000 + 3000 * r(), 5 * (2 * torch.rand(B, 4, generator=g, device=dev) - 1), n))
            y = soft_clip(y, 1.0 + 1.5 * r())
            y = fft_conv(y, self.rir[torch.randint(0, len(self.rir), (B,), generator=g, device=dev)])
            y = freq_gain(y, band_response(60 + 140 * r(), 6000 + 2000 * r(), 2 * (2 * torch.rand(B, 3, generator=g, device=dev) - 1), n))
            y = add_noise(y, self._noise(B, n, g), 15 + 10 * r()[:, None])
            y = add_noise(y, pink((B, n), g, dev), torch.full((B, 1), 35.0, device=dev))
        y = y * 10 ** ((6 * r()[:, None] - 3) / 20)
        y = y.clamp(-1, 1)
        if name == "mp3_64k" or name == "ota_mp3":
            y, extra = codec(y, "mp3", "64k")
        elif name == "opus_32k":
            y, extra = codec(y, "opus", "32k")
        return y, off + extra, speed


# ---------------------------------------------------------------- models
def logmag(X):
    lm = torch.log(X.abs() + 1e-5)
    return (lm - lm.mean((-1, -2), keepdim=True)) / (lm.std((-1, -2), keepdim=True) + 1e-5)


class Res(nn.Module):
    def __init__(self, ch, dil=(1, 1)):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch); self.c1 = nn.Conv2d(ch, ch, 3, padding=dil, dilation=dil)
        self.n2 = nn.GroupNorm(8, ch); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.gelu(self.n2(self.c1(F.gelu(self.n1(x))))))


class Encoder(nn.Module):
    """(log|X|, payload) -> complex STFT mask, watermark W = M * X (host-shaped by construction).
    The payload is a per-frequency bias (time invariant) injected before every residual block."""

    def __init__(self, ch=64, nblk=6, rho=1.0):
        super().__init__()
        self.ch, self.rho = ch, rho
        self.bits = nn.Linear(K, ch * NF)
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        dils = [(1, 1), (2, 2), (4, 4), (8, 2), (1, 1), (2, 4)][:nblk] + [(1, 1)] * max(0, nblk - 6)
        self.blocks = nn.ModuleList([Res(ch, d) for d in dils])
        self.gates = nn.Parameter(torch.ones(nblk))
        self.out = nn.Conv2d(ch, 2, 1)

    def forward(self, host, bits):
        X = stft(host)
        b = self.bits(2 * bits.float() - 1).view(-1, self.ch, NF, 1)
        h = self.inp(logmag(X)[:, None])
        for gt, blk in zip(self.gates, self.blocks):
            h = blk(h + gt * b)
        m = self.rho * torch.tanh(self.out(h).float())
        return istft(torch.complex(m[:, 0], m[:, 1]) * X, host.shape[-1])


class Decoder(nn.Module):
    """(log|Y|, unit phasor Re/Im) -> conv trunk (B, C, 17, T) -> mean over time -> head -> K logits.
    ``net`` / ``head`` keep the original interface so the legacy dense sliding decoder works."""

    def __init__(self, chs=(64, 96, 128, 128), c_out=64, hid=512):
        super().__init__()
        layers, cin = [], 3
        for c in chs:
            layers += [nn.Conv2d(cin, c, 3, stride=(2, 1), padding=1), nn.GELU(), Res(c, (1, 2))]
            cin = c
        layers += [nn.GroupNorm(8, cin), nn.GELU(), nn.Conv2d(cin, c_out, 1)]
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(c_out * 17, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, K))

    @staticmethod
    def features_in(y):
        Y = stft(y)
        u = Y / (Y.abs() + 1e-3 * Y.abs().mean((-1, -2), keepdim=True) + 1e-9)
        return torch.stack([logmag(Y), u.real, u.imag], 1)

    def frames(self, y):
        """Per-frame features (B, T, D)."""
        h = self.net(self.features_in(y))
        return h.flatten(1, 2).transpose(1, 2)

    def forward(self, y):
        return self.head(self.frames(y).mean(1))


def scale_to_whr(w, host, whr_db):
    pw = w.pow(2).mean(-1, keepdim=True) + 1e-12
    ph = host.pow(2).mean(-1, keepdim=True)
    return w * torch.sqrt(ph * 10 ** (whr_db / 10) / pw)


def lsd(host, w):
    H = stft(host, 512, 128).abs().pow(2); Y = stft(host + w, 512, 128).abs().pow(2)
    return (10 * torch.log10((Y + 1e-8) / (H + 1e-8))).pow(2).mean(-2).add(1e-9).sqrt().mean(-1)


def embed_chunks(enc, host, bits, whr_db):
    """host (B, S*N) aligned on the chunk grid, bits (B, S, K) -> watermark (B, S*N)."""
    B, S, _ = bits.shape
    h = host.reshape(B * S, N)
    w = scale_to_whr(enc(h, bits.reshape(B * S, K)), h, whr_db)
    return w.reshape(B, S * N)
