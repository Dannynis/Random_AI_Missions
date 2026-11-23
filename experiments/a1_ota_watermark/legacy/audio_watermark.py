"""Audio watermarking: classical spread spectrum vs a learned, channel-trained embedder.

A new concept for this repo (2026-10-14). The CQT fingerprinting work
(``cqt_widefan_matcher_out.ipynb``) identifies audio *passively*; the
adversarial-Whisper work hides an imperceptible perturbation that has to
survive a simulated over-the-air channel. A watermark is the active,
cooperative version of both: hide a K-bit payload in a host recording at a
fixed loudness budget and read it back after the recording has gone through
a channel (offset, reverb, band-limiting, noise, quantisation, codec).

Everything works on 2 s mono clips at 16 kHz and is written in torch so the
same channel simulator serves as an Expectation-over-Transformation (EoT)
layer for training and as the test harness.

Methods (each watermark is scaled to a watermark-to-host ratio, WHR; the
benchmark picks a per-method WHR that gives equal audibility, see
:func:`calibrate_whr`):

* ``dsss``          white +/-1 PN code per bit spread over the whole clip
                    (code division), plain correlation decoder. Baseline.
* ``dsss_ltas``     same embedding, decoder whitens with the received
                    long-term average spectrum (time-invariant, so the PN
                    references are whitened the same way).
* ``dsss_stftw``    same embedding, decoder whitens every STFT frame by its
                    own smoothed magnitude (time-varying, reference left white).
* ``shaped``        PN spectrum multiplied by the host's STFT magnitude
                    (perceptually shaped, Kirovski & Malvar style) + the
                    STFT-whitening decoder.
* ``ripple``        time-invariant log-magnitude ripple: each bit adds a
                    random +/-1 pattern over frequency to a multiplicative
                    gain (phase free); decoder correlates the frame-averaged,
                    detrended log spectrum with each pattern.
* ``learned``       a small 2-D conv encoder predicts a complex STFT mask
                    M(t, f) from (log|X|, payload) and embeds W = M * X; a conv
                    decoder pools over time to K logits; trained end to end
                    through the random channel below.

Data: LibriSpeech test-clean (speech) and ESC-50 shard 0 (environmental
sounds), downloaded at runtime and cached as an int16 npz; hosts are split by
speaker / ESC fold.
"""

from __future__ import annotations

import io
import math
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
DUR = 2.0
N = int(SR * DUR)
K = 32                     # payload bits per 2 s clip (16 bit/s)
N_FFT, HOP = 512, 128
MAX_OFFSET = 1600          # random playback offset, samples (100 ms)
SEARCH = 3200              # DSSS lag search range (covers offset + codec delay)
HOST_RMS = 0.05

CACHE_DIR = os.environ.get("WM_CACHE", os.path.expanduser("~/.cache/audio_wm"))
HF = "https://huggingface.co/datasets"
URLS = {
    "ls_test.parquet": f"{HF}/openslr/librispeech_asr/resolve/main/all/test.clean/0000.parquet",
    "esc0.parquet": f"{HF}/ashraq/esc50/resolve/main/data/train-00000-of-00002-2f1ab7b824ec751f.parquet",
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _download(name: str) -> str:
    path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(CACHE_DIR, exist_ok=True)
        urllib.request.urlretrieve(URLS[name], path + ".part")
        os.replace(path + ".part", path)
    return path


def _decode(b: bytes) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        g = math.gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def _crops(y: np.ndarray, max_crops: int) -> list[np.ndarray]:
    """Non-overlapping 2 s crops that are mostly active (>= 50 % frames within 30 dB of peak)."""
    out = []
    for s in range(0, len(y) - N + 1, N):
        c = y[s:s + N]
        fr = (c[: N // 400 * 400].reshape(-1, 400) ** 2).mean(1)
        e = 10 * np.log10(fr + 1e-12)
        if (e > e.max() - 30).mean() >= 0.5 and fr.max() > 1e-7:
            out.append(c * (HOST_RMS / (np.sqrt((c ** 2).mean()) + 1e-9)))
        if len(out) >= max_crops:
            break
    return out


def build_cache() -> str:
    """Decode the public sources into ``CACHE_DIR/wm_hosts.npz`` (2 s, 16 kHz crops)."""
    out = os.path.join(CACHE_DIR, "wm_hosts.npz")
    if os.path.exists(out):
        return out
    import pyarrow.parquet as pq
    clips, src, group = [], [], []
    t = pq.read_table(_download("ls_test.parquet"), columns=["audio", "speaker_id", "id"]).to_pylist()
    per_spk: dict[int, int] = {}
    for r in sorted(t, key=lambda r: r["id"]):
        s = int(r["speaker_id"])
        if per_spk.get(s, 0) >= 40:
            continue
        for c in _crops(_decode(r["audio"]["bytes"]), 4):
            clips.append(c); src.append(0); group.append(s)
            per_spk[s] = per_spk.get(s, 0) + 1
    t = pq.read_table(_download("esc0.parquet"), columns=["audio", "fold", "category"]).to_pylist()
    for r in t:
        for c in _crops(_decode(r["audio"]["bytes"]), 2):
            clips.append(c); src.append(1); group.append(int(r["fold"]))
    x = np.stack(clips)
    sc = 32767 / max(1e-6, np.abs(x).max())
    np.savez_compressed(out, x=(x * sc).astype(np.int16), scale=np.float32(sc),
                        src=np.array(src, np.int8), group=np.array(group, np.int32))
    return out


@dataclass
class Hosts:
    """Host clips split into train / test by LibriSpeech speaker and ESC-50 fold."""
    train: torch.Tensor
    test: torch.Tensor
    test_src: np.ndarray          # 0 = speech, 1 = ESC-50
    info: dict = field(default_factory=dict)


def load_hosts(test_spk_frac: float = 0.3, test_folds=(3,), seed: int = 0) -> Hosts:
    d = np.load(build_cache())
    x = d["x"].astype(np.float32) / d["scale"]
    src, grp = d["src"], d["group"]
    rng = np.random.default_rng(seed)
    spk = np.unique(grp[src == 0])
    test_spk = set(rng.choice(spk, int(round(len(spk) * test_spk_frac)), replace=False).tolist())
    is_test = np.where(src == 0, np.isin(grp, list(test_spk)), np.isin(grp, list(test_folds)))
    info = {"n_train": int((~is_test).sum()), "n_test": int(is_test.sum()),
            "n_speakers": int(len(spk)), "n_test_speakers": len(test_spk),
            "n_train_speech": int(((~is_test) & (src == 0)).sum()),
            "n_test_speech": int((is_test & (src == 0)).sum())}
    return Hosts(torch.from_numpy(x[~is_test]), torch.from_numpy(x[is_test]), src[is_test], info)


# --------------------------------------------------------------------------
# STFT helpers and budget
# --------------------------------------------------------------------------

_WIN = torch.hann_window(N_FFT)


def stft(x: torch.Tensor) -> torch.Tensor:
    return torch.stft(x, N_FFT, HOP, window=_WIN, return_complex=True)


def istft(X: torch.Tensor, length: int = N) -> torch.Tensor:
    return torch.istft(X, N_FFT, HOP, window=_WIN, length=length)


def scale_to_whr(w: torch.Tensor, host: torch.Tensor, whr_db: float) -> torch.Tensor:
    """Scale each watermark so that power(w) / power(host) = whr_db exactly."""
    pw = w.pow(2).mean(-1, keepdim=True) + 1e-12
    ph = host.pow(2).mean(-1, keepdim=True)
    return w * torch.sqrt(ph * 10 ** (whr_db / 10) / pw)


def audibility(host: torch.Tensor, w: torch.Tensor, margin_db: float = 10.0) -> dict:
    """Crude masking proxy on a 512-pt STFT.

    ``exposed``: fraction of T-F bins (of bins with any host or watermark energy)
    where the watermark is within ``margin_db`` of the host in that bin, i.e.
    is not buried under it. ``lsd``: log-spectral distance (dB) between host and
    host + watermark. Lower is less audible for both.
    """
    H = stft(host).abs().pow(2); Wm = stft(w).abs().pow(2); Y = stft(host + w).abs().pow(2)
    act = (H + Wm) > 1e-10
    exposed = ((Wm > H * 10 ** (-margin_db / 10)) & act).float().sum((-1, -2)) / act.float().sum((-1, -2))
    lsd = (10 * torch.log10((Y + 1e-8) / (H + 1e-8))).pow(2).mean(-2).sqrt().mean(-1)
    return {"exposed": exposed, "lsd": lsd}


def lsd_torch(host: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Differentiable per-clip log-spectral distance, same definition as :func:`audibility`."""
    H = stft(host).abs().pow(2); Y = stft(host + w).abs().pow(2)
    return (10 * torch.log10((Y + 1e-8) / (H + 1e-8))).pow(2).mean(-2).add(1e-9).sqrt().mean(-1)


# --------------------------------------------------------------------------
# Channel simulator (torch, batched, differentiable except codecs)
# --------------------------------------------------------------------------

def synth_rir(g: torch.Generator, rt60: float | None = None, length: int = 8000) -> torch.Tensor:
    """Exponentially decaying noise tail + a few early reflections, direct path at 0."""
    if rt60 is None:
        rt60 = 0.15 + 0.55 * torch.rand(1, generator=g).item()
    t = torch.arange(length) / SR
    tail = torch.randn(length, generator=g) * torch.exp(-6.9 * t / rt60)
    tail[:int(0.003 * SR)] = 0
    h = 0.35 * tail / (tail.abs().max() + 1e-9)
    for _ in range(4):
        i = int(torch.randint(int(0.002 * SR), int(0.03 * SR), (1,), generator=g))
        h[i] += (0.2 + 0.5 * torch.rand(1, generator=g).item()) * (1 if torch.rand(1, generator=g) > 0.5 else -1)
    h[0] = 1.0
    return h / h.norm()


def rir_bank(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.stack([synth_rir(g) for _ in range(n)])


def fft_conv(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1] + h.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(h, n), n)[..., :x.shape[-1]]


def freq_gain(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Apply a per-sample magnitude response ``gain`` (B, N//2+1)."""
    return torch.fft.irfft(torch.fft.rfft(x) * gain, x.shape[-1])


def band_response(lo: torch.Tensor, hi: torch.Tensor, eq_db: torch.Tensor | None = None) -> torch.Tensor:
    """Smooth band-pass (2nd-order-like roll-offs) with optional cosine-series EQ (B, 3) in dB."""
    f = torch.fft.rfftfreq(N, 1 / SR)[None]
    r = 1 / torch.sqrt(1 + (lo[:, None] / (f + 1e-3)) ** 4) / torch.sqrt(1 + (f / hi[:, None]) ** 8)
    if eq_db is not None:
        u = f / (SR / 2)
        basis = torch.stack([torch.cos(math.pi * (k + 1) * u) for k in range(eq_db.shape[1])], -1)
        r = r * 10 ** ((basis * eq_db[:, None, :]).sum(-1) / 20)
    return r


def warp(x: torch.Tensor, speed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """y[n] = x[(n - offset) * speed] by linear interpolation; zeros outside."""
    n = torch.arange(x.shape[-1], dtype=torch.float32)[None]
    pos = (n - offset[:, None]) * speed[:, None]
    i0 = pos.floor().long(); frac = pos - i0
    valid = (i0 >= 0) & (i0 < x.shape[-1] - 1)
    i0c = i0.clamp(0, x.shape[-1] - 2)
    y = x.gather(-1, i0c) * (1 - frac) + x.gather(-1, i0c + 1) * frac
    return y * valid


def mulaw(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Peak-normalised mu-law quantisation with a straight-through gradient."""
    mu = 2 ** bits - 1
    pk = x.abs().amax(-1, keepdim=True).detach() + 1e-9
    u = x / pk
    c = torch.sign(u) * torch.log1p(mu * u.abs()) / math.log1p(mu)
    q = c + (torch.round(c * (mu / 2)) / (mu / 2) - c).detach()
    return torch.sign(q) * ((1 + mu) ** q.abs() - 1) / mu * pk


def codec(x: torch.Tensor, kind: str = "mp3", bitrate: str = "64k") -> torch.Tensor:
    """Round-trip through ffmpeg (not differentiable). Encoder delay is kept as part of the channel."""
    from concurrent.futures import ThreadPoolExecutor
    enc = {"mp3": ["-c:a", "libmp3lame", "-f", "mp3"], "opus": ["-c:a", "libopus", "-f", "ogg"]}[kind]

    def one(row: np.ndarray) -> np.ndarray:
        pcm = (np.clip(row, -1, 1) * 32767).astype("<i2").tobytes()
        a = subprocess.run(["ffmpeg", "-v", "quiet", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                            *enc, "-b:a", bitrate, "-"], input=pcm, capture_output=True, check=True).stdout
        b = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "-", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-"],
                           input=a, capture_output=True, check=True).stdout
        y = np.frombuffer(b, "<i2").astype(np.float32) / 32767
        return np.pad(y, (0, max(0, N - len(y))))[:N]

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(one, x.detach().numpy()))
    return torch.from_numpy(np.stack(out))


@dataclass
class Condition:
    """One fixed test channel. Every condition also applies a random offset and gain."""
    name: str
    snr_db: float | None = None
    lowpass_hz: float | None = None
    rir: bool = False
    band: bool = False
    mulaw: bool = False
    speed: float = 1.0
    codec: str | None = None        # "mp3" or "opus"
    bitrate: str = "64k"


TEST_CONDITIONS = [
    Condition("none"),
    Condition("awgn 20 dB", snr_db=20),
    Condition("awgn 10 dB", snr_db=10),
    Condition("low-pass 4 kHz", lowpass_hz=4000),
    Condition("mu-law 8 bit", mulaw=True),
    Condition("reverb", rir=True),
    Condition("speed +0.5 %", speed=1.005),
    Condition("mp3 64k", codec="mp3", bitrate="64k"),
    Condition("over-the-air", rir=True, band=True, snr_db=20, mulaw=True),
]


def apply_condition(x: torch.Tensor, c: Condition, g: torch.Generator, rirs: torch.Tensor) -> torch.Tensor:
    """Apply test condition ``c`` to a batch; randomness (offset, gain, RIR pick, noise) from ``g``."""
    B = x.shape[0]
    off = torch.randint(0, MAX_OFFSET + 1, (B,), generator=g).float()
    y = warp(x, torch.full((B,), c.speed), off)
    if c.rir:
        idx = torch.randint(0, len(rirs), (B,), generator=g)
        y = fft_conv(y, rirs[idx])
    if c.band:
        lo = 150 + 150 * torch.rand(B, generator=g); hi = 5000 + 2000 * torch.rand(B, generator=g)
        y = freq_gain(y, band_response(lo, hi, 3 * (2 * torch.rand(B, 3, generator=g) - 1)))
    if c.lowpass_hz:
        f = torch.fft.rfftfreq(N, 1 / SR)
        y = freq_gain(y, (f <= c.lowpass_hz).float()[None].expand(B, -1))
    y = y * 10 ** ((6 * torch.rand(B, 1, generator=g) - 3) / 20)
    if c.snr_db is not None:
        p = y.pow(2).mean(-1, keepdim=True)
        y = y + torch.randn(y.shape, generator=g) * torch.sqrt(p * 10 ** (-c.snr_db / 10))
    if c.mulaw:
        y = mulaw(y)
    if c.codec:
        y = codec(y, c.codec, c.bitrate)
    return y


def random_channel(x: torch.Tensor, g: torch.Generator, rirs: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """EoT training channel: each op on a random subset, probabilities scaled by ``strength`` in [0, 1].

    Covers offset (always), speed 0.99-1.01, train-bank RIRs, band + EQ, low-pass 3-8 kHz,
    gain, AWGN 5-40 dB, mu-law 8 bit. Codecs are not used (not differentiable): MP3 is a
    truly unseen attack at test time.
    """
    B = x.shape[0]
    r = lambda: torch.rand(B, generator=g)
    p = lambda q: (r() < q * strength).float()[:, None]
    off = torch.randint(0, MAX_OFFSET + 1, (B,), generator=g).float()
    speed = 1 + (0.02 * r() - 0.01) * p(0.5)[:, 0]
    y = warp(x, speed, off)
    m = p(0.5)
    y = m * fft_conv(y, rirs[torch.randint(0, len(rirs), (B,), generator=g)]) + (1 - m) * y
    m = p(0.5)
    resp = band_response(100 + 200 * r(), 4000 + 4000 * r(), 4 * (2 * torch.rand(B, 3, generator=g) - 1))
    y = m * freq_gain(y, resp) + (1 - m) * y
    m = p(0.3)
    f = torch.fft.rfftfreq(N, 1 / SR)[None]
    lp = torch.sigmoid((3000 + 5000 * r()[:, None] - f) / 50)
    y = m * freq_gain(y, lp) + (1 - m) * y
    y = y * 10 ** ((6 * r()[:, None] - 3) / 20)
    m = p(0.7)
    snr = 5 + 35 * r()[:, None]
    pw = y.pow(2).mean(-1, keepdim=True).detach()
    y = y + m * torch.randn(y.shape, generator=g) * torch.sqrt(pw * 10 ** (-snr / 10))
    m = p(0.3)
    y = m * mulaw(y) + (1 - m) * y
    return y


# --------------------------------------------------------------------------
# Classical spread spectrum
# --------------------------------------------------------------------------

_KEYS: dict[int, int] = {}


def pn_codes(key: int, k: int = K) -> torch.Tensor:
    g = torch.Generator().manual_seed(10_000 + key)
    c = (torch.randint(0, 2, (k, N), generator=g) * 2 - 1).float()
    _KEYS[id(c)] = key
    return c


def codes_key(codes: torch.Tensor) -> int:
    """Recover the integer key a PN code set was made from (so every method shares one key)."""
    return _KEYS[id(codes)]


def dsss_embed(host: torch.Tensor, bits: torch.Tensor, codes: torch.Tensor, whr_db: float,
               shaped: bool = False) -> torch.Tensor:
    """Return the watermark w (not host + w). bits in {0,1}, shape (B, K)."""
    w = (2 * bits.float() - 1) @ codes                       # (B, N)
    if shaped:
        X = stft(host)
        w = istft(stft(w) * X.abs() / (X.abs().pow(2).mean((-1, -2), keepdim=True).sqrt() + 1e-9))
    return scale_to_whr(w, host, whr_db)


def _xcorr_lags(y: torch.Tensor, refs: torch.Tensor, max_lag: int) -> torch.Tensor:
    """c[b, k, d] = sum_n y[b, n + d] * refs[(b,) k, n] for d in [0, max_lag]."""
    n = y.shape[-1] + refs.shape[-1]
    Y = torch.fft.rfft(y, n)[:, None]
    R = torch.fft.rfft(refs, n)
    R = R[None] if R.dim() == 2 else R
    return torch.fft.irfft(Y * R.conj(), n)[..., :max_lag + 1]


def stft_whiten(y: torch.Tensor, smooth: int = 9) -> torch.Tensor:
    """Divide each STFT bin by a frequency-smoothed magnitude of the same frame."""
    Y = stft(y)
    mag = F.avg_pool1d(Y.abs().transpose(1, 2).reshape(-1, 1, Y.shape[1]), smooth, 1, smooth // 2)
    mag = mag.reshape(Y.shape[0], Y.shape[2], Y.shape[1]).transpose(1, 2)
    return istft(Y / (mag + 1e-3 * mag.mean((-1, -2), keepdim=True) + 1e-9), y.shape[-1])


def ltas_whiten(y: torch.Tensor, refs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Whiten y and the PN refs with 1/sqrt(PSD_y) (smoothed long-term spectrum of y)."""
    Yf = torch.fft.rfft(y)
    psd = F.avg_pool1d(Yf.abs().pow(2)[:, None], 65, 1, 32)[:, 0]
    h = 1 / torch.sqrt(psd + 1e-6 * psd.mean(-1, keepdim=True))
    yw = torch.fft.irfft(Yf * h, y.shape[-1])
    rw = torch.fft.irfft(torch.fft.rfft(refs)[None] * h[:, None], y.shape[-1])
    return yw, rw


def dsss_decode(y: torch.Tensor, codes: torch.Tensor, whiten: str = "none",
                max_lag: int = SEARCH) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (soft bit scores (B, K), detection statistic (B,)).

    Lag search: the lag maximising sum_k c_k(lag)^2 (energy across all codes);
    soft scores are the correlations at that lag, normalised by their std.
    """
    refs = codes
    if whiten == "ltas":
        y, refs = ltas_whiten(y, codes)
    elif whiten == "stft":
        y = stft_whiten(y)
    c = _xcorr_lags(y, refs, max_lag)                      # (B, K, L)
    e = c.pow(2).sum(1)                                    # (B, L)
    lag = e.argmax(-1)
    cb = c.gather(-1, lag[:, None, None].expand(-1, c.shape[1], 1))[..., 0]
    z = cb / (c.std((-1,), keepdim=False) + 1e-9)
    det = (e.max(-1).values - e.median(-1).values) / (e.std(-1) + 1e-9)
    return z, det


def ripple_patterns(key: int, k: int = K, f_lo: float = 200.0, f_hi: float = 7000.0) -> torch.Tensor:
    """K random +/-1 patterns over the STFT bins in [f_lo, f_hi] (zero elsewhere), shape (K, F)."""
    g = torch.Generator().manual_seed(20_000 + key)
    f = torch.fft.rfftfreq(N_FFT, 1 / SR)
    band = ((f >= f_lo) & (f <= f_hi)).float()
    return (torch.randint(0, 2, (k, len(f)), generator=g) * 2 - 1).float() * band


def ripple_embed(host: torch.Tensor, bits: torch.Tensor, pats: torch.Tensor, whr_db: float) -> torch.Tensor:
    """Time-invariant spectral ripple: W = X * G(f), G = sum_k s_k q_k(f) (a multiplicative 1 + G gain).

    Phase-free and identical in every frame, so it is unaffected by offsets and,
    to first order, by time-invariant filters whose response is smooth over a few bins.
    """
    G = (2 * bits.float() - 1) @ pats                      # (B, F)
    X = stft(host)
    return scale_to_whr(istft(X * G[:, :, None]), host, whr_db)


def ripple_decode(y: torch.Tensor, pats: torch.Tensor, smooth: int = 9) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean log-magnitude over frames, minus its moving average over frequency, correlated with each pattern."""
    Y = stft(y).abs()
    act = Y.pow(2).sum(1, keepdim=True)
    w = (act > 1e-3 * act.amax(-1, keepdim=True)).float()             # skip silent frames
    L = (torch.log(Y + 1e-6) * w).sum(-1) / w.sum(-1).clamp_min(1)    # (B, F)
    Ld = L - F.avg_pool1d(L[:, None], smooth, 1, smooth // 2, count_include_pad=False)[:, 0]
    sc = Ld @ pats.T                                                  # (B, K)
    return sc, sc.pow(2).sum(-1) / (Ld.pow(2).sum(-1) * pats.abs().sum(-1).mean() + 1e-9)


# --------------------------------------------------------------------------
# Learned embedder / decoder
# --------------------------------------------------------------------------

def _logmag(X: torch.Tensor) -> torch.Tensor:
    lm = torch.log(X.abs() + 1e-5)
    return (lm - lm.mean((-1, -2), keepdim=True)) / (lm.std((-1, -2), keepdim=True) + 1e-5)


MODEL_HOP = 256            # the learned models use a coarser STFT (half the frames) for speed


def mstft(x: torch.Tensor) -> torch.Tensor:
    return torch.stft(x, N_FFT, MODEL_HOP, window=_WIN, return_complex=True)


def mistft(X: torch.Tensor, length: int = N) -> torch.Tensor:
    return torch.istft(X, N_FFT, MODEL_HOP, window=_WIN, length=length)


class Encoder(nn.Module):
    """(log|X|, payload) -> complex mask M (2 channels, rho * tanh); watermark W = M * X.

    ``rho`` < 1 bounds the per-bin gain change |1 + M| away from 0, so the encoder
    cannot notch host bins out (which the power-budget model learned to do).

    The payload enters as a learned per-frequency bias (ch x F) added after the
    first conv, i.e. the same bit pattern is written into every frame, so the
    decoder can average it over time and tolerate offsets and cropping.
    """

    def __init__(self, k: int = K, ch: int = 16, n_freq: int = N_FFT // 2 + 1, rho: float = 1.0):
        super().__init__()
        self.ch, self.n_freq, self.rho = ch, n_freq, rho
        self.bits = nn.Linear(k, ch * n_freq)
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=(2, 2), dilation=2), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=(4, 4), dilation=4), nn.GELU(),
            nn.Conv2d(ch, 2, 1))

    def forward(self, host: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        X = mstft(host)
        b = self.bits(2 * bits.float() - 1).view(-1, self.ch, self.n_freq, 1)
        m = self.rho * torch.tanh(self.net(self.inp(_logmag(X)[:, None]) + b))
        return mistft(torch.complex(m[:, 0], m[:, 1]) * X)


class Decoder(nn.Module):
    """(log|Y|, whitened Re/Im) -> conv -> mean over time -> K logits (shift tolerant)."""

    def __init__(self, k: int = K, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, stride=(2, 1), padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=(2, 1), padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=(2, 1), padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=(2, 1), padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(ch * 17, 256), nn.GELU(), nn.Linear(256, k))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        Y = mstft(y)
        u = Y / (Y.abs() + 1e-3 * Y.abs().mean((-1, -2), keepdim=True) + 1e-9)
        h = self.net(torch.stack([_logmag(Y), u.real, u.imag], 1))
        return self.head(h.mean(-1).flatten(1))


@dataclass
class TrainLog:
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    ber: list[float] = field(default_factory=list)
    lsd: list[float] = field(default_factory=list)
    seconds: float = 0.0


def train_learned(train_hosts: torch.Tensor, whr_db: float, steps: int, seed: int, eot: bool = True,
                  batch: int = 16, lr: float = 2e-3, log_every: int = 50,
                  rirs: torch.Tensor | None = None, lsd_target: float | None = None,
                  lsd_weight: float = 1.0, rho: float = 1.0,
                  lsd_start: float | None = None) -> tuple[Encoder, Decoder, TrainLog]:
    """Train encoder + decoder with BCE on the payload through ``random_channel`` (if eot).

    Budget: every watermark is rescaled to ``whr_db`` (power budget). If ``lsd_target``
    is set, the loss also adds ``lsd_weight * relu(mean LSD - lsd_target)`` (dB): at a
    fixed power the encoder then has to learn *where* it can hide energy (audibility
    budget). ``rho`` bounds the mask (see :class:`Encoder`). With ``lsd_start`` the
    penalty is a curriculum: off for the first 25 % of steps, then the target anneals
    linearly from ``lsd_start`` to ``lsd_target`` until 75 % of steps. (In pilots, a
    penalty active from step 0 kept the code at chance: the encoder first finds a
    loud code and only then learns to hide it.)

    Channel strength ramps linearly from 0 to 1 over the first 40 % of steps (curriculum);
    without EoT only a random offset is applied.
    """
    import time
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if rirs is None:
        rirs = rir_bank(400, seed=1 + seed)
    enc, dec = Encoder(rho=rho), Decoder()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    log, t0, run_l, run_b, run_d = TrainLog(), time.time(), 0.0, 0.0, 0.0
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(train_hosts), (batch,), generator=g)
        host = train_hosts[idx] * torch.where(torch.rand(batch, 1, generator=g) < 0.5, -1.0, 1.0)
        bits = torch.randint(0, 2, (batch, K), generator=g)
        w = enc(host, bits)
        w = scale_to_whr(w, host, whr_db)
        x = host + w
        if eot:
            y = random_channel(x, g, rirs, strength=min(1.0, step / (0.4 * steps)))
        else:
            y = warp(x, torch.ones(batch), torch.randint(0, MAX_OFFSET + 1, (batch,), generator=g).float())
        logits = dec(y)
        loss = F.binary_cross_entropy_with_logits(logits, bits.float())
        if lsd_target is not None:
            lsd = lsd_torch(host, w).mean()
            frac = step / steps
            if lsd_start is None:
                tgt, on = lsd_target, 1.0
            else:
                a = min(1.0, max(0.0, (frac - 0.25) / 0.5))
                tgt, on = lsd_start + a * (lsd_target - lsd_start), float(frac >= 0.25)
            loss = loss + on * lsd_weight * F.relu(lsd - tgt)
            run_d += lsd.item()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step(); sched.step()
        run_l += loss.item(); run_b += ((logits > 0).long() != bits).float().mean().item()
        if step % log_every == 0:
            log.steps.append(step); log.loss.append(run_l / log_every); log.ber.append(run_b / log_every)
            log.lsd.append(run_d / log_every)
            run_l = run_b = run_d = 0.0
    log.seconds = time.time() - t0
    enc.eval(); dec.eval()
    return enc, dec, log


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

METHODS = ["dsss", "dsss_ltas", "dsss_stftw", "shaped", "ripple", "learned"]


def embed(method: str, host: torch.Tensor, bits: torch.Tensor, whr_db: float, codes: torch.Tensor,
          model: tuple | None = None) -> torch.Tensor:
    if method == "shaped":
        return dsss_embed(host, bits, codes, whr_db, shaped=True)
    if method == "ripple":
        return ripple_embed(host, bits, ripple_patterns(codes_key(codes)), whr_db)
    if method.startswith("dsss"):
        return dsss_embed(host, bits, codes, whr_db)
    with torch.no_grad():
        return scale_to_whr(model[0](host, bits), host, whr_db)


def decode(method: str, y: torch.Tensor, codes: torch.Tensor, model: tuple | None = None):
    """Return (soft scores > 0 means bit 1, detection statistic)."""
    if method == "ripple":
        return ripple_decode(y, ripple_patterns(codes_key(codes)))
    if method in ("dsss", "dsss_ltas", "dsss_stftw", "shaped"):
        wh = {"dsss": "none", "dsss_ltas": "ltas", "dsss_stftw": "stft", "shaped": "stft"}[method]
        return dsss_decode(y, codes, wh)
    with torch.no_grad():
        lg = model[1](y)
    return lg, lg.abs().mean(-1)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney statistic (ties count half)."""
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort").astype(float) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@dataclass
class EvalResult:
    method: str
    condition: str
    seed: int
    whr_db: float
    ber: float
    ber_per_clip: list
    msg_ok: float          # fraction of clips with every bit right
    det_auc: float
    ber_speech: float
    ber_esc: float


def evaluate(method: str, hosts: torch.Tensor, src: np.ndarray, cond: Condition, seed: int,
             whr_db: float, rirs: torch.Tensor, model: tuple | None = None, batch: int = 25) -> EvalResult:
    """Watermark every test host with a random payload, pass marked and unmarked copies
    through the same channel draw, decode, and score BER / message success / detection AUC."""
    codes = pn_codes(seed)
    g_bits = torch.Generator().manual_seed(100 + seed)
    bits_all = torch.randint(0, 2, (len(hosts), K), generator=g_bits)
    errs, dpos, dneg = [], [], []
    for s in range(0, len(hosts), batch):
        h = hosts[s:s + batch]; b = bits_all[s:s + batch]
        w = embed(method, h, b, whr_db, codes, model)
        # identical channel draws for marked and unmarked copies
        gm = torch.Generator().manual_seed(1000 * seed + s + 7)
        gu = torch.Generator().manual_seed(1000 * seed + s + 7)
        ym = apply_condition(h + w, cond, gm, rirs)
        yu = apply_condition(h, cond, gu, rirs)
        sm, dm = decode(method, ym, codes, model)
        _, du = decode(method, yu, codes, model)
        errs.append(((sm > 0).long() != b).float().mean(-1)); dpos.append(dm); dneg.append(du)
    e = torch.cat(errs).numpy()
    return EvalResult(method, cond.name, seed, whr_db, float(e.mean()), e.round(5).tolist(),
                      float((e == 0).mean()), auc(torch.cat(dpos).numpy(), torch.cat(dneg).numpy()),
                      float(e[src == 0].mean()), float(e[src == 1].mean()))


def audibility_table(method: str, hosts: torch.Tensor, whr_db: float, seed: int,
                     model: tuple | None = None) -> dict:
    codes = pn_codes(seed)
    bits = torch.randint(0, 2, (len(hosts), K), generator=torch.Generator().manual_seed(100 + seed))
    w = embed(method, hosts, bits, whr_db, codes, model)
    a = audibility(hosts, w)
    return {k: float(v.mean()) for k, v in a.items()}


def calibrate_whr(method: str, hosts: torch.Tensor, target_lsd: float, seed: int = 0,
                  model: tuple | None = None, lo: float = -80.0, hi: float = 0.0, iters: int = 14) -> float:
    """Bisect the watermark-to-host ratio (dB) at which mean LSD on ``hosts`` equals ``target_lsd``.

    This is how methods are put on an equal-audibility footing: a white watermark
    fills the host's quiet bins and pauses, so at equal WHR it is far more audible
    than a host-shaped one.
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        if audibility_table(method, hosts, mid, seed, model)["lsd"] > target_lsd:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
