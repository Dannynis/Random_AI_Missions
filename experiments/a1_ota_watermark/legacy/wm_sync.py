"""Explicit synchronisation for the chunked audio watermark (PR #19 physical layer, PR #33 link layer).

PR #33 turned the learned watermark into a message channel: a 16-bit message is
convolutionally coded over the 4 x 32 slots of four consecutive 2 s chunks and
decoded with calibrated LLRs + soft Viterbi (0.81 message success over the air).
Its receiver was handed each chunk already cut at the right place and knew which
chunk of the message cycle it was looking at. A real receiver gets a recording
that starts anywhere. This module adds the missing synchronisation layer:

* **Transmitter.** The 4-chunk codeword is looped over a continuous host stream
  (chunk ``j`` of the stream carries cycle chunk ``j mod P``), each chunk embedded by
  the unchanged PR #19 encoder on its own 2 s window. Optional *pilot* slots: the
  first ``n_pilot`` of each chunk's 32 slots carry a known +/-1 word that depends on
  the cycle position, the remaining slots carry the code.
* **Dense sliding decoder** (:func:`dense_logits`). The PR #19 decoder is
  conv -> mean over time -> MLP head. Running its conv trunk once over the whole
  recording and averaging the per-frame features over every 125-frame window
  (cumulative sums) gives the decoder's 32 logits for a 2 s window starting at
  *every* frame (16 ms grid) for the price of one pass.
* **Receivers** (:func:`receive`), all on the same window logits:

  ``naive``       PR #33's receiver as is: chunk boundary at the first sample, cycle phase 0.
  ``oracle``      true boundary and phase (upper bound for the same recording).
  ``blind``       timing = the 125-frame grid offset that maximises mean |logit|
                  (a window straddling two chunks mixes two bit patterns and loses
                  confidence), then cycle phase by maximum Viterbi path metric
                  (ML frame sync over the P cyclic hypotheses).
  ``joint``       exhaustive ML over (grid offset on a 4-frame grid, phase), Viterbi metric.
  ``pilot``       pilot correlation over (offset, phase); needs the pilot layout.
  ``+partial``    additionally uses the partial chunks at both ends of the recording
                  (window shortened to the frames inside, LLR scaled by the fraction).
  ``+track``      re-centres every window within +/- ``track`` frames on its own
                  confidence peak (speed drift).

Everything is torch for the physical layer and numpy for the link layer; the
coding helpers (calibration, interleaver, convolutional code) come from ``wm_coding``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

import audio_watermark as aw
import wm_coding as wc

FPC = aw.N // aw.MODEL_HOP          # decoder frames per 2 s chunk (125)
MP3_DELAY = 1105                    # samples; measured lame 64k round-trip delay at 16 kHz


# --------------------------------------------------------------------------
# Link-layer layout
# --------------------------------------------------------------------------

@dataclass
class Layout:
    """How a ``k_msg``-bit message is laid out over a ``P``-chunk cycle.

    ``n_pilot`` slots per chunk (the first ones) carry a known word that depends on
    the cycle position; the other ``K - n_pilot`` slots per chunk carry the
    convolutional codeword (randomly interleaved over the whole cycle, as in PR #33).
    """

    k_msg: int = 16
    P: int = 4
    n_pilot: int = 0
    seed: int = 0
    code: wc.ConvCode = field(init=False)
    il: wc.Interleaver = field(init=False)
    pilots: np.ndarray = field(init=False)       # (P, n_pilot) in {0, 1}

    def __post_init__(self):
        self.n_data = (aw.K - self.n_pilot) * self.P
        self.code = wc.ConvCode(self.k_msg, self.n_data)
        self.il = wc.Interleaver.make(self.n_data, "random", seed=self.seed + 7)
        rng = np.random.default_rng(4242 + self.n_pilot)
        self.pilots = rng.integers(0, 2, (self.P, self.n_pilot))

    @property
    def name(self) -> str:
        return f"pilot{self.n_pilot}" if self.n_pilot else "nopilot"

    def chunk_bits(self, m: np.ndarray) -> np.ndarray:
        """Messages (B, k_msg) -> slot bits of one cycle (B, P, K)."""
        B = m.shape[0]
        data = self.il.spread(self.code.encode(m)).reshape(B, self.P, aw.K - self.n_pilot)
        pil = np.broadcast_to(self.pilots[None], (B, self.P, self.n_pilot))
        return np.concatenate([pil, data], axis=2)

    def decode(self, data_llr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Data-slot LLRs of one cycle (B, n_data) in slot order -> (messages, path metric)."""
        return viterbi_metric(self.code, self.il.gather(data_llr))


def viterbi_metric(code: wc.ConvCode, llr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Soft Viterbi of ``wc.ConvCode`` that also returns the surviving path metric.

    The metric is ``sum_l L_l (2 c_l - 1)`` over the combined codeword LLRs of the
    best zero-tailed codeword: the ML statistic used to choose between sync hypotheses.
    """
    L = code._combine(llr, True)
    B, T = L.shape[0], code.k_msg + code.mem
    metric = np.full((B, code.n_state), -1e12)
    metric[:, 0] = 0.0
    back = np.zeros((B, T, code.n_state), dtype=np.int8)
    bip = 2 * code.out_prev - 1
    for t in range(T):
        bm = L[:, 2 * t, None, None] * bip[None, :, :, 0] + L[:, 2 * t + 1, None, None] * bip[None, :, :, 1]
        cand = metric[:, code.prev] + bm
        metric = cand.max(2)
        back[:, t] = cand.argmax(2)
    state = np.zeros(B, dtype=np.int64)
    bits = np.zeros((B, T), dtype=np.int64)
    rows = np.arange(B)
    for t in range(T - 1, -1, -1):
        bits[:, t] = code.in_bit[state]
        state = code.prev[state, back[rows, t, state].astype(np.int64)]
    return bits[:, :code.k_msg], metric[:, 0]


# --------------------------------------------------------------------------
# Transmitter and channel on long streams
# --------------------------------------------------------------------------

def embed_stream(host: torch.Tensor, slot_bits: np.ndarray, g0: np.ndarray, enc, whr_db: float,
                 batch: int = 50) -> torch.Tensor:
    """Embed chunk ``j`` (bits ``slot_bits[:, j]``) on samples ``[g0 + j N, g0 + (j+1) N)``.

    ``host`` (B, T) continuous host streams, ``slot_bits`` (B, S, K), ``g0`` (B,) grid
    offsets. Each chunk is embedded by the PR #19 encoder on its own 2 s window and
    scaled to ``whr_db`` against that window, exactly like a single clip.
    """
    B, S, _ = slot_bits.shape
    out = host.clone()
    win = torch.stack([torch.stack([host[b, g0[b] + j * aw.N: g0[b] + (j + 1) * aw.N] for j in range(S)])
                       for b in range(B)]).reshape(B * S, aw.N)
    bits = torch.from_numpy(np.ascontiguousarray(slot_bits.reshape(B * S, aw.K)))
    ws = []
    with torch.no_grad():
        for s in range(0, B * S, batch):
            w = enc(win[s:s + batch], bits[s:s + batch])
            ws.append(aw.scale_to_whr(w, win[s:s + batch], whr_db))
    w = torch.cat(ws).reshape(B, S, aw.N)
    for b in range(B):
        out[b, g0[b]: g0[b] + S * aw.N] += w[b].reshape(-1)
    return out


def _band_response(lo, hi, eq_db, n):
    f = torch.fft.rfftfreq(n, 1 / aw.SR)[None]
    r = 1 / torch.sqrt(1 + (lo[:, None] / (f + 1e-3)) ** 4) / torch.sqrt(1 + (f / hi[:, None]) ** 8)
    u = f / (aw.SR / 2)
    basis = torch.stack([torch.cos(math.pi * (k + 1) * u) for k in range(eq_db.shape[1])], -1)
    return r * 10 ** ((basis * eq_db[:, None, :]).sum(-1) / 20)


def _codec_long(x: torch.Tensor, bitrate: str = "64k") -> torch.Tensor:
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    n = x.shape[-1]

    def one(row):
        pcm = (np.clip(row, -1, 1) * 32767).astype("<i2").tobytes()
        a = subprocess.run(["ffmpeg", "-v", "quiet", "-f", "s16le", "-ar", str(aw.SR), "-ac", "1", "-i", "-",
                            "-c:a", "libmp3lame", "-f", "mp3", "-b:a", bitrate, "-"],
                           input=pcm, capture_output=True, check=True).stdout
        b = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "-", "-f", "s16le", "-ar", str(aw.SR), "-ac", "1", "-"],
                           input=a, capture_output=True, check=True).stdout
        y = np.frombuffer(b, "<i2").astype(np.float32) / 32767
        return np.pad(y, (0, max(0, n - len(y))))[:n]

    with ThreadPoolExecutor(4) as ex:
        return torch.from_numpy(np.stack(list(ex.map(one, x.numpy()))))


def channel_long(x: torch.Tensor, c: aw.Condition, g: torch.Generator, rirs: torch.Tensor):
    """``aw.apply_condition`` for streams of any length. Returns (y, offset, speed).

    Received sample ``n`` holds source sample ``(n - offset) * speed`` (plus the MP3
    delay for the codec condition), so a source position ``p`` arrives at
    ``p / speed + offset``.
    """
    B, n = x.shape
    off = torch.randint(0, aw.MAX_OFFSET + 1, (B,), generator=g).float()
    y = aw.warp(x, torch.full((B,), c.speed), off)
    if c.rir:
        y = aw.fft_conv(y, rirs[torch.randint(0, len(rirs), (B,), generator=g)])
    if c.band:
        lo = 150 + 150 * torch.rand(B, generator=g); hi = 5000 + 2000 * torch.rand(B, generator=g)
        y = aw.freq_gain(y, _band_response(lo, hi, 3 * (2 * torch.rand(B, 3, generator=g) - 1), n))
    if c.lowpass_hz:
        f = torch.fft.rfftfreq(n, 1 / aw.SR)
        y = aw.freq_gain(y, (f <= c.lowpass_hz).float()[None].expand(B, -1))
    y = y * 10 ** ((6 * torch.rand(B, 1, generator=g) - 3) / 20)
    if c.snr_db is not None:
        p = y.pow(2).mean(-1, keepdim=True)
        y = y + torch.randn(y.shape, generator=g) * torch.sqrt(p * 10 ** (-c.snr_db / 10))
    if c.mulaw:
        y = aw.mulaw(y)
    if c.codec:
        y = _codec_long(y, c.bitrate)
        off = off + MP3_DELAY
    return y, off.numpy().astype(np.float64), float(c.speed)


# --------------------------------------------------------------------------
# Dense sliding decoder
# --------------------------------------------------------------------------

def frame_features(dec, y: torch.Tensor, batch: int = 16) -> torch.Tensor:
    """PR #19 decoder trunk on a whole recording: (B, T) -> per-frame features (B, T_f, D).

    Run in mini-batches: the first conv's activations of a 16 s recording are ~17 MB per clip.
    """
    out = []
    with torch.no_grad():
        for s in range(0, y.shape[0], batch):
            Y = aw.mstft(y[s:s + batch])
            u = Y / (Y.abs() + 1e-3 * Y.abs().mean((-1, -2), keepdim=True) + 1e-9)
            h = dec.net(torch.stack([aw._logmag(Y), u.real, u.imag], 1))  # (b, ch, 17, T_f)
            out.append(h.flatten(1, 2).transpose(1, 2).contiguous())
    return torch.cat(out)


def window_logits(dec, feats: torch.Tensor, starts: np.ndarray, length: int = FPC,
                  min_frames: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Decoder logits for windows ``[s, s + length)`` clipped to the recording.

    ``starts`` (B, W) frame indices (may run off either end). Returns
    (logits (B, W, K), fraction of the window inside the recording (B, W)).
    """
    B, T, D = feats.shape
    cs = torch.cat([torch.zeros(B, 1, D), feats.cumsum(1)], 1)                # (B, T+1, D)
    s = torch.from_numpy(np.clip(starts, 0, T)).long()
    e = torch.from_numpy(np.clip(starts + length, 0, T)).long()
    cnt = (e - s).clamp(min=min_frames).float()
    gat = lambda idx: cs.gather(1, idx[..., None].expand(-1, -1, D))
    mean = (gat(e) - gat(s)) / cnt[..., None]
    with torch.no_grad():
        z = dec.head(mean.reshape(-1, D)).reshape(B, -1, aw.K)
    return z.numpy().astype(np.float64), ((e - s).float() / length).numpy()


def dense_logits(dec, feats: torch.Tensor, block: int = 128) -> np.ndarray:
    """Logits of full 2 s windows starting at every frame: (B, T_f - FPC + 1, K)."""
    B, T, _ = feats.shape
    nW = T - FPC + 1
    out = []
    for s in range(0, nW, block):          # blocks keep the gathered (B, W, D) tensors small
        idx = np.arange(s, min(nW, s + block))
        out.append(window_logits(dec, feats, np.broadcast_to(idx[None], (B, len(idx))))[0])
    return np.concatenate(out, 1)


# --------------------------------------------------------------------------
# Receivers
# --------------------------------------------------------------------------

@dataclass
class Truth:
    f0: np.ndarray          # (B,) frame (float) of the first chunk boundary >= 0 in the recording
    phase: np.ndarray       # (B,) cycle index of the chunk that starts at f0


def grid_starts(f: np.ndarray, T: int, partial: bool, min_frac: float = 0.25) -> np.ndarray:
    """Window starts ``f + FPC j`` (B,) offsets -> (B, W) with W fixed; invalid = -10**6."""
    jmin = -1 if partial else 0
    jmax = T // FPC + 1
    j = np.arange(jmin, jmax + 1)
    st = np.round(f[:, None] + FPC * j[None]).astype(np.int64)
    inside = np.clip(st + FPC, 0, T) - np.clip(st, 0, T)
    need = int(math.ceil(min_frac * FPC)) if partial else FPC
    st[inside < need] = -10 ** 6
    return st, j


def assemble(lay: Layout, llr: np.ndarray, j: np.ndarray, valid: np.ndarray, phase: np.ndarray):
    """Sum window LLRs (B, W, K) into one cycle's slots according to ``phase`` (B,)."""
    B, W, _ = llr.shape
    acc = np.zeros((B, lay.P, aw.K))
    cyc = (j[None, :] + phase[:, None]) % lay.P                               # (B, W)
    for w in range(W):
        np.add.at(acc, (np.arange(B), cyc[:, w]), llr[:, w] * valid[:, w, None])
    return acc


def decode_cycle(lay: Layout, acc: np.ndarray):
    data = acc[:, :, lay.n_pilot:].reshape(acc.shape[0], -1)
    return lay.decode(data)


def receive(method: str, lay: Layout, dec, cal: wc.LLRCalibration, feats: torch.Tensor,
            truth: Truth, partial: bool = False, track: int = 0, joint_step: int = 4) -> dict:
    """Run one receiver over a batch of recordings. Returns per-recording decisions."""
    B, T, _ = feats.shape
    P = lay.P

    def llrs_at(f, partial_=partial):
        st, j = grid_starts(f, T, partial_)
        ok = st > -10 ** 5
        if track:
            # re-centre each window on its own confidence peak within +/- track frames
            cand = st[..., None] + np.arange(-track, track + 1)[None, None]
            zc, frc = window_logits(dec, feats, cand.reshape(B, -1))
            conf = np.abs(zc).mean(-1).reshape(cand.shape) * (frc.reshape(cand.shape) >= 0.999)
            st = np.where(ok, np.take_along_axis(cand, conf.argmax(-1)[..., None], -1)[..., 0], st)
        z, frac = window_logits(dec, feats, np.where(ok, st, 0))
        llr = cal(z.reshape(-1, aw.K)).reshape(z.shape) * frac[..., None]
        return llr, j, ok

    if method == "oracle":
        f = truth.f0
        llr, j, ok = llrs_at(f)
        ph = truth.phase.astype(np.int64)
    elif method == "naive":
        f = np.zeros(B)
        llr, j, ok = llrs_at(f)
        ph = np.zeros(B, np.int64)
    elif method in ("blind", "blind_phase_oracle", "timing_oracle"):
        if method == "timing_oracle":
            f = truth.f0
        else:
            Z = dense_logits(dec, feats)                                      # (B, T-FPC+1, K)
            conf = np.abs(Z).mean(-1)
            nW = conf.shape[1]
            score = np.full((B, FPC), -np.inf)
            for o in range(FPC):
                idx = np.arange(o, nW, FPC)
                score[:, o] = conf[:, idx].mean(1)
            f = score.argmax(1).astype(np.float64)
        llr, j, ok = llrs_at(f)
        if method == "blind_phase_oracle":
            # the oracle phase refers to the true grid; shift by whole chunks if f landed one chunk off
            ph = (truth.phase + np.round((f - truth.f0) / FPC)).astype(np.int64) % P
        else:
            ph = _ml_phase(lay, llr, j, ok)
    elif method == "joint":
        best = np.full(B, -np.inf); f = np.zeros(B); ph = np.zeros(B, np.int64)
        for o in range(0, FPC, joint_step):
            fo = np.full(B, float(o))
            l_, j_, ok_ = llrs_at(fo)
            for p in range(P):
                _, met = decode_cycle(lay, assemble(lay, l_, j_, ok_, np.full(B, p)))
                better = met > best
                best[better] = met[better]; f[better] = o; ph[better] = p
        llr, j, ok = llrs_at(f)
    elif method == "pilot":
        if lay.n_pilot == 0:
            raise ValueError("pilot receiver needs a pilot layout")
        Z = dense_logits(dec, feats)
        Lp = cal(Z.reshape(-1, aw.K)).reshape(Z.shape)[..., :lay.n_pilot]   # (B, nW, n_pilot)
        sgn = 2.0 * lay.pilots - 1                                            # (P, n_pilot)
        corr = np.einsum("bwk,pk->bwp", Lp, sgn)                              # window w as cycle chunk p
        nW = corr.shape[1]
        score = np.full((B, FPC, P), -np.inf)
        for o in range(FPC):
            idx = np.arange(o, nW, FPC)
            jj = np.arange(len(idx))
            for p in range(P):
                score[:, o, p] = corr[:, idx, (jj + p) % P].sum(1)
        flat = score.reshape(B, -1).argmax(1)
        f = (flat // P).astype(np.float64); ph = flat % P
        llr, j, ok = llrs_at(f)
    else:
        raise ValueError(method)

    acc = assemble(lay, llr, j, ok, ph)
    msg, met = decode_cycle(lay, acc)
    L = lay.code._combine(lay.il.gather(acc[:, :, lay.n_pilot:].reshape(B, -1)), True)
    return {"msg": msg, "f": f, "phase": ph, "metric": met,
            "norm_metric": met / (np.abs(L).sum(1) + 1e-9), "n_windows": ok.sum(1)}


def _ml_phase(lay: Layout, llr, j, ok) -> np.ndarray:
    B = llr.shape[0]
    mets = np.stack([decode_cycle(lay, assemble(lay, llr, j, ok, np.full(B, p)))[1] for p in range(lay.P)], 1)
    return mets.argmax(1)


def timing_error(f_hat: np.ndarray, f_true: np.ndarray) -> np.ndarray:
    """Circular grid-offset error in frames, in (-FPC/2, FPC/2]."""
    d = (f_hat - f_true) % FPC
    return np.where(d > FPC / 2, d - FPC, d)


def phase_correct(res: dict, truth: Truth, P: int) -> np.ndarray:
    """Was the cycle position right? Accounts for a timing estimate one chunk off the true grid."""
    f_shift = np.round((res["f"] - truth.f0) / FPC).astype(np.int64)
    return (res["phase"] - (truth.phase + f_shift)) % P == 0
