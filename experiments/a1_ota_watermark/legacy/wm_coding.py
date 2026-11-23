"""From bits to messages: LLR calibration, outer codes and soft decoding for the watermark.

Companion of ``audio_watermark.py`` (2026-10-14, PR #19). That run compared
watermark methods by *bit* error rate on single 2 s clips: the best learned,
EoT-trained code reached BER 0.223 over the air, but delivered all 32 bits
correctly on only 0.7 % of clips, i.e. it was useless as a *message* channel.

This module turns the same physical layer into a message channel:

* :func:`fit_llr` calibrates a decoder's raw soft scores into log-likelihood
  ratios (per payload slot, logistic regression on held-out clips), so that
  reliability differences between slots and between methods become usable.
* :class:`Repetition`, :class:`ConvCode` and :class:`Uncoded` are outer codes
  over the ``n_chunks * K`` payload slots of a multi-chunk recording, with
  soft-decision decoding (LLR summing / soft Viterbi) and hard-decision
  variants for the ablation.
* :class:`Interleaver` spreads each codeword bit over chunks and slots, which
  matters because slot reliability is very uneven.
* :func:`simulate_messages` replays measured per-slot LLRs onto new codewords
  using the binary-input-symmetric assumption (an LLR measured for a
  transmitted bit ``b`` is reused as ``(2c-1) * |evidence|`` for a codeword bit
  ``c``), so one embed/decode pass over the test set scores every coding
  scheme. :func:`capacity_bits` gives the matching soft-information budget.

The symmetry assumption is validated end to end in the notebook by embedding
real codewords for one scheme and comparing the message success rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


# --------------------------------------------------------------------------
# LLR calibration
# --------------------------------------------------------------------------

@dataclass
class LLRCalibration:
    """Per-slot affine map from a decoder's soft score to a log-likelihood ratio.

    ``llr_k = a_k * z_k + c_k``, fitted by logistic regression (Newton steps) on
    calibration clips so that ``sigmoid(llr)`` is the probability that the slot
    carried a 1. ``a`` absorbs the arbitrary scale of correlation scores and the
    overconfidence of a BCE-trained decoder; it also encodes how reliable each
    slot is, which is what the outer code needs.
    """

    a: np.ndarray                      # (K,)
    c: np.ndarray                      # (K,)
    method: str = ""
    n_cal: int = 0
    nll: float = 0.0                   # mean calibration NLL in bits/slot

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.a[None, :] * np.asarray(z, dtype=np.float64) + self.c[None, :]


def fit_llr(z: np.ndarray, bits: np.ndarray, method: str = "", iters: int = 50,
            l2: float = 1e-3) -> LLRCalibration:
    """Fit ``LLRCalibration`` on calibration scores ``z`` (n, K) and true bits (n, K)."""
    z = np.asarray(z, dtype=np.float64)
    b = np.asarray(bits, dtype=np.float64)
    n, k = z.shape
    a = np.ones(k) / (z.std(0) + 1e-9)
    c = np.zeros(k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a[None] * z + c[None])))
        w = np.clip(p * (1 - p), 1e-6, None)
        r = b - p
        # per-slot 2x2 Newton system over (a, c)
        gaa = (w * z * z).sum(0) + l2
        gac = (w * z).sum(0)
        gcc = w.sum(0) + l2
        ga = (r * z).sum(0) - l2 * a
        gc = r.sum(0) - l2 * c
        det = gaa * gcc - gac ** 2 + 1e-12
        a += (gcc * ga - gac * gc) / det
        c += (gaa * gc - gac * ga) / det
    p = 1 / (1 + np.exp(-(a[None] * z + c[None])))
    nll = float(-(b * np.log(p + 1e-12) + (1 - b) * np.log(1 - p + 1e-12)).mean() / np.log(2))
    return LLRCalibration(a, c, method, n, nll)


def evidence(llr: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """LLR in favour of the bit that was actually transmitted (``+`` = correct, ``-`` = error)."""
    return llr * (2 * np.asarray(bits, dtype=np.float64) - 1)


def capacity_bits(ev: np.ndarray) -> float:
    """Soft-information rate of a binary-input symmetric channel, bits per slot.

    ``I = E[1 - log2(1 + exp(-s))]`` for the LLR ``s`` of the transmitted bit
    (the mutual information of a channel whose LLRs are correctly calibrated).
    """
    s = np.clip(np.asarray(ev, dtype=np.float64), -60, 60)
    return float(np.mean(1.0 - np.log2(1.0 + np.exp(-s))))


# --------------------------------------------------------------------------
# Interleaving
# --------------------------------------------------------------------------

@dataclass
class Interleaver:
    """Permutation of ``n`` codeword bits over the payload slots of a recording.

    ``identity`` keeps codeword order (consecutive codeword bits land in
    neighbouring slots of the same chunk); ``random`` spreads them, which helps
    whenever reliability is clustered (weak slots, a bad chunk).
    """

    perm: np.ndarray
    kind: str = "random"

    @staticmethod
    def make(n: int, kind: str = "random", seed: int = 0) -> "Interleaver":
        if kind == "identity":
            return Interleaver(np.arange(n), kind)
        return Interleaver(np.random.default_rng(seed).permutation(n), kind)

    def spread(self, x: np.ndarray) -> np.ndarray:
        """Codeword order -> slot order."""
        out = np.empty_like(x)
        out[..., self.perm] = x
        return out

    def gather(self, x: np.ndarray) -> np.ndarray:
        """Slot order -> codeword order."""
        return x[..., self.perm]


# --------------------------------------------------------------------------
# Outer codes
# --------------------------------------------------------------------------

class Code:
    """Outer code mapping ``k_msg`` message bits to ``n`` channel bits."""

    name = "code"

    def __init__(self, k_msg: int, n: int):
        self.k_msg, self.n = k_msg, n

    @property
    def rate(self) -> float:
        return self.k_msg / self.n

    def encode(self, m: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, llr: np.ndarray, soft: bool = True) -> np.ndarray:
        raise NotImplementedError


class Uncoded(Code):
    """No protection: the message occupies the first ``k_msg`` slots, the rest are random filler.

    This is PR #19's implicit scheme (32 bits per clip, all of them payload).
    """

    name = "uncoded"

    def encode(self, m: np.ndarray) -> np.ndarray:
        B = m.shape[0]
        rng = np.random.default_rng(abs(int(m.sum())) + 12345)
        pad = rng.integers(0, 2, (B, self.n - self.k_msg))
        return np.concatenate([m, pad], axis=1)

    def decode(self, llr: np.ndarray, soft: bool = True) -> np.ndarray:
        return (llr[:, :self.k_msg] > 0).astype(np.int64)


class Repetition(Code):
    """Each message bit repeated over as many slots as fit (round robin).

    Soft decoding sums the LLRs of the repeats (optimal for independent slots);
    hard decoding takes a majority vote, which is what a BER-only pipeline can do.
    """

    name = "repetition"

    def __init__(self, k_msg: int, n: int):
        super().__init__(k_msg, n)
        self.idx = np.arange(n) % k_msg          # slot -> message bit

    def encode(self, m: np.ndarray) -> np.ndarray:
        return m[:, self.idx]

    def decode(self, llr: np.ndarray, soft: bool = True) -> np.ndarray:
        x = llr if soft else np.sign(llr)
        acc = np.zeros((llr.shape[0], self.k_msg))
        np.add.at(acc.T, self.idx, x.T)
        return (acc > 0).astype(np.int64)


_POLY = (0o133, 0o171)                            # rate 1/2, constraint length 7 (NASA standard)


class ConvCode(Code):
    """Rate-1/2 K=7 convolutional code (zero-tailed), repeated to fill ``n`` slots.

    Encoding: ``k_msg`` bits + 6 tail zeros -> ``2*(k_msg+6)`` codeword bits,
    cyclically repeated to ``n`` slots. Decoding: repeats are LLR-combined
    (or majority-voted when ``soft=False``) and run through a soft-output
    Viterbi decoder (hard Hamming metric when ``soft=False``).
    """

    name = "conv"

    def __init__(self, k_msg: int, n: int, mem: int = 6):
        super().__init__(k_msg, n)
        self.mem = mem
        self.n_state = 1 << mem
        self.n_code = 2 * (k_msg + mem)
        if n < self.n_code:
            raise ValueError(f"{n} slots < {self.n_code} codeword bits")
        self.idx = np.arange(n) % self.n_code     # slot -> codeword bit
        # Shift register r = (u_t, u_{t-1}, ..., u_{t-mem}); state s = r >> 1 holds the
        # last ``mem`` inputs, so next_state = ((u << mem) | s) >> 1.
        s = np.arange(self.n_state)[:, None]
        u = np.arange(2)[None, :]
        regs = (u << mem) | s
        self.out = np.array([[bin(int(r) & p).count("1") & 1 for p in _POLY]
                             for r in regs.ravel()]).reshape(self.n_state, 2, 2)
        self.nxt = regs >> 1                                             # (n_state, 2)
        # Trellis seen from the next state: its MSB is the input bit, and its two
        # predecessors differ only in the bit that falls out of the register.
        ns = np.arange(self.n_state)
        self.in_bit = ns >> (mem - 1)
        self.prev = np.stack([((ns << 1) & (self.n_state - 1)) | j for j in range(2)], axis=1)
        self.out_prev = self.out[self.prev, self.in_bit[:, None]]        # (n_state, 2, 2)

    def encode(self, m: np.ndarray) -> np.ndarray:
        B = m.shape[0]
        u = np.concatenate([m, np.zeros((B, self.mem), dtype=m.dtype)], axis=1)
        state = np.zeros(B, dtype=np.int64)
        cw = np.empty((B, self.n_code), dtype=np.int64)
        for t in range(u.shape[1]):
            ut = u[:, t].astype(np.int64)
            cw[:, 2 * t:2 * t + 2] = self.out[state, ut]
            state = self.nxt[state, ut]
        return cw[:, self.idx]

    def _combine(self, llr: np.ndarray, soft: bool) -> np.ndarray:
        x = llr if soft else np.sign(llr)
        acc = np.zeros((llr.shape[0], self.n_code))
        np.add.at(acc.T, self.idx, x.T)
        return acc if soft else np.sign(acc)

    def decode(self, llr: np.ndarray, soft: bool = True) -> np.ndarray:
        """Viterbi over the combined codeword LLRs (branch metric = sum of matched LLRs)."""
        L = self._combine(llr, soft)
        B, T = L.shape[0], self.k_msg + self.mem
        NEG = -1e12
        metric = np.full((B, self.n_state), NEG)
        metric[:, 0] = 0.0                                   # encoder starts in the zero state
        back = np.zeros((B, T, self.n_state), dtype=np.int8)
        bipolar = 2 * self.out_prev - 1                      # (n_state, 2 predecessors, 2 out bits)
        for t in range(T):
            l0, l1 = L[:, 2 * t, None, None], L[:, 2 * t + 1, None, None]
            bm = l0 * bipolar[None, :, :, 0] + l1 * bipolar[None, :, :, 1]      # (B, n_state, 2)
            cand = metric[:, self.prev] + bm
            metric = cand.max(2)
            back[:, t] = cand.argmax(2)
        # Zero-tailed: the surviving path must end in state 0.
        state = np.zeros(B, dtype=np.int64)
        bits = np.zeros((B, T), dtype=np.int64)
        rows = np.arange(B)
        for t in range(T - 1, -1, -1):
            bits[:, t] = self.in_bit[state]
            state = self.prev[state, back[rows, t, state].astype(np.int64)]
        return bits[:, :self.k_msg]


def make_code(kind: str, k_msg: int, n: int) -> Code:
    return {"uncoded": Uncoded, "repetition": Repetition, "conv": ConvCode}[kind](k_msg, n)


# --------------------------------------------------------------------------
# Message-level simulation on measured LLRs
# --------------------------------------------------------------------------

@dataclass
class MessageResult:
    """Message-level score of one (code, interleaver, decoder) configuration."""

    scheme: str
    code: str
    k_msg: int
    n_chunks: int
    soft: bool
    interleave: str
    n_messages: int
    success: float                     # fraction of messages decoded exactly
    msg_ber: float                     # message-bit error rate after decoding
    slot_ber: float                    # raw slot BER of the same slots (physical layer)
    rate_bps: float                    # message bits per second of audio
    extra: dict = field(default_factory=dict)


def simulate_messages(ev: np.ndarray, code: Code, n_chunks: int, k_slots: int,
                      soft: bool = True, interleave: str = "random", n_messages: int = 400,
                      seed: int = 0, chunk_sec: float = 2.0,
                      valid_starts: np.ndarray | None = None) -> MessageResult:
    """Score ``code`` by replaying measured per-slot evidence onto random messages.

    ``ev`` is (n_clips, k_slots) LLR-of-the-transmitted-bit, measured once per test
    clip under one channel. Each message takes ``n_chunks`` consecutive clips (same
    source recording, independent channel draws) and uses their
    ``n_chunks * k_slots`` slots. Under the binary-input-symmetric assumption the
    received LLR for a codeword bit ``c`` is ``(2c - 1) * ev``. ``valid_starts``
    restricts the windows (e.g. to chunks of one recording).
    """
    rng = np.random.default_rng(seed)
    n_clips = ev.shape[0]
    if valid_starts is None:
        starts = rng.integers(0, max(1, n_clips - n_chunks + 1), n_messages)
    else:
        starts = rng.choice(np.asarray(valid_starts), n_messages, replace=True)
    pool = np.stack([ev[s:s + n_chunks].reshape(-1) for s in starts])     # (M, n_chunks*k_slots)
    il = Interleaver.make(pool.shape[1], interleave, seed=seed + 7)
    m = rng.integers(0, 2, (n_messages, code.k_msg))
    cw = il.spread(code.encode(m))                                        # slot order
    llr = (2 * cw - 1) * pool
    hat = code.decode(il.gather(llr), soft=soft)
    ok = (hat == m).all(1)
    return MessageResult(
        f"{code.name}/{'soft' if soft else 'hard'}/{interleave}", code.name, code.k_msg, n_chunks,
        soft, interleave, n_messages, float(ok.mean()), float((hat != m).mean()),
        float((pool < 0).mean()), code.k_msg / (n_chunks * chunk_sec),
        {"rate": code.rate, "capacity_bits_per_slot": capacity_bits(pool)})
