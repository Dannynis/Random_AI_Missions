"""CRC-aided synchronisation and false-lock rejection for the looped audio watermark.

PR #45 (``wm_sync.py``) decodes a 16-bit message, convolutionally coded over a
4-chunk (8 s) cycle that is looped through a host stream, from a recording that
starts anywhere: blind timing from decoder confidence, ML cycle phase from the
Viterbi path metric, partial edge chunks. Two gaps were left open:

1. the best timing statistic (exhaustive ML over the Viterbi metric, ``joint``) was
   never combined with partial edge chunks, and costs 32 x 4 Viterbi runs;
2. the receiver always outputs a message, so it cannot say "no message" and it
   cannot tell a false lock (wrong timing / phase / bits) from a right one.

This module adds an *outer CRC* and a *list decoder* on top of the unchanged
physical and link layers:

* :class:`CRCLayout` appends ``r`` CRC bits to the 16 info bits before the
  convolutional code (so the code carries ``16 + r`` bits in the same 128 slots:
  the CRC is paid for in repetition gain, like PR #45's pilots were).
* :func:`list_viterbi` is a parallel list Viterbi decoder (top-``L`` zero-tailed
  paths per recording, exact for the path metric).
* :func:`candidates` runs a timing search (``blind`` / ``joint`` / coarse-to-fine
  ``c2f``, optionally with partial edge chunks) and returns, per recording, every
  (hypothesis, list rank) candidate message with its raw and normalised metric.
* :func:`decide` turns a candidate list into a decision under a rule:
  ``top`` (best metric, always outputs), ``crc@M`` (first of the top-``M`` distinct
  messages whose CRC checks; "no message" if none) and a normalised-metric
  threshold on top of either.

Every decision is one of *correct*, *wrong* (undetected error) or *none*; on
unmarked audio any output is a false alarm. The trade-off this module exposes:
each extra hypothesis tested against the CRC spends roughly one bit of its
protection (false pass probability ~ ``M * 2^-r``), so the CRC length, the list
size and the search width have to be chosen together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import audio_watermark as aw
import wm_coding as wc
import wm_sync as ws

FPC = ws.FPC

# CRC generator polynomials without the leading x^r term (bit i = coefficient of x^i).
CRC_POLY = {4: 0b0011,          # CRC-4-ITU   x^4 + x + 1
            6: 0b000011,        # CRC-6-ITU   x^6 + x + 1
            8: 0b00000111}      # CRC-8-CCITT x^8 + x^2 + x + 1


def crc_bits(info: np.ndarray, r: int) -> np.ndarray:
    """Bitwise CRC of ``info`` (B, k) in {0,1} -> (B, r), MSB first.

    The register starts at all ones (and the remainder is not inverted), so the
    all-zero message does *not* have an all-zero CRC: the all-zero codeword is what
    a decoder tends to output on silence, and it must not pass the check.
    """
    if r == 0:
        return np.zeros((info.shape[0], 0), dtype=np.int64)
    poly = CRC_POLY[r]
    reg = np.full(info.shape[0], (1 << r) - 1, dtype=np.int64)
    top = 1 << (r - 1)
    mask = (1 << r) - 1
    for t in range(info.shape[1]):
        fb = ((reg & top) > 0).astype(np.int64) ^ info[:, t].astype(np.int64)
        reg = ((reg << 1) & mask) ^ (fb * poly)
    return ((reg[:, None] >> np.arange(r - 1, -1, -1)[None]) & 1).astype(np.int64)


def crc_ok(msg: np.ndarray, k_info: int, r: int) -> np.ndarray:
    """(..., k_info + r) decoded bits -> (...) bool, CRC checks."""
    if r == 0:
        return np.ones(msg.shape[:-1], dtype=bool)
    flat = msg.reshape(-1, msg.shape[-1])
    ok = (crc_bits(flat[:, :k_info], r) == flat[:, k_info:]).all(1)
    return ok.reshape(msg.shape[:-1])


@dataclass
class CRCLayout(ws.Layout):
    """PR #45's pilot-free layout, with ``r`` CRC bits appended to ``k_info`` info bits."""

    k_info: int = 16
    r: int = 0
    k_msg: int = field(init=False)

    def __post_init__(self):
        self.k_msg = self.k_info + self.r
        super().__post_init__()

    @property
    def name(self) -> str:
        return f"crc{self.r}"

    def frame(self, info: np.ndarray) -> np.ndarray:
        """Info bits (B, k_info) -> message bits with CRC (B, k_msg)."""
        return np.concatenate([info, crc_bits(info, self.r)], 1)


def list_viterbi(code: wc.ConvCode, L_comb: np.ndarray, n_list: int) -> tuple[np.ndarray, np.ndarray]:
    """Parallel list Viterbi over combined codeword LLRs (B, n_code).

    Keeps the ``n_list`` best partial paths into every state; returns the
    ``n_list`` best zero-tailed paths: messages (B, n_list, k_msg) and path metrics
    (B, n_list), sorted best first. Rank 0 equals the plain Viterbi decision.
    Paths that do not exist (early trellis) carry metric -1e12.
    """
    B, T, S, Lst = L_comb.shape[0], code.k_msg + code.mem, code.n_state, n_list
    NEG = -1e12
    metric = np.full((B, S, Lst), NEG)
    metric[:, 0, 0] = 0.0
    back = np.zeros((B, T, S, Lst), dtype=np.int16)
    bip = 2 * code.out_prev - 1                                           # (S, 2, 2)
    for t in range(T):
        bm = L_comb[:, 2 * t, None, None] * bip[None, :, :, 0] + L_comb[:, 2 * t + 1, None, None] * bip[None, :, :, 1]
        cand = metric[:, code.prev, :] + bm[..., None]                    # (B, S, 2, Lst)
        cand = cand.reshape(B, S, 2 * Lst)
        idx = np.argsort(-cand, axis=2, kind="stable")[:, :, :Lst]
        metric = np.take_along_axis(cand, idx, 2)
        back[:, t] = idx
    fin = metric[:, 0, :]                                                 # sorted already
    rows = np.arange(B)[:, None]
    state = np.zeros((B, Lst), dtype=np.int64)
    rank = np.tile(np.arange(Lst), (B, 1))
    bits = np.zeros((B, Lst, T), dtype=np.int64)
    for t in range(T - 1, -1, -1):
        bits[:, :, t] = code.in_bit[state]
        i = back[rows, t, state, rank].astype(np.int64)
        state = code.prev[state, i // Lst]
        rank = i % Lst
    return bits[:, :, :code.k_msg], fin


@dataclass
class Candidates:
    """Per recording: every candidate message from the timing search, best first."""

    msg: list            # list of (n_c, k_msg) int arrays
    metric: list         # list of (n_c,) raw path metrics
    norm: list           # list of (n_c,) metric / sum |combined LLR| of the hypothesis
    rank: list           # list of (n_c,) list-Viterbi rank of the candidate within its hypothesis
    f: np.ndarray        # (B,) timing of the top candidate
    phase: np.ndarray    # (B,) cycle phase of the top candidate
    n_hyp: int           # timing x phase hypotheses searched


def _hyp_llrs(lay, dec, cal, feats, f, partial):
    """Window LLRs on the grid ``f + FPC j`` (same as ``wm_sync.receive``'s llrs_at, no tracking)."""
    B, T, _ = feats.shape
    st, j = ws.grid_starts(f, T, partial)
    ok = st > -10 ** 5
    z, frac = ws.window_logits(dec, feats, np.where(ok, st, 0))
    llr = cal(z.reshape(-1, aw.K)).reshape(z.shape) * frac[..., None]
    return llr, j, ok


def _blind_timing(dec, feats) -> np.ndarray:
    Z = ws.dense_logits(dec, feats)
    conf = np.abs(Z).mean(-1)
    nW = conf.shape[1]
    score = np.stack([conf[:, np.arange(o, nW, FPC)].mean(1) for o in range(FPC)], 1)
    return score.argmax(1).astype(np.float64)


def candidates(search: str, lay: CRCLayout, dec, cal, feats: torch.Tensor, truth: ws.Truth,
               partial: bool = True, n_list: int = 4, joint_step: int = 4,
               fine_radius: int = 6) -> Candidates:
    """Timing search + list decoding. ``search`` in {oracle, blind, joint, c2f}.

    * ``oracle``: true timing, all 4 phases (the CRC still has to pick the phase).
    * ``blind``: PR #45 confidence timing, all 4 phases.
    * ``joint``: every offset on a ``joint_step`` grid x 4 phases (PR #45's joint ML).
    * ``c2f``: coarse-to-fine, PR #45's blind confidence peak as the coarse
      estimate, then every offset within ``fine_radius`` frames of it (1-frame
      steps) x 4 phases, scored by the Viterbi metric.

    Each (offset, phase) hypothesis contributes its ``n_list`` best paths.
    """
    B = feats.shape[0]
    P = lay.P
    hyps: list[np.ndarray] = []         # list of (B,) offsets, evaluated over all phases

    def run(offsets_list):
        out = []
        for fo in offsets_list:
            llr, j, ok = _hyp_llrs(lay, dec, cal, feats, fo, partial)
            for p in range(P):
                acc = ws.assemble(lay, llr, j, ok, np.full(B, p))
                Lc = lay.code._combine(lay.il.gather(acc.reshape(B, -1)), True)
                m, met = list_viterbi(lay.code, Lc, n_list)
                norm = met / (np.abs(Lc).sum(1, keepdims=True) + 1e-9)
                out.append((fo, p, m, met, norm))
        return out

    if search == "oracle":
        res = run([truth.f0])
    elif search == "blind":
        res = run([_blind_timing(dec, feats)])
    elif search == "joint":
        res = run([np.full(B, float(o)) for o in range(0, FPC, joint_step)])
    elif search == "c2f":
        f_c = _blind_timing(dec, feats)
        res = run([(f_c + d) % FPC for d in range(-fine_radius, fine_radius + 1)])
    else:
        raise ValueError(search)

    met = np.stack([r[3] for r in res], 1)                                  # (B, H, Lst)
    norm = np.stack([r[4] for r in res], 1)
    msg = np.stack([r[2] for r in res], 1)                                  # (B, H, Lst, k)
    fo = np.stack([np.broadcast_to(r[0], (B,)) for r in res], 1)            # (B, H)
    ph = np.array([r[1] for r in res])
    H = met.shape[1]
    rk = np.broadcast_to(np.arange(n_list), (H, n_list)).ravel()
    out_m, out_met, out_n, out_r = [], [], [], []
    f_top = np.zeros(B); p_top = np.zeros(B, np.int64)
    for b in range(B):
        mt = met[b].ravel(); order = np.argsort(-mt, kind="stable")
        order = order[mt[order] > -1e11]
        out_m.append(msg[b].reshape(-1, lay.k_msg)[order])
        out_met.append(mt[order]); out_n.append(norm[b].ravel()[order]); out_r.append(rk[order])
        h_top = order[0] // n_list
        f_top[b] = fo[b, h_top]; p_top[b] = ph[h_top]
    return Candidates(out_m, out_met, out_n, out_r, f_top, p_top, H)


RULES = ("top", "crc@1", "crc@4", "crc@16", "crc@64")


def decide(c: Candidates, lay: CRCLayout, rule: str, list_size: int = 1
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a decision rule. Returns (info bits (B, k_info), output? (B,), normalised metric (B,)).

    Only candidates of list rank < ``list_size`` are considered (1 = plain Viterbi
    per hypothesis), and each distinct message is counted once, at its best metric.
    ``top``: best-metric candidate, always output. ``crc@M``: first of the best ``M``
    distinct candidate messages whose CRC checks, otherwise no output. The returned
    normalised metric belongs to the chosen candidate (for a threshold on top).
    """
    B = len(c.msg)
    info = np.zeros((B, lay.k_info), dtype=np.int64)
    out = np.zeros(B, dtype=bool)
    nm = np.full(B, -np.inf)
    M = 1 if rule == "top" else int(rule.split("@")[1])
    w = 1 << np.arange(lay.k_msg, dtype=np.int64)
    for b in range(B):
        sel = np.where(c.rank[b] < list_size)[0]
        packed = c.msg[b][sel] @ w
        _, first = np.unique(packed, return_index=True)
        sel = sel[np.sort(first)][:M]                     # best M distinct messages
        mm = c.msg[b][sel]
        ok = np.ones(len(mm), bool) if rule == "top" else crc_ok(mm, lay.k_info, lay.r)
        if ok.any():
            i = int(np.argmax(ok))
            info[b] = mm[i, :lay.k_info]; out[b] = True; nm[b] = c.norm[b][sel[i]]
    return info, out, nm


@dataclass
class OperatingPoint:
    """Outcome rates of one (layout, search, rule, threshold) on marked + unmarked recordings."""

    success: float        # marked: correct message output
    wrong: float          # marked: wrong message output (undetected error)
    none: float           # marked: no output
    false_alarm: float    # unmarked: any output
    threshold: float


def operating_point(correct, out_pos, nm_pos, out_neg, nm_neg, thr: float = -np.inf) -> OperatingPoint:
    acc_p = out_pos & (nm_pos > thr)
    acc_n = out_neg & (nm_neg > thr)
    return OperatingPoint(float((acc_p & correct).mean()), float((acc_p & ~correct).mean()),
                          float((~acc_p).mean()), float(acc_n.mean()), float(thr))


def threshold_for(nm_neg: np.ndarray, out_neg: np.ndarray, fa: float) -> float:
    """Smallest normalised-metric threshold with false-alarm rate <= ``fa`` on these unmarked scores."""
    s = np.sort(np.where(out_neg, nm_neg, -np.inf))[::-1]
    k = int(np.floor(fa * len(s)))
    return float(s[k]) if k < len(s) else -np.inf
