"""Structural-causal-model (SCM) prior over small classification tasks (TabPFN-style).

Ported from the Random-Ideas `llm_weights/hyper/scm_prior.py` (PR #14) with the same mechanism, feature
nuisances, rejection filter and finalisation, made self-contained (no `tabtasks` dependency).

Every task is a random causal MLP: root causes -> L sparse random layers with random activations; features are
a random subset of the units, the target is a quantile cut of another unit (ordinal) or the argmax of C units.
Real-table nuisances (skew, rounding, near-duplicate columns) are added per feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TabTask:
    Xtr: np.ndarray = None
    ytr: np.ndarray = None
    Xte: np.ndarray = None
    yte: np.ndarray = None
    d: int = 0
    C: int = 0
    name: str = ""
    feature_names: list = field(default_factory=list)


ACTS = {
    "tanh": np.tanh,
    "relu": lambda v: np.maximum(v, 0),
    "leaky": lambda v: np.where(v > 0, v, 0.1 * v),
    "sin": np.sin,
    "abs": np.abs,
    "square": lambda v: np.clip(v, -3, 3) ** 2,
    "sigmoid": lambda v: 1 / (1 + np.exp(-v)),
    "identity": lambda v: v,
    "step": lambda v: (v > 0).astype(float),
}
ACT_NAMES = tuple(ACTS)


def _mechanism(rng, n, n_units):
    L = int(rng.integers(2, 5))
    width = int(rng.integers(max(4, n_units // L), max(6, n_units // L) + 8))
    m0 = int(rng.integers(2, width + 1))
    if rng.random() < 0.5:
        z = rng.normal(0, 1, (n, m0)) * rng.uniform(0.5, 1.5)
    else:
        a = rng.uniform(1, 2)
        z = rng.uniform(-a, a, (n, m0))
    nodes = [z]
    noise_sd = rng.uniform(0.0, 0.2)
    for _ in range(L):
        A = rng.normal(0, 1, (z.shape[1], width)) / np.sqrt(z.shape[1])
        A *= rng.random(A.shape) < rng.uniform(0.3, 1.0)
        act = ACTS[ACT_NAMES[rng.integers(len(ACT_NAMES))]]
        z = act(z @ A * rng.uniform(0.7, 2.0) + rng.normal(0, 0.5, width))
        z = z + rng.normal(0, noise_sd, z.shape)
        nodes.append(z)
    V = np.concatenate(nodes, 1)
    return V[:, V.std(0) > 1e-6]


def _warp_feature(rng, x):
    x = (x - x.mean()) / (x.std() + 1e-9)
    u = rng.random()
    if u < 0.2:
        return np.exp(np.clip(x * rng.uniform(0.3, 1.2), -6, 6))
    if u < 0.35:
        return np.log1p(np.abs(x) * rng.uniform(0.5, 3))
    if u < 0.5:
        return np.sign(x) * np.abs(x) ** rng.uniform(1.5, 3)
    if u < 0.65:
        levels = int(rng.integers(2, 8))
        edges = np.quantile(x, np.sort(rng.uniform(0.05, 0.95, levels - 1)))
        return np.searchsorted(edges, x).astype(float)
    if u < 0.75:
        return np.log1p(np.exp(np.clip(x * 2, -30, 30)))
    return x


def _labels(rng, V, C, feat_idx):
    rest = np.setdiff1d(np.arange(V.shape[1]), feat_idx)
    pool = rest if len(rest) >= C else np.arange(V.shape[1])
    if rng.random() < 0.6 or len(pool) < C:
        t = V[:, pool[rng.integers(len(pool))]] + rng.normal(0, 1e-6, len(V))
        qs = np.sort(rng.uniform(0.1, 0.9, C - 1)) if rng.random() < 0.5 else np.linspace(0, 1, C + 1)[1:-1]
        return np.searchsorted(np.quantile(t, qs), t)
    cols = rng.choice(pool, C, replace=False)
    Z = V[:, cols]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    return (Z + rng.gumbel(0, rng.uniform(0.05, 0.5), Z.shape)).argmax(1)


def _learnable(X, y, n_sub=300, margin=0.05):
    idx = np.arange(len(y))[:n_sub]
    Xs, ys = (X[idx] - X[idx].mean(0)) / (X[idx].std(0) + 1e-9), y[idx]
    D = ((Xs[:, None] - Xs[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nn1 = (ys[D.argmin(1)] == ys).mean()
    C = int(ys.max() + 1)
    cen = np.stack([Xs[ys == c].mean(0) if (ys == c).any() else np.full(Xs.shape[1], np.inf) for c in range(C)])
    ncen = (((Xs[:, None] - cen[None]) ** 2).sum(-1).argmin(1) == ys).mean()
    return max(nn1, ncen) >= np.bincount(ys).max() / len(ys) + margin


def _gauss_task(rng, d, C, n):
    mu = rng.normal(0, 1, (C, d))
    y = rng.integers(0, C, 2 * n)
    X = mu[y] + rng.normal(0, 1, (2 * n, d))
    return X, y


def sample_scm_task(rng, d=None, C=None, n=400, warp=True, noise=None, max_tries=30):
    d = int(d or rng.integers(2, 33))
    C = int(C or rng.integers(2, 9))
    for _ in range(max_tries):
        V = _mechanism(rng, 6 * n, n_units=d + C + int(rng.integers(2, 16)))
        if V.shape[1] < d + 1:
            continue
        n_dup = int(rng.integers(0, max(1, d // 4) + 1)) if (warp and d >= 4) else 0
        feat_idx = rng.choice(V.shape[1], d - n_dup, replace=False)
        y = _labels(rng, V, C, feat_idx)
        if (np.bincount(y, minlength=C) >= 8).sum() == C and _learnable(V[:, feat_idx], y):
            X = V[:, feat_idx]
            if n_dup:
                src = rng.integers(0, X.shape[1], n_dup)
                X = np.concatenate([X, X[:, src] + rng.normal(0, 0.1, (len(X), n_dup)) * X[:, src].std(0)], 1)
            X = X[:, rng.permutation(d)]
            if warp:
                X = np.stack([_warp_feature(rng, X[:, j]) for j in range(d)], 1)
            break
    else:
        X, y = _gauss_task(rng, d, C, n)
    return _finalize(rng, X, y, d, C, n, noise)


def _finalize(rng, X, y, d, C, n, noise):
    p = rng.dirichlet(np.full(C, rng.uniform(1.5, 8.0)))
    take = []
    for c in range(C):
        idx = np.flatnonzero(y == c)
        if len(idx):
            take += list(rng.choice(idx, min(len(idx), max(4, int(p[c] * n * 2))), replace=False))
    take = rng.permutation(take)[: 2 * n]
    X, y = X[take], y[take].copy()
    if noise is None:
        noise = float(rng.choice([0.0, 0.0, 0.05, 0.1]))
    if noise > 0:
        flip = rng.random(len(y)) < noise
        y[flip] = rng.integers(0, C, flip.sum())
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    for c in range(C):
        if not (y == c).any():
            y[rng.integers(len(y))] = c
    tr, te = [], []
    for c in range(C):
        idx = rng.permutation(np.flatnonzero(y == c))
        cut = max(2, len(idx) // 2)
        tr += list(idx[:cut]); te += list(idx[cut:]) or list(idx[:2])
    tr, te = rng.permutation(tr), rng.permutation(te)
    return TabTask(X[tr].astype(np.float32), y[tr], X[te].astype(np.float32), y[te], d, C, f"scm_d{d}_c{C}",
                   [f"f{j + 1}" for j in range(d)])
