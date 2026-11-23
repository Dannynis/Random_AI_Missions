"""Held-out real tables and frozen few-shot episodes.

The 18-table suite is the one used by Random-Ideas PR #14 / #34 (so numbers are comparable); five more
OpenML-CC18 tables with numeric features are added as a second, never-before-used group.
Episodes: 5 random 60/40 splits x k in {1,2,5,10,20} x 4 draws of k support rows per class (train side) and
balanced, shuffled queries (test side, up to 40 per class). Identical for every method.
"""
from __future__ import annotations

import os

import numpy as np

SUITE18 = {
    "iris": 61, "wine": 187, "wdbc": 1510, "banknote": 1462, "blood": 1464, "diabetes": 37, "ilpd": 1480,
    "vehicle": 54, "segment": 36, "phoneme": 1489, "wall-robot": 1497, "climate": 1467, "kc1": 1067,
    "pc1": 1068, "heart-statlog": 53, "seeds": 1499, "wilt": 40983, "balance-scale": 11,
}
EXTRA_CC18 = {"credit-g": 31, "cmc": 23, "pendigits": 32, "qsar-biodeg": 1494, "steel-plates": 40982}
SUITE = {**SUITE18, **EXTRA_CC18}
KS_EVAL = (1, 2, 5, 10, 20)
D_MAX, C_MAX = 64, 10


def data_home():
    root = os.environ.get("OPENML_HOME") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache")),
                                                         "openml_sklearn")
    os.makedirs(root, exist_ok=True)
    return root


def load_real(name):
    """(X float, y int, feature names); non-numeric and constant columns dropped, at most D_MAX columns."""
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=SUITE[name], as_frame=True, parser="auto", data_home=data_home())
    X = d.data.select_dtypes("number")
    X = X.loc[:, X.std().to_numpy() > 0].iloc[:, :D_MAX]
    y = np.unique(np.asarray(d.target).astype(str), return_inverse=True)[1]
    cols = [str(c) for c in X.columns]
    X = X.to_numpy(float)
    ok = ~np.isnan(X).any(1)
    return X[ok], y[ok].astype(int), cols


def real_split(X, y, seed, name, test_frac=0.4):
    r = np.random.default_rng([seed, sum(map(ord, name))])
    idx = r.permutation(len(y))
    n_te = int(round(test_frac * len(y)))
    te, tr = idx[:n_te], idx[n_te:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    return (X[tr] - mu) / sd, y[tr], (X[te] - mu) / sd, y[te]


def real_episodes(name, X, y, splits=5, draws=4, ks=KS_EVAL, qmax=40):
    eps = []
    C = int(y.max() + 1)
    for s in range(splits):
        Xtr, ytr, Xte, yte = real_split(X, y, s, name)
        q = int(min(qmax, np.bincount(yte, minlength=C).min()))
        if q < 1:
            continue
        for k in ks:
            if np.bincount(ytr, minlength=C).min() < k:
                continue
            r = np.random.default_rng([s, k, sum(map(ord, name))])
            for dr in range(draws):
                xs = np.stack([Xtr[r.choice(np.flatnonzero(ytr == c), k, replace=False)] for c in range(C)])
                qi = np.concatenate([r.choice(np.flatnonzero(yte == c), q, replace=False) for c in range(C)])
                qi = r.permutation(qi)                     # shuffled, so the first 20 are a random subset
                eps.append(dict(table=name, split=s, k=k, draw=dr, xs=xs.astype(np.float32),
                                xq=Xte[qi].astype(np.float32), yq=yte[qi].astype(np.int64)))
    return eps


def all_episodes(tables=None, draws=4):
    out, names = [], {}
    for t in tables or SUITE:
        X, y, fn = load_real(t)
        if int(y.max() + 1) > C_MAX:
            continue
        names[t] = fn
        out += real_episodes(t, X, y, draws=draws)
    return out, names
