"""Pool-based cold-start active learning on frozen features: learners, baselines, learned policy.

All tensors live on one device. Features are L2-normalised, so cosine similarity is a dot product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------

def knn(X: torch.Tensor, k: int, Q: torch.Tensor | None = None, chunk: int = 4096):
    """Top-k cosine neighbours of every row of Q (default X) among X, excluding self when Q is X."""
    self_q = Q is None
    Q = X if Q is None else Q
    sims, idx = [], []
    for i in range(0, len(Q), chunk):
        s = Q[i:i + chunk] @ X.T
        if self_q:
            s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -2
        v, j = s.topk(k, 1)
        sims.append(v); idx.append(j)
    return torch.cat(sims), torch.cat(idx)


def kmeans(X: torch.Tensor, k: int, iters: int = 30, seed: int = 0):
    """Spherical k-means (k-means++ init). Returns (centroids, assignment)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(X)
    k = min(k, n)
    c = [int(torch.randint(n, (1,), generator=g))]
    d2 = (2 - 2 * X @ X[c[0]]).clamp_min(0)
    for _ in range(1, k):
        p = (d2 / d2.sum()).cpu().double()
        j = int(torch.multinomial(p, 1, generator=g))
        c.append(j)
        d2 = torch.minimum(d2, (2 - 2 * X @ X[j]).clamp_min(0))
    C = X[c].clone()
    for _ in range(iters):
        a = (X @ C.T).argmax(1)
        Cn = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=k)
        empty = cnt == 0
        Cn[empty] = C[empty]
        C = F.normalize(Cn, dim=1)
    return C, (X @ C.T).argmax(1)


# ----------------------------------------------------------------------------
# learners
# ----------------------------------------------------------------------------

def fit_linear(X: torch.Tensor, y: torch.Tensor, n_cls: int, wd: float = 1e-3, iters: int = 100,
               scale: float = 10.0):
    """Multinomial logistic regression (L-BFGS) on scaled, L2-normalised features."""
    W = torch.zeros(X.shape[1], n_cls, device=X.device, requires_grad=True)
    b = torch.zeros(n_cls, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=10, line_search_fn="strong_wolfe")
    Xs = X * scale

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xs @ W + b, y) + wd * W.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    W, b = W.detach(), b.detach()
    return lambda Z: torch.softmax(Z * scale @ W + b, -1)


def label_propagation(nb_sim, nb_idx, lab_idx, lab_y, n_cls, alpha=0.99, iters=20):
    """Zhou et al. label spreading on the (symmetrised by averaging) kNN graph of the pool."""
    n, k = nb_idx.shape
    w = nb_sim.clamp_min(0) ** 3
    deg = torch.zeros(n, device=w.device).index_add_(0, nb_idx.flatten(), w.flatten()) + w.sum(1)
    dinv = deg.clamp_min(1e-8).rsqrt()
    Y0 = torch.zeros(n, n_cls, device=w.device)
    Y0[lab_idx, lab_y] = 1
    Fm = Y0.clone()
    wn = w * dinv[:, None] * dinv[nb_idx]
    for _ in range(iters):
        out = (wn[..., None] * Fm[nb_idx]).sum(1)                        # i <- its neighbours
        inn = torch.zeros_like(Fm).index_add_(0, nb_idx.flatten(), (wn[..., None] * Fm[:, None]).flatten(0, 1))
        Fm = alpha * 0.5 * (out + inn) + (1 - alpha) * Y0
    return Fm


# ----------------------------------------------------------------------------
# environment: one AL task (pool, test set, cached geometry)
# ----------------------------------------------------------------------------

@dataclass
class Task:
    X: torch.Tensor          # pool features (n, d), normalised
    y: torch.Tensor          # pool labels (oracle)
    Xt: torch.Tensor         # test features
    yt: torch.Tensor
    n_cls: int
    nb_sim: torch.Tensor = None
    nb_idx: torch.Tensor = None
    t_sim: torch.Tensor = None   # test -> pool kNN (for the LP read-out)
    t_idx: torch.Tensor = None
    density: torch.Tensor = None
    reg: torch.Tensor = None     # coarse k-means region of each pool point (policy features)
    reg_share: torch.Tensor = None
    pc_delta: float = None
    pc_sim: torch.Tensor = None  # wider kNN lists (k=100) that approximate ProbCover's delta-balls
    pc_idx: torch.Tensor = None

    def prepare(self, k: int = 20, seed: int = 0, n_regions: int | None = None):
        self.nb_sim, self.nb_idx = knn(self.X, k)
        self.pc_sim, self.pc_idx = knn(self.X, min(100, len(self.X) - 1))
        self.t_sim, self.t_idx = knn(self.X, k, self.Xt)
        self.density = self.nb_sim.mean(1)
        _, self.reg = kmeans(self.X, n_regions or 2 * self.n_cls, seed=seed)
        self.reg_share = torch.bincount(self.reg, minlength=int(self.reg.max()) + 1).float() / len(self.X)
        self.pc_delta = probcover_delta(self, seed=seed)
        return self


def evaluate(task: Task, lab: torch.Tensor):
    """Accuracy of (linear probe, label propagation) trained on pool indices ``lab``."""
    yl = task.y[lab]
    pred = fit_linear(task.X[lab], yl, task.n_cls)
    lin = float((pred(task.Xt).argmax(1) == task.yt).float().mean())
    Fm = label_propagation(task.nb_sim, task.nb_idx, lab, yl, task.n_cls)
    w = task.t_sim.clamp_min(0) ** 3
    Ft = (w[..., None] * Fm[task.t_idx]).sum(1)
    lp = float((Ft.argmax(1) == task.yt).float().mean())
    return lin, lp, pred


# ----------------------------------------------------------------------------
# baseline query strategies: fn(task, lab, B, pred, gen) -> new pool indices
# ----------------------------------------------------------------------------

def _unl_mask(task, lab):
    m = torch.ones(len(task.X), dtype=torch.bool, device=task.X.device)
    m[lab] = False
    return m


def q_random(task, lab, B, pred, gen):
    u = _unl_mask(task, lab).nonzero().squeeze(1)
    return u[torch.randperm(len(u), generator=gen)[:B].to(u.device)]


def q_entropy(task, lab, B, pred, gen):
    if pred is None:
        return q_random(task, lab, B, pred, gen)
    p = pred(task.X)
    s = -(p * p.clamp_min(1e-12).log()).sum(1)
    s[~_unl_mask(task, lab)] = -1e9
    return s.topk(B).indices


def q_margin(task, lab, B, pred, gen):
    if pred is None:
        return q_random(task, lab, B, pred, gen)
    t2 = pred(task.X).topk(2, 1).values
    s = -(t2[:, 0] - t2[:, 1])
    s[~_unl_mask(task, lab)] = -1e9
    return s.topk(B).indices


def q_coreset(task, lab, B, pred, gen):
    """Greedy k-center (Sener & Savarese 2018) in cosine distance."""
    X = task.X
    if len(lab) == 0:
        first = q_random(task, lab, 1, pred, gen)
        mind = (2 - 2 * X @ X[first[0]]).clamp_min(0)
        picks = [int(first[0])]
    else:
        mind = torch.full((len(X),), 4.0, device=X.device)
        for i in range(0, len(lab), 1024):
            mind = torch.minimum(mind, (2 - 2 * X @ X[lab[i:i + 1024]].T).min(1).values)
        picks = []
    while len(picks) < B:
        j = int(mind.argmax())
        picks.append(j)
        mind = torch.minimum(mind, (2 - 2 * X @ X[j]).clamp_min(0))
        mind[j] = -1
    return torch.tensor(picks, device=X.device)


def typiclust_order(task, lab, B, a, min_size: int = 5):
    """TypiClust picks (in order) given a cluster assignment ``a`` of the pool."""
    K = int(a.max()) + 1
    size = torch.bincount(a, minlength=K)
    nlab = torch.bincount(a[lab], minlength=K) if len(lab) else torch.zeros_like(size)
    typ = task.density                             # 1 / mean distance to 20 NN, monotone proxy
    unl = _unl_mask(task, lab)
    dead = torch.zeros(K, dtype=torch.bool, device=a.device)   # clusters with no unlabeled point left
    picks = []
    while len(picks) < B and not bool(dead.all()):
        key = nlab.float() * 1e7 - size.float()
        key[size < min_size] = 1e12
        key[dead] = float("inf")
        c = int(key.argmin())
        cand = ((a == c) & unl).nonzero().squeeze(1)
        if len(cand) == 0:
            dead[c] = True
            continue
        j = int(cand[typ[cand].argmax()])
        picks.append(j); unl[j] = False; nlab[c] += 1
    return picks


def q_typiclust(task, lab, B, pred, gen, max_clusters: int = 500, min_size: int = 5):
    """TypiClust (Hacohen et al. 2022): k-means with |L|+B clusters, most typical point of the
    largest clusters that hold the fewest labels."""
    X = task.X
    K = min(len(lab) + B, max_clusters)
    _, a = kmeans(X, K, iters=20, seed=int(torch.randint(1 << 30, (1,), generator=gen)))
    picks = typiclust_order(task, lab, B, a, min_size)
    if len(picks) < B:
        extra = q_random(task, torch.cat([lab, torch.tensor(picks, device=X.device, dtype=torch.long)]),
                         B - len(picks), pred, gen)
        picks += extra.tolist()
    return torch.tensor(picks, device=X.device)


def probcover_delta(task, seed=0, purity=0.95, grid=None):
    """Largest ball radius (cosine distance) whose balls are >= 95 % pure w.r.t. k-means with
    n_cls clusters (Yehuda et al. 2022), estimated on the kNN lists."""
    _, a = kmeans(task.X, task.n_cls, seed=seed)
    d = 1 - task.pc_sim
    same = a[task.pc_idx] == a[:, None]
    best = None
    for delta in grid or [0.05 * i for i in range(1, 15)]:
        inball = d < delta
        impure = (inball & ~same).any(1).float().mean()
        if 1 - float(impure) >= purity:
            best = delta
    return best or 0.05


def q_probcover(task, lab, B, pred, gen):
    """ProbCover greedy max coverage on the delta-ball graph (approximated by the kNN lists)."""
    X = task.X
    E = (1 - task.pc_sim) < task.pc_delta              # (n, k) edge mask i -> nb_idx[i]
    covered = torch.zeros(len(X), dtype=torch.bool, device=X.device)
    if len(lab):
        covered[lab] = True
        covered[task.pc_idx[lab][E[lab]]] = True
    unl = _unl_mask(task, lab)
    picks = []
    for _ in range(B):
        gain = (E & ~covered[task.pc_idx]).sum(1) + (~covered).long()
        gain[~unl] = -1
        j = int(gain.argmax())
        if gain[j] <= 0:
            j = int(q_random(task, torch.cat([lab, torch.tensor(picks + [], device=X.device, dtype=torch.long)]),
                             1, pred, gen)[0])
        picks.append(j); unl[j] = False
        covered[j] = True
        covered[task.pc_idx[j][E[j]]] = True
    return torch.tensor(picks, device=X.device)


def q_badge(task, lab, B, pred, gen):
    """BADGE (Ash et al. 2020): k-means++ seeding on last-layer gradient embeddings
    g_i = (p_i - e_yhat) (x) x_i, using <g_i, g_j> = <a_i, a_j><x_i, x_j>."""
    if pred is None:
        return q_random(task, lab, B, pred, gen)
    X = task.X
    p = pred(X)
    A = p - F.one_hot(p.argmax(1), p.shape[1]).float()
    n2 = A.pow(2).sum(1) * X.pow(2).sum(1)
    unl = _unl_mask(task, lab)
    n2u = torch.where(unl, n2, torch.zeros_like(n2))
    j = int(n2u.argmax())
    picks = [j]
    d2 = (n2 + n2[j] - 2 * (A @ A[j]) * (X @ X[j])).clamp_min(0)
    d2[~unl] = 0
    while len(picks) < B:
        d2[picks[-1]] = 0
        pr = d2.double().cpu()
        if pr.sum() <= 0:
            break
        j = int(torch.multinomial(pr / pr.sum(), 1, generator=gen))
        picks.append(j)
        d2 = torch.minimum(d2, (n2 + n2[j] - 2 * (A @ A[j]) * (X @ X[j])).clamp_min(0))
        d2[~unl] = 0
    return torch.tensor(picks, device=X.device)


BASELINES = {"random": q_random, "entropy": q_entropy, "margin": q_margin, "coreset": q_coreset,
             "typiclust": q_typiclust, "probcover": q_probcover, "badge": q_badge}


# ----------------------------------------------------------------------------
# learned policy: per-candidate features -> MLP score -> Plackett-Luce sampling
# ----------------------------------------------------------------------------

POLICY_FEATURES = ["entropy", "margin", "max_p", "dist_labeled", "density", "region_uncovered",
                   "region_share", "class_share", "progress", "pc_gain", "grad_norm"]


def policy_features(task: Task, lab: torch.Tensor, pred, progress: float):
    X = task.X
    n = len(X)
    if pred is None:
        p = torch.full((n, task.n_cls), 1.0 / task.n_cls, device=X.device)
    else:
        p = pred(X)
    ent = -(p * p.clamp_min(1e-12).log()).sum(1) / math.log(task.n_cls)
    t2 = p.topk(2, 1).values
    if len(lab):
        dl = torch.full((n,), 4.0, device=X.device)
        for i in range(0, len(lab), 1024):
            dl = torch.minimum(dl, (2 - 2 * X @ X[lab[i:i + 1024]].T).min(1).values)
        covered = torch.zeros(len(task.reg_share), dtype=torch.bool, device=X.device)
        covered[task.reg[lab]] = True
        cnt = torch.bincount(task.y[lab], minlength=task.n_cls).float()
        share = cnt[p.argmax(1)] / len(lab)
    else:
        dl = torch.full((n,), 2.0, device=X.device)
        covered = torch.zeros(len(task.reg_share), dtype=torch.bool, device=X.device)
        share = torch.zeros(n, device=X.device)
    E = (1 - task.pc_sim) < task.pc_delta
    cov = torch.zeros(n, dtype=torch.bool, device=X.device)
    if len(lab):
        cov[lab] = True
        cov[task.pc_idx[lab][E[lab]]] = True
    pcg = (E & ~cov[task.pc_idx]).sum(1).float() / E.shape[1]
    A = p - F.one_hot(p.argmax(1), task.n_cls).float()
    gn = A.norm(dim=1)
    f = torch.stack([ent, t2[:, 0] - t2[:, 1], t2[:, 0], dl, task.density,
                     (~covered[task.reg]).float(), task.reg_share[task.reg].log(),
                     share, torch.full_like(ent, progress), pcg, gn], 1)
    # per-round standardisation of the continuous columns (binary / progress kept raw)
    keep = [5, 8]
    mu, sd = f.mean(0), f.std(0).clamp_min(1e-6)
    z = (f - mu) / sd
    z[:, keep] = f[:, keep]
    return z


POLICY_FEATURES_V2 = POLICY_FEATURES + ["clus_unlabeled", "clus_size", "clus_most_typical"]


def round_clusters(task, lab, B, gen, max_clusters: int = 500):
    """Per-round k-means with |L| + B clusters (TypiClust's choice), shared by the sub-batches."""
    K = min(len(lab) + B, max_clusters)
    return kmeans(task.X, K, iters=20, seed=int(torch.randint(1 << 30, (1,), generator=gen)))[1]


def cluster_features(task, cur, a):
    """(n, 3): cluster has no label yet; log cluster size share (z-scored); point is the densest
    unlabeled point of its cluster. Together they let the policy express TypiClust exactly."""
    K = int(a.max()) + 1
    n = len(task.X)
    size = torch.bincount(a, minlength=K).float()
    nlab = torch.bincount(a[cur], minlength=K) if len(cur) else torch.zeros(K, device=a.device)
    unl = _unl_mask(task, cur)
    dens = task.density.masked_fill(~unl, -1e9)
    mx = torch.full((K,), -1e9, device=a.device).scatter_reduce(0, a, dens, reduce="amax")
    top = ((dens >= mx[a]) & unl).float()
    cs = (size[a] / n).log()
    cs = (cs - cs.mean()) / cs.std().clamp_min(1e-6)
    return torch.stack([(nlab[a] == 0).float(), cs, top], 1)


class Policy(nn.Module):
    def __init__(self, n_feat: int = len(POLICY_FEATURES), hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, f):
        return self.net(f).squeeze(-1)


def q_policy_factory(policy: Policy, sub_batches: int = 4, greedy: bool = True, record: list | None = None,
                     n_cand: int | None = None, v2: bool = False, bc_target: bool = False):
    """Picks B points in ``sub_batches`` steps (features recomputed between steps so that
    distance / coverage features see the picks). Sampling is Plackett-Luce over a random
    candidate subset of size ``n_cand`` (default: all unlabeled; greedy top-k at evaluation). ``record`` collects
    (features, candidate ids, picks) for the policy-gradient update.
    ``v2`` adds the per-round cluster features. ``bc_target`` (behaviour cloning) follows
    TypiClust's ordered picks on the same clusters instead of the policy, and records them."""

    def q(task, lab, B, pred, gen, progress=0.0):
        picks = torch.zeros(0, dtype=torch.long, device=task.X.device)
        sizes = [B // sub_batches + (1 if i < B % sub_batches else 0) for i in range(sub_batches)]
        a = round_clusters(task, lab, B, gen) if v2 else None
        target = typiclust_order(task, lab, B, a) if bc_target else None
        for b in sizes:
            if b == 0:
                continue
            cur = torch.cat([lab, picks])
            f = policy_features(task, cur, pred, progress)
            if v2:
                f = torch.cat([f, cluster_features(task, cur, a)], 1)
            unl = _unl_mask(task, cur).nonzero().squeeze(1)
            cand = unl if n_cand is None else unl[torch.randperm(len(unl), generator=gen)[:n_cand].to(unl.device)]
            if bc_target:
                tgt = torch.tensor(target[len(picks):len(picks) + b], device=unl.device, dtype=torch.long)
                pos = torch.full((len(task.X),), -1, dtype=torch.long, device=unl.device)
                pos[cand] = torch.arange(len(cand), device=unl.device)
                sel = pos[tgt]
                sel = sel[sel >= 0]
                if record is not None:
                    record.append((f[cand].detach(), sel.detach()))
                picks = torch.cat([picks, cand[sel]])
                continue
            with torch.no_grad():
                s = policy(f[cand])
            if greedy:
                sel = s.topk(b).indices
            else:
                gmb = -torch.log(-torch.log(torch.rand(len(s), generator=gen).to(s.device).clamp(1e-9, 1 - 1e-9)))
                sel = (s + gmb).topk(b).indices        # Gumbel-top-k = Plackett-Luce sample
            if record is not None:
                record.append((f[cand].detach(), sel.detach()))
            picks = torch.cat([picks, cand[sel]])
        return picks

    return q


def pl_logprob(policy: Policy, feats: torch.Tensor, sel: torch.Tensor):
    """Plackett-Luce log-probability of the ordered selection ``sel`` among the candidates."""
    s = policy(feats)
    lp = 0.0
    mask = torch.zeros_like(s, dtype=torch.bool)
    for j in sel:
        lp = lp + s[j] - torch.logsumexp(s.masked_fill(mask, -1e9), 0)
        mask = mask.clone(); mask[j] = True
    return lp


# ----------------------------------------------------------------------------
# episode runner
# ----------------------------------------------------------------------------

def run_episode(task: Task, strategy, budgets, seed: int, eval_lp: bool = True, init_random: int = 0):
    """Grow the labeled set to each budget in turn; returns per-budget (linear, lp) accuracies."""
    gen = torch.Generator().manual_seed(seed)
    lab = torch.zeros(0, dtype=torch.long, device=task.X.device)
    pred = None
    lin_acc, lp_acc = [], []
    for b in budgets:
        need = b - len(lab)
        if len(lab) == 0 and init_random:
            new = q_random(task, lab, need, pred, gen)
        elif getattr(strategy, "__code__", None) and "progress" in strategy.__code__.co_varnames:
            new = strategy(task, lab, need, pred, gen, progress=len(lab) / budgets[-1])
        else:
            new = strategy(task, lab, need, pred, gen)
        lab = torch.cat([lab, new.long()]).unique()
        if eval_lp:
            l1, l2, pred = evaluate(task, lab)
        else:
            pred = fit_linear(task.X[lab], task.y[lab], task.n_cls)
            l1 = float((pred(task.Xt).argmax(1) == task.yt).float().mean()); l2 = float("nan")
        lin_acc.append(l1); lp_acc.append(l2)
    return lin_acc, lp_acc


def auc(acc, budgets):
    """Mean accuracy over budgets, trapezoid in log(budget) (equal weight per doubling)."""
    lb = [math.log(b) for b in budgets]
    tot = sum((acc[i] + acc[i + 1]) / 2 * (lb[i + 1] - lb[i]) for i in range(len(acc) - 1))
    return tot / (lb[-1] - lb[0])
