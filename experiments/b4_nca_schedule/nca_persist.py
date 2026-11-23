"""Persistent self-classifying NCA: train the vote to be an attractor, not a transient.

The 2026-10-23 NCA classifier (``nca_classifier.py``) peaks at the readout step
it was trained for (~0.90 at 40 steps) and then decays (0.68 at 80, 0.17 at 200):
its classification is a transient. This module adds training schemes that aim to
turn it into an attractor, and diagnostics that probe it:

Training schemes (all keep the exact ``NCA`` rule: 16 ch, 3x3 -> 48 -> 1x1, ~8k params)

* ``"fixed"``  - the 10-23 recipe: rollout from zero state, T ~ U[24, 40], CE at
                 T-12/-8/-4/T. Baseline.
* ``"pool"``   - Mordvintsev-style sample pool: a pool of (digit, label, state)
                 entries persists across iterations. Each batch is drawn from the
                 pool, a few entries (random, or the oldest) are re-seeded with fresh
                 digits at zero state, the
                 rule is rolled out T ~ U[16, 32] more steps *from the stored state*,
                 CE is taken at the same late checkpoints, and the detached final
                 states are written back. Entries that keep being drawn have been
                 rolled for hundreds of steps, so the loss covers long horizons at
                 the cost of short BPTT windows.
  ``reseed_end`` anneals the re-seed count during training (2026-11-23): many
                 fresh digits early (throughput, peak accuracy), few late (old states,
                 persistence).
* ``"warmup"`` - truncated BPTT without a pool: run W ~ U[0, w_max] steps without
                 gradient from zero state, then the usual T ~ U[24, 40] graded steps.
                 Same horizon coverage as the pool for the warm-up range, no reuse.

Options orthogonal to the scheme: ``fire_rate`` (stochastic vs deterministic
updates), ``dilate`` (alive mask dilated by k px so dead neighbours next to ink can
relay messages: the occlusion fix proposed in the 10-23 report), and
``overflow`` (penalty on |state| above a bound, as in growing-NCA work).

Diagnostics

* ``step_curve``   - accuracy/agreement of the per-digit vote vs readout step, to 1000+.
* ``damage_curve`` - at step ``t_damage``, zero the state of a k x k patch of cells
                     (cells stay alive) and track accuracy and the patch's agreement
                     with the digit's vote as the rule re-grows it (self-repair).
* ``swap_curve``   - at step ``t_swap``, replace the digit under the cells by another
                     digit (new alive mask, state kept where still alive) and track how
                     fast the vote re-converges to the new label (re-classification).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca_classifier import ALIVE_THR, N_CLASSES, NCA, pixel_ce, random_shift, vote


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

class PersistentNCA(NCA):
    """``NCA`` with an optional initial state, per-step hook and a dilated alive mask.

    ``dilate = k`` makes every cell within k px (Chebyshev) of ink alive. Votes are
    still read only on ink pixels, so the scoring is unchanged; the extra cells are
    relays that let stroke fragments separated by a small gap exchange messages.
    """

    def __init__(self, channels: int = 16, hidden: int = 48, fire_rate: float = 0.5,
                 dilate: int = 0):
        super().__init__(channels, hidden, fire_rate, use_alive=True)
        self.dilate = dilate

    def alive_mask(self, img: torch.Tensor) -> torch.Tensor:
        """(N,H,W) image -> (N,1,H,W) float alive mask."""
        a = (img.unsqueeze(1) > ALIVE_THR).float()
        if self.dilate > 0:
            a = F.max_pool2d(a, 2 * self.dilate + 1, stride=1, padding=self.dilate)
        return a

    def step(self, x: torch.Tensor, im: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        dx = self.update(F.relu(self.perceive(torch.cat([x, im], 1))))
        if self.fire_rate < 1.0:
            n, _, h, w = x.shape
            dx = dx * (torch.rand(n, 1, h, w, device=x.device) < self.fire_rate).float()
        return (x + dx) * alive

    def rollout(self, img: torch.Tensor, steps: int, record: tuple[int, ...] = (),
                detach: bool = True, x0: torch.Tensor | None = None, hook=None):
        """Run ``steps`` updates from ``x0`` (zeros if None).

        ``hook(t, x, img) -> (x, img) | None`` is called before step t (1-based) and
        may replace the state or the image (used by the damage / swap probes).
        Returns final state and {t: logits} for t in ``record``."""
        n, h, w = img.shape
        x = torch.zeros(n, self.c, h, w) if x0 is None else x0
        im, alive = img.unsqueeze(1), self.alive_mask(img)
        recs = {}
        for t in range(1, steps + 1):
            if hook is not None:
                out = hook(t, x, img)
                if out is not None:
                    x, img = out
                    im, alive = img.unsqueeze(1), self.alive_mask(img)
                    x = x * alive
            x = self.step(x, im, alive)
            if t in record:
                recs[t] = x[:, :N_CLASSES].detach() if detach else x[:, :N_CLASSES]
        return x, recs


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

@dataclass
class PersistConfig:
    scheme: str = "fixed"            # fixed | pool | warmup
    iters: int = 4000
    batch: int = 32
    lr: float = 2e-3
    max_shift: int = 2
    clip: float = 1.0
    steps_min: int = 24              # graded rollout length (fixed / warmup)
    steps_max: int = 40
    pool_steps_min: int = 16         # graded rollout length (pool)
    pool_steps_max: int = 32
    late_steps: tuple = (12, 8, 4, 0)
    pool_size: int = 1024
    reseed: int = 2                  # pool entries per batch replaced by fresh digits
    reseed_mode: str = "random"      # "random" (age ~ geometric, long tail) | "oldest"
    reseed_end: int | None = None    # if set: re-seed count annealed linearly from ``reseed``
                                     # (iteration 0) to ``reseed_end`` (at ``reseed_anneal_frac``
                                     # of training), then held. None = constant ``reseed``.
    reseed_anneal_frac: float = 1.0
    reseed_switch_frac: float | None = None  # if set: step schedule, ``reseed`` until this
                                     # fraction of training, ``reseed_end`` afterwards
    warmup_max: int = 120            # warmup scheme: W ~ U[0, warmup_max]
    overflow: float = 0.0            # weight of mean(relu(|x| - overflow_bound)^2)
    overflow_bound: float = 5.0
    seed: int = 0


@dataclass
class PersistLog:
    losses: list = field(default_factory=list)
    pool_age: list = field(default_factory=list)   # mean / max age (steps) of drawn pool states
    seconds: float = 0.0


def reseed_count(cfg: PersistConfig, it: int) -> int:
    """Pool entries re-seeded with fresh digits at iteration ``it`` (constant or annealed).

    The annealed schedule interpolates linearly between ``cfg.reseed`` and
    ``cfg.reseed_end`` over the first ``reseed_anneal_frac`` of training and rounds to
    the nearest integer, e.g. 8 -> 2 over 4000 iterations gives a mean of ~5 per batch."""
    if cfg.reseed_end is None:
        return cfg.reseed
    if cfg.reseed_switch_frac is not None:
        return cfg.reseed if it < cfg.reseed_switch_frac * cfg.iters else cfg.reseed_end
    span = max(1.0, cfg.reseed_anneal_frac * cfg.iters)
    frac = min(1.0, it / span)
    return int(round(cfg.reseed + frac * (cfg.reseed_end - cfg.reseed)))


def _overflow(x: torch.Tensor, bound: float) -> torch.Tensor:
    return F.relu(x.abs() - bound).pow(2).mean()


def train_persistent(model: PersistentNCA, xtr: torch.Tensor, ytr: torch.Tensor,
                     cfg: PersistConfig) -> PersistLog:
    """Train ``model`` with the scheme in ``cfg`` (see module docstring)."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters)
    log = PersistLog()
    model.train()
    h, w = xtr.shape[1:]

    def fresh(k):
        idx = torch.randint(0, len(xtr), (k,), generator=gen)
        return random_shift(xtr[idx], cfg.max_shift, gen), ytr[idx]

    if cfg.scheme == "pool":
        p_img, p_y = fresh(cfg.pool_size)
        p_x = torch.zeros(cfg.pool_size, model.c, h, w)
        p_age = torch.zeros(cfg.pool_size, dtype=torch.long)

    t0 = time.time()
    for it in range(cfg.iters):
        x0 = None
        if cfg.scheme == "pool":
            sel = torch.randperm(cfg.pool_size, generator=gen)[:cfg.batch]
            img, y = p_img[sel].clone(), p_y[sel].clone()
            x0 = p_x[sel].clone()
            ages = p_age[sel].clone()
            # re-seed some entries of the batch with fresh digits at zero state. "oldest"
            # caps the state age at a few rollouts; "random" gives a geometric age
            # distribution (mean ~ batch / reseed rollouts) with a long tail.
            k = reseed_count(cfg, it)
            if cfg.reseed_mode == "oldest":
                order = ages.argsort(descending=True)[:k]
            else:
                order = torch.randperm(cfg.batch, generator=gen)[:k]
            ni, ny = fresh(k)
            img[order], y[order], x0[order], ages[order] = ni, ny, 0.0, 0
            steps = int(torch.randint(cfg.pool_steps_min, cfg.pool_steps_max + 1, (1,),
                                      generator=gen))
        else:
            img, y = fresh(cfg.batch)
            steps = int(torch.randint(cfg.steps_min, cfg.steps_max + 1, (1,), generator=gen))
            if cfg.scheme == "warmup":
                warm = int(torch.randint(0, cfg.warmup_max + 1, (1,), generator=gen))
                if warm:
                    with torch.no_grad():
                        x0, _ = model.rollout(img, warm)
        marks = tuple(steps - k for k in cfg.late_steps)
        x, recs = model.rollout(img, steps, record=marks, detach=False, x0=x0)
        loss = sum(pixel_ce(recs[t], img, y) for t in marks) / len(marks)
        if cfg.overflow > 0:
            loss = loss + cfg.overflow * _overflow(x, cfg.overflow_bound)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        sched.step()
        if cfg.scheme == "pool":
            p_img[sel], p_y[sel], p_x[sel] = img, y, x.detach()
            p_age[sel] = ages + steps
        if it % 50 == 0:
            log.losses.append(float(loss.detach()))
            if cfg.scheme == "pool":
                log.pool_age.append((float(p_age.float().mean()), int(p_age.max())))
    log.seconds = time.time() - t0
    return log


# ----------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------

@torch.no_grad()
def step_curve(model: PersistentNCA, imgs, labels, steps, seed: int = 123, chunk: int = 500):
    """{t: (accuracy, mean agreement, mean |state| on alive cells)} for t in ``steps``."""
    model.eval()
    torch.manual_seed(seed)
    acc = {t: [] for t in steps}
    agr = {t: [] for t in steps}
    mag = {t: [] for t in steps}
    for s in range(0, len(imgs), chunk):
        im, lab = imgs[s:s + chunk], labels[s:s + chunk]
        masks = (im > ALIVE_THR).unsqueeze(1)
        mags = {}

        def hook(t, x, img):
            if (t - 1) in acc:
                mags[t - 1] = float(x.abs().sum() / (masks.sum() * x.shape[1]))
            return None

        _, recs = model.rollout(im, max(steps) + 1, record=steps, hook=hook)
        for t in steps:
            p, a = vote(recs[t], masks)
            acc[t].append(float((p[:, 0] == lab).float().sum()))
            agr[t].append(float(a.sum()))
            mag[t].append(mags.get(t, float("nan")) * len(im))
    n = len(imgs)
    return {t: (sum(acc[t]) / n, sum(agr[t]) / n, sum(mag[t]) / n) for t in steps}


@torch.no_grad()
def damage_curve(model: PersistentNCA, imgs, labels, t_damage: int, patch: int, steps,
                 seed: int = 123):
    """Zero the state of a ``patch`` x ``patch`` square centred on a random ink pixel at
    step ``t_damage``. Returns {t: (accuracy, agreement of the damaged ink cells with the
    true label)} for t in ``steps`` (absolute step indices, > t_damage)."""
    model.eval()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, h, w = imgs.shape
    dmask = torch.zeros(n, 1, h, w)
    for i in range(n):
        ys, xs = np.nonzero(imgs[i].numpy() > ALIVE_THR)
        j = rng.integers(len(ys))
        y0, x0 = max(ys[j] - patch // 2, 0), max(xs[j] - patch // 2, 0)
        dmask[i, 0, y0:y0 + patch, x0:x0 + patch] = 1
    ink = (imgs > ALIVE_THR).unsqueeze(1)
    dmask_ink = (dmask.bool() & ink)

    def hook(t, x, img):
        if t == t_damage + 1:
            return x * (1 - dmask), img
        return None

    _, recs = model.rollout(imgs, max(steps), record=tuple(steps), hook=hook)
    out = {}
    for t in steps:
        p, _ = vote(recs[t], ink)
        cell = recs[t].argmax(1, keepdim=True) == labels[:, None, None, None]
        frac = (cell & dmask_ink).sum() / dmask_ink.sum().clamp_min(1)
        out[t] = (float((p[:, 0] == labels).float().mean()), float(frac))
    return out


@torch.no_grad()
def swap_curve(model: PersistentNCA, imgs_a, imgs_b, labels_b, t_swap: int, steps,
               seed: int = 123):
    """Roll out on digit A, then at step ``t_swap`` replace the image by digit B (labels
    differ). State is kept where the new alive mask overlaps the old one. Returns
    {t: accuracy w.r.t. B} for t in ``steps`` (absolute step indices > t_swap), plus the
    accuracy of a fresh rollout on B at the same number of post-swap steps."""
    model.eval()
    torch.manual_seed(seed)

    def hook(t, x, img):
        if t == t_swap + 1:
            return x, imgs_b
        return None

    _, recs = model.rollout(imgs_a, max(steps), record=tuple(steps), hook=hook)
    torch.manual_seed(seed)
    rel = tuple(t - t_swap for t in steps)
    _, fresh = model.rollout(imgs_b, max(rel), record=rel)
    ink = (imgs_b > ALIVE_THR).unsqueeze(1)
    out = {}
    for t, r in zip(steps, rel):
        p, _ = vote(recs[t], ink)
        pf, _ = vote(fresh[r], ink)
        out[t] = (float((p[:, 0] == labels_b).float().mean()),
                  float((pf[:, 0] == labels_b).float().mean()))
    return out
