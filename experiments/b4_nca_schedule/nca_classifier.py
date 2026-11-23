"""Self-classifying digits with a neural cellular automaton (NCA), plus baselines.

Every ink pixel of a digit is a *cell* with a small state vector. One shared
local rule (3x3 perception -> 1x1 MLP, residual, stochastic firing) is iterated
T times; after the rollout each cell's first 10 channels are its vote for the
digit class. Cells can only exchange information through other living cells
(the alive mask = ink pixels), so two digits on the same canvas cannot hear
each other unless they touch. This is the Randazzo et al. 2020 "self-classifying
MNIST" setup, re-implemented small enough for a CPU.

Models (all ~8k parameters):

* ``GapCNN``   - ordinary strided CNN with global average pooling (baseline).
                 Fully convolutional, so it runs on any canvas, but it emits one
                 label per image.
* ``DilatedFCN`` - per-pixel classifier, untied dilated convs, receptive field
                 31 px, sees background pixels (no alive mask).
* ``NCA``      - the cellular automaton (options: alive mask on/off, fire rate).

Scoring is per digit: per-pixel models vote by majority over that digit's own
ink pixels; ``GapCNN`` predicts the argmax (single digit) or its top-2 classes
(two-digit scenes).

Data: ``load_mnist`` reads the raw IDX files in ``data/MNIST/raw``. Test
scenes (``canvas_single``, ``canvas_pair``, ``occlude``) are generated in code.
"""

from __future__ import annotations

import gzip
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALIVE_THR = 0.1
N_CLASSES = 10


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def _read_idx(path: Path) -> np.ndarray:
    raw = path.read_bytes() if path.suffix != ".gz" else gzip.decompress(path.read_bytes())
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(root: str | Path = "data/MNIST/raw"):
    """Return (xtr, ytr, xte, yte) as float32 tensors in [0, 1], shape (N, 28, 28)."""
    root = Path(root)

    def get(name):
        p = root / name
        return _read_idx(p if p.exists() else p.with_name(name + ".gz"))

    xtr = torch.tensor(get("train-images-idx3-ubyte"), dtype=torch.float32) / 255
    ytr = torch.tensor(get("train-labels-idx1-ubyte"), dtype=torch.long)
    xte = torch.tensor(get("t10k-images-idx3-ubyte"), dtype=torch.float32) / 255
    yte = torch.tensor(get("t10k-labels-idx1-ubyte"), dtype=torch.long)
    return xtr, ytr, xte, yte


def random_shift(x: torch.Tensor, max_shift: int, gen: torch.Generator) -> torch.Tensor:
    """Shift each (N, H, W) image by up to +-max_shift px (zero fill)."""
    if max_shift == 0:
        return x
    n, h, w = x.shape
    pad = F.pad(x, (max_shift,) * 4)
    dx = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen)
    dy = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen)
    out = torch.empty_like(x)
    for i in range(n):
        out[i] = pad[i, dy[i]:dy[i] + h, dx[i]:dx[i] + w]
    return out


def _bbox(img: np.ndarray):
    ys, xs = np.nonzero(img > ALIVE_THR)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def canvas_single(x: torch.Tensor, size: int, rng: np.random.Generator):
    """Paste each 28x28 digit (ink bbox) at a uniformly random spot of a size x size canvas.

    Returns canvases (N, size, size) and per-digit ink masks (N, 1, size, size)."""
    n = len(x)
    out = torch.zeros(n, size, size)
    xn = x.numpy()
    for i in range(n):
        y0, y1, x0, x1 = _bbox(xn[i])
        crop = x[i, y0:y1, x0:x1]
        h, w = crop.shape
        oy = rng.integers(0, size - h + 1)
        ox = rng.integers(0, size - w + 1)
        out[i, oy:oy + h, ox:ox + w] = crop
    masks = (out > ALIVE_THR).unsqueeze(1)
    return out, masks


def canvas_pair(xa: torch.Tensor, xb: torch.Tensor, gap: int, rng: np.random.Generator,
                height: int = 40, width: int = 72):
    """Two digits side by side with a horizontal gap of ``gap`` px between ink bboxes.

    The pair is vertically jittered and horizontally centred on the canvas.
    Returns canvases (N, H, W) and masks (N, 2, H, W) (one ink mask per digit).
    A negative gap means the boxes overlap; overlapping ink is assigned to
    neither digit (pixels where both are inked are dropped from both masks)."""
    n = len(xa)
    out = torch.zeros(n, height, width)
    masks = torch.zeros(n, 2, height, width, dtype=torch.bool)
    an, bn = xa.numpy(), xb.numpy()
    for i in range(n):
        ya0, ya1, xa0, xa1 = _bbox(an[i])
        yb0, yb1, xb0, xb1 = _bbox(bn[i])
        ca, cb = xa[i, ya0:ya1, xa0:xa1], xb[i, yb0:yb1, xb0:xb1]
        wa, wb = ca.shape[1], cb.shape[1]
        total = wa + gap + wb
        left = (width - total) // 2
        pa = torch.zeros(height, width)
        pb = torch.zeros(height, width)
        oya = rng.integers(0, height - ca.shape[0] + 1)
        oyb = rng.integers(0, height - cb.shape[0] + 1)
        pa[oya:oya + ca.shape[0], left:left + wa] = ca
        pb[oyb:oyb + cb.shape[0], left + wa + gap:left + wa + gap + wb] = cb
        out[i] = torch.maximum(pa, pb)
        ma, mb = pa > ALIVE_THR, pb > ALIVE_THR
        both = ma & mb
        masks[i, 0] = ma & ~both
        masks[i, 1] = mb & ~both
    return out, masks


def occlude(x: torch.Tensor, patch: int, rng: np.random.Generator) -> torch.Tensor:
    """Erase a random patch x patch square centred somewhere on the digit's ink."""
    out = x.clone()
    for i in range(len(x)):
        ys, xs = np.nonzero(x[i].numpy() > ALIVE_THR)
        j = rng.integers(len(ys))
        cy, cx = ys[j], xs[j]
        y0, x0 = max(cy - patch // 2, 0), max(cx - patch // 2, 0)
        out[i, y0:y0 + patch, x0:x0 + patch] = 0
    return out


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------

class GapCNN(nn.Module):
    """Standard small CNN: conv/max-pool x2, conv, global average pool, linear.

    One 10-way prediction per image; fully convolutional up to the pool, so it
    accepts any canvas size."""

    per_pixel = False

    def __init__(self, widths=(12, 20, 28)):
        super().__init__()
        a, b, c = widths
        self.features = nn.Sequential(
            nn.Conv2d(1, a, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(b, c, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(c, N_CLASSES)

    def forward(self, img: torch.Tensor) -> torch.Tensor:  # (N,H,W) -> (N,10)
        return self.head(self.features(img.unsqueeze(1)).mean((2, 3)))


class DilatedFCN(nn.Module):
    """Per-pixel classifier: untied dilated 3x3 convs (RF 31 px), no alive mask."""

    per_pixel = True

    def __init__(self, width: int = 17, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers, cin = [], 1
        for d in dilations:
            layers += [nn.Conv2d(cin, width, 3, padding=d, dilation=d), nn.ReLU()]
            cin = width
        layers.append(nn.Conv2d(width, N_CLASSES, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor) -> torch.Tensor:  # (N,H,W) -> (N,10,H,W)
        return self.net(img.unsqueeze(1))


class NCA(nn.Module):
    """Neural cellular automaton; state channels [0:10] are the class logits.

    x_{t+1} = alive * (x_t + fire_t * MLP(conv3x3([x_t, img])))
    with fire_t ~ Bernoulli(fire_rate) per cell and step.
    """

    per_pixel = True

    def __init__(self, channels: int = 16, hidden: int = 48, fire_rate: float = 0.5,
                 use_alive: bool = True):
        super().__init__()
        self.c = channels
        self.fire_rate = fire_rate
        self.use_alive = use_alive
        self.perceive = nn.Conv2d(channels + 1, hidden, 3, padding=1)
        self.update = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)

    def rollout(self, img: torch.Tensor, steps: int, record: tuple[int, ...] = (),
                detach: bool = True):
        """Run ``steps`` updates. Returns the final state (N,C,H,W) and the logits
        recorded after each step in ``record`` (kept in the graph if ``detach=False``)."""
        n, h, w = img.shape
        im = img.unsqueeze(1)
        alive = (im > ALIVE_THR).float() if self.use_alive else torch.ones_like(im)
        x = torch.zeros(n, self.c, h, w, device=img.device)
        recs = {}
        for t in range(1, steps + 1):
            dx = self.update(F.relu(self.perceive(torch.cat([x, im], 1))))
            if self.fire_rate < 1.0:
                dx = dx * (torch.rand(n, 1, h, w, device=img.device) < self.fire_rate).float()
            x = (x + dx) * alive
            if t in record:
                recs[t] = x[:, :N_CLASSES].detach() if detach else x[:, :N_CLASSES]
        return x, recs

    def forward(self, img: torch.Tensor, steps: int = 40) -> torch.Tensor:
        return self.rollout(img, steps)[0][:, :N_CLASSES]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    iters: int = 2000
    batch: int = 32
    lr: float = 2e-3
    max_shift: int = 2
    steps_min: int = 24          # NCA rollout length sampled in [steps_min, steps_max]
    steps_max: int = 40
    clip: float = 1.0
    late_steps: tuple = (12, 8, 4, 0)   # NCA: CE averaged over steps T-k (k in late_steps)
    seed: int = 0


@dataclass
class TrainLog:
    losses: list = field(default_factory=list)
    seconds: float = 0.0


def pixel_ce(logits: torch.Tensor, img: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged over ink pixels (logits (N,10,H,W), img (N,H,W))."""
    mask = (img > ALIVE_THR).float()
    ce = F.cross_entropy(logits, y[:, None, None].expand_as(img), reduction="none")
    return ((ce * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)).mean()


def train(model: nn.Module, xtr: torch.Tensor, ytr: torch.Tensor, cfg: TrainConfig) -> TrainLog:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters)
    log = TrainLog()
    t0 = time.time()
    model.train()
    for it in range(cfg.iters):
        idx = torch.randint(0, len(xtr), (cfg.batch,), generator=gen)
        img, y = random_shift(xtr[idx], cfg.max_shift, gen), ytr[idx]
        if isinstance(model, NCA):
            steps = int(torch.randint(cfg.steps_min, cfg.steps_max + 1, (1,), generator=gen))
            marks = tuple(steps - k for k in cfg.late_steps)
            _, recs = model.rollout(img, steps, record=marks, detach=False)
            loss = sum(pixel_ce(recs[t], img, y) for t in marks) / len(marks)
        elif model.per_pixel:
            loss = pixel_ce(model(img), img, y)
        else:
            loss = F.cross_entropy(model(img), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        sched.step()
        if it % 50 == 0:
            log.losses.append(float(loss.detach()))
    log.seconds = time.time() - t0
    return log


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

def vote(logits: torch.Tensor, masks: torch.Tensor):
    """Majority vote of per-pixel argmax inside each mask.

    logits (N,10,H,W); masks (N,K,H,W) bool. Returns (pred (N,K), agreement (N,K)):
    agreement = fraction of the mask's pixels voting for the winning class."""
    am = logits.argmax(1)                                     # (N,H,W)
    onehot = F.one_hot(am, N_CLASSES).permute(0, 3, 1, 2)     # (N,10,H,W)
    counts = torch.einsum("nchw,nkhw->nkc", onehot.float(), masks.float())
    pred = counts.argmax(-1)
    agree = counts.max(-1).values / masks.float().sum((2, 3)).clamp_min(1)
    return pred, agree


@torch.no_grad()
def predict(model: nn.Module, imgs: torch.Tensor, masks: torch.Tensor, steps: int = 40,
            chunk: int = 500, seed: int = 123):
    """Per-digit predictions. Per-pixel models vote inside each mask; GapCNN
    returns its top-K classes (K = number of digits in the scene).

    Returns (pred (N,K), agreement (N,K) or None)."""
    model.eval()
    torch.manual_seed(seed)
    k = masks.shape[1]
    preds, agrees = [], []
    for s in range(0, len(imgs), chunk):
        im, m = imgs[s:s + chunk], masks[s:s + chunk]
        if model.per_pixel:
            logits = model(im, steps) if isinstance(model, NCA) else model(im)
            p, a = vote(logits, m)
            preds.append(p)
            agrees.append(a)
        else:
            preds.append(model(im).topk(k, -1).indices)
    return torch.cat(preds), (torch.cat(agrees) if agrees else None)


def digit_accuracy(pred: torch.Tensor, labels: torch.Tensor, set_match: bool = False) -> float:
    """labels (N,K). For GapCNN top-K on pairs use set_match (each true label in the top-K)."""
    if set_match:
        hit = torch.stack([(pred == labels[:, j:j + 1]).any(1) for j in range(labels.shape[1])], 1)
    else:
        hit = pred == labels
    return float(hit.float().mean())


@torch.no_grad()
def nca_step_curve(model: NCA, imgs: torch.Tensor, labels: torch.Tensor, steps: tuple[int, ...],
                   seed: int = 123):
    """Single-digit accuracy and agreement of an NCA read out after each step in ``steps``."""
    model.eval()
    torch.manual_seed(seed)
    masks = (imgs > ALIVE_THR).unsqueeze(1)
    _, recs = model.rollout(imgs, max(steps), record=steps)
    out = {}
    for t in steps:
        p, a = vote(recs[t], masks)
        out[t] = (float((p[:, 0] == labels).float().mean()), float(a.mean()))
    return out
