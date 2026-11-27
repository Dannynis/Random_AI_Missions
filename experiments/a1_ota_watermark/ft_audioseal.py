"""AudioSeal fine-tuned on our training channel (fairness control for A1).

The public AudioSeal 16-bit generator and detector are trained further through exactly the play-and-record channel
that ours is trained on (``wmx.Channel(...).train``, same schedule of channel strength), so the comparison separates
"trained for the acoustic channel" from "our architecture and link layer". Losses: message BCE on the detector's
window-averaged message logits, frame-level detection cross-entropy on the marked recording against an unmarked twin
through the same channel, and a log-spectral-distance hinge that keeps the mark no more visible than the public model's
(its initial LSD on the first batches). Audibility is matched at evaluation by PESQ, as for every method.

usage: python ft_audioseal.py --seed 0 --steps 8000   (writes $RUN_DIR/ckpt/audioseal_ft_seed<seed>.pt)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import run_a1 as R
import wmx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--det_weight", type=float, default=1.0)
    ap.add_argument("--lsd_weight", type=float, default=1.0)
    a = ap.parse_args()
    from audioseal import AudioSeal
    dev = R.DEV
    gen = AudioSeal.load_generator("audioseal_wm_16bits").to(dev).train()
    det = AudioSeal.load_detector("audioseal_detector_16bits").to(dev).train()
    os.makedirs(R.CK, exist_ok=True)
    final = os.path.join(R.CK, f"audioseal_ft_seed{a.seed}.pt")
    if os.path.exists(final):
        R.say("already trained"); return
    params = list(gen.parameters()) + list(det.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    corp = (wmx.Corpus("speech_train"), wmx.Corpus("music_train"))
    chan = wmx.Channel(dev, "train")
    L = 3 * wmx.N
    d0, hist, run, t1 = None, [], {"msg": 0.0, "det": 0.0, "lsd": 0.0, "ber": 0.0, "n": 0}, time.time()
    for step in range(1, a.steps + 1):
        rng = np.random.default_rng([a.seed, 900_000 + step])
        g = torch.Generator(device=dev).manual_seed(a.seed * 10 ** 7 + 900_000 + step)
        host = R.host_batch(corp, rng, a.batch, L)
        bits = torch.randint(0, 2, (a.batch, 16), generator=g, device=dev)
        w = gen.get_watermark(host[:, None], wmx.SR, message=bits)[:, 0]
        strength = min(1.0, step / (0.3 * a.steps))
        g_state = g.get_state()
        y, off = chan.train(host + w, g, strength=strength)
        g.set_state(g_state)                                   # the unmarked twin goes through the same channel draw
        y0, _ = chan.train(host, g, strength=strength)
        # the public 16-bit decoder reads the message only on its 320-sample embedding grid, and the channel shifts the
        # audio by up to MAX_OFFSET samples: decode a crop that starts on the grid (what the sync search finds at test)
        s0 = (torch.round(off).long() % 320)
        Lc = L - wmx.MAX_OFFSET - 320
        idx = s0[:, None] + torch.arange(Lc, device=dev)[None]
        y, y0 = y.gather(1, idx), y0.gather(1, idx)
        raw = det.detector(torch.cat([y, y0])[:, None])        # (2B, 2 + 16, T)
        msg_logit = raw[:a.batch, 2:].mean(-1)
        lm = F.binary_cross_entropy_with_logits(msg_logit, bits.float())
        lab = torch.cat([torch.ones(a.batch, raw.shape[-1]), torch.zeros(a.batch, raw.shape[-1])]).long().to(dev)
        ld = F.cross_entropy(raw[:, :2], lab)
        d = wmx.lsd(host, w).mean()
        if d0 is None:
            d0 = float(d.detach())
        tot = lm + a.det_weight * ld + a.lsd_weight * F.relu(d - d0)
        opt.zero_grad(set_to_none=True); tot.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        run["msg"] += lm.item(); run["det"] += ld.item(); run["lsd"] += d.item(); run["n"] += 1
        run["ber"] += ((msg_logit > 0).long() != bits).float().mean().item()
        if step % 100 == 0 or step == a.steps:
            n = run["n"]
            hist.append({"step": step, **{k: v / n for k, v in run.items() if k != "n"}, "lsd0": d0,
                         "s_per_step": (time.time() - t1) / n})
            R.say("audioseal_ft", ", ".join(f"{k}={v:.4g}" for k, v in hist[-1].items()))
            run = {k: 0.0 for k in run}; run["n"] = 0; t1 = time.time()
    R.save_atomic({"gen": gen.state_dict(), "det": det.state_dict(), "hist": hist, "lsd0": d0}, final)
    R.dump(hist, os.path.join(R.OUT, f"train_audioseal_ft_seed{a.seed}.json"))


if __name__ == "__main__":
    main()
