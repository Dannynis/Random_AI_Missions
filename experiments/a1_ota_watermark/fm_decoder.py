"""A1 decoder on a speech foundation model (WavLM), and the matched control.

The trained embedder (ours, seed s) is frozen; only a new decoder is trained, through the same training channel and
timing jitter as the original. ``--arch wavlm``: a frozen WavLM-base-plus (torchaudio's ``WAVLM_BASE_PLUS`` pipeline;
``--unfreeze N`` also trains its top N transformer layers) gives the 12 transformer layers' outputs at 50 Hz; a learned softmax-weighted sum
of all layers -> linear 768 -> 256 -> resampled to the original decoder's frame grid (hop 256, frame t centred at
t * 256) -> two temporal conv layers. ``--arch ours``: the original decoder architecture, trained from scratch by the same
recipe (the control: same embedder, data, channel, steps). Both keep the decoder interface (``frames`` per-frame
features, ``head`` on a window mean), so the CRC / list-Viterbi link layer and the blind timing search run unchanged.

usage: python fm_decoder.py --seed 0 --arch wavlm --steps 10000   (writes $RUN_DIR/ckpt/dec_<arch>_seed<s>.pt)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_a1 as R

BF16 = False
import wmx



class WavLMDecoder(nn.Module):
    def __init__(self, unfreeze=0, d=256):
        super().__init__()
        import torchaudio
        self.fm = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model()
        for p in self.fm.parameters():
            p.requires_grad = False
        self.unfreeze = unfreeze
        if unfreeze:
            for layer in self.fm.encoder.transformer.layers[-unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True
        nl = len(self.fm.encoder.transformer.layers)
        self.layer_w = nn.Parameter(torch.zeros(nl))
        self.proj = nn.Linear(768, d)
        self.conv = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, wmx.K))

    def frames(self, y):
        """(B, n) -> (B, n // HOP + 1, d) on the original decoder's frame grid."""
        x = (y - y.mean(-1, keepdim=True)) / (y.std(-1, keepdim=True) + 1e-5)
        with torch.set_grad_enabled(self.training and self.unfreeze > 0):
            hs, _ = self.fm.extract_features(x)                                # nl x (B, T50, 768)
        h = (torch.softmax(self.layer_w, 0)[:, None, None, None] * torch.stack(hs)).sum(0)
        h = self.proj(h).transpose(1, 2)                                     # (B, d, T50)
        T = y.shape[-1] // wmx.HOP + 1
        src = (torch.arange(h.shape[-1], device=y.device) * 320 + 200).float()  # WavLM frame centres (samples)
        dst = (torch.arange(T, device=y.device) * wmx.HOP).float()
        pos = ((dst - src[0]) / 320).clamp(0, h.shape[-1] - 1)
        i0 = pos.floor().long(); i1 = (i0 + 1).clamp(max=h.shape[-1] - 1); a = (pos - i0)[None, None]
        h = h[..., i0] * (1 - a) + h[..., i1] * a                            # (B, d, T)
        return self.conv(h).transpose(1, 2)


def build(arch, unfreeze=0):
    return WavLMDecoder(unfreeze) if arch == "wavlm" else wmx.Decoder()


def load_decoder(arch, seed):
    dirs = [d for d in os.environ.get("A1_CKPT", "").split(":") if d] + [R.CK]
    name = f"dec_{arch}_seed{seed}.pt"
    path = next((os.path.join(d, name) for d in dirs if os.path.exists(os.path.join(d, name))), None)
    if path is None:
        raise FileNotFoundError(name)
    st = torch.load(path, map_location=R.DEV, weights_only=False)
    dec = build(arch, st.get("unfreeze", 0)).to(R.DEV)
    dec.load_state_dict(st["dec"], strict=False)
    return dec.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="wavlm", choices=["wavlm", "ours"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unfreeze", type=int, default=0)
    a = ap.parse_args()
    dev = R.DEV
    global BF16
    BF16 = dev.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    R.say("bf16 autocast", BF16)
    os.makedirs(R.CK, exist_ok=True)
    final = os.path.join(R.CK, f"dec_{a.arch}_seed{a.seed}.pt")
    if os.path.exists(final):
        R.say("already trained"); return
    torch.manual_seed(a.seed)
    enc, dec0, ar = R.load_ours(a.seed)                                      # frozen embedder of the main run
    for p in enc.parameters():
        p.requires_grad = False
    dec = build(a.arch, a.unfreeze).to(dev).train()
    params = [p for p in dec.parameters() if p.requires_grad]
    R.say("decoder", a.arch, "trainable params", sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    corp = (wmx.Corpus("speech_train"), wmx.Corpus("music_train"))
    chan = wmx.Channel(dev, "train")
    S, hist, run, t1 = 3, [], {"loss": 0.0, "ber": 0.0, "n": 0}, time.time()
    for step in range(1, a.steps + 1):
        rng = np.random.default_rng([a.seed, 700_000 + step])
        g = torch.Generator(device=dev).manual_seed(a.seed * 10 ** 7 + 700_000 + step)
        host = R.host_batch(corp, rng, a.batch, S * wmx.N)
        bits = torch.randint(0, 2, (a.batch, S, wmx.K), generator=g, device=dev)
        with torch.no_grad():
            w = wmx.embed_chunks(enc, host, bits, ar["whr"]).float()
            y, off = chan.train(host + w, g, strength=min(1.0, step / (0.3 * a.steps)))
        if step == 1 or step % 1000 == 0:                                     # sanity: the original decoder on this batch
            with torch.no_grad():
                s_ = torch.round((wmx.N + off) / wmx.HOP).long()
                l0 = dec0.head(R.window_mean(dec0.frames(y).float(), s_)).float()
                R.say(f"step {step} original decoder ber on this batch: {((l0 > 0).long() != bits[:, 1]).float().mean().item():.4f}")
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=BF16):
            feats = dec.frames(y)
            jit = torch.randint(-3, 4, (a.batch,), generator=g, device=dev)
            s = torch.round((wmx.N + off) / wmx.HOP).long() + jit
            logits = dec.head(R.window_mean(feats.float(), s)).float()
        loss = F.binary_cross_entropy_with_logits(logits, bits[:, 1].float())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        run["loss"] += loss.item(); run["n"] += 1
        run["ber"] += ((logits > 0).long() != bits[:, 1]).float().mean().item()
        if step % 100 == 0 or step == a.steps:
            n = run["n"]
            hist.append({"step": step, "loss": run["loss"] / n, "ber": run["ber"] / n, "s_per_step": (time.time() - t1) / n})
            R.say(f"dec {a.arch} step {step}: loss={hist[-1]['loss']:.4f} ber={hist[-1]['ber']:.4f} s/step={hist[-1]['s_per_step']:.3f}")
            run = {"loss": 0.0, "ber": 0.0, "n": 0}; t1 = time.time()
    sd = {k: v for k, v in dec.state_dict().items() if not k.startswith("fm.") or a.unfreeze}
    R.save_atomic({"dec": sd, "arch": a.arch, "unfreeze": a.unfreeze, "hist": hist}, final)
    R.dump(hist, os.path.join(R.OUT, f"train_dec_{a.arch}_seed{a.seed}.json"))


if __name__ == "__main__":
    main()
