"""LLM weight generator: a pretrained decoder LLM (LoRA) reads a few labelled rows of any table and writes
the weights of a d-128-C ReLU MLP, on top of a fitted ridge read-out (the scaffold).

Layout follows the Random-Ideas `TabHyper` (Part V of the owner's notebook) with PR #34's ridge scaffold:

    [text: task + feature names] [one token per labelled row] [text: " Weights:"]
    [one token per feature] [one token per class] [one bias token]

The last d + C + 1 hidden states go through zero-initialised linear heads:
    feature token j -> column j of the delta on W1 (H values)
    class token c   -> row c of the delta on W2 and b2 (H + 1 values)
    bias token      -> delta on b1 (H values)
W1 starts as [+I; -I] (so ReLU features contain the raw inputs), b1 = 0, and the second layer is a ridge
regression fitted in closed form on the support rows' generated features (R2D2, dual form), so with all
heads silenced the model is exactly the "fitted scaffold" baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

H = 128
D_MAX, C_MAX = 64, 10
G_SCALE = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def ridge_head(feats, lam_rel):
    """feats (B,C,k,H) -> W (B,C,H), b (B,C): ridge onto centred one-hot targets, dual form, bias column."""
    B, C, k, Hh = feats.shape
    X = feats.reshape(B, C * k, Hh)
    X1 = torch.cat([X, X.new_ones(B, C * k, 1)], -1)
    Y = F.one_hot(torch.arange(C, device=X.device).repeat_interleave(k), C).to(X.dtype)
    Y = (Y - 1.0 / C)[None].expand(B, -1, -1)
    G = X1 @ X1.transpose(1, 2)
    lam = lam_rel * G.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-6)
    A = torch.linalg.solve(G + lam[:, None, None] * torch.eye(C * k, dtype=X.dtype, device=X.device), Y)
    Wf = X1.transpose(1, 2) @ A
    return Wf[:, :Hh].transpose(1, 2), Wf[:, Hh].contiguous()


def ridge_loo_acc(feats, lam_rel):
    """Leave-one-out accuracy of the ridge read-out on the support rows (closed form). (B,) tensor."""
    B, C, k, Hh = feats.shape
    X = feats.reshape(B, C * k, Hh)
    X1 = torch.cat([X, X.new_ones(B, C * k, 1)], -1).double()
    lab = torch.arange(C, device=X.device).repeat_interleave(k)
    Y = (F.one_hot(lab, C).double() - 1.0 / C)[None].expand(B, -1, -1)
    G = X1 @ X1.transpose(1, 2)
    lam = lam_rel * G.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-6)
    S = G @ torch.linalg.inv(G + lam[:, None, None] * torch.eye(C * k, dtype=G.dtype, device=G.device))
    P = S @ Y
    s = S.diagonal(dim1=1, dim2=2)[..., None]
    loo = (P - s * Y) / (1 - s).clamp_min(1e-6)
    return (loo.argmax(-1) == lab[None]).double().mean(-1)


def mlp_logits(W, x):
    W1, b1, W2, b2 = W
    h = F.relu(torch.einsum("bmi,bhi->bmh", x, W1) + b1[:, None])
    return torch.einsum("bmh,bch->bmc", h, W2) + b2[:, None]


class Generator(nn.Module):
    def __init__(self, llm_path, lora_r=16, random_init=False, grad_ckpt=False):
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(llm_path)
        if random_init:
            cfg = AutoConfig.from_pretrained(llm_path)
            torch.manual_seed(1234)
            llm = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16)
        else:
            llm = AutoModel.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        if grad_ckpt:
            llm.gradient_checkpointing_enable()
            llm.enable_input_require_grads()
        self.full_ft = random_init
        if random_init:                       # no pretrained weights to adapt: train the whole network
            self.llm = llm.float()
            llm.get_input_embeddings().weight.requires_grad_(False)   # only the few text tokens use it
        else:
            from peft import LoraConfig, get_peft_model
            for p in llm.parameters():
                p.requires_grad_(False)
            self.llm = get_peft_model(llm, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                      target_modules=LORA_TARGETS, bias="none"))
        dm = llm.config.hidden_size
        with torch.no_grad():
            self.escale = float(llm.get_input_embeddings().weight.float().pow(2).mean().sqrt())
        self.dm = dm
        self.feat_slot = nn.Parameter(torch.randn(D_MAX, dm) * 0.5)
        self.cls_slot = nn.Parameter(torch.randn(C_MAX, dm) * 0.5)
        self.bias_q = nn.Parameter(torch.randn(dm) * 0.5)
        self.row_in = nn.Linear(D_MAX, dm)
        self.stat_in = nn.Linear(C_MAX + 2, dm)
        self.out_norm = nn.LayerNorm(dm)
        self.head_f = nn.Linear(dm, H)
        self.head_c = nn.Linear(dm, H + 1)
        self.head_b = nn.Linear(dm, H)
        for hd in (self.head_f, self.head_c, self.head_b):
            nn.init.zeros_(hd.weight)
            nn.init.zeros_(hd.bias)
        w1 = torch.randn(H, D_MAX) * math.sqrt(2.0 / D_MAX)
        n_id = min(D_MAX, H // 2)
        w1[:n_id] = torch.eye(D_MAX)[:n_id]
        w1[n_id:2 * n_id] = -torch.eye(D_MAX)[:n_id]
        self.W1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(torch.zeros(H))
        self.log_lam = nn.Parameter(torch.tensor(math.log(0.1)))
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self._tcache = {}

    # ------------------------------------------------------------------ parameter groups
    def groups(self):
        llm = [p for p in self.llm.parameters() if p.requires_grad]
        scaf = [self.log_lam, self.log_scale, self.W1, self.b1]
        ids = {id(p) for p in llm + scaf}
        new = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        return llm, new, scaf

    def trainable_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: t.detach().cpu() for n, t in self.state_dict().items() if n in keep}

    # ------------------------------------------------------------------ tokens
    def _n(self, v):
        return F.normalize(v, dim=-1) * self.escale * math.sqrt(self.dm)

    def text(self, s, dev):
        if s not in self._tcache:
            self._tcache[s] = torch.tensor(self.tok(s, add_special_tokens=False)["input_ids"], dtype=torch.long)
        return self.llm.get_input_embeddings()(self._tcache[s].to(dev)).float()

    def build(self, xs, names):
        B, C, k, d = xs.shape
        dev = xs.device
        stats = xs.new_zeros(B, d, C_MAX + 2)
        stats[:, :, :C] = xs.mean(2).transpose(1, 2)
        flat = xs.reshape(B, -1, d)
        stats[:, :, C_MAX] = flat.mean(1)
        stats[:, :, C_MAX + 1] = flat.std(1) if k * C > 1 else 0
        feat_tok = self._n(self.feat_slot[:d])[None] + self._n(self.stat_in(stats))
        cls_tok = self._n(self.cls_slot[:C])[None].expand(B, -1, -1)
        bias_tok = self._n(self.bias_q)[None, None].expand(B, 1, -1)
        xp = xs.new_zeros(B, C, k, D_MAX)
        xp[..., :d] = xs
        rows = (self._n(self.row_in(xp.reshape(B, C * k, D_MAX)))
                + self._n(self.cls_slot[:C]).repeat_interleave(k, 0)[None])
        nm = ", ".join(str(n)[:24] for n in names)
        head = self.text(f"Task: write the weights of a classifier with {d} input features ({nm}) and {C} classes."
                         f" Labelled rows:", dev)[None].expand(B, -1, -1)
        mid = self.text(" Weights:", dev)[None].expand(B, -1, -1)
        E = torch.cat([head, rows, mid, feat_tok, cls_tok, bias_tok], 1)
        return E

    def deltas(self, xs, names):
        B, C, k, d = xs.shape
        E = self.build(xs, names)
        T = d + C + 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xs.is_cuda):
            o = self.llm(inputs_embeds=E, use_cache=False)
        t = self.out_norm(o.last_hidden_state.float()[:, -T:])
        return (self.head_f(t[:, :d]) * G_SCALE, self.head_c(t[:, d:d + C]) * G_SCALE, self.head_b(t[:, -1]) * G_SCALE)

    def assemble(self, xs, g=None):
        B, C, k, d = xs.shape
        W1 = self.W1[None, :, :d].expand(B, -1, -1)
        b1 = self.b1[None].expand(B, -1)
        dW2 = db2 = 0
        if g is not None:
            gf, gc, gb = g
            W1 = W1 + gf.transpose(1, 2)
            b1 = b1 + gb
            dW2, db2 = gc[..., :H], gc[..., H]
        feats = F.relu(torch.einsum("bcki,bhi->bckh", xs, W1) + b1[:, None, None])
        W, b = ridge_head(feats, self.log_lam.exp())
        s = self.log_scale.exp()
        return (W1, b1, s * W + dW2, s * b + db2), feats

    def forward(self, xs, names, llm_off=False):
        g = None if llm_off else self.deltas(xs, names)
        return self.assemble(xs, g)
