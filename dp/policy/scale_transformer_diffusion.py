from __future__ import annotations
"""
HybridDiffusionPolicy (v2)
=========================
An updated implementation that **matches the original Diffusion-Policy API**:

* **Forward signature** matches `TransformerForDiffusion.forward`:

```python
pred = model(sample, timestep, cond)
#  sample  : (B, T, input_dim)
#  timestep: scalar int | Tensor[B]
#  cond    : (B, T_cond, cond_dim) or None
```
* **No scheduler logic inside the model** - you can plug this into an external
  `compute_loss()` exactly like you posted (the loss helper drives the
  `noise_scheduler`).
* Keeps the token/positional-embedding pipeline from `TransformerForDiffusion`
  **but** replaces the core with ScaleDP-style adaLN-Zero DiT blocks.
* Still exposes `get_optim_groups()` for AdamW weight-decay filtering.
"""

import math
import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
#                                helpers
# -----------------------------------------------------------------------------

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FiLM modulation for adaLN-Zero."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _approx_gelu():
    return nn.GELU(approximate="tanh")


class _Attention(nn.Module):
    """Lightweight multi-head attention with dropout and optional mask."""
    def __init__(self, dim: int, n_head: int, p_attn: float = 0.0, p_proj: float = 0.0):
        super().__init__()
        assert dim % n_head == 0, "dim must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(p_attn)   # used only if self.training
        self.proj_drop = nn.Dropout(p_proj)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.n_head, self.head_dim

        # qkv: (B, N, 3C) -> (3, B, H, N, D)
        qkv = (
            self.qkv(x)
            .view(B, N, 3, H, D)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]        # (B, H, N, D)

        # SDPA expects (B, H, N, D); good as-is.
        # attn_mask handling:
        # - Bool mask: True = keep, False = mask. Shape broadcastable to (B,H,N,N).
        # - Float mask: additive (e.g., 0 for keep, -inf for mask), broadcastable to (B,H,N,N).
        # If you used an "additive" mask previously (already -inf where masked), pass it directly.
        # If your mask was pre-added to logits, DON'T pre-add; pass it here instead.

        dropout_p = self.attn_drop.p if self.training and self.attn_drop.p > 0 else 0.0

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,      # None, bool, or float mask (broadcastable to (B,H,N,N))
            dropout_p=dropout_p,
            is_causal=False,          # set True if you need causal masking
        )                             # -> (B, H, N, D)

        x = x.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class _ScaleDPBlock(nn.Module):
    """adaLN-Zero transformer block (DiT/ScaleDP style)."""

    def __init__(self, dim: int, n_head: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = _Attention(dim, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            _approx_gelu(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # adaLN modulation MLP: (shift,scale,gate)×2 → 6*dim
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(_modulate(self.norm1(x), s1, sc1), attn_mask)
        x = x + g2.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), s2, sc2))
        return x


class _FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.proj(x)


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):  # (B,)
        half = self.dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        emb = torch.einsum("b,d->bd", t.float(), freq)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# -----------------------------------------------------------------------------
#                     Main HybridDiffusionPolicy class
# -----------------------------------------------------------------------------

class ScaleTransformerDiffusionPolicy(nn.Module):
    """Token-style input ⇢ DiT backbone. Compatible with external loss helper."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: Optional[int] = None,
        cond_dim: int = 0,
        n_emb: int = 768,
        depth: int = 12,
        n_head: int = 12,
        mlp_ratio: float = 4.0,
        p_drop_emb: float = 0.1,
        causal_attn: bool = False,
    ) -> None:
        super().__init__()
        if n_obs_steps is None:
            n_obs_steps = horizon

        # ---------- token bookkeeping (same as TransformerForDiffusion) ----- #
        self.horizon = horizon
        self.obs_as_cond = cond_dim > 0
        T_cond = 1 + (n_obs_steps if self.obs_as_cond else 0)  # time token + obs tokens

        # ---------- embeddings --------------------------------------------- #
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = _SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb) if self.obs_as_cond else None
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        # ---------- DiT backbone ------------------------------------------- #
        self.blocks = nn.ModuleList([
            _ScaleDPBlock(n_emb, n_head, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = _FinalLayer(n_emb, output_dim)

        # ---------- causal mask (optional) ---------------------------------- #
        if causal_attn:
            m = torch.triu(torch.ones(horizon, horizon, dtype=torch.bool))
            m = m.float().masked_fill(~m, float("-inf"))
            self.register_buffer("mask", m, persistent=False)
        else:
            self.mask = None

        # weight init
        self._init_weights()
        logger.info("HybridDiffusionPolicy params: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6)

    # --------------------------------------------------------------------- #
    #                         weight initialisation                         #
    # --------------------------------------------------------------------- #

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.normal_(m.weight, 0.0, 0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
        if hasattr(self, "pos_emb") and self.pos_emb is not None:
            nn.init.normal_(self.pos_emb, 0.0, 0.02)
        if hasattr(self, "cond_pos_emb") and self.cond_pos_emb is not None:
            nn.init.normal_(self.cond_pos_emb, 0.0, 0.02)


    # --------------------------------------------------------------------- #
    #                         helper: build cond tokens                     #
    # --------------------------------------------------------------------- #

    def _make_cond_tokens(self, time_vec: torch.Tensor, cond: Optional[torch.Tensor]):
        # time_vec: (B,E) -> (B,1,E)
        tokens = [time_vec.unsqueeze(1)]
        if self.obs_as_cond and cond is not None:
            tokens.append(self.cond_obs_emb(cond))  # (B,n_obs,E)
        x = torch.cat(tokens, dim=1)  # (B,T_cond,E)
        pos = self.cond_pos_emb[:, : x.size(1)]
        return self.drop(x + pos)

    # --------------------------------------------------------------------- #
    #                            forward                                    #
    # --------------------------------------------------------------------- #

    def forward(
        self,
        sample: torch.Tensor,                          # (B,T,input_dim)
        timestep: Union[torch.Tensor, int, float],     # scalar or (B,)
        cond: Optional[torch.Tensor] = None,           # (B,T_cond,cond_dim)
        **kwargs,
    ) -> torch.Tensor:
        """Predict noise (or sample) given a *noisy* trajectory and conditioning."""
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.size(0))  # (B,)

        # time embedding (vector)
        time_vec = self.time_emb(timestep)  # (B,E)
        cond_tokens = self._make_cond_tokens(time_vec, cond)  # (B,T_cond,E)
        global_cond = cond_tokens.mean(dim=1)  # collapse tokens like ScaleDP (B,E)

        # action tokens + learnable pos
        x = self.drop(self.input_emb(sample) + self.pos_emb[:, : self.horizon])  # (B,T,E)
        for blk in self.blocks:
            x = blk(x, global_cond, attn_mask=self.mask)
        x = self.final_layer(x, global_cond)  # (B,T,out_dim)
        return x

    # --------------------------------------------------------------------- #
    #                     optimizer helper (AdamW groups)                   #
    # --------------------------------------------------------------------- #

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay, no_decay = set(), set()
        whitelist = (nn.Linear, _Attention)
        blacklist = (nn.LayerNorm,)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fullname = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fullname)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fullname)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fullname)
        no_decay.update({"pos_emb", "cond_pos_emb"})

        params = dict(self.named_parameters())
        assert decay.isdisjoint(no_decay)
        groups = [
            {"params": [params[p] for p in sorted(decay)], "weight_decay": weight_decay},
            {"params": [params[p] for p in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return groups

if __name__ == "__main__":
    import torch

    model = ScaleTransformerDiffusionPolicy(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        n_emb=128,
        depth=2,
        n_head=4,
    ).cuda()

    B = 2
    sample = torch.randn(B, 8, 16).cuda()
    cond = torch.randn(B, 4, 10).cuda()
    t = torch.tensor([5, 12], dtype=torch.long).cuda()

    out = model(sample, t, cond)
    print("output shape:", out.shape)  # expected: (B, horizon, output_dim)