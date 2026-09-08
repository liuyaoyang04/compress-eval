"""Canonical low-rank projection modules.

These two modules define the checkpoint layout (see ``lrkv.checkpoint``): a
compressed layer stores only the factors ``U`` and ``VS`` (Sigma already folded
into VS) plus, for K, the per-head split ``head_ranks``. Nothing here carries a
threshold, a mask, or any other training-time state, so any method can emit
this layout and be evaluated through the same inference path.

In "reference" mode the modules simply replace ``k_proj`` / ``v_proj`` inside
the stock HF attention (full-size KV cache, exact same numerics as the base
model's attention). ``lrkv.attention.LowRankAttention`` consumes them to run
with a latent (compressed) KV cache instead.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Joint low-rank projection ``W ~= U @ VS``.

    VS: [rank, in_features]   latent projection (what the V cache stores per token)
    U:  [out_features, rank]  expansion back to the full output
    """

    def __init__(self, in_features: int, out_features: int, rank: int, device=None, dtype=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.VS = nn.Linear(self.in_features, self.rank, bias=False, device=device, dtype=dtype)
        self.U = nn.Linear(self.rank, self.out_features, bias=False, device=device, dtype=dtype)
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(self.VS(x))

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.VS(x)

    @classmethod
    def from_factors(cls, U: torch.Tensor, VS: torch.Tensor, device=None, dtype=None) -> "LowRankLinear":
        out_f, rank = U.shape
        rank2, in_f = VS.shape
        assert rank == rank2, f"U {tuple(U.shape)} and VS {tuple(VS.shape)} disagree on the rank"
        device = device if device is not None else U.device
        dtype = dtype if dtype is not None else U.dtype
        mod = cls(in_f, out_f, rank, device=device, dtype=dtype)
        with torch.no_grad():
            mod.U.weight.copy_(U.to(device=device, dtype=dtype))
            mod.VS.weight.copy_(VS.to(device=device, dtype=dtype))
        return mod

    @classmethod
    def from_dense(cls, linear: nn.Linear) -> "LowRankLinear":
        """Exact full-rank wrapper: U = I, VS = W (no compression)."""
        assert linear.bias is None, "biased projections are not supported"
        W = linear.weight.detach()
        eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
        return cls.from_factors(eye, W)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"


class HeadwiseLowRankLinear(nn.Module):
    """Per-KV-head low-rank projection ``W ~= U @ VS`` with block-diagonal U.

    Heads keep different numbers of directions, so VS is their concatenation
    and ``head_ranks`` (a persistent int32 buffer) records the split:

    VS:         [sum_h r_h, in_features]        head h owns rows [off_h, off_h + r_h)
    U:          [H * head_dim, sum_h r_h]       block h = rows of head h x its rank block
    head_ranks: int32 [H]

    ``block_mask`` is rebuilt from ``head_ranks`` and applied to U's gradient so
    any later fine-tune cannot train cross-head entries that the per-head
    export would silently drop.
    """

    def __init__(self, in_features: int, out_features: int, head_ranks: Sequence[int],
                 head_dim: int, device=None, dtype=None):
        super().__init__()
        ranks = [int(r) for r in head_ranks]
        assert len(ranks) * int(head_dim) == int(out_features), (
            f"out_features={out_features} != {len(ranks)} heads x head_dim={head_dim}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.head_dim = int(head_dim)
        self.num_heads = len(ranks)
        self.rank = int(sum(ranks))
        self.VS = nn.Linear(self.in_features, self.rank, bias=False, device=device, dtype=dtype)
        self.U = nn.Linear(self.rank, self.out_features, bias=False, device=device, dtype=dtype)
        self.register_buffer("head_ranks", torch.tensor(ranks, dtype=torch.int32, device=device))
        self.bias = None

        block_mask = torch.zeros(self.out_features, self.rank, device=device, dtype=dtype)
        row = col = 0
        for r in ranks:
            block_mask[row:row + self.head_dim, col:col + r] = 1.0
            row += self.head_dim
            col += r
        self.register_buffer("block_mask", block_mask, persistent=False)
        self.U.weight.register_hook(lambda g: g * self.block_mask)

    @property
    def head_ranks_list(self) -> list[int]:
        return [int(r) for r in self.head_ranks.tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(self.VS(x))

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.VS(x)

    def head_factors(self):
        """Yield (VS_h [r_h, in], U_h [head_dim, r_h]) per head, sliced from the packed tensors."""
        col = 0
        for h, r in enumerate(self.head_ranks_list):
            yield (self.VS.weight[col:col + r],
                   self.U.weight[h * self.head_dim:(h + 1) * self.head_dim, col:col + r])
            col += r

    @classmethod
    def from_head_factors(cls, VS_blocks: Sequence[torch.Tensor], U_blocks: Sequence[torch.Tensor],
                          device=None, dtype=None) -> "HeadwiseLowRankLinear":
        """Build from per-head (VS_h [r_h, in], U_h [head_dim, r_h]) lists."""
        assert len(VS_blocks) == len(U_blocks) > 0
        head_dim = U_blocks[0].shape[0]
        in_f = VS_blocks[0].shape[1]
        ranks = [int(v.shape[0]) for v in VS_blocks]
        for v, u in zip(VS_blocks, U_blocks):
            assert u.shape == (head_dim, v.shape[0]) and v.shape[1] == in_f
        device = device if device is not None else U_blocks[0].device
        dtype = dtype if dtype is not None else U_blocks[0].dtype
        mod = cls(in_f, head_dim * len(ranks), ranks, head_dim, device=device, dtype=dtype)
        with torch.no_grad():
            mod.VS.weight.copy_(torch.cat([v.to(device=device, dtype=dtype) for v in VS_blocks], dim=0))
            U_full = torch.zeros(mod.out_features, mod.rank, device=device, dtype=dtype)
            col = 0
            for h, (u, r) in enumerate(zip(U_blocks, ranks)):
                U_full[h * head_dim:(h + 1) * head_dim, col:col + r] = u.to(device=device, dtype=dtype)
                col += r
            mod.U.weight.copy_(U_full)
        return mod

    @classmethod
    def from_factors(cls, U: torch.Tensor, VS: torch.Tensor, head_ranks: Sequence[int],
                     device=None, dtype=None) -> "HeadwiseLowRankLinear":
        """Build from packed U [H*head_dim, sum r_h] and VS [sum r_h, in] (off-block entries of U ignored)."""
        ranks = [int(r) for r in head_ranks]
        head_dim = U.shape[0] // len(ranks)
        VS_blocks, U_blocks, col = [], [], 0
        for h, r in enumerate(ranks):
            VS_blocks.append(VS[col:col + r])
            U_blocks.append(U[h * head_dim:(h + 1) * head_dim, col:col + r])
            col += r
        return cls.from_head_factors(VS_blocks, U_blocks, device=device, dtype=dtype)

    @classmethod
    def from_dense(cls, linear: nn.Linear, head_dim: int) -> "HeadwiseLowRankLinear":
        """Exact full-rank wrapper: U_h = I, VS_h = W rows of head h (no compression)."""
        assert linear.bias is None, "biased projections are not supported"
        W = linear.weight.detach()
        H = W.shape[0] // head_dim
        eye = torch.eye(head_dim, device=W.device, dtype=W.dtype)
        return cls.from_head_factors([W[h * head_dim:(h + 1) * head_dim] for h in range(H)],
                                     [eye] * H)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"head_dim={self.head_dim}, head_ranks={self.head_ranks_list}")


def is_lowrank_module(m: nn.Module) -> bool:
    return isinstance(m, (LowRankLinear, HeadwiseLowRankLinear))
