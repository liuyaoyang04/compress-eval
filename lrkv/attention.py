"""Attention over a latent (low-rank) KV cache.

``LowRankAttention`` replaces a Llama-style attention block whose ``k_proj`` /
``v_proj`` are :class:`lrkv.modules.HeadwiseLowRankLinear` /
:class:`lrkv.modules.LowRankLinear`. It caches the latents instead of K/V:

    k_lat = x @ VS_k^T   viewed as [B, Hkv, L, max_rank]   (pre-RoPE, per KV head, zero-padded)
    v_lat = x @ VS_v^T   shaped   [B, L, rank_v]           (joint for the layer)

Prefill (one shot, empty cache) attends with the dense reconstruction
``U @ VS`` computed once at construction, i.e. the same cost as the base
model; the latents are still written to the cache. Decode reads the latents:

    triton  fused Triton kernel: q . RoPE(U_k @ k_lat)^T with per-head rank (default)
    torch   the same in plain PyTorch (reference for the kernel)
    sdpa    reconstruct full K/V from the latents and call SDPA

and aggregates V in latent space (``probs @ v_lat``) before expanding with
U_v per KV head. Chunked prefill or prefill on top of an existing cache falls
back to reconstructing K/V from the latents, which is exact.

Ported from STAR-KV ``LlamaCustomAttention`` / ``_patched_attn_forward``
(upstream f08a62d). Changes: per-module decode mode instead of a module-level
global; both the transformers 4.x (``past_key_value``) and 5.x
(``past_key_values``) call conventions; boolean or additive masks; key RoPE
positions follow ``position_ids`` so left-padded batches are correct; the
V expansion uses a per-KV-head bmm instead of broadcasting a per-query-head
copy of U; chunked prefill is supported instead of silently ignoring the
cache.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels.abx_rope import abx_rope, rope_tables, rotate_half
from .modules import HeadwiseLowRankLinear, LowRankLinear

DECODE_MODES = ("triton", "torch", "sdpa")


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, L, D]; cos/sin [B, L, D] as produced by the HF rotary embedding."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


@torch.no_grad()
def pack_k_factors(k_proj: HeadwiseLowRankLinear, dtype: torch.dtype):
    """Pad every head to the layer's max rank and pack for the decode kernel.

    Returns (VS [H*max_r, in], U_T [H, max_r, head_dim], dense [H*head_dim, in], ranks int32 [H]).
    Padded rows/cols are zero, so padding is exact; ``ranks`` lets the kernel
    skip them. ``dense = U @ VS`` is built once here for prefill.
    """
    ranks = k_proj.head_ranks_list
    H, Hd, max_r = len(ranks), k_proj.head_dim, max(ranks)
    VS_list, U_list = [], []
    for VS_h, U_h in k_proj.head_factors():
        VS_list.append(F.pad(VS_h.float(), (0, 0, 0, max_r - VS_h.shape[0])))
        U_list.append(F.pad(U_h.float(), (0, max_r - U_h.shape[1])))
    VS = torch.cat(VS_list, dim=0)                       # [H*max_r, in]
    U = torch.stack(U_list, dim=0)                        # [H, Hd, max_r]
    dense = torch.einsum("hdr,hri->hdi", U, VS.view(H, max_r, -1)).reshape(H * Hd, -1)
    ranks_t = torch.tensor(ranks, dtype=torch.int32, device=VS.device)
    return VS.to(dtype), U.transpose(1, 2).contiguous().to(dtype), dense.to(dtype), ranks_t


@torch.no_grad()
def pack_v_factors(v_proj: LowRankLinear, num_kv_heads: int, head_dim: int, dtype: torch.dtype):
    """Returns (VS [r_v, in], U [out, r_v], dense [out, in], U_heads [Hkv, r_v, head_dim])."""
    VS = v_proj.VS.weight.float()
    U = v_proj.U.weight.float()
    dense = U @ VS
    U_heads = U.view(num_kv_heads, head_dim, -1).transpose(1, 2).contiguous()
    return VS.to(dtype), U.to(dtype), dense.to(dtype), U_heads.to(dtype)


class LowRankAttention(nn.Module):
    """Llama attention with a latent KV cache. Inference only (no dropout)."""

    def __init__(
        self,
        config,
        layer_idx: int,
        q_proj: nn.Linear,
        o_proj: nn.Linear,
        k_proj: HeadwiseLowRankLinear,
        v_proj: LowRankLinear,
        rope_inv_freq: torch.Tensor,
        rope_attention_scaling: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        decode_mode: str = "triton",
        kernel_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        assert decode_mode in DECODE_MODES, f"decode_mode must be one of {DECODE_MODES}"
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(k_proj.head_dim)
        assert k_proj.num_heads == self.num_kv_heads, (
            f"layer {layer_idx}: K factors have {k_proj.num_heads} heads, config has {self.num_kv_heads} KV heads")
        assert v_proj.out_features == self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True
        self.decode_mode = decode_mode
        self.kernel_dtype = kernel_dtype
        self.rope_attention_scaling = float(rope_attention_scaling)

        self.q_proj = q_proj
        self.o_proj = o_proj
        dtype = dtype if dtype is not None else q_proj.weight.dtype
        device = q_proj.weight.device

        k_VS, k_U_T, k_dense, k_ranks = pack_k_factors(k_proj, dtype)
        v_VS, v_U, v_dense, v_U_heads = pack_v_factors(v_proj, self.num_kv_heads, self.head_dim, dtype)
        for name, t in (("k_VS", k_VS), ("k_U_T", k_U_T), ("k_dense", k_dense), ("k_ranks", k_ranks),
                        ("v_VS", v_VS), ("v_U", v_U), ("v_dense", v_dense), ("v_U_heads", v_U_heads)):
            self.register_buffer(name, t.to(device), persistent=False)
        self.register_buffer("rope_inv_freq", rope_inv_freq.detach().float().to(device), persistent=False)
        self.k_head_ranks = [int(r) for r in k_ranks.tolist()]
        self.k_max_rank = max(self.k_head_ranks)
        self.v_rank = int(v_VS.shape[0])

    # ------------------------------------------------------------------ helpers
    def extra_repr(self) -> str:
        return (f"layer={self.layer_idx}, heads={self.num_heads}/{self.num_kv_heads}, head_dim={self.head_dim}, "
                f"k_head_ranks={self.k_head_ranks}, v_rank={self.v_rank}, decode_mode={self.decode_mode}")

    def latent_kv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = hidden_states.shape
        k_lat = F.linear(hidden_states, self.k_VS).view(B, L, self.num_kv_heads, self.k_max_rank).transpose(1, 2)
        v_lat = F.linear(hidden_states, self.v_VS)
        return k_lat, v_lat

    def _key_positions(self, pos_offset: Optional[torch.Tensor], kv_len: int, device) -> torch.Tensor:
        pos = torch.arange(kv_len, device=device, dtype=torch.float32)[None, :]
        return pos if pos_offset is None else pos + pos_offset[:, None]

    def reconstruct_k(self, k_cache: torch.Tensor, pos_offset: Optional[torch.Tensor]) -> torch.Tensor:
        """Full rotated K [B, Hkv, L, head_dim] from the latent cache."""
        K = torch.matmul(k_cache, self.k_U_T)
        cos, sin = rope_tables(self.rope_inv_freq, self._key_positions(pos_offset, K.shape[-2], K.device),
                               self.rope_attention_scaling, K.dtype)
        return K * cos + rotate_half(K) * sin

    def reconstruct_v(self, v_cache: torch.Tensor) -> torch.Tensor:
        """Full V [B, Hkv, L, head_dim] from the latent cache."""
        B, L, _ = v_cache.shape
        return F.linear(v_cache, self.v_U).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def _sdpa(self, q, K, V, mask, causal: bool):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.to(q.dtype)
        gqa = K.shape[1] != q.shape[1]
        # Same policy as transformers' sdpa_attention_forward: SDPA's native GQA
        # broadcast is only used without a mask, because with a mask it falls
        # back to the math kernel, which materializes the full [B, H, L, L]
        # attention matrix (21 GiB at batch 4 x 9K tokens). With a mask, repeat
        # K/V so the memory-efficient kernel handles it in O(L).
        if gqa and mask is None:
            try:
                return F.scaled_dot_product_attention(q, K, V, is_causal=causal, scale=self.scaling, enable_gqa=True)
            except TypeError:  # torch < 2.5: no enable_gqa
                pass
        if gqa:
            K = K.repeat_interleave(self.num_kv_groups, dim=1)
            V = V.repeat_interleave(self.num_kv_groups, dim=1)
        return F.scaled_dot_product_attention(q, K, V, attn_mask=mask, is_causal=causal, scale=self.scaling)

    @staticmethod
    def _pos_offset(position_ids: Optional[torch.Tensor], kv_len: int, batch: int) -> Optional[torch.Tensor]:
        """Key at cache index j of row b has RoPE position j + offset[b].

        HF derives position_ids from the attention mask, so a left-padded row
        whose query sits at cache index kv_len-1 has position kv_len-1-n_pad.
        """
        if position_ids is None:
            return None
        off = position_ids[:, -1].to(torch.float32) - float(kv_len - 1)
        if off.numel() == 1 and batch > 1:
            off = off.expand(batch)
        return off.contiguous()

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if past_key_values is None:
            past_key_values = kwargs.pop("past_key_value", None)   # transformers < 5
        assert position_embeddings is not None, "position_embeddings (cos, sin) are required"
        B, L_q, _ = hidden_states.shape
        Hq, Hkv, Hd = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L_q, Hq, Hd).transpose(1, 2)
        k_lat, v_lat = self.latent_kv(hidden_states)

        past_len = 0
        if past_key_values is not None:
            past_len = int(past_key_values.get_seq_length(self.layer_idx))
            k_cache, v_cache = past_key_values.update(k_lat, v_lat, self.layer_idx)
        else:
            k_cache, v_cache = k_lat, v_lat
        kv_len = k_cache.shape[-2]

        cos, sin = position_embeddings
        q = _apply_rope(q, cos, sin)
        pos_offset = self._pos_offset(position_ids, kv_len, B)
        mask = attention_mask[:, :, :, :kv_len] if attention_mask is not None else None

        if L_q > 1 and past_len == 0:
            # Single-shot prefill: dense reconstruction, same cost as the base model.
            K = _apply_rope(F.linear(hidden_states, self.k_dense).view(B, L_q, Hkv, Hd).transpose(1, 2), cos, sin)
            V = F.linear(hidden_states, self.v_dense).view(B, L_q, Hkv, Hd).transpose(1, 2)
            attn = self._sdpa(q, K, V, mask, causal=mask is None)
        elif L_q > 1:
            # Prefill on top of an existing cache: reconstruct everything from the latents.
            K = self.reconstruct_k(k_cache, pos_offset)
            V = self.reconstruct_v(v_cache)
            if mask is None:
                qi = torch.arange(L_q, device=q.device)[:, None] + past_len
                kj = torch.arange(kv_len, device=q.device)[None, :]
                mask = (kj <= qi)[None, None]
            attn = self._sdpa(q, K, V, mask, causal=False)
        else:
            attn = self._decode(q, k_cache, v_cache, mask, pos_offset)

        attn = attn.transpose(1, 2).reshape(B, L_q, Hq * Hd)
        return self.o_proj(attn), None

    def _decode(self, q, k_cache, v_cache, mask, pos_offset):
        B, Hq, _, Hd = q.shape
        Hkv, G = self.num_kv_heads, self.num_kv_groups
        if self.decode_mode == "sdpa":
            return self._sdpa(q, self.reconstruct_k(k_cache, pos_offset), self.reconstruct_v(v_cache), mask, causal=False)

        if self.decode_mode == "triton":
            logits = abx_rope(
                q, self.k_U_T, k_cache, ranks=self.k_ranks, inv_freq=self.rope_inv_freq,
                attn_scaling=self.rope_attention_scaling, pos_offset=pos_offset, compute_dtype=self.kernel_dtype,
            ) * self.scaling
        else:  # torch
            K = self.reconstruct_k(k_cache, pos_offset).repeat_interleave(G, dim=1)
            logits = torch.matmul(q, K.transpose(-1, -2)) * self.scaling

        if mask is not None:
            if mask.dtype == torch.bool:
                logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            else:
                logits = logits + mask.to(logits.dtype)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)          # [B, Hq, 1, L]

        # Aggregate V in latent space, then expand with U_v once per KV head.
        pv = torch.matmul(probs.squeeze(2), v_cache)                                  # [B, Hq, r_v]
        pv = pv.view(B, Hkv, G, self.v_rank).transpose(0, 1).reshape(Hkv, B * G, self.v_rank)
        out = torch.bmm(pv, self.v_U_heads)                                           # [Hkv, B*G, Hd]
        return out.view(Hkv, B, G, Hd).permute(1, 0, 2, 3).reshape(B, Hq, 1, Hd)
