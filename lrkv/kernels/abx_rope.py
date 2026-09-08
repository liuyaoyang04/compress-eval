"""Fused decode-step attention logits over a low-rank K cache (Triton).

For one query token per sequence the kernel computes

    out[b, h, 0, l] = q[b, h, :] . RoPE(pos_l)( U_g @ x[b, g, l, :] ),   g = h // GROUP_SIZE

where ``x`` is the compressed, pre-RoPE K cache of KV head ``g``, ``u_t[g]``
is U_g^T with shape [max_rank, head_dim], and RoPE is re-derived on chip from
``inv_freq`` for absolute position ``pos_l = l + pos_offset[b]``.

Per-head dynamic rank: storage stays padded to the layer's max rank with
uniform strides, and ``ranks[g]`` is the GEMM trip count for KV head ``g``.
Padded rows of ``u_t`` and ``x`` are zero, so stopping early is exact; the
``--check`` mode asserts bit-equality against a full-rank run.

Ported from STAR-KV ``abx_rope_batched.py`` (upstream f08a62d). Changes:

* ``head_dim`` is a compile-time parameter (``HALF_D``) rather than a
  hard-coded 128, so 64/128/256 all work.
* ``pos_offset[b]`` shifts the key positions of sequence ``b``. HF assigns
  RoPE positions from the attention mask, so in a left-padded batch the
  query at cache index ``L-1`` sits at position ``L-1-n_pad``; the offset
  keeps the keys on the same position grid. Upstream implicitly assumes
  ``pos_offset == 0`` (batch size 1 or no padding).
* The pure-PyTorch reference takes the same ``inv_freq`` / offsets and does
  not depend on ``transformers``.

Run ``python -m lrkv.kernels.abx_rope --check`` to verify correctness and
``python -m lrkv.kernels.abx_rope --bench`` to benchmark.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_cos_sin(inv_freq_ptr, starting_idx, pos_offset,
                  NB_TOKENS: tl.constexpr, HALF_D: tl.constexpr):
    """cos/sin for NB_TOKENS consecutive positions from a precomputed inv_freq.

    inv_freq comes from the model's rotary embedding rather than a closed form
    1/theta**(2i/d): llama3, yarn, linear and dynamic scaling all reshape it
    per dimension, and reproducing that in-kernel would silently drift from
    the RoPE applied to the queries.
    """
    inv_freq = tl.load(inv_freq_ptr + tl.arange(0, HALF_D))
    pos = (tl.arange(0, NB_TOKENS) + starting_idx).to(tl.float32) + pos_offset
    freqs = pos[:, None] * inv_freq[None, :]
    return tl.extra.cuda.libdevice.fast_cosf(freqs), tl.extra.cuda.libdevice.fast_sinf(freqs)


def _configs():
    return [triton.Config({"BLOCK_SIZE_L": 64, "BLOCK_SIZE_R": 16}, num_warps=4, num_stages=1)]


@triton.autotune(configs=_configs(), key=["seq_len"])
@triton.jit
def _abx_rope_fwd(
    a_ptr, b_ptr, x_ptr, out_ptr, r_ptr, inv_freq_ptr, pos_off_ptr,
    stride_ab, stride_az, stride_ad,
    stride_bz, stride_br, stride_bd,
    stride_xb, stride_xhg, stride_xl, stride_xr,
    stride_ob, stride_oz, stride_ol,
    seq_len,
    dtype_tl: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ATTN_SCALING: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_l = tl.program_id(axis=2)

    # Query head pid_h reads the compressed K cache of KV head pid_h // GROUP_SIZE.
    kv_head = pid_h // GROUP_SIZE
    # Ranks are stored per KV head, so every query head in a group shares R.
    R = tl.load(r_ptr + kv_head)
    pos_offset = tl.load(pos_off_ptr + pid_b)

    offs_ds = tl.arange(0, HALF_D)
    offs_rs = tl.arange(0, BLOCK_SIZE_R)
    offs_ls = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)

    A_ptrs = a_ptr + pid_b * stride_ab + pid_h * stride_az + offs_ds[None, :] * stride_ad
    B_ptrs = b_ptr + kv_head * stride_bz + (offs_rs[:, None] * stride_br + offs_ds[None, :] * stride_bd)
    X_ptrs = x_ptr + pid_b * stride_xb + kv_head * stride_xhg + (offs_ls[:, None] * stride_xl + offs_rs[None, :] * stride_xr)
    O_ptrs = out_ptr + pid_b * stride_ob + pid_h * stride_oz + offs_ls[None, :] * stride_ol

    xb_0 = tl.zeros((BLOCK_SIZE_L, HALF_D), dtype=tl.float32)
    xb_1 = tl.zeros((BLOCK_SIZE_L, HALF_D), dtype=tl.float32)

    # The KV length is rarely a multiple of BLOCK_SIZE_L; mask the tail.
    ls_mask = offs_ls < seq_len

    for r in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
        r_mask = offs_rs < R - r * BLOCK_SIZE_R
        x = tl.load(X_ptrs, mask=r_mask[None, :] & ls_mask[:, None], other=0.0)
        b_0 = tl.load(B_ptrs, mask=r_mask[:, None], other=0.0)
        b_1 = tl.load(B_ptrs + HALF_D * stride_bd, mask=r_mask[:, None], other=0.0)
        xb_0 = tl.dot(x, b_0, xb_0)
        xb_1 = tl.dot(x, b_1, xb_1)
        B_ptrs += BLOCK_SIZE_R * stride_br
        X_ptrs += BLOCK_SIZE_R * stride_xr

    xb_0 = xb_0.to(dtype_tl)
    xb_1 = xb_1.to(dtype_tl)

    cos, sin = _rope_cos_sin(inv_freq_ptr, pid_l * BLOCK_SIZE_L, pos_offset,
                             NB_TOKENS=BLOCK_SIZE_L, HALF_D=HALF_D)
    # yarn scales cos/sin by an attention factor; default and llama3 use 1.0.
    cos = (cos * ATTN_SCALING).to(dtype_tl)
    sin = (sin * ATTN_SCALING).to(dtype_tl)

    xb_rope_0 = xb_0 * cos - xb_1 * sin
    xb_rope_1 = xb_1 * cos + xb_0 * sin
    xb_0 = xb_rope_0.to(dtype_tl)
    xb_1 = xb_rope_1.to(dtype_tl)

    a_0 = tl.load(A_ptrs)
    a_1 = tl.load(A_ptrs + HALF_D * stride_ad)
    abx = tl.sum(a_0 * xb_0, 1) + tl.sum(a_1 * xb_1, 1)
    tl.store(O_ptrs, abx[None, :], mask=ls_mask[None, :])


_TL_DTYPES = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


def default_inv_freq(head_dim: int, theta: float = 10000.0, device=None) -> torch.Tensor:
    i = torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
    return 1.0 / (theta ** (i * 2 / head_dim))


def abx_rope(
    q: torch.Tensor,
    u_t: torch.Tensor,
    x: torch.Tensor,
    ranks: Optional[torch.Tensor] = None,
    inv_freq: Optional[torch.Tensor] = None,
    attn_scaling: float = 1.0,
    pos_offset: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Fused ``q @ RoPE(U @ x^T)`` for one decode step.

    Args:
        q: query states, [batch, num_heads, 1, head_dim] (already rotated).
        u_t: K factor U^T per KV head, [num_kv_heads, max_rank, head_dim].
        x: compressed pre-RoPE K cache, [batch, num_kv_heads, seq_len, max_rank].
        ranks: int32 [num_kv_heads], real rank kept per KV head. Entries past
            ``ranks[g]`` must be zero in ``u_t[g]`` and ``x[:, g]``. ``None``
            means every head runs at the padded rank.
        inv_freq: RoPE inverse frequencies, [head_dim // 2]. Pass the model's
            own ``rotary_emb.inv_freq``; ``None`` falls back to theta=10000,
            which is wrong for any model with a different theta or a scaled
            rope type.
        attn_scaling: cos/sin multiplier (``rotary_emb.attention_scaling``).
        pos_offset: float32 [batch]; key at cache index ``l`` of sequence
            ``b`` gets position ``l + pos_offset[b]``. ``None`` means 0.
        compute_dtype: dtype of the RoPE multiply (float16 keeps more
            mantissa than bfloat16 for cos/sin; upstream default).

    Returns:
        Unscaled attention logits, [batch, num_heads, 1, seq_len], in ``x.dtype``.
    """
    assert q.dim() == 4 and u_t.dim() == 3 and x.dim() == 4, "q [B,Hq,1,D], u_t [Hkv,R,D], x [B,Hkv,L,R]"
    batch, num_heads, q_len, head_dim = q.shape
    num_kv_heads, max_rank, d2 = u_t.shape
    bx, gx, seq_len, rx = x.shape
    assert q_len == 1, "decode kernel handles exactly one query token per sequence"
    assert d2 == head_dim and bx == batch and gx == num_kv_heads and rx == max_rank, (
        f"shape mismatch: q {tuple(q.shape)} u_t {tuple(u_t.shape)} x {tuple(x.shape)}")
    assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
    half_d = head_dim // 2
    assert half_d >= 16 and (half_d & (half_d - 1)) == 0, "head_dim must be 32, 64, 128, 256, ..."

    device = x.device
    if ranks is None:
        ranks = torch.full((num_kv_heads,), max_rank, dtype=torch.int32, device=device)
    else:
        assert ranks.numel() == num_kv_heads, f"ranks has {ranks.numel()} entries for {num_kv_heads} KV heads"
        ranks = ranks.to(device=device, dtype=torch.int32).contiguous()

    if inv_freq is None:
        inv_freq = default_inv_freq(head_dim, device=device)
    inv_freq = inv_freq.to(device=device, dtype=torch.float32).contiguous()
    assert inv_freq.numel() == half_d, f"inv_freq has {inv_freq.numel()} entries, expected {half_d}"

    if pos_offset is None:
        pos_offset = torch.zeros(batch, dtype=torch.float32, device=device)
    else:
        assert pos_offset.numel() == batch
        pos_offset = pos_offset.to(device=device, dtype=torch.float32).contiguous()

    out = torch.empty((batch, num_heads, 1, seq_len), dtype=x.dtype, device=device)
    grid = lambda META: (batch, num_heads, triton.cdiv(seq_len, META["BLOCK_SIZE_L"]))  # noqa: E731
    with torch.cuda.device(device):
        _abx_rope_fwd[grid](
            q, u_t, x, out, ranks, inv_freq, pos_offset,
            q.stride(0), q.stride(1), q.stride(3),
            u_t.stride(0), u_t.stride(1), u_t.stride(2),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(3),
            seq_len=seq_len,
            dtype_tl=_TL_DTYPES[compute_dtype],
            HALF_D=half_d,
            GROUP_SIZE=num_heads // num_kv_heads,
            ATTN_SCALING=float(attn_scaling),
        )
    return out


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_tables(inv_freq: torch.Tensor, positions: torch.Tensor,
                attn_scaling: float = 1.0, dtype: torch.dtype = torch.float32):
    """cos/sin tables for explicit positions.

    positions: [batch, seq_len] (float or int). Returns (cos, sin) shaped
    [batch, 1, seq_len, head_dim], matching ``LlamaRotaryEmbedding`` up to the
    unsqueezed head axis.
    """
    freqs = positions.to(torch.float32)[..., None] * inv_freq.to(torch.float32)[None, None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos() * attn_scaling).to(dtype).unsqueeze(1), (emb.sin() * attn_scaling).to(dtype).unsqueeze(1)


@torch.no_grad()
def abx_rope_reference(q, u_t, x, ranks=None, inv_freq=None, attn_scaling=1.0, pos_offset=None):
    """Pure-PyTorch fp32 reference of :func:`abx_rope` (same arguments)."""
    batch, num_heads, _, head_dim = q.shape
    num_kv_heads, max_rank, _ = u_t.shape
    seq_len = x.shape[2]
    group = num_heads // num_kv_heads
    xf, uf = x.float(), u_t.float()
    if ranks is not None:
        keep = torch.arange(max_rank, device=x.device)[None, :] < ranks.to(x.device)[:, None]  # [Hkv, R]
        xf = xf * keep[None, :, None, :]
        uf = uf * keep[:, :, None]
    if inv_freq is None:
        inv_freq = default_inv_freq(head_dim, device=x.device)
    if pos_offset is None:
        pos_offset = torch.zeros(batch, device=x.device)
    k = xf @ uf.unsqueeze(0)                                            # [B, Hkv, L, D]
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)[None, :] + pos_offset.float()[:, None]
    cos, sin = rope_tables(inv_freq, positions, attn_scaling)          # [B, 1, L, D]
    k = k * cos + rotate_half(k) * sin
    k = k.repeat_interleave(group, dim=1)
    return q.float() @ k.transpose(-1, -2)                              # [B, Hq, 1, L]


# ---------------------------------------------------------------------------
# Self-check and benchmark
# ---------------------------------------------------------------------------

def run_check(num_heads=32, num_kv_heads=8, head_dim=128, max_rank=96, seq_len=1000, batch=3,
              dtype=torch.float16, device="cuda") -> bool:
    torch.manual_seed(0)
    ok = True
    q = torch.randn(batch, num_heads, 1, head_dim, dtype=dtype, device=device)
    u_t = torch.randn(num_kv_heads, max_rank, head_dim, dtype=dtype, device=device) * 0.1
    x = torch.randn(batch, num_kv_heads, seq_len, max_rank, dtype=dtype, device=device)
    inv_freq = default_inv_freq(head_dim, theta=500000.0, device=device)
    pos_offset = torch.tensor([0.0, -7.0, 3.0][:batch], device=device)

    ref = abx_rope_reference(q, u_t, x, inv_freq=inv_freq, pos_offset=pos_offset)
    ours = abx_rope(q, u_t, x, inv_freq=inv_freq, pos_offset=pos_offset, compute_dtype=dtype)
    err = (ref - ours.float()).abs().max().item()
    scale = ref.abs().max().item()
    print(f"[check] uniform rank: heads={num_heads}/{num_kv_heads} D={head_dim} R={max_rank} L={seq_len}  "
          f"max|diff|={err:.3e} (max|ref|={scale:.2f})")
    ok &= err <= 2e-2 * scale

    ranks = torch.randint(8, max_rank + 1, (num_kv_heads,), dtype=torch.int32, device=device)
    keep = torch.arange(max_rank, device=device)[None, :] < ranks[:, None]
    u_d = u_t * keep[:, :, None]
    x_d = x * keep[None, :, None, :]
    full = abx_rope(q, u_d, x_d, inv_freq=inv_freq, pos_offset=pos_offset, compute_dtype=dtype)
    dyn = abx_rope(q, u_d, x_d, ranks=ranks, inv_freq=inv_freq, pos_offset=pos_offset, compute_dtype=dtype)
    bit = (full.float() - dyn.float()).abs().max().item()
    print(f"[check] dynamic rank {ranks.tolist()} vs padded: max|diff|={bit:.1e} -> {'PASS' if bit == 0 else 'FAIL'}")
    ok &= bit == 0.0

    ref_d = abx_rope_reference(q, u_d, x_d, ranks=ranks, inv_freq=inv_freq, pos_offset=pos_offset)
    err_d = (ref_d - dyn.float()).abs().max().item()
    ok &= err_d <= 2e-2 * ref_d.abs().max().item()
    print(f"[check] dynamic rank vs reference: max|diff|={err_d:.3e}")
    print("[check] " + ("PASS" if ok else "FAIL"))
    return ok


def run_bench(num_heads, num_kv_heads, head_dim, max_rank, seq_lens, batch=1, dtype=torch.float16, device="cuda"):
    print(f"{'seq_len':>8} {'dense QK^T (us)':>16} {'torch ref (us)':>15} {'fused (us)':>11}")
    for seq_len in seq_lens:
        q = torch.randn(batch, num_heads, 1, head_dim, dtype=dtype, device=device)
        u_t = torch.randn(num_kv_heads, max_rank, head_dim, dtype=dtype, device=device)
        x = torch.randn(batch, num_kv_heads, seq_len, max_rank, dtype=dtype, device=device)
        k_full = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        inv_freq = default_inv_freq(head_dim, device=device)
        t_dense = triton.testing.do_bench(lambda: torch.matmul(q, k_full.transpose(-1, -2)), warmup=25, rep=100)
        t_ref = triton.testing.do_bench(lambda: abx_rope_reference(q, u_t, x, inv_freq=inv_freq), warmup=25, rep=100)
        t_fused = triton.testing.do_bench(lambda: abx_rope(q, u_t, x, inv_freq=inv_freq, compute_dtype=dtype), warmup=25, rep=100)
        print(f"{seq_len:>8} {t_dense * 1000:>16.1f} {t_ref * 1000:>15.1f} {t_fused * 1000:>11.1f}")
        del k_full, x


def main():
    p = argparse.ArgumentParser(description="Check or benchmark the fused low-rank K + RoPE decode kernel.")
    p.add_argument("--check", action="store_true", help="Run the correctness check")
    p.add_argument("--bench", action="store_true", help="Run the benchmark")
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-rank", type=int, default=96)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 16384, 65536])
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = p.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if not (args.check or args.bench):
        args.check = True
    if args.check:
        ok = run_check(args.num_heads, args.num_kv_heads, args.head_dim, args.max_rank, dtype=dtype)
        ok &= run_check(args.num_heads, args.num_heads, args.head_dim, args.max_rank // 4, dtype=dtype)  # MHA
        if not ok:
            raise SystemExit(1)
    if args.bench:
        run_bench(args.num_heads, args.num_kv_heads, args.head_dim, args.max_rank, args.seq_lens, dtype=dtype)


if __name__ == "__main__":
    main()
