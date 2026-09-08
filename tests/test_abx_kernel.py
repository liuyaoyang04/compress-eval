import os
import sys

import pytest
import torch

from conftest import STARKV_DIR, requires_cuda

from lrkv.kernels.abx_rope import abx_rope, abx_rope_reference, default_inv_freq


def _inputs(batch, hq, hkv, hd, max_r, L, dtype, device, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, hq, 1, hd, dtype=dtype, device=device)
    u_t = torch.randn(hkv, max_r, hd, dtype=dtype, device=device) * 0.1
    x = torch.randn(batch, hkv, L, max_r, dtype=dtype, device=device)
    return q, u_t, x


@requires_cuda
@pytest.mark.parametrize("hq,hkv,hd", [(32, 8, 128), (8, 8, 128), (4, 2, 64), (4, 1, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matches_reference(hq, hkv, hd, dtype):
    q, u_t, x = _inputs(3, hq, hkv, hd, 80, 777, dtype, "cuda")
    inv_freq = default_inv_freq(hd, theta=500000.0, device="cuda")
    off = torch.tensor([0.0, -5.0, 2.0], device="cuda")
    ref = abx_rope_reference(q, u_t, x, inv_freq=inv_freq, pos_offset=off)
    out = abx_rope(q, u_t, x, inv_freq=inv_freq, pos_offset=off, compute_dtype=dtype)
    tol = 2e-2 if dtype == torch.float16 else 6e-2
    assert (out.float() - ref).abs().max() <= tol * ref.abs().max()


@requires_cuda
def test_dynamic_rank_is_bit_exact():
    q, u_t, x = _inputs(2, 32, 8, 128, 96, 1000, torch.float16, "cuda")
    ranks = torch.tensor([71, 12, 61, 66, 96, 23, 46, 1], dtype=torch.int32, device="cuda")
    keep = torch.arange(96, device="cuda")[None, :] < ranks[:, None]
    u_d, x_d = u_t * keep[:, :, None], x * keep[None, :, None, :]
    full = abx_rope(q, u_d, x_d)
    dyn = abx_rope(q, u_d, x_d, ranks=ranks)
    assert torch.equal(full, dyn)
    ref = abx_rope_reference(q, u_d, x_d, ranks=ranks)
    assert (dyn.float() - ref).abs().max() <= 2e-2 * ref.abs().max()


@requires_cuda
def test_position_offset_shifts_keys():
    """A row with offset -n must equal a row whose cache is shifted by n positions."""
    q, u_t, x = _inputs(1, 8, 2, 128, 32, 200, torch.float16, "cuda")
    inv_freq = default_inv_freq(128, device="cuda")
    n = 7
    # Row A: keys at cache index j with positions j - n; compare against the same keys placed at index j - n.
    out_a = abx_rope(q, u_t, x, inv_freq=inv_freq, pos_offset=torch.tensor([-float(n)], device="cuda"))
    out_b = abx_rope(q, u_t, x[:, :, n:].contiguous(), inv_freq=inv_freq)
    assert (out_a[..., n:].float() - out_b.float()).abs().max() <= 2e-2 * out_b.abs().max()


@requires_cuda
@pytest.mark.skipif(not os.path.exists(os.path.join(STARKV_DIR, "abx_rope_batched.py")), reason="upstream clone missing")
def test_bit_exact_with_upstream_starkv():
    sys.path.insert(0, STARKV_DIR)
    try:
        from abx_rope_batched import abx as upstream_abx
    finally:
        sys.path.remove(STARKV_DIR)
    for hq, hkv in ((32, 8), (16, 16)):
        q, u_t, x = _inputs(2, hq, hkv, 128, 96, 1234, torch.float16, "cuda", seed=hq)
        ranks = torch.randint(8, 97, (hkv,), dtype=torch.int32, device="cuda")
        keep = torch.arange(96, device="cuda")[None, :] < ranks[:, None]
        u_d, x_d = u_t * keep[:, :, None], x * keep[None, :, None, :]
        inv_freq = default_inv_freq(128, theta=500000.0, device="cuda")
        ours = abx_rope(q, u_d, x_d, ranks=ranks, inv_freq=inv_freq, attn_scaling=1.0)
        theirs = upstream_abx(q, u_d, x_d, ranks=ranks, inv_freq=inv_freq, attn_scaling=1.0, dtype=torch.float16)
        assert torch.equal(ours, theirs)
