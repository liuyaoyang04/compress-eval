"""LowRankAttention against the stock HF attention.

With identity factors (U = I, VS = W) the latent path is mathematically the
dense path, so in fp32 with the torch decode mode every output must match the
base model to numerical noise: single-shot prefill, cached decode, left-padded
batches, chunked prefill. The Triton and SDPA decode modes are then compared
against the torch mode in bf16 on partial-rank factors.
"""

import copy

import pytest
import torch

from conftest import clone, requires_cuda, tiny_model

from lrkv.checkpoint import load_lowrank_checkpoint
from lrkv.factorize import svd_factorize_model
from lrkv.model import enable_cache, install_lowrank_attention, lowrank_attention_layers, set_decode_mode
from lrkv.modules import HeadwiseLowRankLinear, LowRankLinear


def _identity_lowrank(model, layers=(1, 2)):
    for i in layers:
        attn = model.model.layers[i].self_attn
        attn.k_proj = HeadwiseLowRankLinear.from_dense(attn.k_proj, 128)
        attn.v_proj = LowRankLinear.from_dense(attn.v_proj)
    return model


def _generate(model, ids, mask=None, n=6):
    out = model.generate(ids, attention_mask=mask, max_new_tokens=n, do_sample=False, min_new_tokens=n,
                         output_logits=True, return_dict_in_generate=True, pad_token_id=0)
    return out.sequences[:, ids.shape[1]:], torch.stack(out.logits, dim=1)


@pytest.mark.parametrize("rope", ["default", "llama3"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_identity_factors_match_stock_fp32(rope, device):
    ref = tiny_model(rope=rope, device=device)
    lr = _identity_lowrank(clone(ref))
    assert install_lowrank_attention(lr, decode_mode="torch") == [1, 2]
    enable_cache(lr)
    torch.manual_seed(3)
    ids = torch.randint(1, 512, (2, 37), device=device)

    # single-shot prefill
    a = ref(ids, use_cache=False).logits
    b = lr(ids, use_cache=False).logits
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-3), (a - b).abs().max()

    # cached greedy decode
    ta, sa = _generate(ref, ids)
    tb, sb = _generate(lr, ids)
    assert torch.allclose(sa, sb, atol=1e-3, rtol=1e-3), (sa - sb).abs().max()
    assert torch.equal(ta, tb)

    # chunked prefill on top of an existing cache
    cache_out = lr(ids[:, :20], use_cache=True)
    c = lr(ids[:, 20:], past_key_values=cache_out.past_key_values, use_cache=True).logits
    assert torch.allclose(a[:, 20:], c, atol=1e-3, rtol=1e-3), (a[:, 20:] - c).abs().max()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_left_padded_batch_matches_stock_fp32(device):
    ref = tiny_model(device=device)
    lr = _identity_lowrank(clone(ref))
    install_lowrank_attention(lr, decode_mode="torch")
    enable_cache(lr)
    torch.manual_seed(4)
    ids = torch.randint(1, 512, (3, 30), device=device)
    mask = torch.ones_like(ids)
    ids[0, :9] = 0; mask[0, :9] = 0
    ids[2, :3] = 0; mask[2, :3] = 0
    ta, sa = _generate(ref, ids, mask)
    tb, sb = _generate(lr, ids, mask)
    assert torch.allclose(sa, sb, atol=1e-3, rtol=1e-3), (sa - sb).abs().max()
    assert torch.equal(ta, tb)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_modes_agree_bf16(dtype):
    base = tiny_model(rope="llama3", device="cuda", dtype=dtype)
    sd = svd_factorize_model(base, k_rank=48, v_rank=200, skip_layers=(0,), dtype=dtype)
    with torch.no_grad():  # unequal per-head ranks so the dynamic-rank path is exercised
        pre = "model.layers.2.self_attn."
        sd[pre + "k_proj.VS.weight"] = sd[pre + "k_proj.VS.weight"][:-16].clone()
        sd[pre + "k_proj.U.weight"] = sd[pre + "k_proj.U.weight"][:, :-16].clone()
        sd[pre + "k_proj.head_ranks"] = torch.tensor([48, 32], dtype=torch.int32)
    model = clone(base)
    load_lowrank_checkpoint(model, sd)
    install_lowrank_attention(model, decode_mode="torch")
    enable_cache(model)
    assert lowrank_attention_layers(model)[1][1].k_head_ranks == [48, 32]

    torch.manual_seed(5)
    ids = torch.randint(1, 512, (3, 40), device="cuda")
    mask = torch.ones_like(ids)
    ids[1, :11] = 0; mask[1, :11] = 0
    outs = {}
    for mode in ("torch", "triton", "sdpa"):
        set_decode_mode(model, mode)
        outs[mode] = _generate(model, ids, mask, n=5)
    scale = outs["torch"][1].abs().max()
    for mode in ("triton", "sdpa"):
        diff = (outs[mode][1].float() - outs["torch"][1].float()).abs().max()
        assert diff <= 0.05 * scale, (mode, diff, scale)
        agree = (outs[mode][0] == outs["torch"][0]).float().mean()
        assert agree >= 0.8, (mode, agree)


@requires_cuda
def test_reference_mode_equals_latent_prefill_bf16():
    """'reference' (stock attention + low-rank modules) and the latent path agree on prefill logits."""
    base = tiny_model(device="cuda", dtype=torch.bfloat16)
    sd = svd_factorize_model(base, k_rank=64, v_rank=128, skip_layers=(0,), dtype=torch.bfloat16)
    ref = clone(base); load_lowrank_checkpoint(ref, sd)
    lat = clone(base); load_lowrank_checkpoint(lat, sd); install_lowrank_attention(lat, decode_mode="triton")
    ids = torch.randint(1, 512, (2, 33), device="cuda")
    a = ref(ids, use_cache=False).logits.float()
    b = lat(ids, use_cache=False).logits.float()
    assert (a - b).abs().max() <= 0.05 * a.abs().max()
