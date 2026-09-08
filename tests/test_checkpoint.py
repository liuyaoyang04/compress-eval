import copy
import os
import sys

import pytest
import torch

from conftest import STARKV_DIR, tiny_model

from lrkv.checkpoint import (compressed_layers, convert_legacy_starkv, factors_only, infer_skip_layers,
                             is_legacy_starkv_state_dict, is_lowrank_state_dict, load_lowrank_checkpoint,
                             load_state_dict_file, save_state_dict)
from lrkv.compression import kv_cache_report
from lrkv.factorize import svd_factorize_model
from lrkv.model import describe_ranks
from lrkv.modules import HeadwiseLowRankLinear, LowRankLinear


def test_factorize_roundtrip(tmp_path):
    base = tiny_model()
    sd = svd_factorize_model(base, k_rank=48, v_rank=200, skip_layers=(0,), dtype=torch.float32)
    assert is_lowrank_state_dict(sd) and not is_legacy_starkv_state_dict(sd)
    assert compressed_layers(sd) == {1: {"k": True, "v": True}, 2: {"k": True, "v": True}}
    assert infer_skip_layers(sd, 3) == (0,)
    path = str(tmp_path / "ckpt.pt")
    save_state_dict(sd, path)

    model = copy.deepcopy(base)
    info = load_lowrank_checkpoint(model, path)
    assert info.skip_layers == (0,) and not info.converted_legacy and info.overridden_dense_keys == 0
    assert isinstance(model.model.layers[1].self_attn.k_proj, HeadwiseLowRankLinear)
    assert isinstance(model.model.layers[1].self_attn.v_proj, LowRankLinear)
    assert describe_ranks(model) == {1: {"k_head_ranks": [48, 48], "v_rank": 200},
                                     2: {"k_head_ranks": [48, 48], "v_rank": 200}}
    # Truncation error shrinks with rank (random weights have a flat spectrum, so only monotonicity is checked).
    x = torch.randn(2, 5, 512)
    dense = base.model.layers[1].self_attn.k_proj(x)
    err48 = (model.model.layers[1].self_attn.k_proj(x) - dense).norm()
    model96 = copy.deepcopy(base)
    load_lowrank_checkpoint(model96, svd_factorize_model(base, k_rank=96, v_rank=200, skip_layers=(0,), dtype=torch.float32))
    err96 = (model96.model.layers[1].self_attn.k_proj(x) - dense).norm()
    assert err96 < err48 < dense.norm()
    assert isinstance(model.model.layers[0].self_attn.k_proj, torch.nn.Linear)
    rep = kv_cache_report(sd, 3, 2, 128)
    assert rep["per_token"]["k_padded"] == 256 + 2 * 96 and rep["per_token"]["v"] == 256 + 2 * 200


def test_full_rank_factorization_is_exact():
    base = tiny_model()
    sd = svd_factorize_model(base, k_rank="full", v_rank="full", skip_layers=(), dtype=torch.float32)
    model = copy.deepcopy(base)
    load_lowrank_checkpoint(model, sd)
    x = torch.randn(1, 7, 512)
    for i in range(3):
        a, b = base.model.layers[i].self_attn, model.model.layers[i].self_attn
        assert torch.allclose(a.k_proj(x), b.k_proj(x), atol=1e-4)
        assert torch.allclose(a.v_proj(x), b.v_proj(x), atol=1e-4)


def test_partial_layer_k_only_v_only():
    base = tiny_model()
    sd_k = svd_factorize_model(base, k_rank=32, v_rank=None, skip_layers=(0, 2), dtype=torch.float32)
    sd_v = svd_factorize_model(base, k_rank=None, v_rank=100, skip_layers=(0, 1), dtype=torch.float32)
    sd = {**sd_k, **sd_v}
    assert compressed_layers(sd) == {1: {"k": True, "v": False}, 2: {"k": False, "v": True}}
    model = copy.deepcopy(base)
    load_lowrank_checkpoint(model, sd)
    assert isinstance(model.model.layers[1].self_attn.k_proj, HeadwiseLowRankLinear)
    assert isinstance(model.model.layers[1].self_attn.v_proj, torch.nn.Linear)
    assert isinstance(model.model.layers[2].self_attn.v_proj, LowRankLinear)


def test_missing_factor_raises():
    base = tiny_model()
    sd = svd_factorize_model(base, k_rank=16, v_rank=64, skip_layers=(0,), dtype=torch.float32)
    del sd["model.layers.1.self_attn.k_proj.U.weight"]
    with pytest.raises((RuntimeError, KeyError)):
        load_lowrank_checkpoint(copy.deepcopy(base), sd)


@pytest.mark.skipif(not os.path.exists(os.path.join(STARKV_DIR, "model.py")), reason="upstream clone missing")
def test_legacy_conversion_matches_upstream_fuse_and_prune(tmp_path):
    """Our state-dict conversion must produce the tensors STAR-KV's fuse_and_prune produces."""
    sys.path.insert(0, STARKV_DIR)
    try:
        import model as starkv
    finally:
        sys.path.remove(STARKV_DIR)
    base = tiny_model()
    m = copy.deepcopy(base)
    starkv.replace_linear_layer(m, m.config, skip_layers=(0,), init_svd=True)
    # Push the thresholds up so some directions are pruned, differently per head.
    torch.manual_seed(1)
    with torch.no_grad():
        for i in (1, 2):
            attn = m.model.layers[i].self_attn
            for sb in attn.k_proj.Sigma_blocks:
                qv = torch.quantile(sb.diag, float(torch.rand(1) * 0.6 + 0.2))
                sb.soft_thres_layer.alpha.fill_(float(qv))
            qv = torch.quantile(attn.v_proj.Sigma.diag, 0.6)
            attn.v_proj.Sigma.soft_thres_layer.alpha.fill_(float(qv))
            attn.k_proj.U.weight.add_(0.01 * torch.randn_like(attn.k_proj.U.weight))  # cross-head noise in U
    legacy_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    assert is_legacy_starkv_state_dict(legacy_sd)

    fused = starkv.fuse_and_prune(copy.deepcopy(m), skip_layers=(0,))
    up_sd = fused.state_dict()
    ours, report = convert_legacy_starkv(legacy_sd)
    assert not is_legacy_starkv_state_dict(ours) and is_lowrank_state_dict(ours)
    for i in (1, 2):
        pre = f"model.layers.{i}.self_attn."
        assert torch.equal(ours[pre + "k_proj.head_ranks"], up_sd[pre + "k_proj.head_ranks"].to(torch.int32))
        for key in ("k_proj.VS.weight", "k_proj.U.weight", "v_proj.VS.weight", "v_proj.U.weight"):
            assert torch.allclose(ours[pre + key], up_sd[pre + key], atol=1e-6), key
    assert report.max_offblock_energy() > 0  # the injected cross-head noise is reported ...
    assert all(r.k_offblock_energy < 0.05 for r in report.layers)  # ... and is small here
    # Non-factor tensors pass through untouched, and the file loader converts transparently.
    assert torch.equal(ours["lm_head.weight"], legacy_sd["lm_head.weight"])
    path = str(tmp_path / "legacy.pt")
    torch.save(legacy_sd, path)
    sd2, rep2 = load_state_dict_file(path)
    assert rep2 is not None and set(sd2) == set(ours)
    assert len(factors_only(sd2)) == 2 * 5


def test_flattened_legacy_sigma_recovers_rank_from_zero_rows():
    """Older STAR-KV 'fused' files reset Sigma to ones/alpha 0 but keep pruned V rows at zero."""
    base = tiny_model()
    m = copy.deepcopy(base)
    sys.path.insert(0, STARKV_DIR)
    try:
        import model as starkv
    finally:
        sys.path.remove(STARKV_DIR)
    starkv.replace_linear_layer(m, m.config, skip_layers=(0, 2), init_svd=True)
    attn = m.model.layers[1].self_attn
    with torch.no_grad():
        # emulate the old fusion: fold, zero dead rows, flatten Sigma bookkeeping
        d = attn.v_proj.Sigma.diag
        keep = torch.zeros_like(d, dtype=torch.bool)
        keep[:150] = True
        attn.v_proj.V.weight.mul_(torch.where(keep, d, torch.zeros_like(d))[:, None])
        d.fill_(1.0)
        attn.v_proj.Sigma.soft_thres_layer.alpha.fill_(0.0)
    sd, report = convert_legacy_starkv(m.state_dict())
    assert report.layers[0].v_flattened and report.layers[0].v_rank == 150
    assert sd["model.layers.1.self_attn.v_proj.VS.weight"].shape[0] == 150
