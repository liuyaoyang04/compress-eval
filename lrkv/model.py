"""High-level loading: base model, low-rank checkpoint, latent-cache attention.

    model, tok, info = load_compressed_model(base, ckpt, decode_mode="triton")

``decode_mode``:
    "reference"  low-rank modules inside the stock HF attention (full-size KV
                 cache; the accuracy reference, STAR-KV's default eval path)
    "triton" / "torch" / "sdpa"
                 latent KV cache through :class:`lrkv.attention.LowRankAttention`
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .attention import DECODE_MODES, LowRankAttention
from .checkpoint import LoadInfo, head_dim_of, load_lowrank_checkpoint
from .modules import HeadwiseLowRankLinear, LowRankLinear, is_lowrank_module

ALL_MODES = ("reference",) + DECODE_MODES

_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
           "fp32": torch.float32, "float32": torch.float32}


def parse_dtype(name) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    return _DTYPES[str(name).lower()]


def _transformers_major() -> int:
    import transformers
    return int(transformers.__version__.split(".")[0])


def load_base_model(model_path: str, dtype=torch.bfloat16, device_map="auto",
                    attn_implementation: Optional[str] = "sdpa", trust_remote_code: bool = False):
    """Load a Llama-family causal LM and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = parse_dtype(dtype)
    kwargs = dict(device_map=device_map, trust_remote_code=trust_remote_code)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if _transformers_major() >= 5:
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    # Tokenizer first: it is cheap and it lists the directory, so an unreadable
    # or unlistable model path fails here instead of after minutes of weight loading.
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    _assert_llama_layout(model)
    return model, tokenizer


def _assert_llama_layout(model: nn.Module) -> None:
    ok = hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "rotary_emb")
    if ok:
        attn = model.model.layers[0].self_attn
        ok = all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    if not ok:
        raise TypeError(f"{type(model).__name__} does not have the Llama layout "
                        "(model.model.layers[i].self_attn.{q,k,v,o}_proj and model.model.rotary_emb)")


def enable_cache(model: nn.Module) -> None:
    """Turn the KV cache on for both forward and generate.

    Loading with ``use_cache=False`` and then flipping ``config.use_cache`` leaves
    ``generation_config.use_cache`` False under transformers 5.x, so a plain
    ``generate()`` silently recomputes the whole sequence every step.
    """
    model.config.use_cache = True
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.use_cache = True


def lowrank_attention_layers(model: nn.Module) -> List[Tuple[int, LowRankAttention]]:
    return [(i, b.self_attn) for i, b in enumerate(model.model.layers) if isinstance(b.self_attn, LowRankAttention)]


def install_lowrank_attention(model: nn.Module, dtype: Optional[torch.dtype] = None,
                              decode_mode: str = "triton", kernel_dtype: torch.dtype = torch.float16) -> List[int]:
    """Swap every layer holding low-rank K and/or V modules to :class:`LowRankAttention`.

    A layer with only one low-rank projection gets the other wrapped at full
    rank (exact), so the whole layer runs on the latent cache. Layers with
    neither are left untouched. Returns the replaced layer indices.
    """
    assert decode_mode in DECODE_MODES, f"decode_mode must be one of {DECODE_MODES}"
    config = model.config
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None or not hasattr(rotary, "inv_freq"):
        raise RuntimeError("model.model.rotary_emb.inv_freq not found; the decode kernel needs the model's RoPE table")
    inv_freq = rotary.inv_freq.detach().float()
    attn_scaling = float(getattr(rotary, "attention_scaling", 1.0))
    head_dim = head_dim_of(config)

    replaced = []
    for i, block in enumerate(model.model.layers):
        attn = block.self_attn
        if isinstance(attn, LowRankAttention):
            attn.decode_mode = decode_mode
            attn.kernel_dtype = kernel_dtype
            replaced.append(i)
            continue
        k_lr, v_lr = is_lowrank_module(attn.k_proj), is_lowrank_module(attn.v_proj)
        if not (k_lr or v_lr):
            continue
        k_proj = attn.k_proj if k_lr else HeadwiseLowRankLinear.from_dense(attn.k_proj, head_dim)
        v_proj = attn.v_proj if v_lr else LowRankLinear.from_dense(attn.v_proj)
        block.self_attn = LowRankAttention(
            config, i, attn.q_proj, attn.o_proj, k_proj, v_proj, inv_freq, attn_scaling,
            dtype=dtype, decode_mode=decode_mode, kernel_dtype=kernel_dtype,
        )
        replaced.append(i)
    return replaced


def set_decode_mode(model: nn.Module, mode: str) -> None:
    assert mode in DECODE_MODES, f"mode must be one of {DECODE_MODES}"
    for _, attn in lowrank_attention_layers(model):
        attn.decode_mode = mode


def describe_ranks(model: nn.Module) -> dict:
    """{layer: {"k_head_ranks": [...], "v_rank": r}} for every low-rank layer (either mode)."""
    out = {}
    for i, block in enumerate(model.model.layers):
        attn = block.self_attn
        if isinstance(attn, LowRankAttention):
            out[i] = {"k_head_ranks": attn.k_head_ranks, "v_rank": attn.v_rank}
        elif is_lowrank_module(attn.k_proj) or is_lowrank_module(attn.v_proj):
            out[i] = {
                "k_head_ranks": attn.k_proj.head_ranks_list if is_lowrank_module(attn.k_proj) else None,
                "v_rank": attn.v_proj.rank if is_lowrank_module(attn.v_proj) else None,
            }
    return out


def load_compressed_model(
    model_path: str,
    checkpoint,
    dtype=torch.bfloat16,
    device_map="auto",
    decode_mode: str = "reference",
    kernel_dtype: torch.dtype = torch.float16,
    attn_implementation: Optional[str] = "sdpa",
    verbose: bool = True,
):
    """Base model + low-rank checkpoint (+ latent-cache attention unless ``decode_mode == "reference"``).

    ``checkpoint`` may be a path (canonical or legacy STAR-KV) or a state dict.
    Returns (model, tokenizer, LoadInfo).
    """
    assert decode_mode in ALL_MODES, f"decode_mode must be one of {ALL_MODES}"
    dtype = parse_dtype(dtype)
    model, tokenizer = load_base_model(model_path, dtype=dtype, device_map=device_map,
                                       attn_implementation=attn_implementation)
    info: LoadInfo = load_lowrank_checkpoint(model, checkpoint)
    if verbose:
        n = len(info.compressed)
        print(f"  checkpoint: {n} compressed layers, uncompressed {list(info.skip_layers)}"
              f"{' (converted from legacy STAR-KV format)' if info.converted_legacy else ''}"
              f", {info.overridden_dense_keys} dense tensors overridden")
        if info.legacy_report is not None and info.legacy_report.max_offblock_energy() > 1e-4:
            print(f"  note: training-time K U carried cross-head entries (max relative energy "
                  f"{info.legacy_report.max_offblock_energy():.2e}); they are dropped, as in STAR-KV's export")
    if decode_mode != "reference":
        install_lowrank_attention(model, dtype=dtype, decode_mode=decode_mode, kernel_dtype=kernel_dtype)
    model.eval()
    enable_cache(model)
    return model, tokenizer, info
