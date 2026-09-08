"""lrkv: latent (low-rank) KV-cache inference framework and evaluation harness.

The inference path is extracted from STAR-KV (ICML 2026, upstream commit
f08a62d, kept under ``baselines/STAR-KV``) and restructured so any method that
produces per-KV-head low-rank K factors and a joint low-rank V factor per layer
can be run and evaluated through the same code:

    lrkv.modules      canonical low-rank projection modules (checkpoint layout)
    lrkv.checkpoint   checkpoint format, loading, legacy STAR-KV conversion
    lrkv.kernels      fused Triton decode kernel (A @ (B @ X^T) + RoPE)
    lrkv.attention    attention over the latent KV cache (prefill + decode)
    lrkv.model        high-level loading and attention replacement
    lrkv.factorize    plain SVD factorizer (synthetic / baseline checkpoints)
    lrkv.compression  KV-cache size accounting for a checkpoint
"""

__version__ = "0.1.0"

_LAZY = {
    "HeadwiseLowRankLinear": ("lrkv.modules", "HeadwiseLowRankLinear"),
    "LowRankLinear": ("lrkv.modules", "LowRankLinear"),
    "LowRankAttention": ("lrkv.attention", "LowRankAttention"),
    "load_base_model": ("lrkv.model", "load_base_model"),
    "load_compressed_model": ("lrkv.model", "load_compressed_model"),
    "install_lowrank_attention": ("lrkv.model", "install_lowrank_attention"),
    "set_decode_mode": ("lrkv.model", "set_decode_mode"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'lrkv' has no attribute {name!r}")


__all__ = list(_LAZY) + ["__version__"]
