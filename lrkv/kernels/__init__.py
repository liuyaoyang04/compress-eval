"""Triton kernels. Import lazily so ``python -m lrkv.kernels.abx_rope`` does not double-import."""


def __getattr__(name):
    if name in ("abx_rope", "abx_rope_reference", "rope_tables", "rotate_half", "default_inv_freq"):
        from . import abx_rope as _m
        return getattr(_m, name)
    raise AttributeError(name)
