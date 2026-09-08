"""Plain truncated-SVD factorizer: the simplest producer of the checkpoint layout.

Per compressed layer, K is factorized per KV head (``W_h = U_h S_h V_h``,
keep ``k_rank`` directions) and V jointly (keep ``v_rank``). Sigma is folded
into VS. Useful for synthetic checkpoints (latency benchmarks, tests) and as
the weight-SVD baseline; it involves no data and no training.

    python -m lrkv.factorize --model MODEL_DIR --k-rank 32 --v-rank 1024 \
        --skip-layers 0 1 31 --output ckpt.pt
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence, Union

import torch

from .checkpoint import head_dim_of, layer_prefix, save_state_dict

RankSpec = Union[int, str, None]


def _resolve(spec: RankSpec, full: int) -> Optional[int]:
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec == "full":
            return full
        if spec.endswith("%"):
            return max(1, round(full * float(spec[:-1]) / 100))
        spec = int(spec)
    return max(1, min(int(spec), full))


@torch.no_grad()
def svd_factorize_layer(W_k: Optional[torch.Tensor], W_v: Optional[torch.Tensor], head_dim: int,
                        k_rank: RankSpec, v_rank: RankSpec, device=None) -> Dict[str, torch.Tensor]:
    """Factor one layer's projections. Returns {relative key: tensor} in float32 on CPU."""
    out: Dict[str, torch.Tensor] = {}
    if W_k is not None and k_rank is not None:
        W = W_k.detach().float().to(device)
        H = W.shape[0] // head_dim
        r = _resolve(k_rank, head_dim)
        VS_blocks, U_blocks = [], []
        for h in range(H):
            U, S, Vh = torch.linalg.svd(W[h * head_dim:(h + 1) * head_dim], full_matrices=False)
            U_blocks.append(U[:, :r])
            VS_blocks.append(S[:r, None] * Vh[:r])
        U_full = torch.zeros(W.shape[0], H * r, device=device)
        for h in range(H):
            U_full[h * head_dim:(h + 1) * head_dim, h * r:(h + 1) * r] = U_blocks[h]
        out["k_proj.VS.weight"] = torch.cat(VS_blocks, 0).cpu()
        out["k_proj.U.weight"] = U_full.cpu()
        out["k_proj.head_ranks"] = torch.full((H,), r, dtype=torch.int32)
    if W_v is not None and v_rank is not None:
        W = W_v.detach().float().to(device)
        r = _resolve(v_rank, min(W.shape))
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        out["v_proj.VS.weight"] = (S[:r, None] * Vh[:r]).cpu()
        out["v_proj.U.weight"] = U[:, :r].cpu()
    return out


@torch.no_grad()
def svd_factorize_model(model, k_rank: RankSpec, v_rank: RankSpec, skip_layers: Sequence[int] = (),
                        dtype: torch.dtype = torch.bfloat16, device=None) -> Dict[str, torch.Tensor]:
    """Factor-only state dict for every layer not in ``skip_layers``."""
    head_dim = head_dim_of(model.config)
    skip = set(int(i) for i in skip_layers)
    sd: Dict[str, torch.Tensor] = {}
    for i, block in enumerate(model.model.layers):
        if i in skip:
            continue
        attn = block.self_attn
        dev = device if device is not None else attn.k_proj.weight.device
        part = svd_factorize_layer(attn.k_proj.weight, attn.v_proj.weight, head_dim, k_rank, v_rank, device=dev)
        for k, v in part.items():
            sd[layer_prefix(i) + k] = v.to(dtype) if v.is_floating_point() else v
    return sd


def main():
    p = argparse.ArgumentParser(description="Truncated-SVD low-rank K/V checkpoint.")
    p.add_argument("--model", required=True, help="Base model directory")
    p.add_argument("--k-rank", default="32", help="Rank per KV head for K: int, 'N%%' of head_dim, or 'full'")
    p.add_argument("--v-rank", default="1024", help="Joint rank for V: int, 'N%%' of min(out,in), or 'full'")
    p.add_argument("--no-k", action="store_true", help="Leave K dense")
    p.add_argument("--no-v", action="store_true", help="Leave V dense")
    p.add_argument("--skip-layers", type=int, nargs="*", default=[0, 1, 31])
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from .model import load_base_model, parse_dtype
    model, _ = load_base_model(args.model, dtype=torch.float32, device_map="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = svd_factorize_model(model, None if args.no_k else args.k_rank, None if args.no_v else args.v_rank,
                             args.skip_layers, dtype=parse_dtype(args.dtype), device=device)
    save_state_dict(sd, args.output)
    print(f"wrote {args.output}: {len(sd)} tensors, layers skipped {args.skip_layers}")


if __name__ == "__main__":
    main()
