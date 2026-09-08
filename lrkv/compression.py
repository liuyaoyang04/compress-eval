"""KV-cache size encoded in a checkpoint or a loaded model (no GPU needed for the former).

Worth running before any long eval: a checkpoint can be numerically perfect
while carrying no compression at all. Two K numbers are reported:

    padded   Hkv * max_h r_h   what the latent cache actually stores per token
                               (heads padded to the layer's max rank)
    compact  sum_h r_h         what a rank-packed layout would store

    python -m lrkv.compression --checkpoint ckpt.pt --model MODEL_DIR [--per-layer] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import torch

from .checkpoint import layer_prefix, load_state_dict_file


def ranks_from_state_dict(sd: Dict[str, torch.Tensor], num_layers: int) -> Dict[int, dict]:
    out = {}
    for i in range(num_layers):
        pre = layer_prefix(i)
        k = sd[pre + "k_proj.head_ranks"].tolist() if pre + "k_proj.head_ranks" in sd else None
        v = int(sd[pre + "v_proj.VS.weight"].shape[0]) if pre + "v_proj.VS.weight" in sd else None
        if k is not None or v is not None:
            out[i] = {"k_head_ranks": k, "v_rank": v}
    return out


def report_from_ranks(ranks: Dict[int, dict], num_layers: int, num_kv_heads: int, head_dim: int) -> dict:
    """``ranks`` as returned by :func:`ranks_from_state_dict` or ``lrkv.model.describe_ranks``."""
    full = num_kv_heads * head_dim
    rows = []
    k_pad = k_cmp = v_tot = 0
    for i in range(num_layers):
        r = ranks.get(i) or {}
        k_ranks, v_rank = r.get("k_head_ranks"), r.get("v_rank")
        kp = num_kv_heads * max(k_ranks) if k_ranks else full
        kc = sum(k_ranks) if k_ranks else full
        vr = v_rank if v_rank is not None else full
        k_pad += kp
        k_cmp += kc
        v_tot += vr
        rows.append({"layer": i, "k_head_ranks": k_ranks, "k_padded": kp, "k_compact": kc, "v_rank": vr,
                     "compressed": bool(k_ranks or v_rank is not None)})
    full_total = 2 * num_layers * full
    return {
        "num_layers": num_layers, "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        "per_token": {
            "k_padded": k_pad, "k_compact": k_cmp, "v": v_tot,
            "kv_padded": k_pad + v_tot, "kv_compact": k_cmp + v_tot, "uncompressed": full_total,
        },
        "compression_padded": 1.0 - (k_pad + v_tot) / full_total,
        "compression_compact": 1.0 - (k_cmp + v_tot) / full_total,
        "layers": rows,
    }


def kv_cache_report(sd: Dict[str, torch.Tensor], num_layers: int, num_kv_heads: int, head_dim: int) -> dict:
    return report_from_ranks(ranks_from_state_dict(sd, num_layers), num_layers, num_kv_heads, head_dim)


def model_report(model) -> dict:
    from .checkpoint import head_dim_of
    from .model import describe_ranks
    cfg = model.config
    return report_from_ranks(describe_ranks(model), len(model.model.layers), cfg.num_key_value_heads, head_dim_of(cfg))


def format_report(rep: dict, per_layer: bool = False) -> str:
    lines = []
    full = rep["num_kv_heads"] * rep["head_dim"]
    if per_layer:
        lines.append(f"{'layer':>6} {'k rank/kv-head':<48} {'k max':>6} {'v rank':>7}")
        for r in rep["layers"]:
            if not r["compressed"]:
                lines.append(f"{r['layer']:>6} {'(uncompressed)':<48} {rep['head_dim']:>6} {full:>7}")
            else:
                kr = r["k_head_ranks"]
                lines.append(f"{r['layer']:>6} {str(kr) if kr else '(dense)':<48} "
                             f"{max(kr) if kr else rep['head_dim']:>6} {r['v_rank']:>7}")
        lines.append("")
    pt = rep["per_token"]
    lines += [
        f"  K per token : {pt['k_padded']:8d} padded / {pt['k_compact']:8d} compact   (uncompressed {rep['num_layers'] * full})",
        f"  V per token : {pt['v']:8d}                       (uncompressed {rep['num_layers'] * full})",
        f"  K+V         : {pt['kv_padded']:8d} padded / {pt['kv_compact']:8d} compact   (uncompressed {pt['uncompressed']})",
        f"  COMPRESSION : {100 * rep['compression_padded']:.1f}% (padded)  {100 * rep['compression_compact']:.1f}% (compact)",
    ]
    if rep["compression_padded"] <= 0:
        lines.append("  WARNING: no compression encoded in this checkpoint")
    return "\n".join(lines)


def model_dims(model_dir: str):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    return cfg["num_hidden_layers"], cfg.get("num_key_value_heads", cfg["num_attention_heads"]), head_dim


def main():
    p = argparse.ArgumentParser(description="Report the KV-cache compression encoded in a checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default=None, help="Model directory (config.json) for layer/head counts")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--per-layer", action="store_true")
    p.add_argument("--json", default=None, help="Write the report to this JSON file")
    args = p.parse_args()

    if args.model:
        nl, nkv, hd = model_dims(args.model)
    else:
        if None in (args.num_layers, args.num_kv_heads, args.head_dim):
            raise SystemExit("pass --model MODEL_DIR or all of --num-layers/--num-kv-heads/--head-dim")
        nl, nkv, hd = args.num_layers, args.num_kv_heads, args.head_dim
    sd, legacy = load_state_dict_file(args.checkpoint)
    rep = kv_cache_report(sd, nl, nkv, hd)
    rep["checkpoint"] = args.checkpoint
    rep["converted_from_legacy"] = legacy is not None
    print(f"checkpoint : {args.checkpoint}" + ("  (legacy STAR-KV format, converted)" if legacy else ""))
    if legacy is not None and legacy.max_offblock_energy() > 1e-4:
        print(f"  note: training-time K U had cross-head entries (max relative energy {legacy.max_offblock_energy():.2e})")
    print(format_report(rep, args.per_layer))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
