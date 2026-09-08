"""Decode latency: end-to-end single-step decode and synthetic per-layer attention.

    e2e        one decode step (batch 1, one new token) per KV length; stock
               model vs latent cache in each decode mode
    layerwise  single-layer attention on synthetic tensors at the checkpoint's
               real per-head ranks: SDPA on full K/V, torch reconstruction,
               fused kernel at padded rank, fused kernel at dynamic rank

Ported from STAR-KV ``latency.py``. Changes: no CUDA_VISIBLE_DEVICES override
(set it in the environment); every decode mode is timed from one loaded
model; ``--cache-reset crop`` restores the cache with ``crop`` instead of the
upstream deep copy (default ``deepcopy``, the upstream protocol), which lets a
64K stock baseline run without OOM on the copy.

    python -m evals.latency --model MODEL_DIR --baseline --output-dir results/latency
    python -m evals.latency --model MODEL_DIR --checkpoint ckpt.pt --mode both \
        --ctx-lens 1024 4096 16384 32000 --output-dir results/latency
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _median(ts: List[float]) -> float:
    ts = sorted(ts)
    return ts[len(ts) // 2]


@torch.no_grad()
def bench_decode(model, tokenizer, ctx_lens, n_runs: int, device, cache_reset: str = "deepcopy",
                 verbose: bool = True) -> Dict[int, Optional[float]]:
    """Median milliseconds of one decode step (batch 1) at each KV length; None on OOM.

    ``cache_reset="deepcopy"`` re-copies the prefill cache before every timed
    step (STAR-KV's protocol); ``"crop"`` drops the appended token instead.
    """
    assert cache_reset in ("deepcopy", "crop")
    results: Dict[int, Optional[float]] = {}
    for ctx_len in ctx_lens:
        torch.cuda.empty_cache()
        gc.collect()
        prompt = "Hello world. " * (ctx_len // 3)
        inp = tokenizer(prompt, return_tensors="pt", max_length=ctx_len, truncation=True).to(device)
        tok = inp.input_ids[:, -1:]
        prompt_len = inp.input_ids.shape[1]
        try:
            base_cache = model(**inp, use_cache=True).past_key_values
            cache = base_cache
            ts = []
            for _ in range(n_runs):
                if cache_reset == "deepcopy":
                    cache = copy.deepcopy(base_cache)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(tok, past_key_values=cache, use_cache=True)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
                if cache_reset == "crop":
                    cache.crop(prompt_len)
            del cache, base_cache
            results[ctx_len] = _median(ts)
            if verbose:
                print(f"  ctx={ctx_len:7d}  {results[ctx_len]:8.2f} ms")
        except torch.cuda.OutOfMemoryError:
            results[ctx_len] = None
            if verbose:
                print(f"  ctx={ctx_len:7d}  OOM")
            torch.cuda.empty_cache()
            gc.collect()
    return results


def print_speedup_table(ctx_lens, baseline: Optional[dict], modes: Dict[str, dict]):
    cols = ["stock"] + list(modes)
    print(f"\n{'ctx':>8}" + "".join(f"{c + ' (ms)':>14}" for c in cols) + "".join(f"{'x ' + m:>12}" for m in modes))
    for c in ctx_lens:
        b = baseline.get(c) if baseline else None
        cells = [f"{b:14.2f}" if b else f"{'OOM' if baseline else '-':>14}"]
        for m in modes:
            v = modes[m].get(c)
            cells.append(f"{v:14.2f}" if v else f"{'OOM':>14}")
        for m in modes:
            v = modes[m].get(c)
            cells.append(f"{b / v:11.2f}x" if (b and v) else f"{'-':>12}")
        print(f"{c:8d}" + "".join(cells))


@torch.no_grad()
def bench_layerwise(model, seq: int, batch: int, verbose: bool = True) -> List[dict]:
    import triton.testing as tt
    from lrkv.kernels.abx_rope import abx_rope, default_inv_freq
    from lrkv.model import lowrank_attention_layers

    cfg = model.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    layers = lowrank_attention_layers(model)
    if not layers:
        raise RuntimeError("no LowRankAttention layers; load a checkpoint with a latent decode mode")
    Hd = layers[0][1].head_dim
    scale = 1.0 / math.sqrt(Hd)
    dtype = torch.bfloat16
    device = next(model.parameters()).device

    def ms(fn):
        return tt.do_bench(fn, quantiles=[0.5, 0.2, 0.8], warmup=10, rep=30)[0] * 1000  # us

    rows = []
    for i, attn in layers:
        ranks, rv = attn.k_head_ranks, attn.v_rank
        max_r = max(ranks)
        torch.cuda.empty_cache()
        gc.collect()
        try:
            r_vec = torch.tensor(ranks, dtype=torch.int32, device=device)
            Q = torch.randn(batch, Hq, 1, Hd, dtype=dtype, device=device)
            K_c = torch.randn(batch, Hkv, seq, max_r, dtype=dtype, device=device)
            V_c = torch.randn(batch, seq, rv, dtype=dtype, device=device)
            U_T = torch.randn(Hkv, max_r, Hd, dtype=dtype, device=device)
            U_v = torch.randn(Hkv, rv, Hd, dtype=dtype, device=device)
            for h, r in enumerate(ranks):
                U_T[h, r:, :] = 0
                K_c[:, h, :, r:] = 0
            inv_freq = default_inv_freq(Hd, device=device)
            G = Hq // Hkv

            def pv_expand(probs):
                pv = torch.matmul(probs.squeeze(2), V_c)
                pv = pv.view(batch, Hkv, G, rv).transpose(0, 1).reshape(Hkv, batch * G, rv)
                return torch.bmm(pv, U_v)

            try:
                K_full = torch.randn(batch, Hkv, seq, Hd, dtype=dtype, device=device)
                V_full = torch.randn(batch, Hkv, seq, Hd, dtype=dtype, device=device)
                t_sdpa = ms(lambda: F.scaled_dot_product_attention(Q, K_full, V_full, enable_gqa=True))
                del K_full, V_full
            except torch.cuda.OutOfMemoryError:
                t_sdpa = float("nan")
                torch.cuda.empty_cache()

            def torch_path():
                K = torch.matmul(K_c, U_T).repeat_interleave(G, dim=1)
                probs = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) * scale, dim=-1, dtype=torch.float32).to(dtype)
                return pv_expand(probs)

            def fused(ranks_arg):
                logits = abx_rope(Q, U_T, K_c, ranks=ranks_arg, inv_freq=inv_freq, compute_dtype=torch.float16) * scale
                probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(dtype)
                return pv_expand(probs)

            t_torch = ms(torch_path)
            t_uni = ms(lambda: fused(None))
            t_dyn = ms(lambda: fused(r_vec))
            row = dict(layer=i, rank_k_min=min(ranks), rank_k_max=max_r, rank_v=rv, sdpa_us=t_sdpa,
                       torch_us=t_torch, triton_uniform_us=t_uni, triton_dynamic_us=t_dyn,
                       sp_torch=t_sdpa / t_torch, sp_triton_uniform=t_sdpa / t_uni, sp_triton_dynamic=t_sdpa / t_dyn)
            rows.append(row)
            if verbose:
                print(f"  layer {i:2d} rk={min(ranks):3d}..{max_r:3d} rv={rv:4d}  sdpa={t_sdpa:8.1f}us  "
                      f"torch={t_torch:8.1f}us  tri-uni={t_uni:8.1f}us  tri-dyn={t_dyn:8.1f}us  "
                      f"speedup(tri-dyn)={row['sp_triton_dynamic']:.2f}x")
            del Q, K_c, V_c, U_T, U_v
        except torch.cuda.OutOfMemoryError:
            print(f"  layer {i:2d}: OOM, skipped")
            torch.cuda.empty_cache()
            gc.collect()
    if rows and verbose:
        n = len(rows)
        print(f"  mean speedup vs SDPA: torch {sum(r['sp_torch'] for r in rows) / n:.2f}x, "
              f"fused padded {sum(r['sp_triton_uniform'] for r in rows) / n:.2f}x, "
              f"fused dynamic {sum(r['sp_triton_dynamic'] for r in rows) / n:.2f}x")
    return rows


def _write_csv(rows: List[dict], path: str):
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Decode latency benchmarks.")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", default=None, help="Low-rank checkpoint (omit with --baseline)")
    p.add_argument("--baseline", action="store_true", help="Time the stock model only")
    p.add_argument("--mode", choices=["e2e", "layerwise", "both"], default="e2e")
    p.add_argument("--decode-modes", default="triton,torch", help="Comma list of latent decode modes to time")
    p.add_argument("--ctx-lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32000, 64000])
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--cache-reset", choices=["deepcopy", "crop"], default="deepcopy",
                   help="How the prefill cache is restored between timed steps (deepcopy = upstream protocol)")
    p.add_argument("--lw-seq", type=int, default=32768)
    p.add_argument("--lw-batch", type=int, default=16)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--output-dir", default="results/latency")
    p.add_argument("--tag", default=None, help="Filename tag (default: checkpoint basename)")
    return p.parse_args()


def main():
    from lrkv.model import load_base_model, load_compressed_model, set_decode_mode
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    if args.baseline:
        model, tok = load_base_model(args.model, dtype=args.dtype)
        model.eval()
        print("\n=== stock model (bf16 SDPA, full KV cache) ===")
        res = bench_decode(model, tok, args.ctx_lens, args.runs, device, args.cache_reset)
        path = os.path.join(args.output_dir, "baseline_latency.json")
        json.dump(res, open(path, "w"), indent=2)
        print(f"saved {path}")
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --baseline")
    tag = args.tag or os.path.splitext(os.path.basename(args.checkpoint))[0]
    modes = [m.strip() for m in args.decode_modes.split(",") if m.strip()]
    model, tok, info = load_compressed_model(args.model, args.checkpoint, dtype=args.dtype, decode_mode=modes[0])

    if args.mode in ("e2e", "both"):
        per_mode = {}
        for m in modes:
            set_decode_mode(model, m)
            print(f"\n=== latent cache, decode_mode={m} ===")
            per_mode[m] = bench_decode(model, tok, args.ctx_lens, args.runs, device, args.cache_reset)
            path = os.path.join(args.output_dir, f"{tag}_{m}_latency.json")
            json.dump(per_mode[m], open(path, "w"), indent=2)
            print(f"saved {path}")
        base_path = os.path.join(args.output_dir, "baseline_latency.json")
        baseline = {int(k): v for k, v in json.load(open(base_path)).items()} if os.path.exists(base_path) else None
        print_speedup_table(args.ctx_lens, baseline, per_mode)

    if args.mode in ("layerwise", "both"):
        print(f"\n=== layer-wise attention, seq={args.lw_seq} batch={args.lw_batch} ===")
        rows = bench_layerwise(model, args.lw_seq, args.lw_batch)
        path = os.path.join(args.output_dir, f"{tag}_layerwise_seq{args.lw_seq}_b{args.lw_batch}.csv")
        _write_csv(rows, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
