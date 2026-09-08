"""Evaluate a base model or a low-rank checkpoint: PPL, zero-shot, LongBench, RULER.

    python -m evals.run --model MODEL_DIR --checkpoint ckpt.pt --ppl --ppl-datasets wikitext2,c4
    python -m evals.run --model MODEL_DIR --baseline --tasks zero-shot --batch-size 32
    python -m evals.run --model MODEL_DIR --checkpoint ckpt.pt --decode-mode triton \
        --longbench --longbench-tasks star-paper-7 --long-batch-size 4 --max-length 31500 \
        --output results/evals/run.json
    python -m evals.run --model MODEL_DIR --baseline --ruler --long-batch-size 4 --output results/evals/ruler.json

``--decode-mode reference`` (default) keeps the stock HF attention with the
low-rank projections inside it (full-size KV cache): STAR-KV's accuracy
reference. ``triton`` / ``torch`` / ``sdpa`` run on the latent KV cache;
perplexity is prefill-only so they only matter for generation tasks.

Protocol pinned to STAR-KV Table 1: perplexity windows of 4096 tokens
(``evals.ppl.PPL_SEQLEN``) and the mixed acc / acc_norm zero-shot convention
(``evals.harness.ZERO_SHOT_METRIC``), reported under ``zero_shot_scores``.
``--longbench`` runs the Table 2 seven tasks (batch 4, max_length 31500,
``evals.harness.LONGBENCH_PAPER_TASKS``) and reports ``longbench_scores``;
``--ruler`` runs the Table 3 nine-task set at 4096 (``evals.ruler``) and
reports ``ruler_scores``; external files come from ``evals.sources``.

Data comes from the repo-local pinned store (``python -m evals.sources`` once);
runs are offline. GPU selection is left to CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

from .sources import configure_offline, require  # noqa: I001

configure_offline()  # repo-local pinned data, no network: before anything imports datasets / lm_eval

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from lrkv.compression import format_report, model_report
from lrkv.model import ALL_MODES, enable_cache, load_base_model, load_compressed_model

from .meta import environment_fingerprint  # noqa: E402
from .harness import (ZERO_SHOT_LABELS, ZERO_SHOT_METRIC, ZERO_SHOT_TASKS, longbench_scores,
                      resolve_longbench_tasks, run_lm_eval, task_scores, zero_shot_scores)
from .ppl import PPL_SEQLEN, evaluate_ppl
from .ruler import RULER_SEQLEN, build_tasks as build_ruler_tasks, format_scores as format_ruler_scores
from .ruler import install_sources as install_ruler_sources
from .ruler import resolve_ruler_tasks, ruler_scores, task_metadata as ruler_task_metadata


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a low-rank KV cache model.")
    p.add_argument("--model", required=True, help="Base model directory or HF id")
    p.add_argument("--checkpoint", default=None, help="Low-rank checkpoint (canonical or legacy STAR-KV)")
    p.add_argument("--baseline", action="store_true", help="Evaluate the uncompressed model")
    p.add_argument("--decode-mode", choices=ALL_MODES, default="reference")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--ppl", action="store_true")
    p.add_argument("--ppl-datasets", default="wikitext2,c4")
    p.add_argument("--ppl-seqlen", type=int, nargs="+", default=[PPL_SEQLEN],
                   help=f"Window length(s); {PPL_SEQLEN} is the STAR-KV Table 1 protocol")
    p.add_argument("--ppl-limit", type=int, default=None, help="Max windows per dataset")

    p.add_argument("--tasks", default=None, help="Comma list of lm-eval tasks, or 'zero-shot' for the STAR-KV six")
    p.add_argument("--batch-size", default="32", help="int or 'auto'")

    p.add_argument("--longbench", action="store_true")
    p.add_argument("--longbench-tasks", default="star-paper-7", help="'star-paper-7' (Table 2) or a comma list")
    p.add_argument("--ruler", action="store_true")
    p.add_argument("--ruler-tasks", default="paper-9", help="Preset (see evals.ruler) or comma list")
    p.add_argument("--ruler-seqlen", type=int, nargs="+", default=[RULER_SEQLEN],
                   help=f"RULER context length(s); {RULER_SEQLEN} is the STAR-KV Table 3 protocol")
    p.add_argument("--long-batch-size", default="4")
    p.add_argument("--max-length", type=int, default=31500)
    p.add_argument("--limit", type=int, default=None, help="lm-eval per-task example limit")
    p.add_argument("--log-samples", action="store_true")

    p.add_argument("--output", default=None, help="JSON file for all results")
    return p.parse_args(argv)


def _bs(s):
    return int(s) if str(s).isdigit() else s


def main(argv=None):
    args = parse_args(argv)
    if not args.baseline and not args.checkpoint:
        raise SystemExit("pass --checkpoint or --baseline")
    torch.manual_seed(args.seed)
    t0 = time.time()
    for suite, wanted in (("ppl", args.ppl), ("zero-shot", args.tasks == "zero-shot"), ("longbench", args.longbench),
                          ("ruler", args.ruler)):
        if wanted:
            require(suite)  # fail before loading the model if pinned data is missing
    if args.ruler:
        # Before any TaskManager is built: lm-eval's ruler module looks for NLTK data on import.
        install_ruler_sources()  # pinned local essays / SQuAD / punkt_tab, fetched once if missing

    print(f"loading {args.model}" + (f" + {args.checkpoint} [{args.decode_mode}]" if args.checkpoint else " (baseline)"))
    if args.baseline:
        model, tok = load_base_model(args.model, dtype=args.dtype, device_map=args.device_map)
        model.eval()
        enable_cache(model)
        compression = None
    else:
        model, tok, info = load_compressed_model(args.model, args.checkpoint, dtype=args.dtype,
                                                 device_map=args.device_map, decode_mode=args.decode_mode)
        compression = model_report(model)
        print(format_report(compression))

    out = {
        "model": args.model, "checkpoint": args.checkpoint, "baseline": args.baseline,
        "decode_mode": None if args.baseline else args.decode_mode, "dtype": args.dtype,
        "argv": sys.argv[1:], "compression": compression, "environment": environment_fingerprint(),
    }

    if args.ppl:
        print("\n=== perplexity ===")
        out["ppl"] = {}
        for seqlen in args.ppl_seqlen:
            out["ppl"][f"seqlen{seqlen}"] = evaluate_ppl(model, tok, args.ppl_datasets, seqlen=seqlen, limit=args.ppl_limit)

    if args.tasks:
        tasks = ZERO_SHOT_TASKS if args.tasks == "zero-shot" else args.tasks
        print("\n=== lm-eval ===")
        res = run_lm_eval(model, tok, tasks, _bs(args.batch_size), limit=args.limit, log_samples=args.log_samples)
        out["lmeval"] = res
        out["lmeval_scores"] = task_scores(res)
        zs = zero_shot_scores(res)
        if zs:
            out["zero_shot_scores"] = zs
            cols = [t for t in ZERO_SHOT_METRIC if t in zs]
            print("zero-shot (Table 1 convention): " + "  ".join(f"{ZERO_SHOT_LABELS[t]} {zs[t]:.2f}" for t in cols)
                  + f"  | avg {zs['average']:.2f} over {len(cols)} tasks")

    if args.longbench:
        print("\n=== LongBench ===")
        res = run_lm_eval(model, tok, resolve_longbench_tasks(args.longbench_tasks), _bs(args.long_batch_size),
                          max_length=args.max_length, limit=args.limit, log_samples=args.log_samples)
        out["longbench"] = res
        out["longbench_scores"] = longbench_scores(res)
        if out["longbench_scores"]:
            avg = sum(out["longbench_scores"].values()) / len(out["longbench_scores"])
            print(f"LongBench average over {len(out['longbench_scores'])} tasks: {avg:.2f}")

    if args.ruler:
        print("\n=== RULER ===")
        from lm_eval.tasks import TaskManager
        tasks = resolve_ruler_tasks(args.ruler_tasks)
        tm = TaskManager(metadata=ruler_task_metadata(args.model, args.ruler_seqlen))
        task_objs = build_ruler_tasks(tm, tasks)  # group-order samples, independent of sharding
        res = run_lm_eval(model, tok, task_objs, _bs(args.long_batch_size), max_length=args.max_length,
                          limit=args.limit, log_samples=args.log_samples, task_manager=tm)
        out["ruler"] = res
        out["ruler_scores"] = {str(L): ruler_scores(res, L, tasks) for L in args.ruler_seqlen}
        for L, sc in out["ruler_scores"].items():
            if sc:
                print(f"RULER@{L} (Table 3 convention): {format_ruler_scores(sc)}")

    out["elapsed_s"] = time.time() - t0
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nresults saved to {args.output}")


if __name__ == "__main__":
    main()
