"""Compare result files with the STAR-KV paper's reference rows.

    python -m evals.compare results/table1/llama2_7b.json --paper llama2_7b
    python -m evals.compare results/table2/llama31_baseline/ --paper llama31_8b_instruct   # sharded directory
    python -m evals.compare run.json --paper starkv60_llama31   # a compressed checkpoint against the paper's STAR-KV row

A path is either an ``evals.run`` JSON (uses its ``ppl``, ``zero_shot_scores``,
``longbench_scores``, ``ruler_scores``) or a directory of shard JSONs (merged the
way ``evals.summarize`` does). Reference numbers are typed from STARKV.pdf:
Table 1 (PPL + zero-shot), Table 8 (LLaMA-3.1 zero-shot), Table 2 / 14
(LongBench) and Table 3 (RULER); see docs/*.md for which rows are reproducible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

from .harness import LONGBENCH_LABELS, LONGBENCH_PAPER_TASKS, ZERO_SHOT_LABELS, ZERO_SHOT_METRIC
from .ppl import PPL_SEQLEN
from .ruler import RULER_LABELS, RULER_PAPER_TASKS, RULER_SEQLEN
from .summarize import collect_scores

ZS = list(ZERO_SHOT_METRIC)  # paper column order: OBQA PIQA ARC-e ARC-c Hella Wino


def _zs(*v):
    return dict(zip(ZS, v))


def _lb(*v):
    return dict(zip(LONGBENCH_PAPER_TASKS, v))


def _ru(*v):
    return dict(zip(RULER_PAPER_TASKS, v))


PAPER: Dict[str, Dict[str, object]] = {
    "llama2_7b": {
        "source": "Table 1, LLaMA-2-7B (0%)",
        "ppl": {"wikitext2": 5.12, "c4": 7.04}, "zero_shot": _zs(44.20, 78.07, 76.30, 46.42, 76.00, 69.30),
    },
    "llama31_8b_instruct": {
        "source": "Table 8 zero-shot (no PPL in the paper); Table 2 / 14 LongBench; Table 3 RULER",
        "zero_shot": _zs(42.60, 80.96, 81.73, 54.86, 79.17, 73.72),
        "longbench": _lb(25.19, 23.20, 92.00, 39.90, 72.50, 26.90, 15.91),
        "ruler": _ru(100.0, 99.8, 99.9, 98.95, 100.0, 100.0, 99.6, 96.07, 78.12),
    },
    "starkv60_llama31": {
        "source": "STAR-KV 60% on LLaMA-3.1-8B-Instruct: Table 8 (FineWeb-Edu) zero-shot, Table 2 LongBench, Table 3 RULER",
        "zero_shot": _zs(41.80, 79.27, 81.02, 51.45, 75.52, 69.53),
        "longbench": _lb(23.2, 22.4, 89.06, 36.34, 67.0, 25.89, 13.2),
        "ruler": _ru(98.4, 86.0, 85.1, 84.0, 100.0, 100.0, 92.0, 87.73, 68.05),
    },
    "palu50_llama31": {
        "source": "Palu 50% on LLaMA-3.1-8B-Instruct: Table 2 LongBench, Table 3 RULER",
        "longbench": _lb(15.38, 22.10, 73.36, 27.58, 63.5, 21.66, 1.95),
        "ruler": _ru(98.60, 99.80, 75.35, 69.65, 99.80, 96.60, 88.40, 85.13, 58.58),
    },
}


def load_results(path: str) -> Dict[str, Dict[str, float]]:
    """{suite: {task: value}} from an evals.run JSON or a directory of shard JSONs."""
    out: Dict[str, Dict[str, float]] = {}
    if os.path.isdir(path):
        merged = collect_scores([path])
        lb = {t[len("longbench_"):]: m["score"] for t, m in merged.items() if t.startswith("longbench_") and "score" in m}
        ru = {t: m[str(RULER_SEQLEN)] for t, m in merged.items() if t in RULER_PAPER_TASKS and str(RULER_SEQLEN) in m}
        if lb:
            out["longbench"] = lb
        if ru:
            out["ruler"] = ru
        for f in sorted(glob.glob(os.path.join(path, "*.json"))):
            try:
                d = json.load(open(f))
            except json.JSONDecodeError:
                continue
            _merge_run_json(out, d)
        return out
    _merge_run_json(out, json.load(open(path)))
    return out


def _merge_run_json(out: dict, d: dict) -> None:
    if "ppl" in d:
        for k, block in d["ppl"].items():
            if k == f"seqlen{PPL_SEQLEN}":
                out.setdefault("ppl", {}).update({ds: v["ppl"] for ds, v in block.items()})
    if "zero_shot_scores" in d:
        out.setdefault("zero_shot", {}).update({t: v for t, v in d["zero_shot_scores"].items() if t != "average"})
    if "longbench_scores" in d:
        out.setdefault("longbench", {}).update(d["longbench_scores"])
    if "ruler_scores" in d and str(RULER_SEQLEN) in d["ruler_scores"]:
        out.setdefault("ruler", {}).update({t: v for t, v in d["ruler_scores"][str(RULER_SEQLEN)].items() if t != "average"})


def _table(title: str, cols: List[str], labels: Dict[str, str], ours: Dict[str, float], ref: Optional[Dict[str, float]],
           lower_is_better: bool = False, avg: bool = True) -> str:
    present = [c for c in cols if c in ours]
    head = "| | " + " | ".join(labels.get(c, c) for c in present) + (" | Avg |" if avg else " |")
    lines = [f"**{title}**", "", head, "|---|" + "---|" * (len(present) + (1 if avg else 0))]
    def row(name, vals, fmt="{:.2f}"):
        cells = [fmt.format(vals[c]) if c in vals and vals[c] is not None else "-" for c in present]
        if avg:
            full = [vals[c] for c in present if c in vals and vals[c] is not None]
            cells.append(fmt.format(sum(full) / len(full)) if len(full) == len(present) and full else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    row("ours", ours)
    if ref:
        row("paper", ref)
        delta = {c: ours[c] - ref[c] for c in present if c in ref}
        row("delta", delta, "{:+.2f}")
        missing = [c for c in cols if c not in ours]
        if missing:
            lines.append(f"| (not evaluated) | {', '.join(labels.get(c, c) for c in missing)} |")
    return "\n".join(lines)


def compare(results: Dict[str, Dict[str, float]], paper: Optional[str]) -> str:
    ref = PAPER.get(paper or "", {})
    parts = []
    if "ppl" in results:
        parts.append(_table(f"Perplexity @ {PPL_SEQLEN}", ["wikitext2", "c4"], {"wikitext2": "Wiki2", "c4": "C4"},
                            results["ppl"], ref.get("ppl"), avg=False))
    if "zero_shot" in results:
        parts.append(_table("Zero-shot (Table 1 convention)", ZS, ZERO_SHOT_LABELS, results["zero_shot"], ref.get("zero_shot")))
    if "longbench" in results:
        parts.append(_table("LongBench (Table 2 seven tasks)", LONGBENCH_PAPER_TASKS, LONGBENCH_LABELS, results["longbench"],
                            ref.get("longbench")))
    if "ruler" in results:
        parts.append(_table(f"RULER @ {RULER_SEQLEN} (Table 3 nine tasks)", RULER_PAPER_TASKS, RULER_LABELS, results["ruler"],
                            ref.get("ruler")))
    if ref:
        parts.append(f"paper reference: {paper} = {ref['source']}")
    return "\n\n".join(parts) if parts else "no recognised results in the given paths"


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare evaluation results with the STAR-KV paper.")
    p.add_argument("paths", nargs="+", help="evals.run JSON files or shard directories (merged)")
    p.add_argument("--paper", default=None, choices=list(PAPER), help="reference row")
    p.add_argument("--markdown", default=None, help="also write the tables to this file")
    args = p.parse_args(argv)
    results: Dict[str, Dict[str, float]] = {}
    for path in args.paths:
        for suite, vals in load_results(path).items():
            results.setdefault(suite, {}).update(vals)
    text = compare(results, args.paper)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
