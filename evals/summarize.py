"""Merge lm-eval result JSONs (e.g. per-GPU shards) and average a task set.

Accepts the files ``evals.run`` writes (``{"longbench": {...lm-eval...}}``,
``{"ruler": ...}``), raw lm-eval dicts, and the historical STAR-KV shard files.

    python -m evals.summarize RESULT_DIR_OR_FILES... [--preset star-paper-7 | --tasks a,b,c] [--json out.json]
    python -m evals.summarize results/ruler_shards --preset ruler-paper-9      # metric "4096"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Iterable, List, Optional

from .harness import LONGBENCH_PRESETS
from .ruler import RULER_PRESETS, RULER_SEQLEN

PRESETS = {**LONGBENCH_PRESETS, **{f"ruler-{k}": v for k, v in RULER_PRESETS.items()}}


def _iter_result_blocks(payload):
    """Yield every dict that looks like an lm-eval result (has a 'results' key)."""
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], dict):
            yield payload
        for v in payload.values():
            if isinstance(v, dict):
                yield from _iter_result_blocks(v)


def collect_scores(paths: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """{task: {metric: value}} merged over files; LongBench 'score' is scaled to 0-100."""
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.json")))
        else:
            files.append(p)
    scores: Dict[str, Dict[str, float]] = {}
    for f in files:
        try:
            payload = json.loads(open(f, encoding="utf-8").read())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for block in _iter_result_blocks(payload):
            for task, res in block["results"].items():
                if not isinstance(res, dict):
                    continue
                entry = scores.setdefault(task, {})
                for k, v in res.items():
                    if isinstance(v, (int, float)) and not k.endswith("_stderr,none"):
                        metric = k.split(",")[0]
                        if metric.isdigit() and float(v) < 0:
                            continue  # RULER: -1 marks a length that was not evaluated
                        as_percent = (task.startswith("longbench_") and metric == "score") or metric.isdigit()
                        entry[metric] = float(v) * (100 if as_percent else 1)
    return scores


def summarize(scores: Dict[str, Dict[str, float]], tasks: List[str], metric: str = "score") -> dict:
    rows, missing = {}, []
    for t in tasks:
        key = t if t in scores else (f"longbench_{t}" if f"longbench_{t}" in scores else None)
        if key is None or metric not in scores[key]:
            missing.append(t)
        else:
            rows[t] = scores[key][metric]
    avg = sum(rows.values()) / len(rows) if rows else float("nan")
    return {"tasks": rows, "missing": missing, "average": avg, "metric": metric, "n": len(rows)}


def main():
    p = argparse.ArgumentParser(description="Summarize lm-eval result JSONs.")
    p.add_argument("paths", nargs="+", help="Result files or directories of *.json")
    p.add_argument("--preset", default="star-paper-7", help=f"Preset: {', '.join(PRESETS)}")
    p.add_argument("--tasks", default=None, help="Explicit comma list instead of --preset")
    p.add_argument("--metric", default=None, help=f"lm-eval metric; default 'score' (LongBench) or '{RULER_SEQLEN}' (ruler-*)")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    scores = collect_scores(args.paths)
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else PRESETS[args.preset]
    metric = args.metric or (str(RULER_SEQLEN) if args.preset.startswith("ruler-") and not args.tasks else "score")
    s = summarize(scores, tasks, metric)
    for t, v in s["tasks"].items():
        print(f"{t:22s} {v:10.4f}")
    if s["missing"]:
        print(f"missing: {', '.join(s['missing'])}")
    print(f"{'average':22s} {s['average']:10.4f}  ({s['n']} tasks)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"scores": scores, "summary": s}, f, indent=2)
    if s["missing"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
