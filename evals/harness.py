"""lm-evaluation-harness adapter: zero-shot, LongBench, RULER.

Same invocation as STAR-KV ``eval.py`` / the 36.62 reproduction evaluator:
``HFLM(pretrained=model, add_bos_token=False, batch_size, max_length)``,
greedy generation, no chat template. LongBench is the paper's seven tasks
through lm-eval's ``longbench_*`` (dataset ``Xnhyacinth/LongBench``, metric
``score``); the only preset is ``star-paper-7``, comma lists exist for sharding.

The six zero-shot tasks are reported the way STAR-KV Table 1 (and Palu)
report them, ``ZERO_SHOT_METRIC``: ``acc_norm`` for OpenBookQA, ARC-c and
HellaSwag, ``acc`` for PIQA, ARC-e and WinoGrande. The paper does not say so,
but its uncompressed rows match lm-eval only under this choice (Llama-2-7B
average 65.05 vs 64.88 measured; plain ``acc`` averages are 5 points lower).

    python -m evals.harness --list-tasks star-paper-7

RULER (task set, data files, scoring) lives in ``evals.ruler``.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import torch

ZERO_SHOT_TASKS = ["piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "hellaswag"]
# lm-eval metric behind each Table 1 column, in the paper's column order.
ZERO_SHOT_METRIC = {"openbookqa": "acc_norm", "piqa": "acc", "arc_easy": "acc", "arc_challenge": "acc_norm",
                    "hellaswag": "acc_norm", "winogrande": "acc"}
ZERO_SHOT_LABELS = {"openbookqa": "OBQA", "piqa": "PIQA", "arc_easy": "ARC-e", "arc_challenge": "ARC-c",
                    "hellaswag": "Hella", "winogrande": "Wino"}

# STAR-KV Table 2 (paper column order): Qasper, QMSum, TriviaQA, MultiQA (= multifieldqa_en), TREC, MultiNews, VCSum.
# lm-eval 0.4.12's ``longbench_*`` tasks (dataset Xnhyacinth/LongBench, official LongBench scorers and generation
# caps), batch 4, max_length 31500, add_bos_token=False, no chat template, bf16 greedy: the protocol that reproduces
# the paper's Llama-3.1-8B-Instruct row (docs/longbench_table2.md); tests/test_evals.py pins the lm-eval side.
LONGBENCH_PAPER_TASKS = ["qasper", "qmsum", "triviaqa", "multifieldqa_en", "trec", "multi_news", "vcsum"]
LONGBENCH_LABELS = {"qasper": "Qasper", "qmsum": "QMSum", "triviaqa": "TriviaQA", "multifieldqa_en": "MultiQA",
                    "trec": "TREC", "multi_news": "MultiNews", "vcsum": "VCSum"}
LONGBENCH_PRESETS: Dict[str, List[str]] = {"star-paper-7": LONGBENCH_PAPER_TASKS}
from .ruler import RULER_PAPER_TASKS as RULER_TASKS  # noqa: E402  (STAR-KV Table 3 nine tasks)


def resolve_longbench_tasks(spec: Optional[str]) -> List[str]:
    """Preset name or comma list (with or without the ``longbench_`` prefix) -> lm-eval task names."""
    spec = (spec or "star-paper-7").strip()
    names = LONGBENCH_PRESETS[spec] if spec in LONGBENCH_PRESETS else [t.strip() for t in spec.split(",") if t.strip()]
    return [t if t.startswith("longbench_") else f"longbench_{t}" for t in names]


def run_lm_eval(model, tokenizer, tasks, batch_size, max_length: Optional[int] = None,
                limit: Optional[int] = None, add_bos_token: bool = False, log_samples: bool = False,
                set_max_position_embeddings: bool = True, verbose: bool = True, task_manager=None) -> dict:
    """Run lm-eval on an already loaded model.

    ``tasks``: comma string, list of task names, or pre-built lm-eval Task objects (RULER: always
    ``evals.ruler.build_tasks``), which are evaluated as they are. ``task_manager`` reuses an
    existing manager (RULER tasks need the one that built them; each ``TaskManager()`` scans every
    task YAML).
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table

    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    names = [t if isinstance(t, str) else t.config.task for t in tasks]
    kwargs = {"pretrained": model, "tokenizer": tokenizer, "add_bos_token": add_bos_token, "batch_size": batch_size}
    if max_length is not None:
        kwargs["max_length"] = max_length
        if set_max_position_embeddings:
            # Kept from the historical protocol; harmless for default and llama3 RoPE.
            model.config.max_position_embeddings = max_length
    lm = HFLM(**kwargs)
    if verbose:
        print(f"lm-eval tasks: {names}  batch={batch_size} max_length={max_length} limit={limit}")
    if task_manager is None:
        task_manager = TaskManager()
    with torch.no_grad():
        results = lm_eval.simple_evaluate(model=lm, tasks=tasks, task_manager=task_manager,
                                          log_samples=log_samples, limit=limit)
    if verbose:
        try:
            print(make_table(results))
        except Exception as e:  # noqa: BLE001 - a pretty-print must never cost the results
            print(f"(make_table failed: {type(e).__name__}: {e})")
    return results


def task_scores(results: dict) -> Dict[str, Dict[str, float]]:
    """{task: {metric: value}} with the ',none' filter suffix stripped and non-numeric entries dropped."""
    out = {}
    for task, res in results.get("results", {}).items():
        out[task] = {k.split(",")[0]: float(v) for k, v in res.items()
                     if isinstance(v, (int, float)) and not k.endswith("_stderr,none") and k != "alias"}
    return out


def zero_shot_scores(results: dict) -> Dict[str, float]:
    """{task: accuracy * 100} under ``ZERO_SHOT_METRIC`` for the zero-shot tasks present, plus
    ``"average"`` over them (STAR-KV Table 1 columns; the average covers only the tasks found)."""
    out = {}
    for task, metric in ZERO_SHOT_METRIC.items():
        res = results.get("results", {}).get(task)
        if res is not None and f"{metric},none" in res:
            out[task] = float(res[f"{metric},none"]) * 100
    if out:
        out["average"] = sum(out.values()) / len(out)
    return out


def longbench_scores(results: dict) -> Dict[str, float]:
    """{task without prefix: score * 100} for the LongBench tasks in an lm-eval result dict."""
    out = {}
    for task, res in results.get("results", {}).items():
        if task.startswith("longbench_") and "score,none" in res:
            out[task[len("longbench_"):]] = float(res["score,none"]) * 100
    return out


def main():
    p = argparse.ArgumentParser(description="LongBench task presets.")
    p.add_argument("--list-tasks", default=None, help="Preset name or comma list; prints the lm-eval task names")
    p.add_argument("--presets", action="store_true", help="List preset names")
    args = p.parse_args()
    if args.presets or args.list_tasks is None:
        for k, v in LONGBENCH_PRESETS.items():
            print(f"{k:22s} {','.join(v)}")
    if args.list_tasks:
        print(",".join(resolve_longbench_tasks(args.list_tasks)))


if __name__ == "__main__":
    main()
