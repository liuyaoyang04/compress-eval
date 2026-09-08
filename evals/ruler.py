"""RULER as reported in STAR-KV Table 3.

Nine of the thirteen ``ruler`` tasks in lm-eval 0.4.12 (the paper omits
niah_multikey_3, ruler_vt, ruler_cwe and ruler_qa_hotpot), 500 synthetic
samples per task at one sequence length (4096 in the paper), scored with
lm-eval's own string-match metrics; the average is the equal-weight mean of
the nine. Prompts, needle generation, ``max_gen_toks`` and the metrics are
untouched lm-eval code.

lm-eval synthesizes the samples at task-load time and needs (a) the tokenizer
and target lengths through task metadata, (b) the Paul Graham essays for the
essay haystacks (HF dataset ``baber/paul_graham_essays``), (c) NLTK
``punkt_tab`` to split those essays into sentences and (d) SQuAD v2 dev for
``ruler_qa_squad``. ``install_sources`` points all three at the pinned local
copies from ``evals.sources`` (fetched once, hash-checked).

The samples are drawn from Python's global RNG while the tasks are being
instantiated, so a task's sample set depends on the tasks generated before it
in the same process. STAR-KV's ``eval.py --ruler`` loads the whole ``ruler``
group, and that is the only generation protocol implemented here:
``build_tasks`` seeds exactly as ``simple_evaluate`` does and generates the
group prefix (``ruler.yaml`` order) up to the last requested task, then only
the requested tasks are evaluated. Any subset or sharding therefore scores the
same samples as a full-group run (``ruler_qa_hotpot`` is last in the group and
is never generated).

    python -m evals.ruler --list-tasks paper-9
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import re
import zipfile
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import sources

RULER_SEQLEN = 4096

# STAR-KV Table 3 column order.
RULER_PAPER_TASKS = ["niah_multikey_1", "niah_multikey_2", "niah_multiquery", "niah_multivalue",
                     "niah_single_1", "niah_single_2", "niah_single_3", "ruler_fwe", "ruler_qa_squad"]
RULER_LABELS = {"niah_multikey_1": "MK1", "niah_multikey_2": "MK2", "niah_multiquery": "MQ",
                "niah_multivalue": "MV", "niah_single_1": "S1", "niah_single_2": "S2", "niah_single_3": "S3",
                "ruler_fwe": "FWE", "ruler_qa_squad": "SQ"}
RULER_PRESETS: Dict[str, List[str]] = {"paper-9": RULER_PAPER_TASKS}
# Order in which lm-eval's ruler.yaml group instantiates (and therefore synthesizes) the tasks.
RULER_GROUP_ORDER = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
                     "niah_multikey_3", "niah_multiquery", "niah_multivalue", "ruler_vt", "ruler_cwe", "ruler_fwe",
                     "ruler_qa_squad", "ruler_qa_hotpot"]
# simple_evaluate's defaults (DEFAULT_RANDOM_SEED, DEFAULT_OTHER_SEED), applied before task loading.
LM_EVAL_SEEDS = (0, 1234)


def resolve_ruler_tasks(spec: Optional[str]) -> List[str]:
    spec = (spec or "paper-9").strip()
    if spec in RULER_PRESETS:
        return list(RULER_PRESETS[spec])
    return [t.strip() for t in spec.split(",") if t.strip()]


def task_metadata(tokenizer_path: str, seqlens: Iterable[int] = (RULER_SEQLEN,)) -> dict:
    """The ``TaskManager(metadata=...)`` dict lm-eval's ruler tasks read their tokenizer and lengths from."""
    return {"tokenizer": tokenizer_path, "max_seq_lengths": [int(s) for s in seqlens]}


_installed = False


def install_sources(download: bool = True) -> Dict[str, str]:
    """Route lm-eval's ruler data loading to the pinned local files. Idempotent; returns the paths."""
    global _installed
    paths = {n: sources.ensure(n, download=download) for n in ("paul_graham_essays", "squad", "punkt_tab")}

    nltk_dir = os.path.join(sources.data_dir(), "nltk_data")
    if not os.path.isdir(os.path.join(nltk_dir, "tokenizers", "punkt_tab", "english")):
        with zipfile.ZipFile(paths["punkt_tab"]) as z:
            z.extractall(os.path.join(nltk_dir, "tokenizers"))
    import nltk
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)
    if _installed:
        return paths
    # prepare_niah calls nltk.download("punkt_tab") on import; the data is already in place.
    nltk.download = lambda *a, **k: True  # type: ignore[assignment]

    from lm_eval.tasks.ruler import niah_utils, prepare_niah, qa_utils

    essays_path = paths["paul_graham_essays"]
    upstream_get_haystack = prepare_niah.get_haystack

    @functools.cache
    def get_haystack(type_haystack):
        if type_haystack != "essay":
            return upstream_get_haystack(type_haystack)
        import pyarrow.parquet as pq
        # Same join and whitespace normalisation as upstream, which reads the same parquet from the Hub.
        essay = " ".join(pq.read_table(essays_path).column("text").to_pylist())
        return re.sub(r"\s+", " ", essay).split(" ")

    prepare_niah.get_haystack = get_haystack
    niah_utils.get_haystack = get_haystack

    squad_path = paths["squad"]
    upstream_download_json = qa_utils.download_json

    @functools.cache
    def download_json(url):
        if url == sources.SQUAD_URL:
            with open(squad_path, encoding="utf-8") as f:
                return json.load(f)
        return upstream_download_json(url)

    qa_utils.download_json = download_json
    _installed = True
    return paths


def group_prefix(tasks: Iterable[str]) -> List[str]:
    """The ``ruler`` group prefix that must be generated so ``tasks`` get their group-order samples."""
    tasks = list(tasks)
    unknown = [t for t in tasks if t not in RULER_GROUP_ORDER]
    if unknown:
        raise ValueError(f"not RULER tasks: {unknown}")
    last = max(RULER_GROUP_ORDER.index(t) for t in tasks)
    return RULER_GROUP_ORDER[: last + 1]


def build_tasks(task_manager, tasks: Iterable[str], verbose: bool = True) -> list:
    """Instantiate ``tasks`` (lm-eval Task objects) with the sample sets a full-group run gives them.

    ``task_manager`` must carry the metadata from ``task_metadata``. Pass the returned objects to
    ``run_lm_eval`` together with the same manager; they are evaluated as-is, without regeneration.
    This is the only way RULER tasks are built in this repository.
    """
    tasks = list(tasks)
    prefix = group_prefix(tasks)
    if verbose:
        skipped = [t for t in prefix if t not in tasks]
        print(f"RULER: generating {len(prefix)} tasks in group order ({len(skipped)} only to advance the RNG)")
    random.seed(LM_EVAL_SEEDS[0])
    np.random.seed(LM_EVAL_SEEDS[1])
    loaded = task_manager.load(prefix)["tasks"]
    return [loaded[t] for t in tasks]


def ruler_scores(results: dict, seqlen: int = RULER_SEQLEN, tasks: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """{task: accuracy * 100} at ``seqlen`` for the tasks present, plus ``"average"`` over them."""
    key = f"{seqlen},none"
    out = {}
    for task in (list(tasks) if tasks else RULER_PAPER_TASKS):
        res = results.get("results", {}).get(task)
        if res is not None and key in res and float(res[key]) >= 0:
            out[task] = float(res[key]) * 100
    if out:
        out["average"] = sum(out.values()) / len(out)
    return out


def format_scores(scores: Dict[str, float]) -> str:
    cols = [t for t in scores if t != "average"]
    body = "  ".join(f"{RULER_LABELS.get(t, t)} {scores[t]:.2f}" for t in cols)
    return f"{body}  | avg {scores.get('average', float('nan')):.2f} over {len(cols)} tasks"


def main():
    p = argparse.ArgumentParser(description="RULER task presets and source files.")
    p.add_argument("--list-tasks", default=None, help="Preset name or comma list; prints lm-eval task names")
    p.add_argument("--presets", action="store_true")
    p.add_argument("--install", action="store_true", help="Fetch the source files and print their paths")
    args = p.parse_args()
    if args.presets:
        for k, v in RULER_PRESETS.items():
            print(f"{k:10s} {','.join(v)}")
    if args.list_tasks:
        print(",".join(resolve_ruler_tasks(args.list_tasks)))
    if args.install:
        for k, v in install_sources().items():
            print(f"{k:20s} {v}")


if __name__ == "__main__":
    main()
