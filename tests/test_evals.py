"""Evaluation protocol pins: PPL window and the Table 1 zero-shot metric convention (CPU only)."""

from evals.harness import ZERO_SHOT_METRIC, ZERO_SHOT_TASKS, task_scores, zero_shot_scores
from evals.ppl import PPL_SEQLEN
from evals.run import parse_args


def _fake_lmeval(acc, acc_norm):
    res = {}
    for t in ZERO_SHOT_TASKS:
        res[t] = {"alias": t, "acc,none": acc[t], "acc_stderr,none": 0.01}
        if t != "winogrande":  # winogrande has no acc_norm in lm-eval
            res[t].update({"acc_norm,none": acc_norm[t], "acc_norm_stderr,none": 0.01})
    return {"results": res}


def test_ppl_window_is_table1_protocol():
    assert PPL_SEQLEN == 4096
    assert parse_args(["--model", "m", "--baseline"]).ppl_seqlen == [4096]


def test_zero_shot_metric_convention():
    assert set(ZERO_SHOT_METRIC) == set(ZERO_SHOT_TASKS)
    assert {t for t, m in ZERO_SHOT_METRIC.items() if m == "acc_norm"} == {"openbookqa", "arc_challenge", "hellaswag"}


def test_zero_shot_scores_picks_paper_metric_and_averages():
    acc = {t: 0.10 * (i + 1) for i, t in enumerate(ZERO_SHOT_TASKS)}
    acc_norm = {t: acc[t] + 0.05 for t in ZERO_SHOT_TASKS}
    zs = zero_shot_scores(_fake_lmeval(acc, acc_norm))
    for t, m in ZERO_SHOT_METRIC.items():
        expected = (acc_norm if m == "acc_norm" else acc)[t] * 100
        assert abs(zs[t] - expected) < 1e-9
    assert abs(zs["average"] - sum(zs[t] for t in ZERO_SHOT_TASKS) / 6) < 1e-9
    assert set(task_scores(_fake_lmeval(acc, acc_norm))["piqa"]) == {"acc", "acc_norm"}


def test_zero_shot_scores_partial_run():
    partial = {"results": {"piqa": {"acc,none": 0.5, "acc_norm,none": 0.6}}}
    assert zero_shot_scores(partial) == {"piqa": 50.0, "average": 50.0}
    assert zero_shot_scores({"results": {"longbench_trec": {"score,none": 0.7}}}) == {}


# ---- RULER (STAR-KV Table 3) and pinned source files ----

import hashlib
import json
import os

import pytest

from evals import sources
from evals.ruler import RULER_LABELS, RULER_PAPER_TASKS, resolve_ruler_tasks, ruler_scores, task_metadata


def test_ruler_task_set_is_table3():
    assert RULER_PAPER_TASKS == ["niah_multikey_1", "niah_multikey_2", "niah_multiquery", "niah_multivalue",
                                 "niah_single_1", "niah_single_2", "niah_single_3", "ruler_fwe", "ruler_qa_squad"]
    assert [RULER_LABELS[t] for t in RULER_PAPER_TASKS] == ["MK1", "MK2", "MQ", "MV", "S1", "S2", "S3", "FWE", "SQ"]
    assert resolve_ruler_tasks("paper-9") == RULER_PAPER_TASKS
    assert resolve_ruler_tasks("niah_single_1, ruler_fwe") == ["niah_single_1", "ruler_fwe"]
    assert task_metadata("/m", [4096]) == {"tokenizer": "/m", "max_seq_lengths": [4096]}


def test_ruler_scores_reads_one_length_and_averages():
    fake = {"results": {t: {"alias": t, "4096,none": 0.5, "8192,none": -1.0} for t in RULER_PAPER_TASKS}}
    fake["results"]["niah_single_1"]["4096,none"] = 1.0
    s = ruler_scores(fake, 4096)
    assert s["niah_single_1"] == 100.0 and s["ruler_fwe"] == 50.0
    assert abs(s["average"] - (100 + 50 * 8) / 9) < 1e-9
    assert ruler_scores(fake, 8192) == {}  # -1 marks a length that was not evaluated
    assert ruler_scores({"results": {"niah_single_1": {"4096,none": 0.2}}}, 4096) == {"niah_single_1": 20.0, "average": 20.0}


def test_sources_hash_check_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv(sources.DATA_DIR_ENV, str(tmp_path))
    data = b"hello"
    monkeypatch.setitem(sources.SOURCES, "fake", sources.Source("fake", "fake.bin", (), hashlib.sha256(data).hexdigest()))
    with pytest.raises(FileNotFoundError):
        sources.ensure("fake", download=False)
    with pytest.raises(RuntimeError):  # missing and no URL works -> error, nothing fetched
        sources.ensure("fake", verbose=False)
    dst = sources.path_of("fake")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(data)
    assert sources.ensure("fake") == dst and sources.is_present("fake")
    open(dst, "wb").write(b"corrupt")
    assert not sources.is_present("fake")
    with pytest.raises(RuntimeError):
        sources.ensure("fake", verbose=False)


def test_ruler_group_prefix_follows_group_order():
    from evals.ruler import RULER_GROUP_ORDER, group_prefix
    assert group_prefix(["niah_single_1"]) == ["niah_single_1"]
    assert group_prefix(["ruler_fwe", "niah_multikey_2"]) == RULER_GROUP_ORDER[:RULER_GROUP_ORDER.index("ruler_fwe") + 1]
    assert group_prefix(RULER_PAPER_TASKS) == RULER_GROUP_ORDER[:-1]  # everything but ruler_qa_hotpot
    with pytest.raises(ValueError):
        group_prefix(["piqa"])



# ---- LongBench (STAR-KV Table 2): the lm-eval 0.4.12 side that produced the reproduced baseline ----

LONGBENCH_LM_EVAL_PIN = "119361f03a4df77621fd4f50360d109282ae5f9e7a6dbae886f51c2123fcc06f"  # sha256 over the 7 doc_to_text templates, generation kwargs, scorer names, metrics.py


def _load_task_yaml(path):
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("!", lambda loader, suffix, node: loader.construct_scalar(node))
    with open(path) as f:
        return yaml.load(f, Loader=Loader)


def test_longbench_task_set_is_table2():
    from evals.harness import LONGBENCH_LABELS, LONGBENCH_PAPER_TASKS, LONGBENCH_PRESETS, resolve_longbench_tasks
    assert LONGBENCH_PAPER_TASKS == ["qasper", "qmsum", "triviaqa", "multifieldqa_en", "trec", "multi_news", "vcsum"]
    assert [LONGBENCH_LABELS[t] for t in LONGBENCH_PAPER_TASKS] == ["Qasper", "QMSum", "TriviaQA", "MultiQA", "TREC", "MultiNews", "VCSum"]
    assert list(LONGBENCH_PRESETS) == ["star-paper-7"]
    assert resolve_longbench_tasks(None) == [f"longbench_{t}" for t in LONGBENCH_PAPER_TASKS]
    assert resolve_longbench_tasks("trec, longbench_vcsum") == ["longbench_trec", "longbench_vcsum"]
    args = parse_args(["--model", "m", "--baseline"])
    assert (args.longbench_tasks, args.long_batch_size, args.max_length, args.dtype) == ("star-paper-7", "4", 31500, "bf16")


def test_longbench_lm_eval_side_is_pinned():
    import os
    import lm_eval
    from evals.harness import LONGBENCH_PAPER_TASKS
    assert lm_eval.__version__ == "0.4.12"
    root = os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "longbench")
    gen = {"qasper": 128, "qmsum": 512, "triviaqa": 32, "multifieldqa_en": 64, "trec": 64, "multi_news": 512, "vcsum": 512}
    h = hashlib.sha256()
    for t in LONGBENCH_PAPER_TASKS:
        c = _load_task_yaml(os.path.join(root, f"{t}.yaml"))
        assert c["task"] == f"longbench_{t}" and c["dataset_path"] == "Xnhyacinth/LongBench" and c["dataset_name"] == t
        assert c["generation_kwargs"]["max_gen_toks"] == gen[t] and c["generation_kwargs"]["do_sample"] is False
        assert c["generation_kwargs"]["until"] == (["\n"] if t in ("trec", "triviaqa") else [])
        h.update(c["doc_to_text"].encode()); h.update(str(c["generation_kwargs"]).encode()); h.update(c["process_results"].encode())
    with open(os.path.join(root, "metrics.py"), "rb") as f:
        h.update(f.read())
    assert h.hexdigest() == LONGBENCH_LM_EVAL_PIN, "lm-eval's longbench templates/scorers changed; Table 2 numbers are no longer comparable"


# ---- pinned data store and paper comparison ----

def test_sources_tables_cover_every_suite():
    from evals import sources
    assert set(sources.REQUIRED) == {"ppl", "zero-shot", "longbench", "ruler"}
    for suite, names in sources.REQUIRED.items():
        for n in names:
            assert n in sources.FILES or n in sources.DATASETS, (suite, n)
    for d in sources.DATASETS.values():
        assert len(d.revision) == 40 and all(c in "0123456789abcdef" for c in d.revision), d.name
    for f in sources.FILES.values():
        assert len(f.sha256) == 64 and f.urls, f.name
    assert sources.dataset_cache_path("longbench_qasper").endswith(
        "Xnhyacinth___long_bench/qasper/0.0.0/2e9ade51ebf45d98942056c0716234f9d5d257d5")
    assert sources.dataset_cache_path("piqa").endswith("baber___piqa/default/0.0.0/142f6d7367fd9877f0fb3b5734ea6a545f54cdd1")


def test_compare_reads_run_json_and_matches_paper_columns(tmp_path):
    from evals.compare import PAPER, compare, load_results
    from evals.harness import LONGBENCH_PAPER_TASKS
    run = {"ppl": {"seqlen4096": {"wikitext2": {"ppl": 5.12}, "c4": {"ppl": 7.04}}},
           "zero_shot_scores": {t: 50.0 for t in ZERO_SHOT_TASKS} | {"average": 50.0},
           "longbench_scores": {t: 10.0 for t in LONGBENCH_PAPER_TASKS},
           "ruler_scores": {"4096": {t: 90.0 for t in RULER_PAPER_TASKS} | {"average": 90.0}}}
    f = tmp_path / "run.json"; f.write_text(json.dumps(run))
    r = load_results(str(f))
    assert r["ppl"] == {"wikitext2": 5.12, "c4": 7.04} and "average" not in r["zero_shot"] and len(r["ruler"]) == 9
    text = compare(r, "llama2_7b")
    assert "| delta | +0.00 | +0.00 |" in text and "Zero-shot" in text and "RULER" in text
    for key, ref in PAPER.items():
        for suite, cols in (("zero_shot", ZERO_SHOT_TASKS), ("longbench", LONGBENCH_PAPER_TASKS), ("ruler", RULER_PAPER_TASKS)):
            if suite in ref:
                assert set(ref[suite]) == set(cols), (key, suite)
