# Baseline repositories

Upstream code for the external baselines this repository compares against, kept as full
git clones so they can be updated in place instead of being re-cloned into a
session scratchpad. Nothing in `lrkv/` or `evals/` imports from these clones:
the inference framework and the evaluation harness were extracted from
`STAR-KV/` into the repository's own packages (mapping and changes in
`docs/starkv_extraction.md`), and the tests under `tests/` import the clone
only to check bit-exactness against the upstream kernel and fusion code.

## Layout

| Path | Upstream | Pinned state (2026-09-01) |
|---|---|---|
| `STAR-KV/` | https://github.com/PriyanshBhatnagar/STAR-KV (ICML 2026 spotlight, arXiv 2606.08382) | `main` @ `f08a62d` (2026-09-01), 18 commits |
| `Palu/` | https://github.com/shadowpa0327/Palu | `main` @ `bb22666` (2025-02-19), 42 commits; submodule `3rdparty/fast-hadamard-transform` @ `d1a56ee` initialized; `3rdparty/lm-evaluation-harness` not initialized (ssh URL, and this repository uses its own pinned lm_eval 0.4.12) |

## Updating

```bash
git -C baselines/STAR-KV pull --ff-only
git -C baselines/Palu pull --ff-only
git -C baselines/Palu submodule update --init 3rdparty/fast-hadamard-transform
```

Record the new commit in the table above after pulling. Do not clone these
repositories into `/tmp` or a scratchpad again; read and diff against these
copies.

## Palu's LongBench scorer

`baselines/Palu/longbench_utils` (commit `bb22666`) is the THUDM LongBench
reference scorer the previous codebase vendored under `third_party/`.
This repository evaluates LongBench through lm-eval 0.4.12's `longbench_*`
tasks (the STAR-KV protocol, see `evals/harness.py`); the Palu scorer is not
imported anywhere here. If a native LongBench runner is added later, vendor
the scorer deliberately (hash-checked) rather than importing from the clone.

## STAR-KV: what the released code does and does not contain

Audited at `f08a62d`:

- Low-rank decomposition, soft-threshold training, PPL / zero-shot / LongBench /
  RULER evaluation, and the Triton latency kernels are complete.
- Quantization is partial. `bx_quant.py` (fused dequant + B@X for V) and
  `LlamaLoRaAttention_headwise_quant.py` (int8-upper / int4-lower split with
  per-token scales, `UPPER_RANKS_K=16`, `UPPER_RANKS_V=256`) both import
  `quant_utils` and `abx_rope_batched_quant`, which are not in the repository;
  the module docstring says they are not included in this release. There is no
  Hadamard transform, no 3-bit path, and nothing quantization-related in
  `model.py`, `eval.py`, `train.py`, or `latency.py`. The README TODO still
  lists wiring the quantized kernels into train/eval/latency. The `bx_quant.py`
  benchmark path calls `torch_bx`/`bx` with the wrong arity; only `--check`
  runs.
- Consequence: the paper's Table 4 quantizer (block-wise Hadamard, top 20% of
  latent channels in 4-bit, the rest in 3-bit, 3.2-bit average) has to be
  reimplemented from Section 4.4 / Eq. 12 if it is needed as a running
  baseline, in the same way the fake-quantized KV baselines were
  reimplemented in the previous codebase. Trained weights are not
  released either.

## STAR-KV: non-quantized Triton pipeline, completeness check (2026-09-02)

Re-checked against upstream on 2026-09-02: `origin/main` is still `f08a62d`
(no commits after 2026-09-01). Every local import of the non-quantized path
resolves (`train.py`, `eval.py`, `latency.py`, `model.py`,
`LlamaLoRaAttention_headwise.py`, `abx_rope_batched.py`, `soft_thres_layer.py`,
`check_compression.py`); only `bx_quant.py` and
`LlamaLoRaAttention_headwise_quant.py` import the missing `quant_utils` and
`abx_rope_batched_quant`. Verified by running in a transformers 5.9 environment:

- `abx_rope_batched.py --check` passes for GQA (32/8 heads) and MHA (32 heads).
- `eval.py --triton --ppl` on a synthetic fused checkpoint runs end to end.
- `model.generate(..., use_cache=True)` on the exported model calls the Triton
  kernel once per compressed layer per decode step and produces the same
  tokens as the `no_triton` path.
- `latency.py --mode e2e` and `--mode layerwise` run.
- `train.py` streams a local FineWeb-Edu parquet directory when passed as
  `--dataset <dir> --dataset-config default`; a 24-step smoke run on
  Llama-2-7B produced a fused checkpoint that `check_compression.py` and
  `eval.py --triton` accept.

Caveats found while doing this:

- `eval.py` loads with `use_cache=False` and then sets `model.config.use_cache =
  True`, but under transformers 5.x `generation_config.use_cache` stays False.
  lm_eval's `HFLM._model_generate` passes `use_cache=True` explicitly, so
  LongBench and RULER through `eval.py` do use the cache and the Triton decode;
  a plain `model.generate()` on the same object silently recomputes the whole
  sequence every step and never touches the kernel.
- `eval.py --ruler` builds a bare `TaskManager()`, so lm-eval's RULER tasks get no
  tokenizer metadata and fail to load; this repository passes the metadata
  (`evals/ruler.py`).
- `eval.py --longbench` hard-codes 11 tasks that do not include multifieldqa_en,
  so the paper's seven-task LongBench average cannot be produced from it unmodified.
- `train.py`, `eval.py` and `latency.py` overwrite `CUDA_VISIBLE_DEVICES` with
  `--cuda-devices` (default `0` or `0,1`).
- The `abx_rope_batched.py` benchmark path (no `--check`) builds a 3-D query
  tensor and trips the kernel's `a.dim() == 4` assertion; only `--check` runs.
- `check_compression.py` defaults to 8 KV heads and skip layers 0/1/2/31
  (Llama-3 style); pass `--num-kv-heads 32 --skip-layers 0 1 31` for Llama-2.
- `train.py` defaults to `--skip-layers 0 1 2 31` while `eval.py` and
  `latency.py` default to `0 1 31`; the loader infers skip layers from the
  checkpoint, so the mismatch is harmless at load time.
- `eval.py` loads the baseline model without `torch_dtype`, i.e. fp32 under
  transformers 4.x and the checkpoint dtype (bf16) under 5.x; measured on
  Llama-3.1 LongBench qasper/trec the two differ by < 0.6 points.
- No trained weights are released (README TODO still open) and there is no
  Table 5 (throughput) script.

## Palu: quantization in the paper and in the code

Paper (arXiv 2407.21118, Sec. 3.4 and 4.1): latents are quantized per token with
asymmetric integer quantization, Eq. 8 `clamp(round(X/s)+z, 0, 2^B-1)`, at 2/3/4
bits. SVD ordering puts outliers in the first latent channels of every head
group, so a Walsh-Hadamard matrix R is folded offline into both factors,
`W = AB = (AR)(R^T B)`, and quantization happens on the rotated latent at no
runtime cost. Table 1 (Llama-2-7B, WikiText-2, 4096 window): Palu-50% 3-bit
5.77 PPL at 90.63% compression, 2-bit 6.41 at 93.75%; baselines are Atom and
KVQuant (3-bit 5.35 at 81.25%, 2-bit 6.95 at 87.50%). Hadamard ablation on
Palu-30%: 3-bit 5.52 -> 5.33, 2-bit 9.48 -> 5.76.

Code at `bb22666`: `palu/model/modules/quant.py::quantize_tensor` is plain
round-to-nearest, asymmetric by default (`--lt_sym` switches to symmetric),
`--lt_group_size 0` means one (scale, zero) per token per head-group rank
slice, `--lt_clip_ratio` defaults to 1.0. `--lt_hadamard` calls
`HeadwiseLowRankModule.fused_hadamard_matrix`, which applies the QuIP#-style
block Hadamard (`hadamard_utils.apply_hadamard`, backed by
`fast_hadamard_transform`) to `VT` and `U` in place. Everything is fake
quantization in the PyTorch eval path; `kernel/abx_rope.py` is fp16 only and
the README TODO "update reconstruction kernel with quantization integrated" is
unchecked, so the fused dequantization kernel described in the paper's latency
section is not in the repository.

## Historical 36.62 reproduction (recipe archived)

The frozen recipe of a local STAR-KV run that reached a 36.62 seven-task LongBench
average on Llama-3.1-8B-Instruct (2026-07-17: training scripts, pinned upstream
sources, expected scores) was removed from this repository on 2026-09-07 and archived
outside it: its evaluation path is superseded by `evals.run` / `scripts/eval_sharded.sh`.
What still matters:

- The checkpoint (16.8 GB, legacy U/Sigma/V layout; `lrkv.checkpoint` converts it on
  load; 49.2% padded / 58.4% compact KV compression) scores, under the lm-eval 0.4.12
  protocol this repository uses: qasper 17.16, qmsum 21.94, triviaqa 83.67,
  multifieldqa_en 36.01, trec 61.0, multi_news 24.94, vcsum 11.58, average 36.62
  (paper STAR-KV 60%: 39.58).
- It was trained with pinned copies of `train.py`, `model.py` and `abx_rope_batched.py`
  that differ from upstream `f08a62d`; retraining must use those archived sources, not
  `STAR-KV/`.
