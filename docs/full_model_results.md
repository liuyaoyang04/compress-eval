# 未压缩模型（full model）最终结果

用本仓库 commit `5beea44954b9` 的脚本一次性跑出（2026-09-07），作为后续所有压缩方法对比的参照基线。环境：Python 3.10.13、torch 2.5.1+cu121、transformers 5.9.0、lm-eval 0.4.12、datasets 2.19.2，NVIDIA NVIDIA H800，bf16。数据为 `python -m evals.sources` 固定的版本。原始 JSON 与日志在 `results/table1/`、`results/table2/llama31_8b_instruct_baseline/`、`results/table3/llama31_8b_instruct_baseline/`（不进版本库）。

命令：

```bash
GPUS=2,3           bash scripts/table1_ppl_zeroshot.sh   # 4.7 分钟
GPUS=4,5,6,7       bash scripts/table2_longbench.sh      # 38 分钟（最长分片 qmsum）
GPUS=2,3,4,5,6,7   bash scripts/table3_ruler.sh          # 15 分钟（最长分片）
```

## Table 1：Llama-2-7B

**Perplexity @ 4096**

| | Wiki2 | C4 |
|---|---|---|
| ours | 5.12 | 7.04 |
| paper | 5.12 | 7.04 |
| delta | -0.00 | -0.00 |

**Zero-shot (Table 1 convention)**

| | OBQA | PIQA | ARC-e | ARC-c | Hella | Wino | Avg |
|---|---|---|---|---|---|---|---|
| ours | 44.00 | 77.64 | 76.26 | 46.16 | 75.98 | 69.22 | 64.88 |
| paper | 44.20 | 78.07 | 76.30 | 46.42 | 76.00 | 69.30 | 65.05 |
| delta | -0.20 | -0.43 | -0.04 | -0.26 | -0.02 | -0.08 | -0.17 |

论文参考行：Table 1, LLaMA-2-7B (0%)

## Table 1 / 8：Llama-3.1-8B-Instruct

**Perplexity @ 4096**

| | Wiki2 | C4 |
|---|---|---|
| ours | 6.75 | 11.10 |

**Zero-shot (Table 1 convention)**

| | OBQA | PIQA | ARC-e | ARC-c | Hella | Wino | Avg |
|---|---|---|---|---|---|---|---|
| ours | 43.00 | 79.92 | 81.82 | 55.12 | 79.20 | 73.56 | 68.77 |
| paper | 42.60 | 80.96 | 81.73 | 54.86 | 79.17 | 73.72 | 68.84 |
| delta | +0.40 | -1.04 | +0.09 | +0.26 | +0.03 | -0.16 | -0.07 |

论文参考行：Table 8 zero-shot (no PPL in the paper); Table 2 / 14 LongBench; Table 3 RULER

论文没有 Llama-3.1 的困惑度；zero-shot 对照的是附录 Table 8 的基线行。

## Table 2：LongBench，Llama-3.1-8B-Instruct

**LongBench (Table 2 seven tasks)**

| | Qasper | QMSum | TriviaQA | MultiQA | TREC | MultiNews | VCSum | Avg |
|---|---|---|---|---|---|---|---|---|
| ours | 33.89 | 23.99 | 93.08 | 39.29 | 74.00 | 27.08 | 16.10 | 43.92 |
| paper | 25.19 | 23.20 | 92.00 | 39.90 | 72.50 | 26.90 | 15.91 | 42.23 |
| delta | +8.70 | +0.79 | +1.08 | -0.61 | +1.50 | +0.18 | +0.19 | +1.69 |

论文参考行：Table 8 zero-shot (no PPL in the paper); Table 2 / 14 LongBench; Table 3 RULER

Qasper 一项论文值无法复现（排查见 `longbench_table2.md`），其余六项差在 1.5 以内。

## Table 3：RULER @4096，Llama-3.1-8B-Instruct

**RULER @ 4096 (Table 3 nine tasks)**

| | MK1 | MK2 | MQ | MV | S1 | S2 | S3 | FWE | SQ | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | 100.00 | 100.00 | 100.00 | 98.65 | 100.00 | 100.00 | 99.80 | 96.40 | 77.23 | 96.90 |
| paper | 100.00 | 99.80 | 99.90 | 98.95 | 100.00 | 100.00 | 99.60 | 96.07 | 78.12 | 96.94 |
| delta | +0.00 | +0.20 | +0.10 | -0.30 | +0.00 | +0.00 | +0.20 | +0.33 | -0.89 | -0.04 |

论文参考行：Table 8 zero-shot (no PPL in the paper); Table 2 / 14 LongBench; Table 3 RULER

## 一行汇总

| 模型 | Wiki2 | C4 | zero-shot 均值 | LongBench 均值 | RULER 均值 |
|---|---|---|---|---|---|
| Llama-2-7B | 5.12 | 7.04 | 64.88 | – | – |
| Llama-3.1-8B-Instruct | 6.75 | 11.10 | 68.77 | 43.92 | 96.90 |
