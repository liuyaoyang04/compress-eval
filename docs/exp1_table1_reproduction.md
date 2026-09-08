# 实验 1：评测代码与 STAR-KV Table 1 对齐

日期：2026-09-07。目的只有一个：确认本仓库的 PPL + zero-shot 评测代码与 STAR-KV 论文（`baselines/STAR-KV/STARKV.pdf`）Table 1 的口径一致。（脚本名后来统一为 `scripts/table1_ppl_zeroshot.sh`，汇总由 `python -m evals.compare` 完成；下文的旧脚本名指同一流程。）研究的模型是 Llama-2-7B 和 Llama-3.1-8B-Instruct（未压缩）。

## 固定下来的设置

| 项目 | 取值 | 代码位置 |
|---|---|---|
| 困惑度窗口 | 4096 token，非重叠 | `evals/ppl.py::PPL_SEQLEN`，`evals.run --ppl-seqlen` 默认值 |
| 困惑度数据 | WikiText-2 test（`\n\n` 拼接）；C4 固定分片 `en/c4-validation.00000-of-00008.json.gz` 前 1100 篇、最多 256×4096 token | `evals/ppl.py`，Hub 不可达时用 `LRKV_C4_FILE` 指向本地副本 |
| zero-shot 任务 | OBQA、PIQA、ARC-e、ARC-c、HellaSwag、WinoGrande | `evals/harness.py::ZERO_SHOT_TASKS` |
| zero-shot 指标 | OBQA、ARC-c、HellaSwag 取 `acc_norm`；PIQA、ARC-e、WinoGrande 取 `acc`；六项等权平均 | `evals/harness.py::ZERO_SHOT_METRIC`、`zero_shot_scores()`，结果写在 JSON 的 `zero_shot_scores` |
| lm-eval 调用 | lm_eval 0.4.12，`add_bos_token=False`，batch 32，无 chat template | `evals/harness.py::run_lm_eval` |
| 模型加载 | bf16，sdpa 注意力，transformers 5.9.0 | `lrkv/model.py::load_base_model` |

这些是从下面的对照实验里反推出来的，论文正文都没有写明；`tests/test_evals.py` 把窗口长度和指标口径钉住了。

## 为什么是 4096 和混合口径

论文 5.1 节说困惑度按上游 `eval.py` 的协议算，但 `eval.py` 默认窗口是 1024 和 2048，只有 13B 的注释提到"因显存用 1024"。用三个窗口各跑一遍，只有 4096 与 Table 1 完全吻合：

| 模型 | Wiki2@1024 | Wiki2@2048 | Wiki2@4096 | C4@1024 | C4@2048 | C4@4096 | Table 1 |
|---|---|---|---|---|---|---|---|
| Llama-2-7B | 6.10 | 5.47 | **5.12** | 7.47 | 7.27 | **7.04** | 5.12 / 7.04 |
| Llama-3-8B-Instruct（仅用于定协议） | 9.27 | 8.29 | **7.74** | 13.36 | 13.01 | **12.61** | 7.74 / 12.61 |

zero-shot 方面，Table 1 只写"accuracy (%)"。Llama-2-7B 基线在纯 `acc` 下均值 59.09，与论文 65.05 差 6 个点；OBQA、ARC-c、HellaSwag 换成 `acc_norm` 后六项全部落在论文 ±0.5 以内（这也是 Palu 表格的口径）。

注意 Table 1 第三块的 "LLaMA-3-8B-Inst" 是 Llama-3，不是 3.1；Llama-3.1-8B-Instruct 的 zero-shot 基线在附录 Table 8（均值 68.84），论文没有它的 PPL。Llama-3-8B-Instruct 只在定协议时跑过一次（原始输出已清理，数字如上表），不属于本课题的研究模型。

## 固定设置下的结果

`GPUS=2,3 bash scripts/table1_ppl_zeroshot.sh`，单模型约 4 分钟（一张 H800）。

| 模型 | Wiki2 | C4 | OBQA | PIQA | ARC-e | ARC-c | Hella | Wino | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Llama-2-7B（本仓库） | 5.12 | 7.04 | 44.00 | 77.64 | 76.26 | 46.16 | 75.98 | 69.22 | 64.88 |
| Llama-2-7B（Table 1） | 5.12 | 7.04 | 44.20 | 78.07 | 76.30 | 46.42 | 76.00 | 69.30 | 65.05 |
| 差 | 0.00 | 0.00 | −0.20 | −0.43 | −0.04 | −0.26 | −0.02 | −0.08 | −0.17 |
| Llama-3.1-8B-Instruct（本仓库） | 6.75 | 11.10 | 43.00 | 79.92 | 81.82 | 55.12 | 79.20 | 73.56 | 68.77 |
| Llama-3.1-8B-Instruct（Table 8） | – | – | 42.60 | 80.96 | 81.73 | 54.86 | 79.17 | 73.72 | 68.84 |
| 差 | – | – | +0.40 | −1.04 | +0.09 | +0.26 | +0.03 | −0.16 | −0.07 |

lm-eval 的标准误：OBQA 2.2、ARC-c 1.5、Wino 1.3、PIQA 1.0、ARC-e 0.9、Hella 0.4 个百分点。除 Llama-3.1 的 PIQA 外，所有单项偏差都在一个标准误内；均值偏差 0.2 以内。结论：**评测代码与论文对齐。**

`acc` / `acc_norm` 两种口径的完整数字在结果 JSON 的 `lmeval_scores` 里；与论文的逐项对照用 `python -m evals.compare <json> --paper llama2_7b`。

## 附：本地 STAR-KV checkpoint

驱动脚本加 `CHECKPOINT=<ckpt> GPUS=2,3,4` 会在 Llama-3.1 上多跑一个压缩模型。用本地那个 36.62 的 STAR-KV checkpoint（见 `baselines/README.md`）测过一次：Wiki2 10.33、C4 16.19、zero-shot 均值 62.04，比 Table 8 的 STAR-KV 60%（66.43）低 4.4 个点。论文没有公开权重，这只说明本地训出来的 checkpoint 没达到论文水平，与评测代码无关。
