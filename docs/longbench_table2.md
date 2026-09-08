# LongBench：对照 STAR-KV Table 2

日期：2026-09-07。目标：用本仓库的评测代码复现论文 Table 2 / Table 14 中 Llama-3.1-8B-Instruct 的行（七任务均值 42.23）。LongChat-7B-v1.5-32K 不在本课题范围内。

## 论文协议

| 项目 | 取值 | 代码位置 |
|---|---|---|
| 任务 | Qasper、QMSum、TriviaQA、MultiQA（= MultiFieldQA-en）、TREC、MultiNews、VCSum，七任务等权平均 | `evals/harness.py::LONGBENCH_PAPER_TASKS` |
| 框架 | STAR-KV `eval.py --longbench` 调 lm-eval 的 `longbench_*` 任务：数据集 `Xnhyacinth/LongBench`，官方判分函数（F1 / ROUGE-L / ROUGE-L-zh / 分类），官方生成上限（128 / 512 / 32 / 64 / 64 / 512 / 512） | lm-eval 0.4.12 `tasks/longbench/` |
| 模型调用 | `add_bos_token=False`、无 chat template、batch 4、`max_length` 31500、贪心 | `evals/run.py --longbench` |
| 截断 | lm-eval 只保留末尾 `31500 − 生成上限` 个 token；官方 LongBench 是掐中间。七任务 1350 条里只有 1 条 vcsum 超过 31500，所以两者在这个长度下没有区别 | |

论文 Table 2 与附录 Table 14 的 Llama-3.1 基线行一致（25.19 / 23.20 / 92.00 / 39.90 / 72.50 / 26.90 / 15.91，均值 42.23）。注意 Table 14 里 "STAR-KV 60%" 那一行的数字与 Table 2 里 LongChat 的 STAR-KV 60% 行完全相同（39.09），应是论文排版错误；Table 2 里 Llama-3.1 的 STAR-KV 60% 行是 39.58。

## 固定下来的版本

2026-09-07 起代码只保留跑出上述基线的这一套：`evals.harness.LONGBENCH_PAPER_TASKS`（唯一预设 `star-paper-7`，其余预设删除），`evals.run --longbench` 的默认参数就是这次运行的参数（`--longbench-tasks star-paper-7 --long-batch-size 4 --max-length 31500 --dtype bf16`），分片入口只有 `scripts/eval_sharded.sh`。`tests/test_evals.py` 钉住了两侧：本仓库的任务集、列序和默认参数；lm-eval 侧的版本号 0.4.12、七个任务的数据集、生成上限、停止符，以及七个 `doc_to_text` 模板加 `metrics.py` 的 SHA256。lm-eval 升级后若模板或判分变了，测试会先报错，避免与这行数字失去可比性。

## lm-eval 0.4.12 的 LongBench 提示词与官方不一致

`Xnhyacinth/LongBench` 这个镜像把官方数据的 `input` 拆成了 `question` 和 `answer_prefix`，并且 `context` 字段里已经拼进了官方的指令语；lm-eval 的 `doc_to_text` 模板又把指令语和 "Question:" 再包一层，`answer_prefix` 则没有用。用官方 `THUDM/LongBench` 的 `data.zip` 逐条对比，七个任务 1350 条样本的差异：

| 任务 | 指令语出现两次 | "Question: Question:" / "Query: Query:" | 结尾缺少官方答案前缀 |
|---|---|---|---|
| qasper | 200/200 | 200/200 | 否 |
| qmsum | 200/200 | 200/200 | 否 |
| triviaqa | 200/200 | 否 | 否 |
| multifieldqa_en | 150/150 | 150/150 | 否 |
| trec | 200/200 | 否 | 200/200（缺 "Type:"） |
| multi_news | 200/200 | 否 | 否 |
| vcsum | 200/200 | 否 | 否 |

上下文正文和参考答案本身与官方一致，差异只在提示词的包装。之前代码库的协议文档把这记为"双提示 bug"并弃用了这个镜像；但 STAR-KV 的 `eval.py` 用的就是这条流水线，论文数字大概率含这个偏差。判据是基线：lm-eval 原样跑出的七任务均值若等于 42.23，说明论文协议就是含双提示的 lm-eval 版本，评测代码保持原样即对齐；若对不上，再用官方提示词的版本对照。

## 结果

Llama-3.1-8B-Instruct 未压缩基线，`GPUS=4,5,6,7 bash scripts/table2_longbench.sh`（内部按 qmsum | multi_news | vcsum+triviaqa | 其余三个 QA 分四卡，qmsum 单卡 43 分钟最慢）。原始 JSON 和日志在 `results/table2/llama31_8b_instruct_baseline/`。

| | Qasper | QMSum | TriviaQA | MultiQA | TREC | MultiNews | VCSum | Avg |
|---|---|---|---|---|---|---|---|---|
| 本仓库（lm-eval 原样） | 33.89 | 23.99 | 93.08 | 39.29 | 74.00 | 27.08 | 16.10 | 43.92 |
| 论文 Table 2 | 25.19 | 23.20 | 92.00 | 39.90 | 72.50 | 26.90 | 15.91 | 42.23 |
| 差 | +8.70 | +0.79 | +1.08 | −0.61 | +1.50 | +0.18 | +0.19 | +1.69 |
| 官方 THUDM 数据 + 官方 prompt（对照） | 15.07 | 23.75 | 92.36 | 28.31 | 73.00 | 26.89 | 16.18 | 39.37 |

七项里六项与论文的差在 1.5 以内（生成式任务用 ROUGE-L / F1，样本 150 到 200 条，这个量级属于正常波动），只有 Qasper 高了 8.7，均值的 +1.69 几乎全部来自它。官方 prompt 版本（均值 39.37）在 Qasper（15.07）和 MultiQA（28.31）上离论文更远，其余五项与 lm-eval 版相差不到 0.7，反过来证明论文用的就是 lm-eval 这条含双提示的流水线。结论：**除 Qasper 外，LongBench 评测代码与论文对齐；Qasper 的论文值无法用 lm-eval 的任何变体复现。**

### Qasper 的排查

在 lm-eval 流水线上试了能想到的变体（原始预测记录已清理，分数如下）：

| Qasper 变体 | F1 |
|---|---|
| lm-eval 原样（双提示） | 33.89 |
| 去掉重复的 "Question:" | 35.58 |
| 生成到换行即停 | 35.51 |
| LongBench-E 版 `qasper_e`（224 条） | 36.56 |
| 官方 THUDM 数据 + 官方 prompt | 15.07 |
| fp32 加载（上游 `eval.py` 在 transformers 4.x 下的默认）| 34.43（TREC 74.00） |
| 论文 | 25.19 |

原样流水线的预测本身是正常的：200 条里 105 条首行不超过 12 个词，答案中位数 16.5 个词，模型说 "unanswerable" 的 16 条对应 22 条标准答案为 unanswerable。论文的 25.19 落在官方 prompt（15）和 lm-eval（34 到 37）之间，任何单一设置都对不上；论文其它六项都与 lm-eval 原样吻合，所以更可能是论文这一格来自不同的运行或口径，而不是我们代码的问题。这一格在做压缩方法对比时应以本仓库自己跑出的基线为准。
