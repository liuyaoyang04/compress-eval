# RULER：对齐 STAR-KV Table 3

日期：2026-09-07。目标：用本仓库的评测代码复现论文 Table 3（RULER，4K 长度）中 Llama-3.1-8B-Instruct 的行。LongChat-7B-v1.5-32K 不在本课题范围内，不跑。

## 协议

| 项目 | 取值 | 代码位置 |
|---|---|---|
| 框架 | lm-eval 0.4.12 的 `ruler` 任务（合成数据、`max_gen_toks` 128、string-match 判分），不改判分逻辑 | `evals/harness.py::run_lm_eval` |
| 任务集 | 论文的九个：MK1、MK2、MQ、MV、S1、S2、S3、FWE、SQ（去掉 lm-eval 组里的 MK3、VT、CWE、QA-hotpot） | `evals/ruler.py::RULER_PAPER_TASKS` |
| 长度 | 4096；每任务 500 条；九任务等权平均 | `evals/ruler.py::RULER_SEQLEN`、`ruler_scores()` |
| 模型调用 | `add_bos_token=False`、无 chat template、batch 4、`max_length` 31500，与 STAR-KV `eval.py` 一致 | `evals/run.py --ruler` |
| tokenizer | 通过 `TaskManager(metadata={"tokenizer": 模型目录, "max_seq_lengths": [4096]})` 传给任务；这是上游 `eval.py --ruler` 缺的一步，缺了任务加载直接报错 | `evals/ruler.py::task_metadata` |

样本是在任务实例化时用 Python 全局随机数合成的，`simple_evaluate` 先 `random.seed(0)`、`np.random.seed(1234)` 再逐个加载任务，所以每个任务拿到的样本取决于它前面生成了哪些任务。论文的 `eval.py --ruler` 加载整个 `ruler` 组（13 个任务按 `ruler.yaml` 顺序），本仓库只实现这一种生成协议：`evals.ruler.build_tasks` 同样设种子，按组顺序生成到所请求的最后一个任务为止（`ruler_qa_hotpot` 排在最后，永远不生成），再只评测请求的任务。无论怎么分片，每个任务的样本都与整组运行一致。这一点影响很大：FWE 若单独生成或排在别的任务之后生成，随机词表会改变每条样本的词数，得分能相差 6 个百分点。代价是每个进程多花约一分钟生成前置任务的数据。

## 数据

RULER 的三份外部数据在上游分别来自 HF Hub、NLTK 下载器和每次运行时的 HTTP GET。本仓库把它们钉成本地文件（`evals/sources.py`），只在缺失或哈希不符时下载（可通过 `LRKV_PROXY` 指定代理）：

| 文件 | 来源 | SHA256 | 用途 |
|---|---|---|---|
| `paul_graham_essays.parquet` | HF 数据集 `baber/paul_graham_essays`，revision `792d672b`（lm-eval 读的就是这个数据集） | `825448033be0…` | S2、S3、MK1、MQ、MV 的 essay haystack |
| `squad-dev-v2.0.json` | rajpurkar.github.io | `80a5225e9490…` | SQ |
| `punkt_tab.zip` | nltk_data gh-pages（jsDelivr、ghfast 镜像作后备） | `e57f64187974…` | essay haystack 分句 |
| `c4-validation.00000-of-00008.json.gz` | `allenai/c4` revision `607bd4c8` | `1f25b6af12da…` | 困惑度（与之前手头的副本逐字节相同） |

`evals.ruler.install_sources()` 在加载任务前把 lm-eval 的 `get_haystack`（essay 分支）和 `qa_utils.download_json`（SQuAD URL）指向这些文件，并把解压后的 `punkt_tab` 加进 `nltk.data.path`。读取方式与上游相同（同样的拼接与空白归一化），所以合成出来的样本与联网运行一致。

## 运行

```bash
python -m evals.sources                      # 一次取齐外部文件（缺失时才下载）
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m evals.run --model <模型目录> --baseline --ruler --long-batch-size 4 --output out.json
# 按任务分到多张 GPU（约 1.9 秒/样本；九任务 4500 条，6 卡约半小时）
GPUS=2,3,4,5,6,7 bash scripts/table3_ruler.sh            # 分片 + 汇总 + 与论文对照（results/table3/<NAME>/compare.md）
python -m evals.compare results/table3/<NAME> --paper llama31_8b_instruct
```

`--ruler-tasks` 接受预设 `paper-9` 或逗号列表（分片时用）；`--ruler-seqlen 4096 16384` 可一次算多个长度（附录 Table 13 的 16K）。压缩模型加 `--checkpoint ckpt.pt --decode-mode reference`（STAR-KV 的精度参考路径）或 `triton`。

## 结果

Llama-3.1-8B-Instruct 未压缩基线，`GPUS=2,3,4,5,6,7 bash scripts/table3_ruler.sh`，最长分片 23 分钟。原始 JSON 和日志在 `results/table3/llama31_8b_instruct_baseline/`。

| | MK1 | MK2 | MQ | MV | S1 | S2 | S3 | FWE | SQ | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| 本仓库 | 100.00 | 100.00 | 100.00 | 98.65 | 100.00 | 100.00 | 99.80 | 96.40 | 77.23 | 96.90 |
| 论文 Table 3 | 100.0 | 99.8 | 99.9 | 98.95 | 100.0 | 100.0 | 99.6 | 96.07 | 78.12 | 96.94 |
| 差 | 0.00 | +0.20 | +0.10 | −0.30 | 0.00 | 0.00 | +0.20 | +0.33 | −0.89 | −0.04 |

九项均在 ±0.9 以内，均值差 0.04。结论：**RULER 评测代码与论文 Table 3 对齐。** SQ 的 −0.89 在 500 条样本、部分匹配判分下属于正常波动（单条样本分数取值离散，标准误约 1.5 个百分点）。
