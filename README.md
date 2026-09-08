# compress-eval

低秩 KV cache 压缩方法的统一评测框架：困惑度（WikiText-2 / C4）、六个 zero-shot 任务、LongBench 七任务、RULER 4K 九任务，协议与 STAR-KV（ICML 2026）论文 Table 1 / 2 / 3 对齐。评测数据一次下载后固定在仓库本地并逐项校验哈希，评测过程完全离线。附带一个从 STAR-KV 开源代码提取并重构的 latent KV cache 推理框架（`lrkv/`），任何能给出"每个 KV head 一组低秩 K 因子、每层一个低秩 V 因子"的压缩 checkpoint 都能走同一条推理和评测路径。

## 快速开始

```bash
git clone --recursive https://github.com/liuyaoyang04/compress-eval.git
cd compress-eval
LOCK=1 bash scripts/setup_env.sh                 # uv + requirements.lock.txt：torch 2.5.1 cu121、transformers 5.9.0、lm-eval 0.4.12
.venv/bin/python -m evals.sources                # 一次下载全部评测数据到 data/（约 270 MB），之后完全离线
ln -s /path/to/Llama-2-7b-hf models/             # 或 HF_TOKEN=... .venv/bin/python scripts/fetch_models.py
ln -s /path/to/Llama-3.1-8B-Instruct models/
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests -q
```

需要 Python 3.10、[uv](https://github.com/astral-sh/uv) 和一张 80 GB 的 NVIDIA GPU。下载需要代理时设置 `LRKV_PROXY=http://host:port`；huggingface.co 不通时自动退到 `hf-mirror.com`。`.venv/bin/python -m evals.sources --check` 随时核对数据是否齐全。**不要升级 lm-eval**：LongBench 和 RULER 的提示词与判分来自它，换版本数字就不可比。

## 运行评测

```bash
# 论文三张表（分片到 GPUS 里的卡，跑完自动汇总并与论文对照，输出在 results/table{1,2,3}/<NAME>/）
GPUS=0,1           bash scripts/table1_ppl_zeroshot.sh     # PPL + zero-shot，两个模型
GPUS=0,1,2,3       bash scripts/table2_longbench.sh        # LongBench 七任务
GPUS=0,1,2,3,4,5   bash scripts/table3_ruler.sh            # RULER 九任务

# 评测一个压缩 checkpoint
CHECKPOINT=ckpt.pt DECODE_MODE=triton NAME=my_method GPUS=0,1,2,3 bash scripts/table2_longbench.sh
CHECKPOINT=ckpt.pt DECODE_MODE=triton NAME=my_method GPUS=0,1,2,3,4,5 bash scripts/table3_ruler.sh

# 单条命令，四个套件可任意组合；--limit N 做 smoke
PY=.venv/bin/python; M=models/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 $PY -m evals.run --model $M --baseline --ppl --tasks zero-shot --output results/zs.json
CUDA_VISIBLE_DEVICES=0 $PY -m evals.run --model $M --checkpoint ckpt.pt --decode-mode triton --longbench --ruler --output results/lb_ruler.json

# 与论文对照 / 合并分片
$PY -m evals.compare results/table2/my_method --paper starkv60_llama31
$PY -m evals.summarize results/table2/my_method --preset star-paper-7
```

`--decode-mode reference` 保留 HF 原生注意力只换低秩投影（精度参考路径）；`triton` / `torch` / `sdpa` 走 latent KV cache。`evals.compare` 的 `--paper` 可选 `llama2_7b`、`llama31_8b_instruct`（论文基线）、`starkv60_llama31`、`palu50_llama31`。结果 JSON 含各套件分数、压缩模型的 cache 占用和环境指纹（库版本、GPU、代码 commit、数据状态）。

## 评测协议

| 套件 | 协议 | 代码 |
|---|---|---|
| 困惑度 | WikiText-2 test、C4 固定分片前 1100 篇，非重叠窗口 4096 | `evals/ppl.py` |
| zero-shot | OBQA、PIQA、ARC-e、ARC-c、HellaSwag、WinoGrande；OBQA / ARC-c / HellaSwag 取 `acc_norm`，其余 `acc`；batch 32 | `evals/harness.py` |
| LongBench | Qasper、QMSum、TriviaQA、MultiFieldQA-en、TREC、MultiNews、VCSum，lm-eval 0.4.12 `longbench_*` 原样；batch 4，`max_length` 31500 | `evals/harness.py` |
| RULER | 九任务 @4096，每任务 500 条，按 lm-eval `ruler` 组顺序合成样本，与分片方式无关 | `evals/ruler.py` |
| 模型调用 | bf16、sdpa、`add_bos_token=False`、无 chat template、贪心 | `lrkv/model.py`、`evals/harness.py` |

这些设置都由 `tests/test_evals.py` 钉住（窗口、指标口径、任务集、默认参数、数据表、lm-eval 侧模板与判分的哈希）。Llama-3.1-8B-Instruct 基线的复现情况：Table 1 困惑度精确一致，zero-shot 均值差 0.1，RULER 九项差 0.9 内，LongBench 六项差 1.5 内但 Qasper 高 8.7（论文值无法复现）。做方法对比请以本框架自己跑出的基线为参照。完整的基线数字见 `docs/full_model_results.md`，细节见 `docs/exp1_table1_reproduction.md`、`docs/longbench_table2.md`、`docs/ruler_table3.md`。

## 数据是怎么固定的

`evals/sources.py` 维护两张表：URL 文件（C4 分片、RULER 的 essays / SQuAD / punkt_tab）钉 SHA256；lm-eval 按 id 读的 HF 数据集（六个 zero-shot 集、WikiText-2、LongBench 七子集）钉数据集仓库的 git revision，下载到仓库自己的 `data/hf_datasets/` 作为 `HF_DATASETS_CACHE`，并记录逐行内容哈希（`--verify` 核对）。`evals.run` 启动时先切换到这个目录并强制离线，缺数据会在加载模型前报错。

## Checkpoint 与推理

checkpoint 是一个 `torch.save` 的 state dict，每个压缩层含 `k_proj.{VS.weight,U.weight,head_ranks}` 和 `v_proj.{VS.weight,U.weight}`（STAR-KV 的 legacy U/Sigma/V 格式可直接加载），格式说明见 `docs/checkpoint_format.md`。

```python
from lrkv import load_compressed_model, set_decode_mode
model, tok, info = load_compressed_model("models/Llama-3.1-8B-Instruct", "ckpt.pt", dtype="bf16", decode_mode="triton")
set_decode_mode(model, "torch")     # triton | torch | sdpa，运行时可切换
```

```bash
.venv/bin/python -m lrkv.checkpoint convert --input trained_weights.pt --output canonical.pt   # legacy -> 标准布局
.venv/bin/python -m lrkv.compression --checkpoint canonical.pt --model models/Llama-3.1-8B-Instruct --per-layer
.venv/bin/python -m lrkv.factorize --model models/Llama-2-7b-hf --k-rank 32 --v-rank 1024 --skip-layers 0 1 31 --output synth.pt
```

## 目录

```
lrkv/       推理框架：低秩模块、checkpoint 格式、Triton decode 核、latent KV cache 注意力、加载器
evals/      评测框架：run（入口）、sources（数据固定）、harness / ruler / ppl（协议）、compare、summarize、meta、latency
scripts/    setup_env.sh、fetch_models.py、table{1,2,3}_*.sh、eval_sharded.sh、check_generation.py
tests/      内核逐位一致性、评测协议钉子、数据表、对照表
docs/       checkpoint 格式、STAR-KV 提取记录、三张表的对齐记录
baselines/  上游 STAR-KV / Palu 子模块（只读，仅测试与文档使用）
data/ models/ results/   数据、模型、输出（gitignored）
```
