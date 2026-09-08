# 从 STAR-KV 提取了什么、改了什么、去掉了什么

来源：`baselines/STAR-KV`，上游 `main @ f08a62d`（2026-09-01），以及 36.62 复现所用的 pinned 源码（原在 `baselines/reproductions/`，2026-09-07 移至 CFS `backups/belt_baselines_reproductions_20260907/`）。上游代码的审计记录在 `baselines/README.md`。

## 对应关系

| 上游（`baselines/STAR-KV`） | 本仓库 | 说明 |
|---|---|---|
| `abx_rope_batched.py`（Triton 核 `abx`、`torch_abx`、`--check`） | `lrkv/kernels/abx_rope.py`（`abx_rope`、`abx_rope_reference`、`--check/--bench`） | head_dim 参数化，逐行位置偏移，参考实现不依赖 transformers，修复 benchmark 路径（上游构造了 3 维张量，触发自己的断言） |
| `model.py::FusedDecomposeLinear`、`FusedDecomposeLinear_headwise` | `lrkv/modules.py::LowRankLinear`、`HeadwiseLowRankLinear` | 张量和键名相同；新增 `from_dense` / `from_factors` / `head_factors` |
| `model.py::is_fused_state_dict`、`infer_skip_layers`、`build_fused_from_state_dict`、`load_compressed_checkpoint` | `lrkv/checkpoint.py` | 允许只压 K 或只压 V 的层；legacy checkpoint 在 state dict 层面转换，不再实例化训练模块 |
| `model.py::export_kproj_for_triton`、`export_vproj_for_triton`、`_pad_and_pack_kproj`、`KProjInferenceWrapper`、`VProjInferenceWrapper` | `lrkv/attention.py::pack_k_factors`、`pack_v_factors` | 打包后的张量作为 buffer 挂在注意力模块上 |
| `LlamaLoRaAttention_headwise.py::LlamaCustomAttention` + `model.py::_patched_attn_forward`、`_bf16_sdpa_fwd` | `lrkv/attention.py::LowRankAttention` | 一个模块、三种 decode 模式（`triton`、`torch`、`sdpa`），按实例选择 |
| `model.py::replace_attn_with_triton`、`set_model_mode` | `lrkv/model.py::install_lowrank_attention`、`set_decode_mode` | 按模块类型决定替换哪些层，而不是靠 `skip_layers` 参数 |
| `eval.py`（模型加载部分） | `lrkv/model.py::load_compressed_model` | `config` 和 `generation_config` 的 `use_cache` 都打开 |
| `check_compression.py` | `lrkv/compression.py` | 维度从模型的 `config.json` 读取；同时给出 padded 和 compact 两个口径 |
| `eval.py::evaluate_ppl`、`_get_ppl_data` | `evals/ppl.py` | 协议不变；增加 `limit` 用于 smoke |
| `eval.py::evaluate_lmeval` 及任务列表；归档的 `reproductions/.../starkv_trained_eval.py` | `evals/harness.py`、`evals/run.py` | 相同的 HFLM 调用（`add_bos_token=False`、`max_length`、贪心）；11 任务与 7 任务列表做成预设 |
| `latency.py`（`_bench_decode`、`run_e2e`、`run_layerwise`） | `evals/latency.py` | 所有 decode 模式共用一次加载的模型；只输出 CSV/JSON（去掉 matplotlib） |
| 归档的 `reproductions/.../eval_36p62.sh`、`summarize_36p62.py` | `scripts/eval_sharded.sh`、`evals/summarize.py` | 推广到任意 GPU 列表 / 任务预设 / decode 模式 |
| （无） | `lrkv/factorize.py` | 截断 SVD 产出 checkpoint 布局，替代此前时延实验里临时写的 `make_fused_ckpt` |

## 去掉的部分

* `train.py`、`soft_thres_layer.py`、训练时模块 `DiagonalLinear` / `MaskedLinear` / `DecomposeLinear(_headwise)`、`replace_linear_layer`、`fuse_and_prune`、`enforce_rank_floor` 以及秩统计函数。它们是 STAR-KV 的方法本身，不属于推理框架。它们产出的 checkpoint 仍能通过 legacy 转换器加载，转换结果与 `fuse_and_prune` 在测试中比对过。
* `bx_quant.py`、`LlamaLoRaAttention_headwise_quant.py`：它们依赖上游从未发布的 `quant_utils` 和 `abx_rope_batched_quant`。
* 覆盖 `CUDA_VISIBLE_DEVICES` 的 `--cuda-devices` 参数。
* `latency.py` 里的 `matplotlib` / `pandas` 画图。

## 行为上的改动（均有测试覆盖）

1. **左 padding 批次中 key 的位置。** HF 根据 attention mask 分配 RoPE 位置，所以一个左 padding 的行里，位于 cache 下标 `L-1` 的 query 实际位置是 `L-1-n_pad`。上游按 cache 下标旋转缓存的 key（核函数和 torch 路径都是），使得 padding 行的所有相对位置整体偏移 `n_pad`。现在核函数接受由 `position_ids` 推出的 `pos_offset[b]`；`test_left_padded_batch_matches_stock_fp32` 证明 latent 路径在三行不同 padding 的批次上与原模型一致。（STAR-KV 论文报告的精度数字用的是纯 PyTorch 参考路径，不受影响；受影响的只有 Triton/no_triton 时延路径。）
2. **分块 prefill。** 上游的 dense prefill 忽略已有的 cache。现在在已有 cache 之上做 prefill 会从 latent 重建 K/V 并使用带偏移的因果 mask；`test_identity_factors_match_stock_fp32` 用 20 + 17 个 token 的拆分与一次性 prefill 的 logits 对比。
3. **Mask。** transformers 5.x 传给 SDPA 的是布尔 mask，上游按浮点数直接相加。现在 prefill 和 decode 两种形式都处理；带 mask 时不再使用 SDPA 的 `enable_gqa`（会退化到 math kernel 并把整个 `[B, H, L, L]` 注意力矩阵实体化，batch 4、9K token 时约 21 GiB），而是像 transformers 一样重复 K/V。
4. **`generation_config.use_cache`。** 上游只设置了 `config.use_cache`，在 transformers 5.x 下 `model.generate()` 每一步都重新计算整个序列，从未用到核函数（只有显式传 `use_cache=True` 的 lm-eval 用到了）。
5. **V 的展开。** 上游通过 `matmul` 广播一份按 query head 复制的 `U_v`（`[Hq, r_v, d]`），会实体化 `B` 份拷贝；现在每个 KV head 做一次 `bmm`。算术结果相同。
6. **核函数通用性。** `head_dim` 是编译期参数（测试了 64/128/256）；`pos_offset = 0` 时 fp16 路径与上游逐位一致（`test_bit_exact_with_upstream_starkv`）。
7. **decode 单步计时。** `evals.latency` 默认保留上游协议（每次计时前深拷贝 prefill cache）。`--cache-reset crop` 可用于拷贝会 OOM 的 KV 长度，但它改变了分配器的行为，会抬高**原模型**的数字（H800 上的 Llama-2-7B：16K 处 39 ms 而非 29 ms，32K 处 66 而非 46），两种协议的结果不能混用。

STAR-KV 路径本身的数值没有改变：同样的补零，同样的 fp32 `dense = U @ VS` 再转 bf16，核函数内同样的 fp16 RoPE，同样的 fp32 softmax，同样先在 latent 空间做 `probs @ v_lat` 再展开 V。

## 真实模型验证（2026-09-06，H800 80 GB，之前代码库的 transformers 5.9 环境）

原始输出已清理。模型副本和 checkpoint 来自本地已有目录。

**测试。** `pytest tests` 全部通过（当时 26 个，评测协议钉子加入后为 38 个）：transformers 5.9.0 下（之前代码库的环境，即下面各项运行所用的环境，以及 `scripts/setup_env.sh` 构建的仓库自带 `.venv`）和 4.51.1 下（`../本仓库/.venv-eval`）。覆盖内容：核函数与 fp32 参考在 head_dim 64/128/256、GQA 与 MHA、fp16 与 bf16、带位置偏移下的对比；逐 head 动态秩与补零满秩逐位一致；与上游 STAR-KV 核函数逐位一致；用单位因子在 fp32 下 latent 路径与原模型对比（一次性 prefill、带 cache 的贪心 decode、分块 prefill、三行左 padding 的批次）；bf16/fp16 下 triton/torch/sdpa 在各 head 秩不等时的一致性；legacy 转换与上游 `fuse_and_prune` 的对比。

**36.62 checkpoint**（`starkv_llama31_comp60/trained_weights.pt`，legacy 格式，Llama-3.1-8B-Instruct）。`torch.load` 之后转换只需约 3 秒 CPU 时间；29 个压缩层，各 head 的 K 秩见 `lrkv.compression` 打印的表，KV 压缩率 49.2%（padded）/ 58.4%（compact），U 的跨 head 能量 2.1e-3。

| 测量项 | 参考路径 | latent 路径 |
|---|---|---|
| WikiText-2 困惑度，seqlen 2048，141 个窗口 | 11.060（loss 2.4033） | 11.060（loss 2.4034），`triton` |
| 未压缩模型，同一协议 | 7.217 | |
| teacher-forced decode，3 个 prompt（20/0/14 个 pad）、48 步，logit 最大绝对偏差（logit 量级 27.5） | 0（循环本身的 sanity check） | triton 0.25 / 0.30 / 0.28，torch 0.25 / 0.20 / 0.37，sdpa 0.25 / 0.26 / 0.38 |
| 同上，每行 top-1 一致率 | | triton 1.00 / 0.98 / 0.96，torch 1.00 / 1.00 / 0.98，sdpa 1.00 / 1.00 / 0.96 |
| LongBench smoke，lm-eval 0.4.12，batch 4，31500，前 8 例：qasper / trec | 16.54 / 50.0 | 17.15 / 50.0，`triton`（batch 4，带左 padding） |
| zero-shot smoke，前 64 例：piqa / arc_easy 准确率 | 0.781 / 0.750 | |
| `scripts/eval_sharded.sh`，multi_news / trec，4 例 | 17.61 / 75.0（汇总已写出） | |

自由生成的贪心解码在三个 prompt 中的两个上，各 latent 模式都在第 23 到 39 步与参考路径分叉（近似平局的翻转；上面 teacher-forced 的行才反映真实的数值偏差）。完整的七任务 36.62 复现这次没有重跑（需要数小时 GPU 时间），协议对应分片脚本的 `TASKS=star-paper-7`。

**时延**（Llama-2-7B 上的 `starkv_synth_rk32_rv1024.pt`：每 head K 秩 32，V 秩 1024，第 0/1/31 层 dense；上游深拷贝协议，30 次，batch 1，单步 decode，毫秒）：

| ctx | 原模型 HF（bf16 SDPA） | latent `triton` | latent `torch` | 加速比（triton） |
|---|---|---|---|---|
| 1024 | 19.4 | 24.6 | 27.9 | 0.79x |
| 4096 | 19.3 | 24.7 | 27.7 | 0.78x |
| 16384 | 28.8 | 24.7 | 51.0 | 1.17x |
| 32000 | 45.6 | 26.0 | 87.2 | 1.76x |

上游 `latency.py` 在同一 checkpoint、同一 GPU 上的结果（2026-09-02，之前代码库的 `results/throughput/starkv_official_latency/`）：4K 处 17.9 / 23.6 ms，32K 处 45.4 / 26.1 ms，即 0.76x 与 1.74x。逐层合成注意力（batch 16，seq 32768，29 层）：SDPA 3.09 到 3.42 ms，torch 重建 8.0 ms，融合核在补零秩与动态秩下均为 2.30 到 2.32 ms，平均加速 1.39x（上游运行：2.49 ms，1.33x；差别来自按 KV head 做 `bmm` 的 V 展开）。
