# 低秩 KV checkpoint 格式

checkpoint 是用 `torch.save` 保存的普通 state dict，以 `strict=False` 加载到基础模型上：里面有的张量覆盖基础模型的对应张量，没有的保持原样。哪些层被压缩，由低秩张量本身决定；加载器（`lrkv.checkpoint.load_lowrank_checkpoint`）先按张量形状建好模块再加载，任何因子加载失败都会直接报错。

## 每层的张量

前缀 `model.layers.{i}.self_attn.`；`H` 是 KV head 数，`d` 是 head_dim。

| 键 | 形状 | 含义 |
|---|---|---|
| `k_proj.VS.weight` | `[sum_h r_h, hidden]` | 每个 KV head 的 latent 投影，按 head 首尾相接（head h 占据行 `[off_h, off_h + r_h)`） |
| `k_proj.U.weight` | `[H*d, sum_h r_h]` | 块对角的展开矩阵：head h 的输出行只读取它自己的秩块 |
| `k_proj.head_ranks` | `int32 [H]` | `r_h`；head 划分只记录在这里 |
| `v_proj.VS.weight` | `[r_v, hidden]` | 整层联合的 latent 投影 |
| `v_proj.U.weight` | `[H*d, r_v]` | 展开回所有 KV head |

语义：`k_proj(x) = U_k @ (VS_k @ x)`，`v_proj(x) = U_v @ (VS_v @ x)`。Sigma 已经折进 `VS`。推理时 cache 每个 token 存的是 `VS @ x`：K 为 `[B, H, L, max_h r_h]`（各 head 补零到该层最大秩；decode 核在每个 head 的真实秩处提前停止，结果精确），V 为 `[B, L, r_v]`。

一层可以只带 K 的键或只带 V 的键。另一个投影保持 dense，`install_lowrank_attention` 会把它按满秩包装（`U = I`，`VS = W`），整层仍走 latent cache。两者都没有的层完全不动（对应 STAR-KV 的 `skip_layers`，通常是 0、1、31）；`lrkv.checkpoint.infer_skip_layers` 直接从 checkpoint 读出这一信息，训练、评测、benchmark 之间不必重复传同一个参数。

## 其余张量

`q_proj`、`o_proj`、MLP、norm、embedding、`lm_head` 都是可选的。STAR-KV 的 KD 会微调全部权重，所以它的 checkpoint 带整个模型（Llama-3.1-8B 为 16.8 GB）；只替换投影的方法可以只交付因子（`lrkv.checkpoint.factors_only`，或转换命令的 `--factors-only`）。

## 模块

`lrkv.modules.HeadwiseLowRankLinear`（K）和 `lrkv.modules.LowRankLinear`（V）是内存中的形式，它们的 `state_dict()` 正是上面的布局（K 模块的 `block_mask` 是非持久化 buffer）。`from_dense`、`from_factors`、`from_head_factors` 用于构造，`head_factors()` 按 head 切分 K 模块。`lrkv.factorize.svd_factorize_model` 是产出这种布局最简单的方式。

## Legacy STAR-KV checkpoint

STAR-KV `train.py` 写出的 `trained_weights.pt` 保存的是训练时的分解 `U / Sigma / V`，每个 head 带一个 soft threshold：

```
k_proj.U.weight, k_proj.U.mask, k_proj.V.weight, k_proj.V.mask
k_proj.Sigma_blocks.{h}.diag, k_proj.Sigma_blocks.{h}.soft_thres_layer.{alpha,s,c}
v_proj.U.weight, v_proj.U.mask, v_proj.V.weight, v_proj.V.mask
v_proj.Sigma.diag, v_proj.Sigma.soft_thres_layer.{alpha,s,c}
```

`lrkv.checkpoint.convert_legacy_starkv` 按 STAR-KV 自己导出函数的规则把它转成标准布局：一个方向保留当且仅当 `diag > alpha`，它的 VS 行是 `V[row] * soft_threshold(diag)`，其中 `x > alpha` 时 `soft_threshold(x) = x * tanh(s * (x - alpha))`，U 列原样复制（K 逐 head，V 整层）。VS 全零的行也一并丢掉，这是精确的，并且顺便恢复了老版"fused"文件的秩——那些文件把 Sigma 重置成了全 1、`alpha = 0`。转换结果在 `tests/test_checkpoint.py` 中与上游 `fuse_and_prune` 逐张量比对过；加载 legacy 文件时会自动完成这一转换。

两个注意点由转换器打印出来，而不是悄悄处理：

* 老版 STAR-KV（归档在 CFS `backups/belt_baselines_reproductions_20260907/reproductions/.../pinned_source/model.py`）没有把 K 的 `U` 约束成块对角，训练可能在里面留下少量跨 head 的混合。逐 head 导出会丢掉这些项（和 `export_kproj_for_triton` 一样），转换器会打印它们的相对能量（36.62 checkpoint 为 2e-3）。
* 秩为 0 的 head 输出为一个零方向，与上游一致。

## 压缩率口径

`lrkv.compression` 按两种口径报告每个 token 的 cache 大小：**padded**（每层 `H * max_h r_h + r_v`，latent cache 实际存储量）和 **compact**（`sum_h r_h + r_v`，按秩紧凑排布时的存储量）。STAR-KV 的 `check_compression.py` 报告的是 padded。
