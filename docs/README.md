# 文档索引

| 文件 | 内容 |
|---|---|
| `full_model_results.md` | 未压缩基线的最终结果（三张表、环境、命令），后续方法对比的参照 |
| `exp1_table1_reproduction.md` | Table 1 对齐：为什么困惑度窗口是 4096、zero-shot 指标口径是混合的 |
| `longbench_table2.md` | Table 2 对齐：lm-eval 0.4.12 的 LongBench 双提示现象、Qasper 排查、固定的版本 |
| `ruler_table3.md` | Table 3 对齐：九任务、数据文件、样本按组顺序合成的原因 |
| `checkpoint_format.md` | 低秩 checkpoint 的张量布局、legacy STAR-KV 格式转换、压缩率口径 |
| `starkv_extraction.md` | 推理框架与评测代码从 STAR-KV 提取时拿了什么、改了什么、验证结果 |

上游代码审计（量化未开源、`eval.py` 的坑）在 `../baselines/README.md`。
