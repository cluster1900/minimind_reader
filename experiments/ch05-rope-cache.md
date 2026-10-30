# 第 05 章位置与缓存验证记录

2026-09-08；MiniMind commit `6fc918beb68a0d8c40452338df6319fe168014ba`。
运行命令：`.venv/bin/python experiments/ch05_rope_cache.py`。CPU、PyTorch 2.6.0、种子 42、单计算线程。
完整输出及哈希见 [结果](ch05-results.json)。

真实 RoPE 旋转满足本次范数与相对内积检查，相对内积误差约 `2.38e-7`。
全量、逐 token、分块缓存前向对应位置的 logits 在容差内一致，最大误差分别约 `5.96e-8` 与 `2.98e-8`。
三次生成后的结果长度 7、返回缓存长度 6，验证最后采出的 token 尚未写入缓存。

固定 logits 的测试保留真实 generate 的分布处理，但替换前向以隔离采样机制。
Top-P 的边界项、Top-K 并列值和正负重复惩罚均符合预期。
YaRN 只做源码公式与触发条件复核，本实验不验证长文本质量或性能。

## 2026-10-08 复核

携完整生成序列与缓存续写，结果与一次生成一致；右 padding 的采样概率来自最后一列而非最后有效位置。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind commit `233d00fa068ec2b8c0d4f07b703934d6391a5a73`，工作区干净。
命令：项目根目录运行 `.venv/bin/python experiments/ch05_rope_cache.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：Python 3.12 venv，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch05-results.json](ch05-results.json)。

真实 RoPE 相对内积误差约 `2.38e-7`。
全量、逐 token、分块缓存前向对应位置的 logits 最大误差分别约 `7.45e-8` 与 `2.24e-8`。
三次生成后结果长度 7、返回缓存长度 6；`resumed_generation_matches=true`，`right_padding_samples_last_column=true`。
Top-P 分布约 `[0.5714, 0.4286, 0, 0]`，Top-K 并列支持数为 3，重复惩罚分布约 `[0.002837, 0.4211, 0.1549, 0.4211]`。

未修改上游；这是当前 CPU 的位置旋转、缓存续写与采样机制检查，不验证长文本质量或性能。
