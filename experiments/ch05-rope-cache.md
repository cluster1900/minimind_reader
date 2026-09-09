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
