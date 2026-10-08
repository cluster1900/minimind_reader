# 第 04 章注意力验证记录

2026-09-08；MiniMind commit `6fc918beb68a0d8c40452338df6319fe168014ba`。
运行命令：`.venv/bin/python experiments/ch04_attention.py`。CPU、PyTorch 2.6.0、种子 42、单计算线程。
完整输出与源文件/脚本哈希见 [结果](ch04-results.json)。

手算两位置注意力与程序结果一致；真实小配置的 SDPA/显式路径最大误差约 `5.96e-8`。
改变未来输入不改变过去输出；改变被 padding 屏蔽的键，不改变其他有效 query 的输出，误差均为零。
记录 API 调用确认 padding 和单 token 时跳过 SDPA；全零 mask 仍得到非零有限输出。
RoPE 在本实验被设为恒等旋转以隔离机制；未测试 GPU 内核速度或正式模型质量。

## 2026-10-08 复核

补充非方阵 SDPA 左上角因果偏置（输出 1、1.5），并验证带缓存多 token 分支未调用 SDPA、与全量结果近似一致。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。
