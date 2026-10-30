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

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind commit `233d00fa068ec2b8c0d4f07b703934d6391a5a73`，工作区干净。
命令：项目根目录运行 `.venv/bin/python experiments/ch04_attention.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：Python 3.12 venv，torch `2.6.0+cpu`，无 CUDA，种子 42、单计算线程。
结构化输出见 [ch04-results.json](ch04-results.json)。

原有手算、因果/padding、SDPA 分支与非方阵对齐断言仍通过。补充低精度显式路径断言，对应新基线 `(xq @ xk.T).float()` 发生在缩放和 mask 之前：

- 手算权重与输出与 4.2 节一致；SDPA/显式路径最大误差约 `8.94e-8`；未来扰动与 padding 有效查询误差均为 0。
- 非方阵 SDPA 输出 `[1.0, 1.5]`；带缓存多 token 未调用 SDPA，与全量误差 0。
- fp16 激活 + fp32 mask：softmax 收到的分数为 float32；被 padding 的键相对 `-1e9` 的残差为 0（有限，不是 `-inf`）；输出 dtype 为 float16 且有限。
- 全零 fp32 mask 在 fp16 显式路径上仍得到有限非零输出。
- fp16 下 `1000` 向量的 QK 乘积溢出为 inf，之后 `.float()` 不能恢复；先转 fp32 再相乘得到 `8e6`。`q_norm`/`k_norm` 权重取 1000 时，真实 `Attention.forward` 输出非有限。

未修改上游；未执行 GPU 内核或正式模型质量评估。
