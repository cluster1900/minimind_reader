# 第 09 章 MoE 实验

命令：`.venv/bin/python experiments/ch09_moe.py`。基线、环境与哈希见 [结果](ch09-results.json)。
固定路由后，输出等于选中专家输出，token 重排保持对应关系。
主损失 gate 最大梯度约 `2.83e-10`，加辅助项后约 `0.007734`；未选专家梯度存在但为零。
均匀概率对照下 K=1/2 的 aux 分别约 0.01/0.02。eval aux 为零，上层模型正确汇总各层辅助项。
这是当前 CPU 模块的路由与梯度验证，不是专家并行或质量 benchmark。

## 2026-10-08 复核

按原实验重新运行，并重新核对正文中使用的入口、数值与结论范围。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind commit `233d00fa068ec2b8c0d4f07b703934d6391a5a73`，工作区干净。
命令：项目根目录运行 `.venv/bin/python experiments/ch09_moe.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：Python 3.12 venv，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch09-results.json](ch09-results.json)。

旧断言“主损失 gate 梯度接近 0”已不成立。新基线 `norm_topk_prob=True` 且 K=1 走直通估计，四路实际反向结果：

- 主损失：gate 最大梯度约 `0.009189`，输入约 `0.018536`，选中专家约 `0.106539`；未选专家梯度存在但为零。
- 主损失的输入梯度与只走选中专家时一致（`top1_main_input_matches_expert_only=true`），与 `gate(x_flat.detach())` 切断重算路径相符。
- 辅助项单独反向：gate 约 `0.007734`，输入约 `0.004283`，专家无梯度。
- 主损失加辅助项后 gate 约 `0.016924`。
- token 重排仍等价；均匀概率下 K=1/2 的 aux 仍约 `0.01`/`0.02`；eval aux 为 0；上层 `aux_loss` 约 `0.001019`，等于各层之和。

未修改上游；这是当前 CPU 模块的路由与梯度验证，不是专家并行或质量 benchmark。
