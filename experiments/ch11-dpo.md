# 第 11 章 DPO 实验

命令：`.venv/bin/python experiments/ch11_dpo.py`。数据与损失版本见 [结果](ch11-results.json)。
相同策略/参考时 loss 约 0.693147；beta=.15 下 chosen/rejected 的有效 log-prob 梯度分别为 -0.075/+0.075。
提高 chosen 分数使 loss 降至约 0.663597，忽略位置不影响结果。
真实 DPODataset 已 shift，y 是合法 ID；全零 mask 仍给出有限 loss。
固定种子 42 时，即使原始回答相同，两边随机空思考处理也产生不同输入和 5/11 个有效目标。
实验验证配对、mask 和目标，不报告 DPO 后质量提升。

## 2026-10-08 复核

按原实验重新运行，并重新核对正文中使用的入口、数值与结论范围。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。
