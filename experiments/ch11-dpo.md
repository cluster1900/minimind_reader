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

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind commit `233d00fa068ec2b8c0d4f07b703934d6391a5a73`，工作区干净。
命令：项目根目录运行 `.venv/bin/python experiments/ch11_dpo.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：Python 3.12 venv，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch11-results.json](ch11-results.json)。

断言已改为成对共享决策，不再沿用“相同回答被两边各自随机处理成 5/11”：

- 零差 DPO loss 仍为 `0.693147`；beta=0.15 时 chosen/rejected 有效项梯度为 `-0.075` / `+0.075`，忽略位置为 0。
- 把一个 chosen log-prob 提高 0.4 后，loss 降至 `0.663597`；全零 mask 仍给出 `0.693147`。
- 模板两侧都含空思考段；`post_processing_chat` 在同一 `remove_empty_think` 下同删或同留，显式相反决策才会得到不同文本。
- 种子 42 下不同回答的一对监督目标均为 5，且都去掉空思考；相同回答的一对 `x`/`y`/`mask` 完全相同，计数为 `[5, 5]`。

未修改上游；本实验验证目标、mask 与成对共享决策，不报告 DPO 后质量提升。
