# 第 12 章 RL 目标实验

命令：`.venv/bin/python experiments/ch12_rl_objectives.py`。原始输出见 [结果](ch12-results.json)。
从锁定函数 AST 抽取 GAE、组内优势与目标计算块，提供合成中间变量执行原语句；未执行完整训练函数。
GAE returns 为 `[1,1,0]`、归一化优势约 `[1,-1,0]`；组内 `[1,3]` 优势约 ±0.9999，等分组为零。
ratio=2、优势 ±1 的对照中，GRPO 梯度约 `[0,1]`，CISPO 为 `[-1,1]`。
真实 Critic 前向观察到两次末尾 norm；真实 log-prob helper 与对应下一 token 对齐。
没有启动奖励模型、SGLang 或 GPU RL 作业。

2026-09-09 补查：从真实 `grpo_train_epoch` 提取 `policy_loss` 归约语句。两条回答长度为 1/3 时，逐 token 损失分别为 1/3，回答平均为 2，全 token 平均为 2.5；反向梯度分别为 0.5 与三个 1/6，mask 位置为零。
