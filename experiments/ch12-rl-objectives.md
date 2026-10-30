# 第 12 章 RL 目标实验

命令：`.venv/bin/python experiments/ch12_rl_objectives.py`。原始输出见 [结果](ch12-results.json)。
从锁定函数 AST 抽取 GAE、组内优势与目标计算块，提供合成中间变量执行原语句；未执行完整训练函数。
GAE returns 为 `[1,1,0]`、归一化优势约 `[1,-1,0]`；组内 `[1,3]` 优势约 ±0.9999，等分组为零。
ratio=2、优势 ±1 的对照中，GRPO 梯度约 `[0,1]`，CISPO 为 `[-1,1]`。
真实 Critic 前向观察到两次末尾 norm；真实 log-prob helper 与对应下一 token 对齐。
没有启动奖励模型、SGLang 或 GPU RL 作业。

2026-09-09 补查：从真实 `grpo_train_epoch` 提取 `policy_loss` 归约语句。两条回答长度为 1/3 时，逐 token 损失分别为 1/3，回答平均为 2，全 token 平均为 2.5；反向梯度分别为 0.5 与三个 1/6，mask 位置为零。

## 2026-10-08 复核

用受控 HTTP 响应验证：两个生成 ID 只返回一个 log-prob 时，适配器在前面补 0，mask 仍有效；没有启动 SGLang 服务。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind commit `233d00fa068ec2b8c0d4f07b703934d6391a5a73`，工作区干净。
命令：`MINIMIND_SOURCE_ROOT=/workspace/repos/minimind_resources /workspace/repos/.venv-mm/bin/python experiments/ch12_rl_objectives.py`。
环境：Python venv `/workspace/repos/.venv-mm`，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch12-results.json](ch12-results.json)。

既有 AST 算术与 Critic/log-prob 对齐在新基线复现：GAE returns 为 `[[1,1,0]]`，归一化优势约 `[1,-1,0]`；组内优势约 `±0.9999`；GRPO/CISPO 梯度约 `[0,1]` 与 `[-1,1]`；回答平均 2、全 token 平均 2.5。

清单第 5 条补充断言：

- 有效 PAD ID：生成 token 等于 PAD 时，朴素 `!= pad` mask 为 `[0,1,1,0,1,1]`，`scatter` 回填后为 `[0,1,1,1,1,1]`。
- EOS：首个 EOS 计入 mask，其后为 0；PPO 策略 mask 为 `[1,1,0]`。
- Torch 左 padding：`prompt_lens` 为矩形宽度 `[4,4]`，`completion_mask` 起始全 1。
- SGLang 右补齐（受控 HTTP，未启动服务）：去掉左 padding 后 `input_ids` 为 `[[1,4],[5,6,7]]`，`prompt_lens` 为 `[2,3]`；completion/output/log-prob 右侧补齐，短行 mask 为 `[1,0]`。
- DDP：CPU gloo、`world_size=2`。包装后 log-prob 解包对齐，训练前向走 DDP；`broadcast_buffers=False` 时 rank0 buffer 增量 1.0、rank1 为 0；反向后两卡梯度相等。未走源码里的 `device_ids=[local_rank]` CUDA 路径。

未修改上游；未执行 NCCL、多 GPU、真实 SGLang 或完整 RL 作业。
