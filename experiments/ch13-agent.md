# 第 13 章 Agent 交互实验

命令：`.venv/bin/python experiments/ch13_agent.py`。完整输出与源码哈希见 [结果](ch13-results.json)。
生成后端按脚本提供两轮文本，实际运行原 rollout_single、模板、解析、工具执行与奖励。
记录 70 个 response 区域 token，其中 39 个生成、31 个观察；token 重建完全一致，观察 mask/log-prob 占位正确。
正常完成奖励为 3，轮数上限时仍执行最后工具并标 unfinished，奖励更低。
非法 JSON 忽略、未知工具 None，以及 gt `3` 匹配“答案是13”的宽松子串反例均已验证。
它验证交互控制流，不证明一个学习策略具备这些行为。

## 2026-10-08 复核

合法 JSON 数组会被 parse_tool_calls 接受，但 rollout_single 随后的 .get 调用报错，验证了语法解析与结构校验的区别。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind commit `233d00fa068ec2b8c0d4f07b703934d6391a5a73`，工作区干净。
命令：项目根目录运行 `.venv/bin/python experiments/ch13_agent.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：Python 3.12 venv，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch13-results.json](ch13-results.json)。

断言已改为新基线行为：

- 两轮轨迹 71 个 response token，其中 39 个生成、32 个观察；完整重编码不再等于保留轨迹。
- 下一轮输入逐 token 等于 `prompt +` 已保留 response；EOS 留在轨迹中，策略 mask 为 0。
- `<tool_call>[1]</tool_call>` 与 `"x"` / `1` / `true` / `null` 均被 `parse_tool_calls` 丢弃，`rollout_single` 不再对非对象调用 `.get`。
- 非 dict `arguments` 被置为 `{}` 后仍执行；marker 丢失与 log-prob 长度不符均抛 `RuntimeError`。
- 非连续 `completion_mask` 按 `sum` 取前缀，保留空洞位置、丢掉尾部有效 token。
- 正常完成奖励为 3.0；`max_turns=1` 仍写入工具观察，unfinished 奖励约 -0.0333。

未修改上游；生成仍是脚本化后端，不证明学习策略已获得相同成功率。
