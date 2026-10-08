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
