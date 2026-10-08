---
title: "第 13 章 Agent 多轮交互与奖励"
chapter_id: ch13
status: verified
audience: advanced
prerequisites: [ch02, ch06, ch12]
sources:
  - repository: jingyaogong/minimind
    commit: 6fc918beb68a0d8c40452338df6319fe168014ba
    dirty: false
verified_at: 2026-10-08
validation: {static: passed, cpu: passed, gpu: not_required}
validation_records:
  - meta/source-audit-2026-10-08.md
  - meta/consistency-review.md
  - experiments/ch13-agent.md
  - meta/ch09-ch15-source-review.md
scope: "真实 rollout_single、工具解析与反馈、生成/观察 mask、打包与奖励；生成使用脚本化响应，不证明策略已学会工具调用"
---

# 第 13 章 Agent 多轮交互与奖励

单轮回答结束后才评分，仍不足以描述工具交互。模型可能先请求计算，收到结果后再继续回答。
这意味着一条训练轨迹交替包含模型自己生成的内容，以及环境提供的观察。
本章要解决的核心问题是：怎样把这些内容放进上下文，又只对模型负责的部分计算策略目标？

我们用“计算 1 加 2”贯穿流程。这个例子足够小，便于观察每个 token、工具返回和奖励，
而不把复杂任务是否成功与轨迹拼接是否正确混在一起。

**主线必读：** 生成请求、执行工具、反馈和 response mask。先跟随本章的小输入，再对照图和源码。

**进阶选读：** 轨迹概率、奖励误匹配和轮数上限；第一次阅读可先掌握输入输出契约。

**需要额外设备/权重：** 训练后策略在真实工具环境中的评估。本章 CPU 实验只支持明确列出的机制结论。

## 13.1 工具定义、调用文本和执行器是三层

工具定义告诉模型可用的名字与参数格式；模型生成的 `<tool_call>` 文本表达请求；
真正执行工作的是外部 Python 代码。聊天模板只负责组织文本，不会因为出现一个标签就自己访问网络或运行函数。

源码入口是 MiniMind
[trainer/train_agent.py](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/trainer/train_agent.py)
中的 `parse_tool_calls`、`execute_tool`、`rollout_single`、`calculate_rewards` 和 `rl_train_epoch`。
`rollout_engine.py` 提供采样后端，但不独自负责全部 Agent 循环。

当前工具集合里，天气、时间、汇率和翻译等来自本地模拟数据；计算器等通过本地函数处理。
所以这个仓库的多轮示例可以教学交互机制，却不能被描述成已经验证了真实在线天气或金融服务。

数据入口是 [AgentRLDataset](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/dataset/lm_dataset.py)。
它读取 conversations 与 gt，从 system 消息提取工具定义，并移除最后一条示例消息供后续生成。
应核对末条确实是参考回答，以及 gt 的具体评判含义；数据类不会自动把任意对话修成合法任务。

## 13.2 一轮调用结束后，模型会收到什么

模型先生成一个合法 JSON 调用，例如请求 `calculate_math`，参数 `expression` 为 `1+2`。
解析器取出标签内的 JSON，执行器找到对应函数，返回 `{"result":"3"}`。
随后程序把 assistant 调用和 tool 结果加入消息列表，再应用模板，形成下一轮 prompt。

```mermaid
sequenceDiagram
    participant R as rollout_single
    participant P as 采样后端
    participant T as 本地工具函数
    R->>P: 问题与工具说明
    P-->>R: tool_call：calculate_math
    R->>T: expression = 1+2
    T-->>R: result = 3
    R->>R: 加入 assistant 与 tool 消息，重建模板
    R->>P: 包含工具观察的新上下文
    P-->>R: 计算结果是3。
```

图 13-1：两轮交互。先追工具请求如何离开模型，再看结果如何变成下一轮模型输入。
同一条轨迹不只有一个生成片段，也包含执行器提供的观察。图中的工具是本地模拟集合，
没有隐藏的联网请求；真实服务接入需要另行实现并验证其契约。

`parse_tool_calls` 对无法解析的 JSON 会忽略，若最终没有解析出调用，循环就可能按普通回答结束。
因此“没有调用”不等于“输出格式一定正确”。执行未知工具返回 None，也不能被当成成功结果。
解析成功还不等于参数结构已经验证：`<tool_call>[1]</tool_call>` 中的 `[1]` 是合法 JSON，
`parse_tool_calls` 会接受其中的数组，随后 `rollout_single` 对它调用 `.get` 就会报错。
执行器按全局工具表找函数，提供给模型的工具定义列表并不是这里的硬性白名单；
奖励阶段才另行检查名称和参数。真实工具接入应在执行边界明确检查这些条件，不能只靠事后扣分。
`execute_tool` 使用 SIGALRM 做时间限制，其平台和线程适用条件需要检查，不是一个完整的任意代码沙箱。

## 13.3 一条轨迹里，谁生成的 token 才参与策略目标

`rollout_single` 维护 `response_ids`、`response_mask` 和 `response_old_logps`。
模型生成的有效 ID 加入轨迹，并标 mask=1；工具反馈与模板补入的观察增量标为 0。
这些观察仍要输入模型，影响后续回答，但不应被当成模型刚刚采样的动作来优化。

| 轨迹部分 | 是否进入上下文 | 策略 mask |
|---|---|---:|
| 初始问题和工具说明 | 是 | 0 |
| 模型生成的工具调用 | 是 | 1 |
| 工具结果与相应模板补入内容 | 是 | 0 |
| 模型随后生成的回答 | 是 | 1 |

```mermaid
flowchart TD
    P["初始 prompt"] --> C["生成调用：mask 1"]
    C --> O["环境观察：mask 0"]
    O --> A["生成回答：mask 1"]
    A --> S["按原顺序打包完整序列"]
    S --> L["shift 后只统计有效生成位置"]
```

图 13-2：输入范围与优化范围的区别。自上而下理解整条轨迹，再看 mask 的 0/1。
观察被 mask 掉的是策略计分位置，不是从上下文删除。这个区别与第 06 章的 prompt 梯度讨论相呼应。

当前代码还过滤生成 ID 中的 PAD 和 EOS，再记录动作段；模板可能随后补入自己的结束标记。
不要把这里的 EOS 处理直接等同于单轮 GRPO 的“保留到第一个 EOS 为止”，否则可能错误截断多轮交互。

## 13.4 重建模板后，旧前缀必须仍然对齐

工具返回加入 messages 后，代码再次 tokenize 新模板，再以当前累计长度取 `obs_delta`。
这种做法依赖“之前累计的 token 确实是重建后序列的前缀”。
模板重写、空白处理或边界编码变化都可能影响这个假设，不能只看最终字符串读起来正常。

本章实验检查了 `prompt_ids + response_ids` 与最终 context 重新编码完全相同。
还检查 response ID、mask 与旧 log-prob 长度一致，观察位置的旧 log-prob 使用占位值且不会计入策略目标。
这些检查对所用样例成立，不自动证明任意格式的生成文本都保持同样性质。

训练侧 `rl_train_epoch` 把 prompt 与 response 合并，再按最终总长度做左侧截断和右侧补齐。
mask 与 log-prob 都必须同步裁剪、错位和补齐。模型 logits 的前一个位置对应后一个 ID，
所以旧 log-prob 的前缀占位数与输入 token 数又差一位，这正是要画对齐表的地方。

## 13.5 reward 并不是“工具已经真正做对”的万能证明

当前奖励区分没有工具调用和有工具调用两条路径。
前者可结合长度、思考格式、重复惩罚及奖励模型分数；后者统计合法名称与参数、调用数量差异，
再检查最终文本是否包含 gt 中的内容或数值，并对未完成轨迹扣分。

`validate_gt_in_text` 使用文本匹配及数值比较，不是在所有任务上重新执行一个严格判题器。
例如 gt 为字符串 `3` 时，“答案是13”也会命中子串条件；本章实验复现了这个反例。
函数所在分支的注释提到执行结果奖励，也不能据此把它描述为完整的工具正确性证明。
训练可能学会满足匹配规则，但最终仍应在独立任务中检验。

单条轨迹得到奖励后，与同一 prompt 的其他候选一起做组内标准化，
再使用第 12 章的 GRPO/CISPO 目标。当前 Agent 入口的默认 `loss_type` 也是 cispo。
有效生成 token 为零的行在 Agent 目标中有单独处理，不能把另一训练器的归约直接照抄过来。

## 13.6 达到轮数上限时发生什么

当最后允许的一轮仍然生成了工具调用，`unfinished` 会被设为 true。
但当前流程仍执行这次工具并写入观察，只是不再开启下一轮模型生成。
因此“达到上限”不能被解释成在解析调用之前就完全停止一切副作用。

本章以 max_turns=1 运行调用样例，观察到工具结果仍进入 context，轨迹被标为 unfinished，奖励低于正常完成情况。
接入真实工具时，轮数、参数校验、失败结果、重复调用和资源边界应有明确设计；
本地模拟函数的成功不能代替这些外部行为的验证。

归约还有一个与第 12 章不同的边界：`rl_train_epoch` 用 `valid_rows = token_counts > 0`
排除没有有效动作 token 的行，再对剩余轨迹求均值；GRPO 的相应实现则仍把空行计入回答数。
因此不能把两个脚本同名的 `policy_loss` 直接当作相同平均口径。

## 13.7 一个可复现的交互实验

运行 `.venv/bin/python experiments/ch13_agent.py`。
它用脚本化后端先返回计算器调用，再返回“计算结果是3。”，但实际执行真实 `rollout_single`、
模板、解析器、计算器和奖励函数。这样可以控制输入，专门验证轨迹，而不需要先训练出 Agent。

本次两轮轨迹记录 70 个 response 区域 token，其中 39 个为生成、31 个为观察。
最终 token 重建完全一致，生成动作中没有保留被过滤的 EOS；完整回答按当前规则得到奖励 3。
非法 JSON 被忽略，未知工具返回 None，轮数上限的 unfinished 分支也已验证。
结果与哈希在 `experiments/ch13-results.json`，不声称一个学习策略已获得相同成功率。

1. 聊天模板、工具请求与执行器分别做什么？
2. 工具观察 mask=0，是否表示模型不能读取它？
3. 仅有最终字符串，能否可靠还原每个训练位置的 log-prob？
4. max_turns 到上限，当前代码是否一定不再执行最后的工具调用？
5. gt 文本匹配成功，是否足以证明任意工具任务正确？

参考解释：第一题模板排列消息，工具请求声明名称和参数，执行器实际调用函数并返回观察。第二题 response mask=0 表示不把观察作为策略自己的动作计入相应目标，观察仍可作为后续生成的上下文。

第三题最终字符串会丢失分词、被过滤的 EOS、工具边界以及采样时的概率，不能可靠还原轨迹。第四题当前到达轮数上限仍可能执行最后一次工具，再按 unfinished 分支收尾。第五题文本包含匹配并非语义判定，例如 gt 为 3 时字符串 13 也可能命中。

可选练习：让脚本化后端返回一次错误参数，再检查观察文本、unfinished 和奖励分别如何变化。
