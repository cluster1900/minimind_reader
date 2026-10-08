# 第 17 章双分支实验

命令：`.venv-omni/bin/python experiments/ch17_thinker_talker.py`，详见 [结果](ch17-results.json)。
真实随机小 Omni 的 thinker/model 为同一对象，四层配置的 Bridge hook 与第二 block 输出一致。
改变输出音频历史或有效 speaker 条件，声学 logits 改变而文本 logits 保持一致；无 speaker marker 时条件不生效。
六层双缓存与全量前向对照通过；纯音频损失到达 Bridge 前缀，但不连接后续 Thinker block。
一层默认 Bridge 为 -1，实际保留进 block 前的状态；无模态注入、eval 的原实验中它与 Embedding 相同。未加载外部编码器或报告音质。

## 2026-10-08 复核

补充模态注入反例：一层、bridge_layer=-1 时保留注入后的状态，并非总是原始 Embedding。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。
