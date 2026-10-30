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

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind-O commit `b5d1bc11ece8de17404cbc034c3e09718d0fecdf`，工作区干净。
命令：项目根目录运行 `.venv-omni/bin/python experiments/ch17_thinker_talker.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：音频依赖环境为 Python 3.12 venv（torch `2.6.0+cpu`，含 soundfile/librosa/scipy；未按 `requirements-omni.lock` 的 Python 3.11 重建），无 CUDA。
结构化输出见 [ch17-results.json](ch17-results.json)。

- 默认 `bridge_layer = 3`；四层小模型的 tiny bridge 索引为 1，hook 与第二 block 输出一致。
- 改变输出音频历史或有效 speaker 条件时，文本 logits 不变、声学 logits 改变；无 speaker marker 时条件不生效。
- 双缓存共 6 层，增量步与全量前向对照通过。
- 纯音频损失到达 Bridge 前缀，不连接后续 Thinker block。
- 一层配置 `invalid_single_layer_bridge = -1`：无模态注入时与 Embedding 相同，注入后保留音频特征而不是原始 Embedding。

未修改上游；未加载外部编码器或报告音质。
