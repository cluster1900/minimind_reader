# 第 06 章样本与监督验证记录

2026-09-08；MiniMind commit `6fc918beb68a0d8c40452338df6319fe168014ba`。
运行命令：`.venv/bin/python experiments/ch06_datasets.py`。
环境为项目 Python 3.12、PyTorch 2.6.0、Transformers 4.57.6、datasets 3.6.0、CPU；种子 42。
合成 JSONL 与 HF 缓存都写入 `build/`，上游保持只读。文件与脚本哈希见 [结果](ch06-results.json)。

PretrainDataset 返回未错位的同长标签，并忽略 PAD。
两轮 SFT 的有效目标解码为 `甲答案<|im_end|>\n乙答案<|im_end|>\n`，共 9 个有效目标。
模型 loss 与独立 shift/交叉熵计算一致，约 8.741042。
忽略目标的 logits 梯度为零，但 prompt 首位置的 Embedding 输出仍有非零梯度。
截断到长度 4 时 labels 全部为 -100，真实 forward loss 为 NaN。

本实验验证数据/监督语义，不报告训练质量；随机后处理固定种子，不代表任意样本的标签范围相同。

## 2026-10-08 复核

对长度与 max_length 一致的人工序列，缺少结束序列时末尾 3 个 PAD 被监督；补齐结束边界后 PAD 被忽略。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind commit `233d00fa068ec2b8c0d4f07b703934d6391a5a73`，工作区干净。
命令：项目根目录运行 `.venv/bin/python experiments/ch06_datasets.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：Python 3.12 venv，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch06-results.json](ch06-results.json)。

PretrainDataset 返回未错位的同长标签，PAD 位置为 -100。
两轮 SFT 有效目标解码为 `甲答案<|im_end|>\n乙答案<|im_end|>\n`，共 9 个有效目标。
模型 loss 与独立 shift/交叉熵一致，约 8.741042。
忽略目标的 logits 梯度为零，prompt 首位置 Embedding 输出仍有非零梯度。
截断到长度 4 时 labels 全为 -100，真实 forward loss 为 NaN。
缺少结束序列时末尾 3 个 PAD 被监督；补齐结束边界后监督 PAD 数为 0。

未修改上游；本实验验证数据/监督语义，不报告训练质量或语言能力。
