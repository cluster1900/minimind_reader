# 第 10 章 LoRA 与蒸馏实验

命令：`.venv/bin/python experiments/ch10_lora_distillation.py`。版本与哈希见 [结果](ch10-results.json)。
两层、隐藏维 32、词表 40、rank 4 的模型只注入 q/o 四处，共 1024 参数。
初始输出一致，首次 A 梯度为零、B 非零，更新 B 后 A 也有梯度。
保存并合并到新模型后，与合并前在线路径 logits 最大差约 `1.59e-4`，不称为逐位无损。
真实 distillation_loss 与教师→学生 KL 手算一致；教师无梯度，列交换使相同 shape 的分布出现明显 KL。
不下载教师权重，不以给定 logits 实验推断实际蒸馏效果。

## 2026-10-08 复核

补充 V=D=32 且共享输出头权重的反例：合并重载会改变输入 Embedding，不能把单层代数等价推广到整网。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。
