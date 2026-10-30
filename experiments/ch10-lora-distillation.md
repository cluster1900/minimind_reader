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

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind commit `233d00fa068ec2b8c0d4f07b703934d6391a5a73`，工作区干净。
命令：`MINIMIND_SOURCE_ROOT=/workspace/repos/minimind_resources /workspace/repos/.venv-mm/bin/python experiments/ch10_lora_distillation.py`。
环境：Python venv `/workspace/repos/.venv-mm`，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch10-results.json](ch10-results.json)。

原有注入、梯度、合并与蒸馏断言仍成立；并补上 `--lora_rank` 训练/评估一致性：

- 注入名单仍是两层 `q_proj`/`o_proj`，可训练参数 1024；首次 A 梯度和为 0，B 梯度和约 `1.118733`。
- 合并后 logits 最大差约 `1.578e-4`；同 rank=4 重新 `load_lora` 最大差约 `1.16e-6`。
- `apply_lora` 默认 rank 与 `train_lora` / `eval_llm` / `serve_openai_api` 的 `--lora_rank` 均为 16，并传入 `apply_lora(model, rank=args.lora_rank)`。
- 用默认 rank=16 加载本次 rank=4 适配器触发 `load_state_dict` 的 `size mismatch`，不会静默错输出。
- 共享输出头反例：合并 logits 差约 `1.821`，Embedding 变化约 `0.125`。
- `distillation_loss` 为 `0.157891`，相同分布残差约 `5.96e-8`，列交换后约 `0.184316`，教师无梯度。

未修改上游；不下载教师权重，不声称真实任务微调或压缩效果。
