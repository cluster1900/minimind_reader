# 第 08 章小模型训练与评估记录

2026-09-08；MiniMind commit `6fc918beb68a0d8c40452338df6319fe168014ba`。
运行命令：`.venv/bin/python experiments/ch08_sft_evaluation.py`。
项目 Python 3.12、PyTorch 2.6.0、Transformers 4.57.6、datasets 3.6.0；CPU、单线程、种子 42。
模型隐藏维度 32、2 层、词表 6400、dropout 0；真实 SFTDataset；固定两条训练记录与一条留出记录。

教学循环执行 80 次 AdamW 更新，学习率 0.003，梯度范数阈值 1。
训练有效目标数 9，NLL 从约 8.7770 降至 0.1119；留出有效目标数 5，NLL 从约 8.7982 降至 6.6260。
两条训练题均生成预期回答并 EOS；留出题“未见问题丙”输出“答甲”，不符合目标“答丙”。
默认空思考前缀与移除空思考的诊断前缀在本次运行中结果相同。

评估采用 NLL 总和除以有效 token 总数；不把 batch 平均直接当成 token 平均。
完整损失轨迹、数据哈希、生成文本与脚本哈希见 [结果](ch08-results.json)。
这是随机小模型的链路及过拟合诊断，不是发布权重、真实预训练能力或大规模泛化测评。

## 2026-10-08 复核

按原实验重新运行，并重新核对正文中使用的入口、数值与结论范围。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind commit `233d00fa068ec2b8c0d4f07b703934d6391a5a73`，工作区干净。
命令：项目根目录运行 `.venv/bin/python experiments/ch08_sft_evaluation.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：Python 3.12 venv，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch08-results.json](ch08-results.json)。

- 训练有效目标数 9，NLL 从 8.7770 降至 0.1137；留出有效目标数 5，NLL 从 8.7982 降至 6.7956。
- 两条训练题贪心生成分别为「答甲」「答乙」并 EOS；留出题「未见问题丙」仍输出「答甲」。
- `eval_llm.init_model`：`load_from` 末段恰为 `model`（含 `../model`、`model/`）走原生权重；含 `model` 子串但末段不是 `model` 的 `foo_model_bar` 等路径走 transformers。

未修改上游；未执行 CUDA 或发布权重评估。
