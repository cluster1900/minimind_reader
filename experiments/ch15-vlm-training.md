# 第 15 章视觉数据与冻结实验

命令：`.venv/bin/python experiments/ch15_vlm_training.py`。版本、参数名单及完整结果见 [记录](ch15-results.json)。
四层 32 维语言模型下，freeze 0/1/2 的可训练参数数目分别为 243648/20224/1632，视觉编码器始终冻结。
使用真实 PNG 字节、Parquet、VLMDataset 与 collate；相同图数形成 `(2,1,3,32,32)`，混合一图/两图直接 stack 失败。
冻结语言模型后，projector 仍有非零梯度并完成一次更新，其他冻结参数梯度为 None。
真实 checkpoint 验证推理导出排除编码器且为 FP16，resume model 包含编码器且本实验为 FP32。
编码器是真实随机小 SigLIP，只有预训练加载被替换，测试不涉及发布模型质量。

## 2026-10-08 复核

按原实验重新运行，并重新核对正文中使用的入口、数值与结论范围。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。
