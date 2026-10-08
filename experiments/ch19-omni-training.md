# 第 19 章训练目标与状态实验

命令：`.venv-omni/bin/python experiments/ch19_omni_training.py`，详见 [结果](ch19-results.json)。
真实初始化将 Thinker 最后两层复制给匹配小 Talker；freeze_backbone 与两种 projector 模式按原代码执行。
原训练损失块配合均匀 logits 得文本 loss 约 1.098612、音频 loss 约 5.263081，stop 梯度比例约 10。
真实 collate 的 dummy 行保持 `(2,3,560)`，改成 None 后音频/长度表行数变为 1/2。
随机 token 扰动保留 image 标记，但没有同样显式排除文本音频标记。
真实 resume 模型状态排除外部编码器并为 FP16；这些测试不代表完整训练收敛或音频质量。

## 2026-10-08 复核

真实损失 AST 中，全 -100 文本及音频标签产生两个零项；有限 loss 不能证明存在有效监督。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。
