# 第 03 章源码与教学复核

日期：2026-09-08；基线：MiniMind `6fc918beb68a0d8c40452338df6319fe168014ba`。
本章不讨论注意力内部、缓存、生成采样或完整训练收敛，因此不以 GPU 质量实验为验收项。

| 章节内容 | 复核位置 | 判断 |
|---|---|---|
| 外壳 → 主干 → block → 输出头 | `MiniMindForCausalLM.forward`、`MiniMindModel.forward`、`MiniMindBlock.forward` | 与调用顺序一致 |
| 默认与小配置分别说明 | `MiniMindConfig` | 768/8/6400 为默认；32/2/32 为实验显式值；默认前馈宽度计算为 2432 |
| 两次 Pre-Norm 与残差 | `MiniMindBlock.forward` | 与公式和图 3-2 一致；Dense/MoE 为替代分支 |
| RMSNorm 的公式、归约轴与 dtype | `RMSNorm.norm/forward` | 手算与 CPU 断言一致；保留零输入和可学习缩放的边界 |
| 三个无 bias Linear 和 SiLU | `FeedForward.__init__/forward` | 行向量转置约定与 PyTorch 权重形状一致 |
| Weight tying | `MiniMindForCausalLM.__init__` | 对象共享而非数值复制；实验检查 `is` |

实际运行：`.venv/bin/python experiments/ch03_model_skeleton.py`。
输入/权重反向、零输入、FP16 中间平方、前馈参考表达式、共享参数与全模型小前向均通过，
详细环境、输出及脚本哈希见 `experiments/ch03-results.json`。

教学复核：每张图均有读图说明，每个公式均解释符号和使用条件；自检题附具体参考解释。
正文先给主干位置，再展开 RMSNorm 与 SwiGLU；Attention 内部明确延后到第 04 章。
本次结论只支持当前 CPU 小配置行为，不包含性能对比和模型质量承诺。

出版复核与章节内容状态分开管理。样章在 EPUB/PDF 中的检查结果另记于 M1 排版报告。
