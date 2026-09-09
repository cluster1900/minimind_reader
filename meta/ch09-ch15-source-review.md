# 第 09～15 章源码与实验复核

日期：2026-09-08。MiniMind 基线 `6fc918beb68a0d8c40452338df6319fe168014ba`；
MiniMind-V 基线 `740d467ece78a0b7d2d976fcb424472095d4a688`。上游只读。

| 章 | 复核的实现与主要证据 |
|---|---|
| 09 | MOEFeedForward 的 N/K/E 维归约、Top-1 归一化、index_add、dummy 梯度及上层 aux 汇总；真实模块实验 |
| 10 | LoRA 的方阵筛选、闭包绑定、A/B 初始化、冻结和合并；真实损失的教师→学生 KL、温度与词表切片 |
| 11 | DPODataset 自行 shift、正 ID gather、前后半排列、序列求和、参考冻结；损失梯度与随机模板反例 |
| 12 | PPO GAE 与 critic 两次 norm，GRPO 组内 std、默认 CISPO、detach、KL 与归约；执行原 AST 计算块，不冒充完整 RL 训练 |
| 13 | AgentRLDataset、rollout_single、观察增量/mask、EOS 过滤、工具与奖励分支；脚本生成配合真实交互代码 |
| 14 | VLMConfig、LayerNorm projector、连续段替换、prefill 图像路径与缓存；真实随机小 SigLIP 接入模型 |
| 15 | 三种 freeze_llm、真实 Parquet Dataset/直接 stack、projector 梯度与两种保存内容；真实工具函数实验 |

本轮没有把注释或名字直接当成行为：默认 GRPO 脚本实际选择 CISPO；critic 并非只按注释删掉旧 head；
`target_tokens` 不代表 projector 自动重采样；视觉 SFT 默认仅开放首尾语言 block。

Agent 的 gt 字符串 `3` 可匹配“答案是13”，已以反例说明此奖励不是严格任务正确性证明。
VLM 相邻标记段和变图数 collate 的失败也作为输入契约案例保留，没有改上游掩盖问题。
图题与说明方向已核对，修正了原草稿中把 TD 流程称作横向的描述。

七个实验均在 CPU 通过。09～13 使用真实本地实现或明确隔离的源码计算块；
14～15 使用真实随机小 SigLIP，只替代预训练权重加载。视觉辅助代码为 `experiments/_tiny_vision.py`。
随机编码器不具备本书宣称的视觉质量，脚本化 Agent 也不证明策略学会工具调用。

算法背景另查原始 DPO、PPO、DeepSeekMath 与 MiniMax-M1 报告，引用已加入正文。
特别区分 MiniMax-M1 报告中的 CISPO 与本地保留 KL、按回答归约的实现，不把二者写成完全相同。
源码映射见 `chapter-source-map.json`，实验 JSON 保留源文件、脚本与通用辅助函数哈希。
出版验证另行记录；本次不包含完整 RL、SGLang 服务或发布视觉权重的质量测评。
