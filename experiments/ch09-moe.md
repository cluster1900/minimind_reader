# 第 09 章 MoE 实验

命令：`.venv/bin/python experiments/ch09_moe.py`。基线、环境与哈希见 [结果](ch09-results.json)。
固定路由后，输出等于选中专家输出，token 重排保持对应关系。
主损失 gate 最大梯度约 `2.83e-10`，加辅助项后约 `0.007734`；未选专家梯度存在但为零。
均匀概率对照下 K=1/2 的 aux 分别约 0.01/0.02。eval aux 为零，上层模型正确汇总各层辅助项。
这是当前 CPU 模块的路由与梯度验证，不是专家并行或质量 benchmark。
