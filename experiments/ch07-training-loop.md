# 第 07 章循环与保存时机验证记录

2026-09-08；MiniMind commit `6fc918beb68a0d8c40452338df6319fe168014ba`。
运行命令：`.venv/bin/python experiments/ch07_training_loop.py`。CPU，禁用 GradScaler，三个固定标量目标。
结果与源代码/脚本哈希见 [结构化记录](ch07-results.json)。

直接导入真实 pretrain 与 full_sft 的 train_epoch，将模型替换为标量平方损失，优化器替换为计数 SGD；
checkpoint 调用在控制流测试中记录调用时的参数和梯度，但真实循环的权重文件仍写入项目 build 目录。
这样观察的是循环时序，不声称已经运行真实语言模型的完整训练入口。

两个循环均在第 2 小批完成一次更新。第 3 小批保存时参数约 0.065，仍有待处理梯度 -0.935；
函数返回前处理尾组后，参数约 0.07435、更新次数变成 2。保存内容落后于尾部更新的边界得到复现。

另调用真实 lm_checkpoint 做独立保存/加载，验证 model 权重保存为 FP16，优化器含 AdamW 动量，
字段为 model、optimizer、epoch、step、world_size、wandb_id、scaler；没有自动加入待累积梯度和 RNG 快照。
get_lr 的端点与 SkipBatchSampler 的尾组/跳过行为也已检查。
未修改上游；未执行 CUDA、NCCL 或性能测试。

## 2026-10-08 复核

按原实验重新运行，并重新核对正文中使用的入口、数值与结论范围。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。
