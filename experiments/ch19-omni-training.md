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

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind-O commit `b5d1bc11ece8de17404cbc034c3e09718d0fecdf`，工作区干净。
命令：`MINIMIND_SOURCE_ROOT=/workspace/repos/minimind_resources /workspace/repos/.venv-mm/bin/python experiments/ch19_omni_training.py`。
环境：Python venv `/workspace/repos/.venv-mm`，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch19-results.json](ch19-results.json)。

断言已改为新基线行为，不再沿用 resume=fp16：

- 初始化复制、冻结与 projector 可训练参数仍为 none/all/last1 = 9270578/9004018/9019450，audio_proj/vision_proj = 18496/27200。
- 均匀 logits 下 text_loss = 1.0986123085021973，audio_loss = 5.263081073760986，stop 梯度比约 9.999981880187988；全 -100 标签两项均为 0。
- collate dummy 批 `(2,3,560)`；混入 None 后音频/长度表为 1/2。
- `omni_checkpoint` resume 排除 `audio_encoder.` / `vision_encoder.` 且为 float32（`1e-8` 得以保留）；推理 `*.pth` 为 float16（该值下溢为 0）。这与 ch15 的 V resume 仍写入完整 state_dict 不同，不能混写。
- `eval_omni.init_model`：`load_from` 末段恰为 `model` 走原生权重；`foo_model_bar` 等含 `model` 子串但末段不是 `model` 的路径走 transformers。
- 尾组（3 小批、K=2）：保存时已更新 2 次，参数约 0.07435，梯度已清空。
- 整组（4 小批、K=2）：末步同时是累积边界与 `iters`，保存时更新 2 次、梯度为空。
- 周期组内保存（K=4、间隔 3、5 小批）：第 3 步仍有待处理梯度 -1.5、更新 0 次；第 5 步先更新再保存。
- `start_step=2` 续训：剩余 2 批从 step 3 起算，保存发生在 step 4，更新 1 次。
- 空 loader：即使 `iters=3` 也不更新、不保存。

未修改上游；未执行 CUDA、完整 T2A/A2A/I2T 训练或音质评价。
