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

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind-V commit `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc`，工作区干净。
命令：项目根目录运行 `.venv/bin/python experiments/ch15_vlm_training.py`。
环境：Python 3.12 venv，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch15-results.json](ch15-results.json)。

断言已按新基线补齐，冻结计数与旧小配置一致：

- freeze 0/1/2 可训练参数仍为 243648/20224/1632；同图数 batch 为 `(2,1,3,32,32)`；混合图数 collate 被拒绝；冻结 LLM 时 projector 梯度为 3.5333178006112576。
- `eval_vlm.py`：末段恰好为 `model` 走原生权重；`foo_model_bar` 及同类子串路径走 transformers。
- 尾组（3 小批、K=2）：pretrain/SFT 均先更新再保存，保存时已更新 2 次，参数约 0.07435，梯度已清空。
- 整组（4 小批、K=2）：末步同时是累积边界与 `iters`，保存时更新 2 次、梯度为空，参数约 0.1278。
- 周期组内保存（K=4、间隔 3、5 小批）：第 3 步仍有待处理梯度 -1.5、更新 0 次；第 5 步先更新再保存，参数约 0.04200。
- `start_step=2` 续训：剩余 2 批从 step 3 起算，保存发生在 step 4，更新 1 次。
- 空 loader：即使 `iters=3` 也不更新、不保存。
- `vlm_checkpoint`：resume 仍为完整 `state_dict`（含 `vision_encoder`，`1e-8` 以 float32 保留）；推理 `*.pth` 排除编码器且为 float16（该值下溢为 0）。这不是 MiniMind/O 的 resume-FP32 修复。

未修改上游；未执行 CUDA 或真实视觉问答评测。
