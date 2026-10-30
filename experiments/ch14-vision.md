# 第 14 章视觉注入实验

命令：`.venv/bin/python experiments/ch14_vision.py`。MiniMind-V 版本、环境与输出见 [结果](ch14-results.json)。
真实随机小 SigLIP 为 32×32 图、patch 16、隐藏维 16，加载环节由 `_tiny_vision.py` 替代；语言模型隐藏维 32。
实际 VLM 输出 `(1,7,6400)`，projector 非零梯度，冻结视觉编码器梯度 None。
decode 不重复编码图片，缓存/全量最后位置误差约 `1.49e-7`。
两段分开的图像标记正确替换；相邻八标记被视为一段，输入长度 10 变为 6，作为契约失败案例。
没有加载默认视觉权重，不报告图像识别或问答质量。

## 2026-10-08 复核

相同像素用四维/五维字典得到一致 logits，直接传五维 Tensor 则在 vision_model(**image_inputs) 处抛 TypeError。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind-V commit `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc`，工作区干净。
命令：项目根目录运行 `.venv/bin/python experiments/ch14_vision.py`。
环境：Python 3.12 venv，torch `2.6.0+cpu`，无 CUDA。
结构化输出见 [ch14-results.json](ch14-results.json)。

断言按新基线通过：

- 真实随机小 SigLIP（32×32、patch 16、隐藏维 16），加载被 `_tiny_vision.py` 替换；VLM logits 形状 `(1, 7, 6400)`。
- projector 梯度非零，视觉编码器梯度为 None。
- decode 不重复编码图片（prefill 仅一次 encoder 调用）；缓存/全量最后位置误差为 `1.1920928955078125e-07`。
- 相同像素的四维/五维字典 logits 一致；直接传五维 Tensor 在 `vision_model(**image_inputs)` 处抛 TypeError（`argument after ** must be a mapping, not Tensor`）。
- 两段分开的图像标记正确替换；相邻八标记被视为一段，输入长度 10 变为 6。

未修改上游；未执行 CUDA 或真实视觉问答评测。

