# 第 14 章视觉注入实验

命令：`.venv/bin/python experiments/ch14_vision.py`。MiniMind-V 版本、环境与输出见 [结果](ch14-results.json)。
真实随机小 SigLIP 为 32×32 图、patch 16、隐藏维 16，加载环节由 `_tiny_vision.py` 替代；语言模型隐藏维 32。
实际 VLM 输出 `(1,7,6400)`，projector 非零梯度，冻结视觉编码器梯度 None。
decode 不重复编码图片，缓存/全量最后位置误差约 `1.49e-7`。
两段分开的图像标记正确替换；相邻八标记被视为一段，输入长度 10 变为 6，作为契约失败案例。
没有加载默认视觉权重，不报告图像识别或问答质量。
