# 第 18 章码本与时序实验

命令：`.venv-omni/bin/python experiments/ch18_audio_codes.py`，详见 [结果](ch18-results.json)。
真实 Dataset 的九通道输入为 `(9,127)`，八路首音频标签位置为 20～27，之前为 pad，之后跟随对应历史 code。
给真实 stream_generate 提供确定 logits，普通/思考两模式均重建同样三帧，文本结束后仍输出音频。
随机小 Mimi 实际解码为 `(1,1,5760)`，对应 24 kHz 下 0.24 秒，不评价音质。
还检查生成器迭代的梯度上下文，以及当前 tokenizer 中 201 不是换行、234 才是换行。
