# 第 18 章码本与时序实验

命令：`.venv-omni/bin/python experiments/ch18_audio_codes.py`，详见 [结果](ch18-results.json)。
真实 Dataset 的九通道输入为 `(9,127)`，八路首音频标签位置为 20～27，之前为 pad，之后跟随对应历史 code。
给真实 stream_generate 提供确定 logits，普通/思考两模式均重建同样三帧，文本结束后仍输出音频。
随机小 Mimi 实际解码为 `(1,1,5760)`，对应 24 kHz 下 0.24 秒，不评价音质。
还检查生成器迭代的梯度上下文，以及当前 tokenizer 中 201 不是换行、234 才是换行。

## 2026-10-08 复核

验证文本 yield 是累计前缀，音频是单帧；非流式仅返回最后一帧且无文本，函数内部迭代关闭梯度。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind-O commit `b5d1bc11ece8de17404cbc034c3e09718d0fecdf`，工作区干净。
命令：项目根目录运行 `.venv-omni/bin/python experiments/ch18_audio_codes.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：音频依赖环境为 Python 3.12 venv（torch `2.6.0+cpu`，含 soundfile/librosa/scipy；未按 `requirements-omni.lock` 的 Python 3.11 重建），无 CUDA。
结构化输出见 [ch18-results.json](ch18-results.json)。

- 真实 Dataset 九通道输入 `dataset_shape = [9, 127]`，`assistant_start = 20`，八路首音频标签位置为 20～27。
- 脚本化 logits 驱动真实 `stream_generate`：普通模式 12 步、思考模式 15 步，均重建同样三帧；文本结束后仍输出音频。
- 非流式最后一帧为 `[12, 112, 212, 312, 412, 512, 612, 712]` 且无文本；函数内部迭代关闭梯度；文本 yield 是累计前缀。
- 随机小 Mimi 解码 `decoded_shape = [1, 1, 5760]`，24 kHz、12.5 fps，时长 0.24 秒。
- tokenizer 中 `234` 为换行，`201` 不是换行。

未修改上游；随机 codec 不评价音质。
