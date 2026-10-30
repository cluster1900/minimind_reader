# 第 16 章音频输入实验

命令：`.venv-omni/bin/python experiments/ch16_audio_input.py`。版本和哈希见 [结果](ch16-results.json)。
真实读入 8 kHz 双声道 800 点 WAV，下混重采样为 16 kHz 单声道 1600 点。
合成前端输出 `(4,560)`、有效长度 3，小 Linear 替代 SenseVoice 后得到 `(3,32)` 投影。
真实 processor/注入路径保留 dummy 行及未覆盖占位，projector 有梯度，冻结编码器无梯度。
未加载 SenseVoice，不评价语音识别；processor 本身没有重采样的行为也已检查。

## 2026-10-08 复核

替代编码器报告输出长度 1 时，现有包装仍按输入有效长度保留 3 帧；不代表真实 SenseVoice 下采样或长度有误。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind-O commit `b5d1bc11ece8de17404cbc034c3e09718d0fecdf`，工作区干净。
命令：项目根目录运行 `.venv-omni/bin/python experiments/ch16_audio_input.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：音频依赖环境为 Python 3.12 venv（torch `2.6.0+cpu`，含 soundfile/librosa/scipy；未按 `requirements-omni.lock` 的 Python 3.11 重建），无 CUDA。
结构化输出见 [ch16-results.json](ch16-results.json)。

- 8 kHz 双声道 800 点 WAV 经真实 `OmniDataset.process_audio` 下混重采样为 16 kHz 单声道 1600 点。
- 合成前端输出 `feature_shape = [4, 560]`、`valid_length = 3`；小 Linear 替代 SenseVoice 后 `encoded_shape = [3, 32]`。
- 替代编码器报告长度 1 时，包装仍按输入有效长度保留 3 帧（`encoder_reported_length = 1`，`retained_length_from_input = 3`）。
- dummy 行保留、注入后序列长度不变、未覆盖占位保留；projector 有梯度，冻结编码器未注册进 `named_parameters`；processor 本身不重采样。

未修改上游；未加载 SenseVoice，不评价语音识别质量。
