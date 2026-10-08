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
