# 第 16 章音频输入实验

命令：`.venv-omni/bin/python experiments/ch16_audio_input.py`。版本和哈希见 [结果](ch16-results.json)。
真实读入 8 kHz 双声道 800 点 WAV，下混重采样为 16 kHz 单声道 1600 点。
合成前端输出 `(4,560)`、有效长度 3，小 Linear 替代 SenseVoice 后得到 `(3,32)` 投影。
真实 processor/注入路径保留 dummy 行及未覆盖占位，projector 有梯度，冻结编码器无梯度。
未加载 SenseVoice，不评价语音识别；processor 本身没有重采样的行为也已检查。
