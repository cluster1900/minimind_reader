# 第 20 章交互与取消实验

命令：`.venv-omni/bin/python experiments/ch20_realtime.py`，详见 [结果](ch20-results.json)。
真实 RealtimeSession 配合可控 VAD，验证 2048 点开始、13312 点静音结束和前后缓冲保留。
4096 点大块触发 interrupt 时仅保留前 2048 点，复现提前返回边界。
实际会话消费/PCM/生成包装函数在可控队列、socket 和解码波形下运行；正常发出 7680/3840 字节，
stop 和新语音打断不发旧 PCM，done 标记正确，包装保持 no_grad。
执行 HTML 的真实 done 处理与 stopPcm，以计数对象替代 WebAudio；两个节点停止、列表清空、计划时间重置。
未执行真实接收线程竞态、麦克风/扬声器或网络时延测试，不把替身边界当作已测设备行为。

## 2026-10-08 复核

除了 interrupt，speech_end 也会提前返回；同一大块末尾的新语音未被处理。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。

## 2026-10-30 按新基线重验

日期：2026-10-30。MiniMind-O commit `b5d1bc11ece8de17404cbc034c3e09718d0fecdf`，工作区干净。
命令：项目根目录运行 `.venv-omni/bin/python experiments/ch20_realtime.py`（`MINIMIND_SOURCE_ROOT` 指向三个上游仓库的父目录）。
环境：音频依赖环境为 Python 3.12 venv（torch `2.6.0+cpu`，含 soundfile/librosa/scipy；未按 `requirements-omni.lock` 的 Python 3.11 重建），无 CUDA；HTML 片段由本机 `node` 执行。
结构化输出见 [ch20-results.json](ch20-results.json)。

- VAD 窗 1024 点，2048 点开始说话、13312 点静音结束；前后缓冲保留 4096 点。
- 4096 点大块 interrupt 仅保留前 2048 点；`speech_end` 输入 15360 点时丢弃末尾 2048 点新语音。
- 正常路径 PCM 7680/3840 字节、模型 yield 3 次、未打断；stop/speech 打断不发 PCM、yield 1 次、`interrupted = true`；包装保持 no_grad。
- 浏览器 done 处理：停止 2 个播放节点、列表清空、`asstPlay = false`、`nextT = 5`。

未修改上游；未执行真实麦克风/扬声器、接收线程竞态或网络时延测试。
