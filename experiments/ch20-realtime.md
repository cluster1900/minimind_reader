# 第 20 章交互与取消实验

命令：`.venv-omni/bin/python experiments/ch20_realtime.py`，详见 [结果](ch20-results.json)。
真实 RealtimeSession 配合可控 VAD，验证 2048 点开始、13312 点静音结束和前后缓冲保留。
4096 点大块触发 interrupt 时仅保留前 2048 点，复现提前返回边界。
实际会话消费/PCM/生成包装函数在可控队列、socket 和解码波形下运行；正常发出 7680/3840 字节，
stop 和新语音打断不发旧 PCM，done 标记正确，包装保持 no_grad。
执行 HTML 的真实 done 处理与 stopPcm，以计数对象替代 WebAudio；两个节点停止、列表清空、计划时间重置。
未执行真实接收线程竞态、麦克风/扬声器或网络时延测试，不把替身边界当作已测设备行为。
