# 第 16～20 章源码与实验复核

日期：2026-09-08。MiniMind-O 基线为 `ccab6d98520ac5ac5d7583a3fa21a65105137e28`。
已完整读取模型、数据集、训练工具/主循环、eval 入口、Web 服务及相关浏览器事件处理。

| 章 | 复核链路 | 验证边界 |
|---|---|---|
| 16 | WAV 读入/下混/重采样 → frontend mask → encoder → projector → 注入 | 重采样与包装/注入真实；前端与 SenseVoice 用明确替身 |
| 17 | Thinker → Bridge → Talker，音频历史/说话人条件，双缓存 | 真实随机小 Omni；hook、梯度和 logits 对照；无默认权重质量声明 |
| 18 | Dataset 延迟/shift → generator 各路时序 → 帧重组 → Mimi | 原 Dataset 与生成循环；logits 脚本化；Mimi 为真实随机小模型 |
| 19 | 初始化复制、冻结模式、CE/stop 权重/8 路归约、collate、保存 | 原函数/AST 计算块配合合成中间变量，不运行完整训练作业 |
| 20 | RealtimeSession → 会话队列消费 → 生成/PCM → browser stop | 实际源代码体；VAD、线程、socket、解码波形和音频节点受控替代 |

以下容易误述的内容均已核实并进入正文：

- processor 的 sampling_rate 参数不代表内部重采样；前端有效长度不能只按注释称为编码器长度。
- 文本音频标记 ID 16 与声学 pad 2049 不是同一词表；音频注入保留未覆盖位置和总长度。
- 默认 8 层 Bridge index 3 对应第 4 block；一层默认 index -1 会退回 Embedding。
- 声学历史和 speaker 条件只影响当前 Talker 分支，输入音频特征则可先进入 Thinker。
- 八路是码本通道，不是八个未来文本时间点；Dataset 和流式生成的延迟/shift 已逐项对齐。
- 默认 enter_token_id=201 不是当前 tokenizer 的换行，换行为 234；装饰生成器创建不自动关闭迭代期间梯度。
- 当前图像数据取第一项；随机 token 扰动不等于从模型预测采样；文本排除项明确保护 image ID。
- stop=2050 在损失中乘 10，按有效数量平均后固定除以 8；配置改变需同步硬编码。
- collate 过滤 None 可破坏批位置，默认 dummy 机制不是任意数据都安全的证明。
- VAD 大块中途返回 interrupt 可留下未处理样本；两条停止路径都需浏览器停止已排队节点。
- Web 的图像+语音实时分支会走 ASR，不能笼统称所有路径无级联。

全部五个 CPU 实验通过。`experiments/_omni_fixture.py` 为共享小配置，音频使用独立环境与锁定依赖。
结果 JSON 保存源文件、实验脚本与通用辅助函数哈希，进一步的出版快照保存音频 fixture 哈希。
没有进行真实麦克风/扬声器、预训练语音质量或 GPU 时延测量；章节不作这些承诺。
