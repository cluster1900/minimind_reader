# 初始书籍蓝图检查记录

- 检查日期：2026-09-08。
- 输入：用户粘贴的《从零手撕大模型》蓝图、RMSNorm 样章及 Prompt 模板。
- 对照：桌面 `minimind_resources` 下三个源码仓库，以及 `vllm_reader` 的项目规范与出版实现。
- 版本：完整 commit 见 [README](../README.md#源码基线)；三个上游工作区在核对时均为 clean。
- 验证范围：静态阅读核心模型、数据集、相关训练入口、工具反馈与实时会话路径；没有执行训练、模型质量测试或出版构建。

## 1. 总体判断

选题成立，三个项目能够串起“模型内核 → 训练和生成 → 后训练 → 多模态输入输出”的工程学习路线。
原蓝图已经具备问题、张量、源码、实验四个重要元素，但还不能直接作为批量生产书稿的依据。
主要风险是入门跨度过大、实现细节混淆，以及把教学直觉写成了确定的技术结论。

建议以 MiniMind 文本闭环作为主书骨架，后训练、视觉与语音作为有前置说明的扩展。
标题采用“源码阅读与工程实践”更贴合内容；如保留“从零”，应解释是从基础理解并实现核心模块，
不表示从零训练所有外部编码器、codec，也不表示已经完成全部内部依赖的讲解。

## 2. 源码核对中必须修正的内容

以下链接均固定在本次本地基线。它们作为可定位证据，不表示本次运行过这些代码。

### 2.1 Dataset 与模型不能重复 shift

蓝图把 PretrainDataset 写成 `tokens[:-1]`、`tokens[1:]`。当前实现实际返回同长度
`input_ids` 与其复制的 `labels`，仅把 padding label 设为 `-100`。
`MiniMindForCausalLM.forward` 再用 `logits[..., :-1, :]` 与 `labels[..., 1:]` 计算 loss。
按蓝图再错位一次会改变学习目标。

SFT 同样由模型 shift；DPODataset 自己拆出 x/y；Omni 另有九通道布局和对应标签构造。
应绘制“原 token / 输入 / 标签 / loss mask”对齐表，逐条训练路径确认，不建立全书通用 shift 假设。

证据：[文本数据集](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/dataset/lm_dataset.py)
的 `PretrainDataset.__getitem__`、`SFTDataset`、`DPODataset.__getitem__`；
[模型](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/model/model_minimind.py)
的 `MiniMindForCausalLM.forward`。

### 2.2 配置不是定律，Dense 与 MoE 是分支

`hidden_size=768`、8 层、8 个查询头、4 个 KV 头和 6400 词表是当前默认配置。
`intermediate_size=ceil(hidden_size*pi/64)*64` 可作为实现事实解释，不能直接称为“黄金定律”。
`use_moe=False` 为默认值，`MiniMindBlock` 在 Dense FFN 与 MoE 之间选择，不能画成必经的两个模块。

“GQA 节省一半内存”需收窄：相同其他条件下，4 个 KV 头相对 8 个 KV 头的缓存元素量减半，
不表示整个模型、整个注意力计算或训练峰值显存减半。还需检查 `repeat_kv` 后续张量布局。

证据：上述模型文件中的 `MiniMindConfig`、`Attention.forward`、`MiniMindBlock.__init__`。

### 2.3 SDPA 不等于已使用 FlashAttention

本地 `self.flash` 检查 API 存在性和配置开关，实际调用还受序列长度、缓存与 mask 条件影响。
即使调用 `F.scaled_dot_product_attention`，也不能仅凭 Python 函数名断言具体硬件内核。
写作应分别讲本项目的分支选择与依赖库的后端选择；后者需结合依赖版本、设备和测量验证。

证据：上述模型文件中的 `Attention.__init__` 与 `Attention.forward`。
YaRN 的 2048、factor=16 与长度缓冲同样只是配置与计算路径，不能据此保证 32k 上下文任务质量。

### 2.4 本地 LoRA 没有显式 alpha/r

通用 LoRA 公式可以先介绍，但本实现直接返回 `B(A(x))`，没有显式 `alpha/r`。
`apply_lora` 只选输入输出维相同的 Linear，也不是对所有投影层统一注入。
`merge_lora` 导出时做 FP16 转换与加法，数学上的可合并性不能描述为浮点结果“完全无损”。

证据：[model_lora.py](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/model/model_lora.py)
的 `LoRA.forward`、`apply_lora`、`merge_lora`。

蒸馏蓝图的 CE/KL 组合方向与当前代码主结构一致，但正式章还需交代温度、有效 token mask、
teacher/student 词表对齐和 MoE 辅助损失，不能只放一条总公式。
证据：[train_distillation.py](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/trainer/train_distillation.py)。

### 2.5 视觉 SFT 默认不是全部解冻

预训练脚本默认 `freeze_llm=2`，只训练 projector；SFT 默认 `freeze_llm=1`，
在 projector 之外解冻语言模型首尾 block。即使取 `freeze_llm=0`，视觉编码器仍冻结。
每个阶段都应列出“模块 / requires_grad / 优化器是否包含 / 权重来源”。

证据：[预训练入口](https://github.com/jingyaogong/minimind-v/blob/740d467ece78a0b7d2d976fcb424472095d4a688/trainer/train_pretrain_vlm.py)、
[SFT 入口](https://github.com/jingyaogong/minimind-v/blob/740d467ece78a0b7d2d976fcb424472095d4a688/trainer/train_sft_vlm.py)、
[init_vlm_model](https://github.com/jingyaogong/minimind-v/blob/740d467ece78a0b7d2d976fcb424472095d4a688/trainer/trainer_utils.py)。

256×256、patch=32、64 个 patch token 与默认编码器配置对应，不写成所有视觉输入的恒定规律。
`count_vision_proj` 按连续占位符段替换特征，需讲清多图、段长度与截断；“原位替换”是序列位置上的描述，
实现用了 `torch.cat`，不能据此断言是原地内存写入。
证据：[model_vlm.py](https://github.com/jingyaogong/minimind-v/blob/740d467ece78a0b7d2d976fcb424472095d4a688/model/model_vlm.py)。

### 2.6 Bridge 的层编号及最佳性

`OmniConfig.bridge_layer` 默认 `num_hidden_layers // 2 - 1`；8 层时为 3。
`forward` 在零基循环 `i==3` 的 block 执行后保存状态，即第 4 个 block 后。
将其写成“第 3 层”会产生 off-by-one 理解错误。“中间层最适合指导声学”只能作为待验证的解释，
若要陈述最优性需引用能支持该结论的消融实验并注明其范围。

证据：[model_omni.py](https://github.com/jingyaogong/minimind-o/blob/ccab6d98520ac5ac5d7583a3fa21a65105137e28/model/model_omni.py)
的 `OmniConfig` 与 `MiniMindOmni.forward`。

### 2.7 八路码本不能省略 delay pattern

`TalkerHead` 的确输出共享 base 加八个 adapter 的 logits，但数据和生成存在分码本的时间错位。
模型复合输入是八个音频通道加一个文本通道，不能只画成一个平铺的多模态 token 串。
需要用对齐网格解释训练输入、音频标签、参考音频、speaker 占位、文本结束、音频结束及帧重组。
“MTP”在这里的码本接口含义，也不能直接等同于文本模型一次预测多个未来 token。

证据：上述模型文件中的 `TalkerHead`、`TalkerEmbedding`、`MiniMindOmni.stream_generate`；
[omni_dataset.py](https://github.com/jingyaogong/minimind-o/blob/ccab6d98520ac5ac5d7583a3fa21a65105137e28/dataset/omni_dataset.py)。

### 2.8 三种数据任务不等于只有三次固定训练

T2A → A2A → I2T 可保留为能力接入主线，但 `trainer/train.sh` 的 full 建议命令含多次阶段切换，
包括 projector 单独对齐、联合训练及任务回访；mini 路线又不同。
应分开记录数据任务、`mode`、`freeze_backbone`、初始化权重和训练预算。
冻结的外部编码器与 codec 也意味着“端到端”需要写明具体指哪些路径。

证据：[训练入口](https://github.com/jingyaogong/minimind-o/blob/ccab6d98520ac5ac5d7583a3fa21a65105137e28/trainer/train_sft_omni.py)、
[建议训练命令](https://github.com/jingyaogong/minimind-o/blob/ccab6d98520ac5ac5d7583a3fa21a65105137e28/trainer/train.sh)。

### 2.9 Agent 工具包含模拟反馈

天气、时间、汇率和翻译等由 `WEATHER_DATA`、`TIME_DATA`、`EXCHANGE_DATA`、`MOCK_RESULTS`
等本地数据构造反馈；数学工具包含受限 `eval`。这可以教学多轮交互，但不是联网查询真实环境的证据。
交互和奖励逻辑位于 `train_agent.py`，`rollout_engine.py` 抽象本地/服务采样后端。

证据：[train_agent.py](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/trainer/train_agent.py)
中的 `execute_tool`、`rollout_single`、`calculate_rewards`；
[rollout_engine.py](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/trainer/rollout_engine.py)。

### 2.10 打断需要追到服务端和播放端

`RealtimeSession.push_chunk` 产生 `interrupt` 事件，只解释这个类不足以说明完整交互。
真实 Web 路径还在 `webui/web_demo.py` 中轮询输入、中止生成并通知客户端。
要分别画采音、VAD、生成、发送、解码/播放及清理队列；未做运行验证前不承诺无缝双工或具体延迟。

证据：上述 `model_omni.py` 的 `RealtimeSession`；
[webui/web_demo.py](https://github.com/jingyaogong/minimind-o/blob/ccab6d98520ac5ac5d7583a3fa21a65105137e28/webui/web_demo.py)
的 `realtime`/`poll_interrupt`，以及
[webui/web_demo.html](https://github.com/jingyaogong/minimind-o/blob/ccab6d98520ac5ac5d7583a3fa21a65105137e28/webui/web_demo.html)
的 `initCall`、`stopPcm`、WebSocket 消息处理与 `stopRT`。浏览器端收到中断后的播放停止是闭环的一部分。

## 3. RMSNorm 样章尚不能作为质量金标准

样章具有直觉、公式、形状与源码四层结构，这个结构值得保留。需要修正：

- “减均值对稳定性几乎没有贡献”“必然节省 10%～50% 时间”缺少限定实验来源，不作为本书结论。
- 不能从 `torch.rsqrt` 推导出 CPU/GPU 上固定单条指令或一定快于另一种写法，编译、融合和 dtype 都需核实。
- `x.float()` 不只是“防下溢”；这里应解释低精度平方、归约的范围和精度问题，并避免把 BF16 与 FP16 混为一谈。
- 独立 `RMSNorm` 默认 `eps=1e-5`，模型配置默认传入 `1e-6`，两者都存在，应说明调用位置。
- 全 1 权重和输入均方值明显大于 epsilon 是输出 RMS 接近 1 的充分条件；不是训练后永远成立的性质。
- `rms.weight.requires_grad` 不证明已经产生梯度；需执行 backward，检查 `.grad` 是否存在且有限。
- 原示例 `from minimind.model...` 依赖本地目录加入搜索路径，不能不说明运行目录就称为完全独立。
- 不需要用户手动“还原反向传播梯度 dtype”；应回到 autograd 与源码中的前向类型转换解释。
- NaN 有多种来源，不能以“通常因为 epsilon 太小”概括诊断。示例应加入零输入与数值边界。

下一步先重写此短样章，完成小数字推演、断言与反向验证，作为教学试样；
另加一页复杂公式、长代码和跨页图表用于出版压力测试。两者都通过后再固定模板。

## 4. 编辑与出版补充

修订目录增加了张量入门、训练闭环、评估、PPO/GRPO 前置概念和音频基础；
把 12 个大章拆为 20 个较小章，视觉及音频路线可绕过 RL。
评估至少区分训练 loss、独立验证集、生成质量、资源与延迟；单次生成样例不能替代质量评估。

出版从第一份样章开始建设：同一份 Markdown 内容清单，EPUB 与 PDF 两个后端，
数学与图源可维护、资源离线可用、内容逐项一致，并做阅读器和页面视觉检查。
具体规范见 [AGENTS](../AGENTS.md#11-epub-与-pdf-出版契约)，实施进度见 [README](../README.md#写作与出版里程碑)。

本次没有重写原蓝图文件，也没有把它的精确行号、技术性能断言或未运行输出迁入正式书稿。

## 5. 第二轮设计复审与修复（2026-09-08）

本轮对象是初次生成的 AGENTS 和 README。保留四篇二十章及单一 Markdown 内容源，
修复以下具体缺口；这不代表开始撰写章节或已经验证出版工具链。

| 问题 | 影响 | 已落实的修改 |
|---|---|---|
| 第 01 章指向真实评测入口，未区分随机模型观察与有质量的生成 | 初学者可能先被权重下载和环境阻塞 | 明确 CPU、合成输入、小配置、随机权重；评测入口只做导航 |
| “随用随讲”没有首次讲解位置，第 06 章又提前带入 DPO | 容易出现未学目标先读数据、先遇公式再补概念 | 新增前置知识与学习结果表；DPO 数据归到第 11 章；03/04 明确抽象与展开边界 |
| LoRA 与蒸馏合章缺少关系说明 | 容易误读为固定连续训练流程 | 保持两个独立小节及实验，并补低秩、温度、KL 前置 |
| 第 20 章漏列浏览器源码 | 仅看服务端不能说明音频播放如何真正停止 | 加入 `webui/web_demo.html` 与播放/消息处理符号，静态核对相关代码 |
| 所有类型结论共用“源码优先”的证据排序 | 可能让实现反过来定义数学，或用代码猜性能 | 分开规定实现、数学、依赖、性能证据 |
| HEAD 变化就要求章节 stale | 破坏固定版本书籍的可维护性 | 仅在迁移未验证或证据出错时 stale；补锁定快照读取与 dirty 引用规则 |
| RMSNorm 近单位 RMS 被写成必要条件，且遗漏小输入边界 | 将充分条件写成“只有”，零输入也可能被误判 | 改成充分条件，说明稳定项相对大小和零输入输出 |
| 评估有目标但缺操作口径 | 可能把训练总 loss、生成样例当质量评估 | 补留出数据、mask/归约口径、不同目标分开报告及分模态测量 |
| 多文件章节、元数据和双后端转换责任不明确 | 可能重复入书、顺序漂移或把 PDF 公式一并变成图像 | 清单列有序文件，元信息做一致性校验；共用 AST 分支复制 |
| 内容比对主要描述中间稿 | 打包丢失正文仍可能通过 | 最终 EPUB 解包回读，PDF 文本与页面并验；校验器用损坏样本反证 |
| 未区分样章预览与正式版门槛 | 要么样章被完整清单阻塞，要么草稿被当整书发行 | 设计 preview/release 模式，保留检查，标明范围；双格式同批发布 |

复审后，书籍主线、跳读路线和交付阶段相互一致。后续实施优先完成 M1 的教学与排版样章，
通过真实 CPU 实验及两种格式检查再推进 M2。静态设计检查不能替代这些运行验收。

本轮实际校验：使用 Python 标准库检查 3 份 Markdown 的围栏和块公式分隔符、9 处本地链接/锚点、
19 个去重后的固定源码引用、三个仓库 HEAD、20 章顺序及浏览器取消播放的关键源码文本，全部通过。
没有执行浏览器交互、训练或 EPUB/PDF 导出；公式与图示的最终视觉效果仍由 M1 验收。
