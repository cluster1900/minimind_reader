# 2026-10-08 全书源码阅读复核

本轮交付为 Markdown 书稿勘误、配套实验及证据更新。已通读二十章、阅读指南、术语表和附录，
按“入口 → 数据 → 模型 → loss/生成 → 输出”核对所讲机制；修订后再次从调用点检查条件、轴和归约口径。
这次复核由同一作者另一次沿入口检查完成，没有把自动哈希校验称为独立语义审阅。

## 版本与工作区

| 仓库 | 实际 HEAD / 书稿绑定 commit | 状态 |
|---|---|---|
| minimind | `6fc918beb68a0d8c40452338df6319fe168014ba` | detached HEAD，clean |
| minimind-v | `740d467ece78a0b7d2d976fcb424472095d4a688` | detached HEAD，clean |
| minimind-o | `ccab6d98520ac5ac5d7583a3fa21a65105137e28` | detached HEAD，clean |

上游只读，无 pull、切换或修改。本书开始于 `5d3f6a81b9eedd7eca0b3742721a5576e210f19d`，
当时章节 04、05、12、14、15、17～20、一致性记录、内容快照及 package-lock 已有未提交修改。
保留并复核已有正文修订，未改动本轮开始前的 package-lock 差异。
修订后的书稿尚未提交，不能只用该 HEAD 表示本轮内容；实验快照记录实际输入 SHA-256。

## 逐章核对与修订

表中“静态”表示固定源码及调用链证据，“CPU”表示声明配置下的实际实验。
源码完整符号与固定链接维护于 [源码映射](chapter-source-map.json) 和书中 [源码索引](../book/appendices/source-index.md)。

| 章 | 对照与结果 |
|---|---|
| 01 | 随机模型、张量广播/矩阵乘法、真实 backward、输出与标签形状重新核对；原结论保留。 |
| 02 | 区分 tokenizer 配置中的边界字符串与模型配置的数值 ID，核对模板、换行、BPE 和 special token 行为。 |
| 03 | 修正 GQA 整除方向：查询头数可被 KV 头数整除；说明 head_dim 可以独立覆盖，不能无条件用 D/Hq。 |
| 04 | 保留因果 mask 的尾部切片；核实带缓存多 token 也走显式分支。非方阵 SDPA 的左上角语义由 PyTorch 2.6 文档与 CPU 反例支持，分支效果与作者动机分开。 |
| 05 | 区分 forward 的未缓存尾部输入与 generate 的完整序列输入；携缓存续写验证一致。补充缓存列表层数、带 layers 的对象被丢弃及右 padding 从最后一列采样的边界。 |
| 06 | 修正“SFT padding 无条件忽略”：标签扫描依赖结束序列，没有独立按 PAD ID 清除。复现缺失结束边界的 3 个 PAD 被监督，以及全空目标 mean CE 的 NaN。 |
| 07 | 复核标签 shift 所属、累积/裁剪/更新、尾部不足累积边界、保存字段与恢复能力；保留原结论。 |
| 08 | 复核真实 SFT 循环及独立小模型训练/留出 NLL 口径；原质量边界保留。 |
| 09 | 修正图 9-1，明确原 token 向量进入专家，Top-K 分别提供选择编号与权重；负载辅助项的可微路径重新核对。 |
| 10 | 澄清 merge_lora 另存合并权重，不会拆掉当前模型适配器；新增方形输出头与输入 Embedding 共享时，整网不等价的反例。普通配置的合并容差和蒸馏 KL 实验仍通过。 |
| 11 | 明确 train_epoch 默认 beta=0.1、CLI 默认 0.15、dpo_loss 无默认 beta；DPO mask 扫描全部 assistant 段，不能无条件说只评分最后候选。 |
| 12 | 复核 Critic 二次 norm/共享权重、CISPO detach 与归约；补充 early-stop 后仍可能 step、no_grad 不等于 eval、EOS mask、远端概率补零和按间隔同步权重的边界。 |
| 13 | 区分合法 JSON 与工具调用结构：数组能解析却在 .get 时报错；Agent 的轨迹均值排除空行，与 GRPO 分母不同；工具反馈仍为原模拟集合。 |
| 14 | 修正普通 Tensor 像素接口的支持描述：锁定版本会在 **kwargs 展开处抛错；字典四维/五维路径实测可用。相邻占位段使 10 个位置缩为 6 的旧反例保留。 |
| 15 | 复核冻结模式对应参数数量、训练状态与编码器梯度；补齐 labels 未 shift、VLM forward shift/归约。明确字典输入、空图片列表与不同图数 stack 的限制。 |
| 16 | 区分前端长度、输入给编码器的长度与编码器返回长度；源码丢弃最后一种。替代编码器返回 1，包装仍取 3 帧；不推断真实 SenseVoice 出错。 |
| 17 | 保留已有修订：无效 Bridge 索引留下进 block 前的状态，可能已经包含模态注入；新增音频注入实验，避免旧记录将它概括为原始 Embedding。 |
| 18 | 复核八路 delay、shift、EOS 后更新标志、201/0 写回前 break；说明 Dataset 仅搜索前 50 个位置的思考结束起点。区分累计文本与单帧音频、非流式最后一次 yield，以及生成器迭代的梯度上下文和单样本限制。 |
| 19 | 保留正确的冻结范围、随机 ID 扰动和 stop 权重说明；补充 token 截断前后“最后一轮”可能不一致、缺失权重路径跳过加载、全空标签零 loss。随机扰动不等同经典 Scheduled Sampling。 |
| 20 | 区分 16 kHz 发送块与原生采样回调；补齐 speech_end 与 interrupt 都会提前返回、get_audio 不等于 reset，以及浏览器依据服务端 VAD 消息停止播放。 |

共用材料同步修订：术语英文/中文、跨仓库标签对齐速查、实验历史 JSON 与本次输出的区别、
源码索引，以及 Python 3.12 文本/视觉环境和 Python 3.11 音频环境的实际复现命令。

## 新增反例的实测结果

以下是机制验证，不是对模型质量或性能的评估。容差沿用各脚本的显式断言；离散 ID/布尔条件则精确比较。

| 位置 | 输入或条件 | 实际观察 |
|---|---|---|
| ch04 | 查询 2、键 4，分数为零、V=1～4 | SDPA 得到 1、1.5；缓存尾部用显式路径，与全量结果在容差内一致 |
| ch05 | 先生成 3 个，完整序列与缓存续写 3 个 | 与一次生成 6 个一致；右 padding 取最后列的采样分布 |
| ch06 | 缺少结束序列，尾部 3 个 PAD | 3 个均被监督；完整结束边界时 0 个被监督 |
| ch10 | V=D=32，共享输出头，明确非零 LoRA 增量 | 输入 Embedding 最大改变量约 0.1251，logits 最大差约 1.7163；普通配置合并误差约 0.000159 |
| ch12 | 两个生成 ID、仅一项 -0.7 log-prob | 记录为 `[0.0,-0.7]`，两位置 mask 都为 1；HTTP 为受控替身 |
| ch13 | tool_call 内是合法 JSON 数组 | 解析通过，真实 rollout 在 `.get` 处抛 AttributeError |
| ch14 | 相同像素，映射与裸 Tensor | 映射四维/五维 logits 一致，裸五维 Tensor 抛 TypeError |
| ch16 | 输入有效长度 3、替代编码器报告 1 | 包装仍投影 3 帧 |
| ch17 | 单层 index=-1，合成音频特征注入 | Bridge 保留注入值，而非该位置原 token Embedding |
| ch18 | 三帧脚本化 logits，非流式消费 | 仅返回 `(None, [12,112,212,312,412,512,612,712])`，并非完整音频；非流式内部关闭梯度 |
| ch19 | 文本和八路标签均为 -100，有限 logits | 文本 loss=0、音频 loss=0；不表示存在有效监督 |
| ch20 | 足够静音触发 speech_end 后，同块还有 2048 点语音 | 提前返回，块末 2048 点未被消费 |

## 验证与证据

统一复验 **20/20 通过**，其中 12 个章节脚本新增了边界断言；
正文结构/链接/公式解析/固定源码符号检查覆盖 26 个出版输入，检查通过。
内容契约、Markdown 解析与审阅契约共 **16 项测试通过**。图 9-1 已重新生成 SVG/PNG，
并目视检查箭头、中文与标签；这不代替新版电子书页宽检查。

完整执行时间、命令、依赖清单、输入哈希和逐章输出保存在
[实验快照](source-audit-2026-10-08-experiments.json)，其输入哈希保留运行开始时的状态。
运行后仅对第 12/13/15 章及术语表做措辞精确化，未改实验代码、公式或运行条件；
最终审阅内容另由 `meta/content-review.json` 绑定，不改写实验时的原始输入记录。
本次检查的命令、结果及日志哈希另见 [检查快照](source-audit-2026-10-08-checks.json)。

从项目根目录执行的主要命令为：

```bash
.venv/bin/python scripts/verify_experiments.py
.venv/bin/python -m unittest discover -s tests -p test_content_contract.py -v
.venv/bin/python -m unittest discover -s tests -p test_markdown_inputs.py -v
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false .venv/bin/python -m unittest discover -s tests -p test_review_contract.py -v
.venv/bin/python scripts/record_content_review.py --record meta/source-audit-2026-10-08.md
.venv/bin/python scripts/check_book.py --edition release
git diff --check
```

这里的 `check_book --edition release` 是全书输入与内容快照门禁，并不表示已经构建或发布新版电子书。

- 运行目录：书稿仓库根目录。
- 数据/权重：脚本内合成数据、真实随机小模型或各脚本明确的替身；不下载大模型权重。
- 随机种子：各脚本为 42，执行真实源码的独立进程避免同名包污染。
- 本机：macOS 26.6.2 arm64，CPU；文本/视觉 Python 3.12.13，音频 Python 3.11.15，PyTorch 2.6.0。
- 自动检查只覆盖结构、链接、公式解析、源符号和内容变更检测；上述语义由人工对照与机制实验支持。
- 首轮 ch06 边界夹具未将 max_length 与人工输入长度对齐，已修正夹具；ch18 最后 yield 的初始预期也按实际执行修正为最后一帧。未改上游来迎合预期。
- 审阅契约测试的临时 Git 提交最初受本机全局 GPG 签名设置影响而失败；使用仅对子进程生效的 `commit.gpgsign=false` 重跑，不改用户 Git 配置。

## 尚未验证与出版状态

GPU 训练、默认视觉/语音权重质量、真实 SGLang/奖励服务、声卡/VAD 模型识别效果和端到端打断延迟未运行。
对应章节只声明 CPU 算术、接口或受控状态机结论，因此 GPU 为 `not_required`；不把这些检查当作质量证明。

本次没有重新导出或发布完整 EPUB/PDF。`publishing/book.yaml` 的 1.0.1 仍描述历史发行版，
`meta/final-review.json`、`dist/`、在线下载及旧实验 JSON 均未被当作修订后产物的验收证据。
更新内容快照只代表正文复核完成；新版页面、资源、内容回读与阅读器检查仍需另行完成。
