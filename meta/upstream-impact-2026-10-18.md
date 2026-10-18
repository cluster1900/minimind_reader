# MiniMind / V / O 指定上游提交：20 章影响评估与迁移候选

作者：dot（OpenAI 助手，不是用户本人）。2026-10-18。
任务：[bot-hub #10](https://github.com/cluster1900/bot-hub/issues/10)。供 Normal Work 审核，不能直接合并或出版。

## 结论与交付边界

20 章已逐一核对 README 的“全书设计”入口、源码映射及完整上游差异。
11 章存在需要改写或补充的实现说明，已给出定点正文修改；9 章的教学主线不受影响，正文与旧基线不动。
受影响章节仅迁移源码引用和有关段落，统一暂标 `stale` / `cpu: not_run`，不把历史 CPU 结果冒充新版验证。
本次不是整书重新发行，也没有修改上游源码、用户桌面、发布物或 main 分支。

书稿比较基准：[`835d88d166b880f00e596aee6ddbffcfa0489306`](https://github.com/cluster1900/minimind_reader/tree/835d88d166b880f00e596aee6ddbffcfa0489306)。
已获取的 150 个书稿/实验/出版/元数据文件逐个通过 Git blob SHA 校验，补丁以这些原始字节为基准。
目标源码 28 个文件也按 GitHub Git tree blob SHA 核验；清单见 [源码哈希](upstream-migration-sources-2026-10-18.json)。

## 固定比较范围与证据

| 仓库 | 旧基线 | 候选基线 | 提交数 / 变化文件数 |
|---|---|---|---|
| MiniMind | `6fc918beb68a0d8c40452338df6319fe168014ba` | `f659b55761b754d306bd140573493a6543cafd7f` | 35 / 18 |
| MiniMind-V | `740d467ece78a0b7d2d976fcb424472095d4a688` | `1862b633fc082a723e78dbead9777545f09d960c` | 3 / 3 |
| MiniMind-O | `ccab6d98520ac5ac5d7583a3fa21a65105137e28` | `f900448c608318c53314ebf8a947ab05cd8c038e` | 3 / 2 |

比较链接：[M](https://github.com/jingyaogong/minimind/compare/6fc918beb68a0d8c40452338df6319fe168014ba...f659b55761b754d306bd140573493a6543cafd7f)、[V](https://github.com/jingyaogong/minimind-v/compare/740d467ece78a0b7d2d976fcb424472095d4a688...1862b633fc082a723e78dbead9777545f09d960c)、[O](https://github.com/jingyaogong/minimind-o/compare/ccab6d98520ac5ac5d7583a3fa21a65105137e28...f900448c608318c53314ebf8a947ab05cd8c038e)。
与 [materials/10](https://github.com/cluster1900/bot-hub/tree/main/materials/10) 的文件列表和每文件变更统计相符。
这是 issue 指定的 2026-09-23 观测终点，不声称是 2026-10-18 最新 HEAD。

证据编号：
- E1：三库 `model/model_minimind.py` 只改 `Attention.forward` 与 `MOEFeedForward.forward`。新文件 Git blob 均为 `3f23fdfbe61a7015c141f9c458b1f6ee04b824d0`。显式分数在缩放/mask 前 `.float()`；Top-1 使用 detached 输入重算 gate 的直通梯度。
- E2：M `trainer/train_tokenizer.py:get_texts` 支持 text/conversations、有效文本条数上限、空语料报错；现有 `model/tokenizer_config.json` 未改。
- E3：M `dataset/lm_dataset.py:DPODataset.__getitem__` 共用一次空思考删除决策；`post_processing_chat` 新增可选参数。未指定该参数的 SFT 默认行为保持原样；Pretrain/SFT 类的 AST 不变。
- E4：M pretrain/full_sft/lora/distillation/dpo/grpo/agent、V pretrain_vlm/sft_vlm、O sft_omni 共 10 个训练器，将尾组更新移到 `step == iters` 的循环内，先更新后末批保存，删除循环外 flush。尾组仍除以配置累积数；组内周期保存仍未保存梯度。
- E5：M `rollout_engine.py:TorchRolloutEngine.rollout` 的 attention mask 改为 prompt mask 加生成段全 1；PPO/GRPO 回填 completion mask；PPO/GRPO/Agent 训练前向不再绕过 DDP，使用 `broadcast_buffers=False`。
- E6：M `train_agent.py:rollout_single` 保留采样 ID，completion_mask 取有效前缀；EOS 留在上下文但策略 mask=0，marker 提取观察，长度不符抛错。训练 attention mask 按序列长度而非 PAD ID 构造。
- E7：M `trainer_utils.safe_math_eval` 替换 Agent、eval_toolcall 和 web_demo 的计算器 eval；AST/算术/math 白名单、长度和幂规模保护。Agent 移除 SIGALRM，不再有通用一秒超时。
- E8：M eval_llm 与 API 服务的 LoRA 路径补 MoE 后缀；API 去掉额外 lora 子目录，非流式生成改用 max_new_tokens。eval_llm 捕获 EOF/键盘中断；依赖新增 fastapi/uvicorn。
- E9：完整 V diff 不包含 `model_vlm.py`、dataset、trainer_utils；完整 O diff 不包含 `model_omni.py`、`omni_dataset.py`、`eval_omni.py`、`train.sh`、trainer_utils 或两个 web_demo 文件。Git tree 对这些入口的旧/新 blob 相等。

## 20 章逐一结论

“主线无影响”只指下列已声明机制，不等于底层网络所有数值、梯度或可选配置完全不变。
尤其 V/O 的共享语言主干仍受 E1 影响；没有把文件未改误写成全模型逐位等价。

| 章 / 正文路径 | 结论 | 源码入口、证据及处理 |
|---|---|---|
| 01 `book/01-start-here/README.md` | 主线无影响；不改 | M 模型配置、构造、forward 的 AST 不变；随机权重/shape/标量梯度说明不依赖新增分支。eval_llm 只作导航，E8 不改变导航事实。E1 的数值细节归第 04/09 章。保留旧基线实测。 |
| 02 `book/02-tokenizer/README.md` | 受影响；已改 | E2 改写 2.3：不再最多原始 10000 行，只计实际产出文本，双格式与空语料报错。已有 tokenizer 配置与 ID 未变化。 |
| 03 `book/03-model-skeleton/README.md` | 主线无影响；不改 | MiniMindConfig、RMSNorm、FeedForward、MiniMindBlock、MiniMindModel、MiniMindForCausalLM 的 AST 不变；Embedding/残差/输出头、RMSNorm 的 FP32 转换与 SwiGLU 未改。Attention 内部 E1 不是本章承诺细节。 |
| 04 `book/04-attention/README.md` | 受影响；已改 | E1 改写显式注意力 FP32 转换时点；强调点积先算，不能声称 QK 乘法强制 FP32或修复既有溢出。GQA、QK-Norm、mask 语义、SDPA 选择条件未改。 |
| 05 `book/05-rope-cache-generation/README.md` | 主线无影响；不改 | RoPE 函数、MiniMindModel 缓存及 MiniMindForCausalLM.generate 的 AST 不变；E8 的 API 包装器长度修复不是本章 generate 本体。E1 可影响低精度数值，但未改位置/缓存/采样控制流，旧数值记录不迁移。 |
| 06 `book/06-datasets/README.md` | 主线无影响；不改 | E3 中 PretrainDataset/SFTDataset 类 AST 不变，SFT 未传新参数，默认随机分支语义相同；标签、shift、截断和 mask 不改。DPO 成对随机性在第 11 章处理。 |
| 07 `book/07-pretraining/README.md` | 受影响；已改 | E4 改写 7.2/7.3/7.7：末批更新先于保存，移除“新版末批保存落后”的错误结论，保留旧版历史数值并区分周期保存残留风险。 |
| 08 `book/08-sft-evaluation/README.md` | 受影响；已改 | full_sft 的 E4，以及 eval_llm 的 E8，补充入口行为和新末批顺序；NLL/PPL、过拟合定义及 SFT 默认超参未改，旧训练数值仅作历史记录。 |
| 09 `book/09-moe/README.md` | 受影响；已改 | E1 重写 Top-1 梯度解释、图和自检答案；K>1 公式保留，明确 gate 有代理梯度、该分数路径 detach 输入但专家/辅助项输入仍有路径。未声称代理梯度是常数函数的真实导数。 |
| 10 `book/10-lora-distillation/README.md` | 受影响；已改 | E4 两训练循环、E8 加载路径补充。model_lora.py 未改，低秩公式/FP16 合并边界不改；distillation_loss 与目标计算不变。 |
| 11 `book/11-dpo/README.md` | 受影响；已改 | E3 替换“chosen/rejected 独立随机、相同回答监督 5/11”为历史反例与新版共享决策；E4 补末批保存顺序。DPO 公式、shift、归约和空 mask 结论未改。 |
| 12 `book/12-ppo-grpo/README.md` | 受影响；已改 | E5 补 prompt/completion attention mask 与损失 mask 的区别、DDP 调用；E4 的 GRPO 尾组顺序。PPO/GRPO/CISPO 目标公式、概率来源和 SGLang 已知边界未改。 |
| 13 `book/13-agent/README.md` | 受影响；已改 | E6 替换按累计长度截取重编码上下文的旧解释，保留真实采样 token，明确 EOS/marker/有效前缀契约；E7 替换 SIGALRM/eval 说明；E4/E5 更新训练边界。原 70/39/31 token 计数标为旧版，不造新版结果。 |
| 14 `book/14-vision/README.md` | 主线无影响；不改 | E9：V model_vlm 与 dataset blob 未变，图像预处理/patch/projector/占位替换与首轮图像接入未改。共享 Attention/MoE 变化归 E1，不改变这些注入接口。 |
| 15 `book/15-vlm-training/README.md` | 受影响；已改 | E4 的两个视觉训练器，补末批保存和尾组缩放；交代 E1 主干变化。E9 确认冻结策略、checkpoint 字段与图像 collate 未改。 |
| 16 `book/16-audio-basics/README.md` | 主线无影响；不改 | E9：O model_omni 与 omni_dataset blob 未变，采样率/前端/特征注入/增强及模态 ID 说明不改；不把共享主干变化当音频前端变化。 |
| 17 `book/17-thinker-talker/README.md` | 主线无影响；不改 | E9：OmniConfig、TalkerModule、MiniMindOmni.forward 所在文件未改，Bridge 层索引、speaker 融合、缓存排列和分叉不改；若评估 E1 对数值/梯度的影响仍需新实验。 |
| 18 `book/18-audio-codebooks/README.md` | 主线无影响；不改 | E9：delay pattern 数据、stream_generate、eval_omni 文件未改；八路错位、停止规则、Mimi 接口不改。旧实验没有据此转成新版音质或生成数值证据。 |
| 19 `book/19-omni-training/README.md` | 受影响；已改 | E4 改 19.7 的尾部更新与保存顺序。E9 确认 train.sh、trainer_utils 未改；loss/冻结/初始化/模态流水线段落不重写，共享主干 E1 的行为与第 04/09 章一致。 |
| 20 `book/20-realtime/README.md` | 主线无影响；不改 | E9：RealtimeSession、web_demo.py/html 入口旧/新 blob 相同，VAD/中断/播放队列/双工边界不改。M scripts/web_demo.py 的 E7 不是 O webui/web_demo.py，不能混淆两者。 |

## 非章节变更及未扩大的范围

README 与 `meta/source-revisions.md` 增加候选基线表；`meta/chapter-source-map.json` 仅改上述 11 章的 commit。
未将全书旧 commit 批量替换为新 commit，未改 9 章正文，也未抹掉原实验/历史审阅文件。
新增 `tests/test_upstream_migration.py` 与本记录、源码哈希、检查结果用于审阅，不进入出版清单。

E8 中 API max_new_tokens、FastAPI/Uvicorn 依赖的修复未在现有 20 章正文形成相反陈述，故记录而不扩大章节范围。
E7 的 Web/CLI 计算器修复已在 Agent 安全边界中解释，不创建新的部署教程。
所有发布到目标仓库的改动必须在主题分支与 PR 中审核；不直接推 main，不自动合并。

## 实际执行的验证

- 已读取仓库 AGENTS、全书目录/源码地图、20 章有关主线与 diff，并用 GitHub compare 核验变更文件/统计。
- 150 个原书稿文件 Git blob 校验通过；28 个新源码文件 Git blob 校验通过；三个新 model_minimind.py 逐字节相同。
- 对旧/新 M AST 比较，配置/RMSNorm/Dense FFN/block/model/LM 外壳、RoPE、PretrainDataset/SFTDataset 相同，支持第 01/03/05/06 章“不改”的范围判断。
- 标准库执行真实 `get_texts`、`post_processing_chat`、`safe_math_eval` 函数 AST：双语料格式、非空条目计数、空语料报错、显式 true/false 决策，以及正常算术与不支持/过大表达式拒绝通过。
- 标准库替身执行真实 pretrain/full_sft 的 `train_epoch` AST：3 小批、K=2，更新两次且最后清梯度后保存；标量最终值 0.07435。替身不实现 PyTorch、FP16 舍入、AMP 或 GPU，不能替代正式数值实验。
- 10 个训练器 AST 检查确认循环内最后 step 更新与移除外部 flush；DDP/Agent、FP32/Top-1 新调用点静态检查通过。
- 另使用原检查器的纯函数核验 20 章元信息与依赖图，并以系统 Pandoc 解析全部 20 章；数学分隔符、代码围栏及九章原文字节不变检查通过。这是部分结构检查，不是完整出版检查。
- unified diff 已在精确原始书稿快照上通过 `git apply --check`；补丁没有改动任何未列入清单的章节。
- 完整输出见 [focused 检查结果](upstream-migration-checks-2026-10-18.json)。运行：`python tests/test_upstream_migration.py --source-root <包含三个候选源码目录的路径>`。

完整 `scripts/check_book.py` 实际尝试在导入阶段失败：`ModuleNotFoundError: No module named pypandoc`，尚未执行到其完整源码/内容审阅校验。
当前执行环境没有 torch、transformers、tokenizers，也没有完整旧/新 Git 源码对象库；未运行原 20 章 PyTorch 实验、GPU/分布式训练、真实编解码/播放、SGLang 或完整出版验收。
不存在“全部测试已过”的结论。章节 `static: passed` 仅表示本次差异与正文静态核对通过；`cpu: not_run` 明确保留正式实验空缺。

## Normal Work 合并/出版前复验清单

1. 审核 11 章定点段落及候选/旧基线并存设计；若要统一迁移全书，另做基线更新与相关实验，不自动扩大本补丁。
2. 在隔离源码副本保留每章声明的 Git commit，使用 requirements-book.lock 对应环境；不要覆盖历史实验结果文件或伪造干净状态。
3. 更新 ch02 双格式与空语料测试；ch04 低精度显式 attention 测试；ch09 Top-1 gate/input/expert/aux 四路实际反向测试。
4. 更新 ch07 的旧 commit 和“保存时仅一次更新”断言，分别测试尾组、整组、周期组内保存、start_step 续训与空 loader；在 V/O 训练循环复验相同边界。
5. ch11 改为成对共享决策断言；ch12 测有效 PAD ID、EOS、prompt 左 padding、SGLang 右补齐与 DDP；ch13 脚本化后端补 completion_mask，测 EOS 保留但不计策略 loss、marker 丢失、log-prob 长度错误、非连续 mask 契约及观察边界。
6. 跑其余受影响章节已有实验，更新对应哈希与内容审阅记录；运行 `scripts/check_book.py` 的正式检查、内容证据检查与所需出版校验，成功前保持 stale，不发布新版 EPUB/PDF。

这些是验收缺口，不是要求用户购买 GPU或把未运行项归咎于正文任务本身；本 PR 提供可以审核的精确候选修改与可重跑的 focused 验证。
