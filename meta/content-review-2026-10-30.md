# 2026-10-30 内容复核记录（bot-hub #111）

范围：相对上次复核快照（[2026-10-08](source-audit-2026-10-08.md)，对应提交 `835d88d`）有正文、公式或图变化的全部文档，
即第 02、04、07、08、09、10、11、12、13、15、19 章与 `book/appendices/source-index.md`。
变化的公式/图条目为 `ch09-math-015`、`ch09-diagram-002`、`ch13-diagram-001`；其余文档的正文与条目哈希未变。

方法：逐章阅读 `git diff 835d88d -- <章节>`，把每条新增或改写的技术断言与锁定提交的上游源码（`git show <commit>:<file>`）
及本轮 `experiments/chNN-results.json` 对照。基线为 MiniMind `233d00fa068ec2b8c0d4f07b703934d6391a5a73`、
MiniMind-V `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc`、MiniMind-O `b5d1bc11ece8de17404cbc034c3e09718d0fecdf`，三个工作区均干净。
实验为 torch `2.6.0+cpu`、无 CUDA；每份结果 JSON 的 `source_commit` 与上述基线一致。

| 文档 | 核对的源码断言 | 实验佐证 | 结论 |
| --- | --- | --- | --- |
| 02 | `train_tokenizer.get_texts` 默认 `max_lines=MAX_LINES=0`、优先 `text` 字段、空输入抛 `ValueError` | `ch02-results.json`（空输入错误、计数） | 通过 |
| 04 | `Attention.forward` 第 128 行 `(xq @ xk.transpose(-2,-1)).float() / sqrt(head_dim)`、SDPA 分支条件 | `ch04-results.json`（误差约 `8.94e-8`、fp16 溢出键） | 通过 |
| 07 | `step % accumulation_steps == 0 or step == iters` 更新位于保存前；`lm_checkpoint` resume 存 CPU 状态、推理 `.pth` 为 `.half()` | `ch07-results.json`（resume float32、推理 float16） | 通过 |
| 08 | `eval_llm` 以目录名等于 `model` 分流、`moe_suffix`、`lora_rank` 默认 16、`EOFError/KeyboardInterrupt` 退出 | `ch08-results.json`（NLL `0.1137` / `6.7956`） | 通过 |
| 09 | `MOEFeedForward` Top-1 直通 `top1 - top1.detach() + 1.0`；`ch09-math-015` 解析梯度 ±0.48；新图与代码路径一致 | `ch09-results.json`（`0.009189`、输入梯度 `assert_close`） | 通过 |
| 10 | `train_lora` 用 `apply_lora(rank=args.lora_rank)`；LoRA/蒸馏 `step == iters` 尾批更新并删除循环外 flush；`serve_openai_api`/`eval_llm` 的 LoRA 路径去掉 `lora/` 子目录并加 `moe_suffix`；`model_lora.py` 与 `distillation_loss` 未改 | `ch10-results.json` | 通过 |
| 11 | `DPODataset.__getitem__` 单次抽取 `remove_empty_think` 同时作用于 chosen/rejected；`train_dpo` 尾批更新 | `ch11-results.json` | 通过 |
| 12 | Torch rollout `full_mask` 拼接；PPO/GRPO 按 `logp_pos + 1` 用 `completion_mask` 回填 attention mask；DDP `broadcast_buffers=False`，训练前向经包装后的 actor/critic；SGLang 适配器左补 0.0 log-prob、批张量右补齐 | `ch12-results.json`（PAD/EOS mask、SGLang 右补齐、gloo `world_size=2`） | 通过 |
| 13 | `parse_tool_calls` 只收 dict；非 dict 参数置 `{}`；计算器改用 `trainer_utils.safe_math_eval`（AST 白名单、512 字符、幂运算守卫），`execute_tool` 无 SIGALRM、异常返回 None；marker 提取观察与长度不符 `RuntimeError`；EOS mask 为 0；`seq_lens` 构造 attention mask；`ch13-diagram-001` 与代码一致 | `ch13-results.json`（71/39/32 token、奖励 3） | 通过 |
| 15 | V 两个训练器 `step == iters` 尾批更新、循环外 flush 删除；`eval_vlm` 目录名分流；`vlm_checkpoint` 未改；语言主干同步 FP32 分数与 Top-1 直通 | `ch15-results.json` | 通过 |
| 19 | `omni_checkpoint`（约 108 行）排除编码器，推理 `.pth` 为 `.half()`、resume 存过滤后的 CPU 状态；`eval_omni.py` 第 19 行目录名分流；`train_sft_omni` 尾批更新位于保存前 | `ch19-results.json` | 通过 |
| source-index | 去除 40 位 SHA 后与 `835d88d` 版本逐行相同，仅为 #110 的链接 SHA 刷新 | — | 通过 |

限制：本复核只覆盖上表变化内容，依赖 CPU 小规模实验与源码阅读；未启动真实 SGLang 服务、NCCL/多 GPU 或完整训练，
不证明训练质量或 GPU 行为。`meta/content-review.json` 仅记录本次复核时的内容快照，用于发现后续未复核的变化。
当时其余 9 章仍保留旧提交 SHA，不在 #111 范围内；现已由下方 bot-hub #116 补充覆盖。

### 补记：`ch07-math-013` 已从清单删除

#111 按当前 Pandoc 清单重写 `meta/content-review.json` 时，过时的 `ch07-math-013` 从公式条目中消失。它不是显示公式，而是旧 §7.7（「实测一个保存时机的边界」）里作为标量参数占位的行内 `$w$`。
Inventory payload 为 `InlineMath` 换行后加 `w`，哈希 `da88938b2680676a440c3ecf9b379bd7fd89eaabce992a8a4be72c89ee054865`。

公式本身在 `54c5d93`（bot-hub #10）改写 §7.7 时离开章节，math 节点 13→12。`7a5fed4`（bot-hub #111）只是按当前章节重新 inventory，因此丢掉了滞留的 id；#111 自己的 ch07 Markdown diff 没有删掉任何公式。本记录原先漏记这一删除，现补记；该条目对应的公式已不在正文中，无需恢复。

## bot-hub #116 补充

范围：此前仍绑定旧 SHA 的 9 章（01、03、05、06、14、16、17、18、20），以及对外基线说明（`README.md`、`docs/index.html`、`book/appendices/reproduction.md`）。
方法：逐个引用文件对比新旧 SHA 的 `git diff` 并按 AST 核对正文引用的符号（逐章结论见下表及 PR 说明），以及各章 `experiments/chNN.md` 的 2026-10-30 复验节与 `experiments/chNN-results.json`。
基线同 #111：MiniMind `233d00fa068ec2b8c0d4f07b703934d6391a5a73`、MiniMind-V `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc`、MiniMind-O `b5d1bc11ece8de17404cbc034c3e09718d0fecdf`。各章结果 JSON 的 `source_commit` 与上述提交一致。

| 文档 | 核对的源码断言 | 实验佐证 | 结论 |
| --- | --- | --- | --- |
| 01 | `MiniMindConfig` / `MiniMindForCausalLM.__init__` / `forward` 与旧 SHA 相同；`eval_llm.init_model` 仍 `half().eval().to(device)`，`main` 仍先 `init_model` 再 tokenize / `generate`。文件级 diff 在 Attention/MoE 与 loader/LoRA/Ctrl-C，不否定本章断言。无 `#L` 锚点。 | `ch01-results.json`（`source_commit` 新基线；logits `[2,4,32]`，loss ≈ `3.571819`，参数 19648） | 通过 |
| 03 | `RMSNorm.norm`/`forward`、`FeedForward.forward`、`MiniMindBlock` 残差、权重 tying 与旧 SHA 相同。Attention fp16 与 MoE Top-1 STE 属第 04/09 章。 | `ch03-results.json`（RMSNorm 手算、输入梯度、FFN 6144、权重共享、logits `[1,4,32]`） | 通过 |
| 05 | `rotate_half`、`start_pos` 切片、`generate` 的 `past_len` 切片与 Top-P 右移与旧 SHA 相同。 | `ch05-results.json`（RoPE 误差约 `2.38e-7`，缓存误差约 `7.45e-8`/`2.24e-8`，`resumed_generation_matches=true`） | 通过 |
| 06 | `PretrainDataset.__getitem__` 仍 `labels = input_ids.clone()`、`max_length - 2`；`MiniMindForCausalLM.forward` 仍 `logits[..., :-1, :]` / `labels[..., 1:]`。`post_processing_chat` 增 DPO 参数，SFT 默认调用路径与空思考移除比例仍约 80%。 | `ch06-results.json`（未 shift 标签、PAD→`-100`、shift_loss ≈ `8.741042`、截断全 `-100` 时 loss 为 NaN） | 通过 |
| 14 | MiniMind-V `model/model_vlm.py` 新旧 SHA 字节相同：`VLMConfig`、`count_vision_proj`、`get_image_embeddings`（`vision_model(**image_inputs)`）、`forward`。 | `ch14-results.json`（logits `(1, 7, 6400)`，projector 有梯度、编码器无梯度，缓存误差约 `1.19e-7`） | 通过 |
| 16 | `model_omni.py` / `omni_dataset.py` 新旧 SHA 相同：`encode_audio_inputs` 仍丢弃编码器长度；`process_audio` 仍读 WAV、下混、重采样到 16 kHz 再送 processor。 | `ch16-results.json`（800 点 8 kHz 立体声 → 1600 点 16 kHz；`encoded_shape = [3, 32]`；编码器报告长度 1 仍保留 3 帧） | 通过 |
| 17 | `OmniConfig.bridge_layer` 默认仍 `num_hidden_layers // 2 - 1`（8 层 → 索引 3）；`TalkerModule` 与 `MiniMindOmni.forward` 未改。 | `ch17-results.json`（`default_bridge_index = 3`，tiny 为 1；文本 logits 不随输出音频历史变化） | 通过 |
| 18 | `stream_generate` 仍用 `text_finished` 与默认 `enter_token_id=201` / `pad_token_id=0`；delay 仍 `assistant_start + layer_idx + 1`；`eval_sample` 函数体相同，调用方仍不传 enter/pad。`eval_omni.init_model` 的目录名分流不在本章描述范围。 | `ch18-results.json`（`dataset_shape = [9, 127]`，八路首标签 20–27；流式 12/15 步重建三帧；`201` 不是换行） | 通过 |
| 20 | `SileroVAD`、`RealtimeSession.push_chunk`/`get_audio`/`reset`、`web_demo.py` 的 `realtime`/`run_generate`/`stream_pcm` 以及 HTML `stopPcm` 均与旧 SHA 相同。 | `ch20-results.json`（VAD 窗 1024；interrupt 保留 2048；stop/speech `interrupted = true`；浏览器停止 2 个节点） | 通过 |
| README.md | 当前基线表改为三仓库新 SHA、二十章均绑定；#111/#116 复验已完成。2026-10-08 锁与 2026-10-18 候选改为历史表，并保留两份 impact 文档链接。已去掉「待审核 / 保留旧版绑定 / 等待新版 CPU 复验」。 | 与本表各章 `source_commit` 及 #111 表一致 | 通过 |
| docs/index.html | 基线行 href 与 7 字符短链改为 `233d00f` / `041d528` / `b5d1bc1`。 | 与当前锁定 commit 一致 | 通过 |
| reproduction.md | 附录 B 基线表改为新 SHA。补充说明 2026-10-30 复验在 Linux、torch 2.6.0+cpu 上完成，音频章使用装有相同音频依赖的 Python 3.12 环境（未按 3.11 重建）；`.venv-omni` 仍为音频命令入口。注明早期验证在 macOS arm64。 | ch16–20 实验记录 `source_commit` 为 `b5d1bc11…` | 通过 |

限制：本补充只覆盖上表 9 章的已核对机制与对外基线说明，证据是 CPU 小配置实验与静态 diff/AST，不证明 GPU 行为、训练质量、语音/视觉质量或真实设备时延。MiniMind 文件级 diff 中的 Attention fp16 与 MoE Top-1 STE 仍归第 04/09 章，不在这 9 章的断言范围内。`ch07-math-013` 的删除已在上方补记。
