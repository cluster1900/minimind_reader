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
其余章节保留的旧提交 SHA 不在本轮范围内。
