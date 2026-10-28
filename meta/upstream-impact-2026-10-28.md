# MiniMind / V / O 指定上游提交：影响评估与迁移候选（2026-10-28）

作者：Coding（Grok Bot，credit Hawk Wu）。任务：[bot-hub #102](https://github.com/cluster1900/bot-hub/issues/102)。
仅主题分支 PR，供 review / Normal Work 审核；不直接改 main、不自动合并、不宣称新版实验已通过。

## 结论与交付边界

在 2026-10-18 候选基线之上继续定点迁移。本轮比较终点由 issue 指定：

| 仓库 | 旧候选基线 | 新候选基线 | 提交数 / 变化文件 |
|---|---|---|---|
| MiniMind | `f659b55761b754d306bd140573493a6543cafd7f` | `233d00fa068ec2b8c0d4f07b703934d6391a5a73` | 13 / 8 |
| MiniMind-V | `1862b633fc082a723e78dbead9777545f09d960c` | `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc` | 1 / 1 |
| MiniMind-O | `f900448c608318c53314ebf8a947ab05cd8c038e` | `b5d1bc11ece8de17404cbc034c3e09718d0fecdf` | 1 / 2 |

比较链接：[M](https://github.com/jingyaogong/minimind/compare/f659b55761b754d306bd140573493a6543cafd7f...233d00fa068ec2b8c0d4f07b703934d6391a5a73)、[V](https://github.com/jingyaogong/minimind-v/compare/1862b633fc082a723e78dbead9777545f09d960c...041d528e037a8a373c3c8aa7278a3ba3bd26b6dc)、[O](https://github.com/jingyaogong/minimind-o/compare/f900448c608318c53314ebf8a947ab05cd8c038e...b5d1bc11ece8de17404cbc034c3e09718d0fecdf)。

受影响章节：07、08、10、13、15、19。均保持 `stale` / `cpu: not_run`；未把 2026-10-18 或更早的 CPU 结果当作新版验证。01–06、09、11、12、14、16–18、20 主线无相反陈述，正文不改；12 章上游仅 README 文档澄清 GRPO loss 曲线读法，公式与训练器未改，故不扩写。

## 逐 commit 影响（MiniMind）

非 merge 实质提交 7 个；6 个 merge 仅合入下列补丁。`233d00fa` 修剪合并残留（删流式 tool 测试与若干注释），**保留** `86aad506` 的流式 tool_call 与 content 分流逻辑及 `043b22a`/`dcbd31b`/`695aa1f`/`cd36b4a` 的行为修复。

| Commit | 摘要 | 影响章节 | 处理 |
|---|---|---|---|
| `dcbd31bf3d6412f0be25bb9bbcda958c1f665c4d` | train_agent：畸形 tool JSON / 非 dict 参数不再崩溃 | 13 | 改写 13.2：数组 JSON 现被丢弃；arguments 非 dict 按空参 |
| `86aad5062149897174a66395e86371ee6ec05ab7` | serve_openai_api：流式时 tool_call 不进 content | （记录） | 现有 20 章无相反陈述；E-API 记入非章节变更，不扩写部署教程 |
| `7bde9752c125759be9bf3be33cde1b2204055681` | README：GRPO policy loss 曲线读法 | 12 | 上游文档；训练器/公式未改，正文不改 |
| `cd36b4a0137490b31cfe535ba8cdb3869a44c457` | load_from 以目录名 `model` 判定原生加载，不再子串匹配 | 08（及 eval_toolcall/API） | 08 评估入口补充；路径含 `model` 子串但末段不是 `model` 时走 transformers |
| `695aa1fdb6bb7ac9fa31cf83ae0b7990c7476d99` | `--lora_rank`（默认 16）传入 `apply_lora` | 10、08 | 10.3 与 eval/API 加载路径同步 |
| `043b22a07326dbdacde770580956fb6b7f75066c` | resume 权重保持 CPU FP32；推理 `*.pth` 另存 FP16 | 07 | 改 7.6 表：resume model ≠ 推理权重 dtype |
| `233d00fa068ec2b8c0d4f07b703934d6391a5a73` | trim leftovers（删测试/注释；LoRA rank 补到 serve） | 上列 | 行为以本 SHA 为准核对 |

## V / O 各 1 commit

| 仓库 | Commit | 变更 | 影响 |
|---|---|---|---|
| MiniMind-V | `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc` | `eval_vlm.py`：同上 basename 判定 | 15：评估入口；`vlm_checkpoint` **未**随 M 改 resume dtype |
| MiniMind-O | `b5d1bc11ece8de17404cbc034c3e09718d0fecdf` | `eval_omni.py` basename；`omni_checkpoint` resume 改 CPU FP32（仍排除 encoder） | 19.7：resume 与推理权重 dtype 分离 |

核对行号均相对各自新 HEAD：`lm_checkpoint` M:65、`parse_tool_calls` M:76、`apply_lora(..., rank=)` train_lora.py:122、`omni_checkpoint` O:108。

## 证据要点（相对新 HEAD）

- **E-FP32**：`lm_checkpoint` / `omni_checkpoint` 先 `{k: v.cpu()}` 作为 `resume_data['model']`，`torch.save({k: v.half() ...})` 只写推理 `*.pth`。旧版把 `.half().cpu()` 后的同一 dict 同时写入 resume，续训会静默抹掉小于半个 fp16 ulp 的更新。
- **E-LORA**：`train_lora.py` / `eval_llm.py` / `serve_openai_api.py` 增加 `--lora_rank`（默认 16），`apply_lora(model, rank=args.lora_rank)`。
- **E-BASENAME**：`os.path.basename(os.path.normpath(args.load_from)) == 'model'` 替代 `'model' in args.load_from`（M eval/API/toolcall；V eval_vlm；O eval_omni）。
- **E-AGENT**：`parse_tool_calls` 仅 `isinstance(call, dict)` 才收录；`rollout_single` / `calculate_rewards` 对非 dict arguments 置 `{}`。
- **E-API**：`generate_stream_response` 用 pending/`in_tool_call` 把 `<tool_call>…</tool_call>` 发到 `delta.tool_calls`，不进入 `content`；`233d00fa` 删除配套测试文件，运行时逻辑仍在。

## 未扩大范围 / 验证缺口

- 未重跑任何 PyTorch/GPU/DDP/真实音频实验；旧 `experiments/*` 数值仅作历史。
- 未改 9 章无影响正文与旧 commit 绑定；未批量替换全书 SHA。
- 仓库无 PR 级 CI workflow（仅 pages-build-deployment）；本 PR 以本地 `check_book` / 构建检查为准。
- 完整 EPUB/PDF 依赖 mermaid CLI、字体子集、epubcheck 等；本环境跑通原稿结构检查与可达构建步骤，不把未安装工具的缺省当成书稿错误。

## 复验清单（合并/出版前）

1. 用新 HEAD 工作副本重跑 ch07 resume dtype（resume 为 fp32、推理 pth 为 fp16）与末批保存顺序。
2. ch10：`--lora_rank` 与训练/评估一致；错 rank 加载应失败或明显错输出。
3. ch13：`<tool_call>[1]</tool_call>` 与非对象 JSON 不再触发 `.get` 崩溃。
4. ch08/V/O：`load_from` 路径含 `foo_model_bar` 但末段不是 `model` 时应走 transformers。
5. ch19：确认 resume 不含 encoder 且为 fp32；与 ch15（V resume 仍为完整 state_dict）对比不要混写。
