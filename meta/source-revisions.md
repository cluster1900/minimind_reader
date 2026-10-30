# 源码与研究环境基线

2026-09-08 再次读取本地 Git HEAD。源码根目录默认 `../minimind_resources`，
也可由 `MINIMIND_SOURCE_ROOT` 指定。上游只读。

| 仓库 | commit | 分支 |
|---|---|---|
| jingyaogong/minimind | `6fc918beb68a0d8c40452338df6319fe168014ba` | master |
| jingyaogong/minimind-v | `740d467ece78a0b7d2d976fcb424472095d4a688` | master |
| jingyaogong/minimind-o | `ccab6d98520ac5ac5d7583a3fa21a65105137e28` | master |

运行实验前再检查 dirty 状态；代码与文档的版本绑定不能只依赖表中的日期。
初始记录时尚无 Git 历史，曾以输入文件 SHA-256 标识具体内容。

## 2026-10-08 同基线复核

三个 HEAD 仍与上表完全一致，当前均为 detached HEAD、工作区 clean。本次没有升级、切分支或修改上游。
书稿已有 Git 历史，本次开始于 `5d3f6a81b9eedd7eca0b3742721a5576e210f19d`，且已有未提交修订；
保留这些修订并逐项复核。修订后的实际文件哈希与实验见 [本次复核](source-audit-2026-10-08.md)。
出版版本仍为 1.0.1；本轮是同一源码基线上的书稿勘误，不是源码版本迁移。


## 2026-10-18 候选迁移（bot-hub #10）

书稿基准提交为 `835d88d166b880f00e596aee6ddbffcfa0489306`。逐章迁移而非全书静默换版：
MiniMind 02、04、07—13 改绑 `f659b55761b754d306bd140573493a6543cafd7f`；V 的 15 改绑
`1862b633fc082a723e78dbead9777545f09d960c`；O 的 19 改绑 `f900448c608318c53314ebf8a947ab05cd8c038e`。
其余九章与旧实验记录保持原引用。受影响章节暂为 stale，cpu 为 not_run，未宣称新数值验证通过。
本次使用 GitHub 固定版本读取，不曾接入、切换或检查用户桌面工作区；不继承历史的本地 clean 声明。
详见 [影响评估、证据与复验要求](upstream-impact-2026-10-18.md)。本次不更新电子书或正式发布版号。

## 2026-10-28 候选迁移（bot-hub #102）

在 2026-10-18 候选终点之上继续定点迁移：
MiniMind `f659b55761b754d306bd140573493a6543cafd7f` → `233d00fa068ec2b8c0d4f07b703934d6391a5a73`（13 commits，改 07/08/10/13）；
MiniMind-V `1862b633fc082a723e78dbead9777545f09d960c` → `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc`（1 commit，改 15 评估入口说明）；
MiniMind-O `f900448c608318c53314ebf8a947ab05cd8c038e` → `b5d1bc11ece8de17404cbc034c3e09718d0fecdf`（1 commit，改 19）。
受影响章保持 stale、cpu: not_run。详见 [upstream-impact-2026-10-28.md](upstream-impact-2026-10-28.md)。不更新电子书正式版号。

## 2026-10-30 CPU 复验（bot-hub #111）

按 [复验清单](upstream-impact-2026-10-18.md) 第 3–6 条，在 torch `2.6.0+cpu` 上对 02、04、07—13、15、19 共 11 章实跑配套实验，
基线为 MiniMind `233d00fa…`、MiniMind-V `041d528e…`、MiniMind-O `b5d1bc11…`；未沿用旧 CPU 结果。
11 章均改为 `verified`（`verified_at: 2026-10-30`，`cpu: passed`）。12 章的 SGLang 右补齐用受控 HTTP 响应、DDP 用 CPU gloo 测，
未启动真实 SGLang 服务，也未走 NCCL/多 GPU。不更新电子书正式版号。
02、04、09、11、12 章的 frontmatter、正文链接与来源映射统一钉到 MiniMind `233d00fa…`（实验即在该提交运行；`f659b557…`→`233d00fa…` 未改这些章引用的文件）。
内容复核记录见 [content-review-2026-10-30.md](content-review-2026-10-30.md)。
