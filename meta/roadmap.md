# 全书实施进度

本文件记录实际交付状态，出版顺序由 `publishing/book.yaml` 管理。

| 阶段 | 状态 | 验收证据 |
|---|---|---|
| M0 设计与两轮复核 | completed | `blueprint-review.md` |
| M1 教学样章与双格式出版试验 | completed | `m1-publication-review.md`；34 页合并预览、EPUBCheck、内容检查及显示复核 |
| M2 文本基础 01～08 | completed | `m2-publication-review.md`；8 章正文与实验、79 页预览、格式/内容/显示检查 |
| M3 后训练与视觉 09～15 | completed | `m3-publication-review.md`；7 章与实验、126 页合并预览、内容与显示核验 |
| M4 音频与交互 16～20 | completed | `ch16-ch20-source-review.md`；五章正文、真实源码/受控替身 CPU 实验全部通过 |
| M5 全书源码复核及 EPUB/PDF 发布 | completed | `final-review.md` / JSON；194 页 PDF、完整 EPUB、20 章 CPU 实验、29 项出版及一致性测试、两种 EPUB 阅读引擎及最终内容/页面核验 |

本书的完整目标保持不变：二十章正文、通俗解释、图示及读图说明、Markdown 原生数学、
配套实验、源码逐章复核、最终 EPUB/PDF 与 Markdown 一致性验证。
短样章与预览文件仅用于建立管线，不作为完整交付。

1.0.1（2026-09-09）完成持续一致性复审：显式损失分母、关键图依赖、长公式分行，固定 Git 对象读取、正文/图式复核快照、PDF 章节顺序检查与可复现发布记录收集。详见 `consistency-review.md`。
