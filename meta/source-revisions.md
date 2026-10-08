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
