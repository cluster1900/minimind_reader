# 附录 B 运行与验证方法

本书将上游源码作为只读基线，实验数据与输出写入书稿项目的 build 目录。
从项目根目录执行命令；不要在多个仓库共用的全局 Python 搜索路径里混合导入同名 model 包。

## 两套验证环境

文本、后训练、视觉和出版使用 Python 3.12 的 `.venv`；音频实验使用独立 `.venv-omni`。
依赖分别由 `requirements-book.lock` 与 `requirements-omni.lock` 固定。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-book.lock
python3.12 -m venv .venv-omni
.venv-omni/bin/python -m pip install -r requirements-omni.lock
```

安装需要网络，已经准备好的小实验通常不再下载模型权重。
声明使用随机模型或替身的实验，不会因为安装了依赖就变成默认预训练模型测试。
CPU 验证环境是 macOS arm64；在其他平台复现时应记录差异，不能直接沿用本机的性能或数值结论。

## 源码路径与版本

默认源码根目录为相邻的 `minimind_resources`；也可以设置 `MINIMIND_SOURCE_ROOT`。
其中的 minimind、minimind-v、minimind-o 都要保持与章节对应的 commit 一致。

| 仓库 | 本书基线 |
|---|---|
| MiniMind | `6fc918beb68a0d8c40452338df6319fe168014ba` |
| MiniMind-V | `740d467ece78a0b7d2d976fcb424472095d4a688` |
| MiniMind-O | `ccab6d98520ac5ac5d7583a3fa21a65105137e28` |

修改代码做练习时，另建自己的工作副本，并记录修改内容。
上游发布新版本不自动使旧版书籍无效，但迁移后必须重新验证受影响的结论。

## 怎样读实验记录

每章在 experiments 目录中有脚本、JSON 结果和说明。先看 scope，再看断言与实际输出。
例如源码 AST 片段实验只验证相应算术，脚本化 Agent 只验证控制流；
随机小 SigLIP/Mimi 则验证真实网络的接口与张量，不证明预训练能力。

```bash
.venv/bin/python experiments/ch08_sft_evaluation.py
.venv-omni/bin/python experiments/ch18_audio_codes.py
```

训练损失、实际生成、GPU 性能和设备交互分别记录。若当前机器无法运行某种资源，
保留待测范围，不伪造输出，也不把相关条件悄悄从结论中删去。

## 出版结果怎样核对

正文来自有序出版清单，不按文件系统顺序拼接。
构建会检查源码符号、链接和原生数学，回读最终 EPUB 的正文、代码、表格、公式与图资源，
并检查 PDF 的文字与页面。页数和字数相同不足以证明内容相同。
书稿仓库的出版说明给出工具版本与检查命令；个人阅读环境仍需支持 EPUB3 或 PDF。
