# 附录 B 运行与验证方法

本书将上游源码作为只读基线，实验数据与输出写入书稿项目的 build 目录。
从项目根目录执行命令；不要在多个仓库共用的全局 Python 搜索路径里混合导入同名 model 包。

## 两套验证环境

文本、后训练、视觉和出版使用 Python 3.12 的 `.venv`；音频实验默认使用 Python 3.11 的独立 `.venv-omni`。
依赖分别由 `requirements-book.lock` 与 `requirements-omni.lock` 固定。
2026-10-30 按当前基线的 CPU 复验在 Linux、torch 2.6.0+cpu 上完成；其中音频章 16—20 是在装有 `requirements-omni.lock` 依赖的 Python 3.12 `.venv-omni` 上重新验证的；上文与出版说明记载的音频环境为 Python 3.11，本次未用 Python 3.11 重建或复跑。结果 JSON 中的 `torch` 字段（ch06 不记录该字段）为 `2.6.0+cpu`：这些结果由装有 CPU 版 PyTorch wheel（`2.6.0+cpu`）的 Python 3.12 环境直接运行实验脚本写出；发布前用 `scripts/verify_experiments.py` 复跑时，01—15 章使用的仓库 `.venv` 装的是 `2.6.0+cu124` wheel（本机无 CUDA，按 CPU 运行），因此 `build/final-audit/experiments.json` 与 `meta/final-experiments.json` 对这些章记录 `2.6.0+cu124`。两者都满足锁定文件的 `torch==2.6.0`，复跑结果除 `torch` 字段外与提交的 JSON 一致。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-book.lock
python3.11 -m venv .venv-omni
.venv-omni/bin/python -m pip install -r requirements-omni.lock
```

安装需要网络，已经准备好的小实验通常不再下载模型权重。
声明使用随机模型或替身的实验，不会因为安装了依赖就变成默认预训练模型测试。
早期 CPU 验证环境是 macOS arm64；2026-10-30 复验在 Linux CPU 上完成。在其他平台复现时应记录差异，不能直接沿用某次本机的性能或数值结论。

## 源码路径与版本

默认源码根目录为相邻的 `minimind_resources`；也可以设置 `MINIMIND_SOURCE_ROOT`。
其中的 minimind、minimind-v、minimind-o 都要保持与章节对应的 commit 一致。

| 仓库 | 本书基线 |
|---|---|
| MiniMind | `233d00fa068ec2b8c0d4f07b703934d6391a5a73` |
| MiniMind-V | `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc` |
| MiniMind-O | `b5d1bc11ece8de17404cbc034c3e09718d0fecdf` |

修改代码做练习时，另建自己的工作副本，并记录修改内容。
上游发布新版本不自动使旧版书籍无效，但迁移后必须重新验证受影响的结论。

## 怎样读实验记录

每章在 experiments 目录中有脚本、历史 JSON 结果和说明。重新运行时结果写到
`build/experiments/chNN-results.json`，不会覆盖历史 JSON。先看验证范围（scope），再看断言与实际输出；
当前修订的复核记录会标出采用哪次运行，不能把旧结果的脚本哈希当成修订后脚本的证据。
例如源码 AST 片段实验只验证相应算术，脚本化 Agent 只验证控制流；
随机小 SigLIP/Mimi 则验证真实网络的接口与张量，不证明预训练能力。

```bash
.venv/bin/python experiments/ch08_sft_evaluation.py
.venv-omni/bin/python experiments/ch18_audio_codes.py
```

训练损失、实际生成、GPU 性能和设备交互分别记录。若当前机器无法运行某种资源，
保留待测范围，不伪造输出，也不把相关条件悄悄从结论中删去。

## 跨仓库的标签对齐速查

学完第 06 章后，可以用此表核对后续训练器。表中结论绑定各章声明的固定 commit，详细证据见对应章节。
shift 指“用当前位置预测下一个 token”，只应在一条训练路径中做一次。

| 路径 | 谁做 shift | 谁选择监督区 | 谁归约目标 |
|---|---|---|---|
| [文本 Pretrain/SFT](../06-datasets/README.md) | 模型 forward | Dataset 的 labels | 模型按有效 token 平均 CE |
| [文本 DPO](../11-dpo/README.md) | DPODataset | Dataset 的 loss_mask | 训练器先对 token 求和，再算偏好目标 |
| [视觉训练](../15-vlm-training/README.md) | VLM forward | VLMDataset 的 labels | VLM 按有效 token 平均 CE |
| [Omni 训练](../19-omni-training/README.md) | OmniDataset | 文本与八路音频各自标签 | 训练器分别归约，再加辅助项 |

注意力 mask 决定模型可以读取哪些位置，loss mask/忽略标签决定哪些预测直接计入目标，
`requires_grad` 决定哪些张量需要求导，`train/eval` 则影响 dropout 等层的行为。
这四者不能互相替代：prompt 的标签被忽略时，回答仍可以读取它并向其表征反传；
冻结视觉编码器也不等于永远保持 eval。

## 出版结果怎样核对

正文来自有序出版清单，不按文件系统顺序拼接。
构建会检查源码符号、链接和原生数学，回读最终 EPUB 的正文、代码、表格、公式与图资源，
并检查 PDF 的文字与页面。页数和字数相同不足以证明内容相同。
书稿仓库的出版说明给出工具版本与检查命令；个人阅读环境仍需支持 EPUB3 或 PDF。
