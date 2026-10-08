# 第 02 章 tokenizer 实验记录

- 日期：2026-09-08；源 commit：`6fc918beb68a0d8c40452338df6319fe168014ba`。
- 命令：项目根目录运行 `.venv/bin/python experiments/ch02_tokenizer.py`。
- 环境：本项目 Python 3.12.14、Transformers 4.57.6、tokenizers 0.22.2；无需 GPU。
- 输入：仓库自带词表、三条固定文本、用户/assistant/tool 定义样本；另用 16 条合成文本训练隔离的小词表。
- 原始词表及模板未修改，文件哈希与具体输出见 [原始结果](ch02-results.json)。

已有词表长度 6400；BOS/EOS/PAD ID 分别为 1/2/0。
“你好，MiniMind！”为 12 字符、8 token；“hello world”为 11 字符、2 token；空格与换行完整往返。
`<think>`、`<tool_call>` 在 skip_special_tokens 解码中保留，`<|image_pad|>` 被跳过。

模板直接编码与先格式化再编码的 ID 完全一致；思考开关产生相应的开段/空段后缀，
运行字符串含真实换行，不含字面量反斜杠加 n。assistant 历史消息含模板插入的思考段；
传入工具定义后 system 说明中包含工具名。

调用真实训练函数，目标词表 384，有限语料实际得到 364 项。小词表能往返“你好 hello”，
但没有替换原词表或加载已有模型权重；不对其编码质量作优劣判断。

## 2026-10-08 复核

按原实验重新运行，并重新核对正文中使用的入口、数值与结论范围。
旧 JSON 保留历史结果；本次完整命令、环境、脚本哈希及实际输出见
[同基线复核记录](../meta/source-audit-2026-10-08.md)及其结果快照。
