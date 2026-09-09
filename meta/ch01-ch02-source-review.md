# 第 01、02 章源码与教学复核

日期：2026-09-08；源基线 MiniMind `6fc918beb68a0d8c40452338df6319fe168014ba`，工作区 clean。

第 01 章重新核对了 `MiniMindConfig` 与模型 `forward`，以及 `eval_llm.init_model/main`。
书中明确区分随机 FP32 小模型和真实入口的权重加载、half、eval 与生成，不将一次前向称为完成训练。
标量导数按平方误差与链式法则手算，再由真实反向计算核对；矩阵例子的行列约定与 Linear 权重一致。
配套实验检查张量索引、广播、Linear、loss、概率归一化和有限梯度，全部通过。

第 02 章逐段读取 `train_tokenizer.py`、当前 `tokenizer_config.json` 的模板和属性，
并检查 `SFTDataset.create_chat_prompt` 的连接位置，SFT 的监督 mask 留到第 06 章。
关键修订为区分单独 ID 与 special 属性，工具的执行与模板包装，以及 tokenizer 配置长度与模型能力。
实际模板后缀、ID、解码、BPE 小训练由实验验证，未替换上游词表。

Markdown 额外复核：特殊 token 表格中的竖线使用 GFM 转义，配套解析测试确认生成的单元格仍为
`<|im_start|>`，没有把内容错误拆成多个列；数学内容仍使用 `$...$` 与 `$$`。

两章每图附读图说明，每个公式就近说明变量，章末提供自检参考解释和可选修改练习。
章节内容检查与出版显示检查分开记录；后者见出版报告。
