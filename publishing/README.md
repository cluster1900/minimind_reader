# 出版与复核

全书出版顺序由 `book.yaml` 管理，该文件采用 JSON 语法（合法 YAML）。
正文只维护一套 Markdown；EPUB 的 SVG 数学与 PDF 的原生数学分别从共用 AST 的副本生成。
当前完整发行版为四篇二十章、阅读指南、术语表及四个附录。最终验收记录见 `meta/final-review.md`。

## 已实现的命令

从项目根目录执行，Python 包分别固定于 `requirements-book.lock` 和 `requirements-omni.lock`：

```bash
.venv/bin/python scripts/check_book.py --edition release
.venv/bin/python scripts/verify_experiments.py
.venv/bin/python scripts/subset_book_font.py
.venv/bin/python scripts/build_book.py --edition release --format all
BOOK_TEST_EDITION=release .venv/bin/python -m unittest discover -s tests -v
node scripts/check_epub_display.cjs release
node scripts/check_epubjs.cjs
node scripts/check_foliate.cjs
```

构建器只生成 `build/release/` 中的候选文件，自动执行最终 EPUB 内容回读、EPUBCheck 和 PDF 文本检查。
还需检查 PDF 页面和实际阅读器，再生成绑定该批输入/产物哈希的最终审阅记录。
`scripts/publish_book.py` 检查该记录及当前文件，两个格式都准备并核验后才更新 dist，并最后写入发布清单。
修改正文后旧审阅记录不能继续授权新产物，需重新构建与检查受影响部分。

预览允许显式选章和 draft，文件名、封面标题与描述明确实际范围：

```bash
.venv/bin/python scripts/build_book.py --edition preview --chapters ch01 ch02 ch03 --smoke --format all
```

`--smoke` 加入排版附页，与章节形成多文件夹具，检查重复标题、跨文件链接、脚注、矩阵和长代码。
它不进入正式书。脚注损坏测试使用保留的预览夹具；其他产物反例测试针对完整发行候选。

## 固定工具与字体

- Python 3.12，PyTorch 2.6.0、Transformers 4.57.6；出版包见 Python 锁定文件。
- Node 包使用 `package-lock.json`；安装方式为 `npm ci --ignore-scripts`。
- Pandoc 3.9、MathJax 3.2.2、Mermaid CLI 11.17.0。
- PDF：Tectonic 0.17.0（XeTeX），A5、11 pt；TeX 资源缓存位于 `.cache/tectonic`。
- EPUBCheck 5.3.0，需要 Java。字体检查另用 Poppler，页面栅格比较用 PyMuPDF。
- 中文字体：项目 `assets/fonts/NotoSansCJKsc-Regular.otf`；代码字体：Source Code Pro。
  EPUB 嵌入改名后的 `MiniMindBookCJK.otf` 子集；PDF 嵌入中文、代码、Latin Modern 及图中文字字形。
  来源、版权与许可见书中附录和 `assets/licenses/`，实际哈希见发布清单。
- 浏览器配置在 `puppeteer.json`，当前指向本机 Chrome；换机器时须设置自己的路径并重验。

Tectonic [官方 0.17.0 发布包](https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-aarch64-apple-darwin.tar.gz)
下载 SHA-256 为 `a3f1cac7c5678f01661a92212f58480ae3b0634115d880dbc59e2953ded45667`，曾与官方发布 API 的 digest 核对。
可执行文件放在 `.tools/tectonic`。
EPUBCheck [官方 5.3.0 ZIP](https://github.com/w3c/epubcheck/releases/download/v5.3.0/epubcheck-5.3.0.zip)
下载 SHA-256 为 `6c07e68584b2e2ce2f89fe06e1246dfead3eb36b46b340e7d93524f29dcff6c5`；这是本地记录，不冒充发布方签名。
目录为 `.tools/epubcheck-5.3.0`。

## 两种独立 EPUB 阅读引擎

EPUB.js 0.3.93 已锁定于 Node 依赖。
第二种是 [Foliate.js](https://github.com/johnfactotum/foliate-js)，固定 commit
`78914aef4466eb960965702401634c2cb348e9b1`，放在 `.tools/foliate-js`。
它与 EPUB.js 分别解析同一最终 EPUB ZIP，处理资源、目录、定位和分页；不把解包后的普通网页冒充第二种阅读器。
测试只提供本地文件，不上传书稿。

```bash
git clone https://github.com/johnfactotum/foliate-js.git .tools/foliate-js
git -C .tools/foliate-js checkout 78914aef4466eb960965702401634c2cb348e9b1
```

这两个测试包装页不是新的出版正文。`check_epub_display.cjs` 则额外对最终 XHTML 检查边界，
覆盖每个文档的窄屏、宽屏和深色主题，不能代替实际阅读器翻页检查。
Apple Books 另完成阅读指南和第 17 章结构/公式页的白色主题抽查；后续本机窗口状态异常，
因此完整兼容性门禁采用上述两种可复现阅读引擎，未将 Books 未完成的字号/主题测试标为通过。

## 实验及结论边界

二十章实验在独立进程运行；音频使用 `.venv-omni`，文本/视觉和出版使用 `.venv`。
`verify_experiments.py` 保存完整命令、执行时间、环境包版本、源版本、文件哈希与逐章结果到 `build/final-audit/experiments.json`。
随机小 SigLIP/Mimi、合成音频、脚本化 logits 与受控外部替身的范围在正文和实验记录中明确。
没有下载整套默认语音权重，也没有测量真实训练质量、GPU 吞吐或麦克风/扬声器端到端延迟。

## 内容一致性与页面审阅

转换前后逐项比较正文、代码、表格、公式和图源；再回读最终 EPUB XHTML、SVG、链接与资源哈希。
损坏测试包含删正文、改公式、坏图像路径、改脚注、改链接和不改元数据却破坏 SVG 字形。
PDF 同时检查文字探针、书签、字体与逐页渲染，不用文字抽取证明数学字形正确。
源 Markdown 的 `$...$` 和独立行 `$$` 与 Pandoc 数学节点相互核对；公式意义另按章节源码和小数值实验复核。

两种格式同批核验后才更新发行文件，`dist/release-manifest.json` 记录输入与输出 SHA-256。
这份检查证明当前书稿在声明范围内通过核对，不承诺未知阅读器或任意自定义模型配置均有相同行为。

## 修改后怎样保持可核对（1.0.1）

`check_book.py --edition release` 现在同时检查章节元信息、前置依赖、证据记录、完整源码符号，
以及 `meta/content-review.json` 中复核过的正文、所有公式与图源。源文件一律从固定 commit 的 Git 对象读取，
上游 HEAD 或工作区发生变化不会自动让仍可重建的旧版章节失效。数值实验仍应使用对应版本的独立工作副本；
可通过 MINIMIND_SOURCE_ROOT 指向该副本，不需要切换原上游仓库。

维护顺序：先修改 Markdown 和必要实验，逐项核对源码、公式及箭头含义，运行相应实验并写复核记录；
完成这一步后，显式记录新的内容快照：

```bash
.venv/bin/python scripts/record_content_review.py --record meta/consistency-review.md
.venv/bin/python scripts/check_book.py --edition release
```

快照命令会先核验元信息和固定源码，随后记录已经完成的内容审阅；它不会替你证明新公式或解释正确。
构建器从不自动更新这个快照。接下来按前述命令构建、运行出版反例测试、检查页面及阅读器，
最后更新绑定这一批产物的 `meta/final-review.json` 并运行 `scripts/publish_book.py`。
发布脚本再次检查内容快照，且自动重打包当前 Markdown、实验、工具及证据，避免电子书已更新而源码 ZIP 仍是旧版。

不要用 `-O` 或设置 `PYTHONOPTIMIZE` 运行数值/产物验证；这会禁用 Python 的 assert。
实验总入口与 EPUB/PDF 校验器现在会明确拒绝这种运行模式，不能静默报告“通过”。

最终审阅记录不必手工拼接检查数字：`collect_release_review.py` 会实际回读产物、运行 EPUBCheck 和 pdffonts，
核对实验脚本哈希、测试结果及两种阅读器和 XHTML 显示报告对应的 EPUB 哈希。
另需提供已经完成的 PDF 视觉复核记录（JSON，至少含 `passed`、`pages`、`pdf_sha256`；其余字段记录受影响页和检查方法）。
该记录由真正看过页面的复核者填写，收集器不会自行把视觉检查标成通过。

```bash
.venv/bin/python scripts/render_pdf_review.py build/release/minimind-source-guide.pdf --output build/consistency-audit/pages --compare dist/minimind-source-guide.pdf
.venv/bin/python scripts/collect_release_review.py --audit-dir build/consistency-audit --visual-record build/consistency-audit/visual-review.json --review-record meta/consistency-review.md
.venv/bin/python scripts/publish_book.py
```

`pdffonts` 需在 PATH 上，也可以用 `--pdffonts` 指定 Poppler 可执行文件。
本次 1.0.1 的实验、工具和产物快照存于 meta；1.0 电子书、源码 ZIP 与发布清单保留在 `dist/archive/1.0/`。
