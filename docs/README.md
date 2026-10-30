# GitHub Pages 站点

这里是 `MiniMind 源码阅读与工程实践` 的 GitHub Pages 静态站点源文件。

页面采用无构建依赖的 HTML/CSS/JavaScript，可直接在 GitHub 仓库设置中发布：

1. 打开仓库 **Settings → Pages**。
2. 在 **Build and deployment** 中选择 **Deploy from a branch**。
3. 选择 `main` 分支和 `/docs` 文件夹，保存后等待 GitHub Pages 构建。

本地预览：

```bash
python3 -m http.server 8000 --directory docs
```

然后访问 <http://localhost:8000/>。站点不依赖 CDN；封面与字体资源从 `docs/assets/` 读取。

页面内容以仓库正文、实验记录和 `publishing/book.yaml` 为准。章节卡片链接到 GitHub 上的 Markdown，
下载按钮链接到 `v1.1.0` Release 的 EPUB/PDF。更新书稿或发行版本时，应同步检查 `index.html` 中的版本号、
Release 地址、章节目录和源码 commit。
