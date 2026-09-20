const chapters = [
  ["01", "第一篇", "环境、张量与一次最小前向", "从随机权重开始，观察输入、输出、shape 与梯度。", "start-here", "环境 张量 梯度"],
  ["02", "第一篇", "从文字到 token", "分词、特殊 token 和聊天模板怎样连接？", "tokenizer", "token 分词 聊天模板"],
  ["03", "第一篇", "搭起语言模型骨架", "配置、Embedding、残差、RMSNorm、SwiGLU 如何合作？", "model-skeleton", "模型 RMSNorm 残差"],
  ["04", "第一篇", "手算一次注意力", "Q/K/V、因果 mask、GQA、QK-Norm 和 SDPA 怎样工作？", "attention", "注意力 QKV mask"],
  ["05", "第一篇", "位置、缓存与逐字生成", "RoPE 如何编码位置？缓存怎样增长？如何停止？", "rope-cache-generation", "RoPE 缓存 生成"],
  ["06", "第一篇", "一条训练样本的旅程", "Pretrain/SFT 的模板、截断、mask 和 shift 各在哪里？", "datasets", "数据集 SFT shift"],
  ["07", "第一篇", "预训练循环与断点恢复", "loss 怎样变成参数更新？checkpoint 保存什么？", "pretraining", "训练 loss checkpoint"],
  ["08", "第一篇", "SFT、评估与训练排错", "怎样训练对话行为，并区分训练 loss 与验证结果？", "sft-evaluation", "SFT 评估 排错"],
  ["09", "第二篇", "MoE 的路由与负载", "为什么只选部分专家？辅助损失怎样连接回模型？", "moe", "MoE 专家 路由"],
  ["10", "第二篇", "LoRA 与知识蒸馏", "少量参数怎样改变模型？学生如何学习教师分布？", "lora-distillation", "LoRA 蒸馏"],
  ["11", "第二篇", "DPO 与偏好数据", "一对 chosen/rejected 如何形成优化目标？", "dpo", "DPO 偏好"],
  ["12", "第二篇", "从 PPO 概念到 GRPO", "policy、reference、reward、value 和 advantage 各是什么？", "ppo-grpo", "PPO GRPO 强化学习"],
  ["13", "第二篇", "Agent 多轮交互与奖励", "生成、工具反馈、response mask 和整轮奖励如何闭环？", "agent", "Agent 工具 奖励"],
  ["14", "第三篇", "图像怎样进入语言模型", "图像预处理、patch、projector 与占位符如何对应？", "vision", "视觉 图像 patch"],
  ["15", "第三篇", "视觉训练与评估", "冻结哪些参数？多图、缺图、截断怎样影响训练？", "vlm-training", "视觉训练 VLM"],
  ["16", "第四篇", "音频入门与特征注入", "波形、采样率、声学特征和离散码本有什么区别？", "audio-basics", "音频 波形 采样率"],
  ["17", "第四篇", "Thinker、Bridge 与 Talker", "语义特征怎样与音色、历史音频共同影响生成？", "thinker-talker", "Thinker Bridge Talker"],
  ["18", "第四篇", "八路码本与时间对齐", "delay pattern 怎样组织训练与生成？Mimi 怎样还原波形？", "audio-codebooks", "码本 delay pattern Mimi"],
  ["19", "第四篇", "Omni 训练流水线", "T2A/A2A/I2T 怎样切换数据、loss、冻结和权重？", "omni-training", "Omni 训练 T2A A2A"],
  ["20", "第四篇", "流式交互、打断与综合实战", "VAD、生成、WebSocket、音频播放怎样协作与取消？", "realtime", "实时 VAD 打断 WebSocket"]
].map(function (item) {
  return {number:item[0], part:item[1], title:item[2], question:item[3], slug:item[4], keywords:item[5]};
});

const chapterGrid = document.querySelector("#chapterGrid");
const resultCount = document.querySelector("#resultCount");
const emptyState = document.querySelector("#emptyState");
const searchInput = document.querySelector("#siteSearch");
const filterChips = Array.from(document.querySelectorAll(".filter-chip"));
let selectedPart = "all";

function chapterUrl(chapter) {
  return "https://github.com/cluster1900/minimind_reader/blob/main/book/" + chapter.number + "-" + chapter.slug + "/README.md";
}

function renderChapters() {
  const query = searchInput.value.trim().toLowerCase();
  const visible = chapters.filter(function (chapter) {
    const matchesPart = selectedPart === "all" || chapter.part === selectedPart;
    const haystack = (chapter.number + " " + chapter.part + " " + chapter.title + " " + chapter.question + " " + chapter.keywords).toLowerCase();
    return matchesPart && (!query || haystack.indexOf(query) !== -1);
  });
  chapterGrid.innerHTML = visible.map(function (chapter) {
    return '<article class="chapter-card">' +
      '<div class="chapter-number">' + chapter.number + '</div>' +
      '<div><p class="chapter-meta">' + chapter.part + ' · ' + chapter.keywords.split(" ").slice(0, 2).join(" / ") + '</p>' +
      '<h3>' + chapter.title + '</h3><p>' + chapter.question + '</p></div>' +
      '<a class="chapter-link" href="' + chapterUrl(chapter) + '" target="_blank" rel="noreferrer" aria-label="阅读第 ' + chapter.number + ' 章">↗</a></article>';
  }).join("");
  resultCount.textContent = "显示 " + visible.length + " 章";
  emptyState.hidden = visible.length !== 0;
}

filterChips.forEach(function (chip) {
  chip.addEventListener("click", function () {
    selectedPart = chip.dataset.part === "all" ? "all" : chip.dataset.part;
    filterChips.forEach(function (item) { item.classList.toggle("is-selected", item === chip); });
    renderChapters();
  });
});
searchInput.addEventListener("input", renderChapters);

document.querySelectorAll("[data-filter]").forEach(function (link) {
  link.addEventListener("click", function () {
    const filter = link.dataset.filter;
    const chip = filterChips.find(function (item) { return item.dataset.part === filter; });
    if (chip) chip.click();
  });
});

document.addEventListener("keydown", function (event) {
  if (event.key === "/" && document.activeElement !== searchInput) { event.preventDefault(); searchInput.focus(); }
  if (event.key === "Escape" && document.activeElement === searchInput) { searchInput.value = ""; searchInput.blur(); renderChapters(); }
});

const root = document.documentElement;
const themeToggle = document.querySelector("#themeToggle");
const storedTheme = localStorage.getItem("minimind-theme");
if (storedTheme === "light") root.classList.add("light");
themeToggle.addEventListener("click", function () {
  root.classList.toggle("light");
  localStorage.setItem("minimind-theme", root.classList.contains("light") ? "light" : "dark");
});

const sidebar = document.querySelector("#sidebar");
const menuToggle = document.querySelector("#menuToggle");
menuToggle.addEventListener("click", function () {
  const open = sidebar.classList.toggle("is-open");
  menuToggle.setAttribute("aria-expanded", String(open));
});
document.querySelectorAll(".sidebar a").forEach(function (link) {
  link.addEventListener("click", function () {
    sidebar.classList.remove("is-open");
    menuToggle.setAttribute("aria-expanded", "false");
  });
});

const navLinks = Array.from(document.querySelectorAll(".nav-link[href^='#']"));
const sections = navLinks.map(function (link) { return document.querySelector(link.getAttribute("href")); }).filter(Boolean);
const observer = new IntersectionObserver(function (entries) {
  entries.forEach(function (entry) {
    if (entry.isIntersecting) {
      navLinks.forEach(function (link) { link.classList.toggle("is-active", link.getAttribute("href") === "#" + entry.target.id); });
    }
  });
}, {rootMargin:"-25% 0px -65% 0px", threshold:0});
sections.forEach(function (section) { observer.observe(section); });

renderChapters();
