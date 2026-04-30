---
layout: page
---

<div style="margin-top: 60px;"></div>

# 目录导航

<div style="margin-top: 60px;"></div>

这是一个傻逼的导航页：

<div style="margin-top: 60px;"></div>

<div class="items-grid">
  <a class="nav-card" href="/LLM-Study-Notes_by-1ndigoRiVow/guide/learning/p1">
    <div class="nav-icon">📒</div>
    <div class="nav-title">学习笔记</div>
    <div class="nav-desc">数学基础 & 核心算法 & 前沿技术</div>
  </a>

  <a class="nav-card" href="/LLM-Study-Notes_by-1ndigoRiVow/guide/llmexp/p1">
    <div class="nav-icon">🛠️</div>
    <div class="nav-title">模型实操</div>
    <div class="nav-desc">记录大模型微调、提示工程的心得</div>
  </a>

  <a class="nav-card" href="/LLM-Study-Notes_by-1ndigoRiVow/guide/projexp/p1">
    <div class="nav-icon">💼</div>
    <div class="nav-title">项目经验</div>
    <div class="nav-desc">一些 KsT 的项目经验（脱敏）</div>
  </a>

  <a class="nav-card" href="/LLM-Study-Notes_by-1ndigoRiVow/guide/industry/p1">
    <div class="nav-icon">📈</div>
    <div class="nav-title">行业趋势</div>
    <div class="nav-desc">个人有感记录（手打）</div>
  </a>

  <a class="nav-card" href="/LLM-Study-Notes_by-1ndigoRiVow/guide/econ/p1">
    <div class="nav-icon">🪙</div>
    <div class="nav-title">经济学研究</div>
    <div class="nav-desc">经济学学术要点记录</div>
  </a>
</div>

<style scoped>
.items-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-top: 24px;
}
.nav-card {
  text-decoration: none !important;
  border: 1px solid var(--vp-c-bg-soft);
  border-radius: 12px;
  padding: 24px;
  background-color: var(--vp-c-bg-soft);
  transition: all 0.3s ease;
  display: block;
}
.nav-card:hover {
  border-color: var(--vp-c-brand-1);
  background-color: var(--vp-c-bg-alt);
  transform: translateY(-5px);
}
.nav-icon {
  font-size: 32px;
  margin-bottom: 12px;
}
.nav-title {
  font-weight: 600;
  font-size: 18px;
  color: var(--vp-c-text-1);
  margin-bottom: 8px;
}
.nav-desc {
  font-size: 14px;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}
</style>