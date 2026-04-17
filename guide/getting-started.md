---
layout: page
---

# 导航指南
欢迎来到我的学习路线图。点击下方板块进入详细内容：

::: info 导航板块
<div class="features">
  <div class="feature">
    <h2>学习笔记</h2>
    <p>数学基础 & 核心算法 & 前沿技术</p>
    <a href="/notes/index">点击进入 →</a>
  </div>

  <div class="feature">
    <h2>模型实操</h2>
    <p>记录大模型微调、提示工程的心得</p>
    <a href="/practice/index">点击进入 →</a>
  </div>

  <div class="feature">
    <h2>项目经验</h2>
    <p>一些 KsT 的项目经验（脱敏）</p>
    <a href="/projects/index">点击进入 →</a>
  </div>

  <div class="feature">
    <h2>行业趋势</h2>
    <p>不作新闻分享，仅个人有感记录（手打）</p>
    <a href="/trends/index">点击进入 →</a>
  </div>
</div>
:::

<style>
.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 16px;
  margin-top: 24px;
}
.feature {
  border: 1px solid var(--vp-c-bg-soft);
  border-radius: 12px;
  padding: 24px;
  background-color: var(--vp-c-bg-soft);
  transition: border-color 0.25s;
}
.feature:hover {
  border-color: var(--vp-c-brand);
}
.feature h2 {
  margin: 0 0 8px 0 !important;
  border: none;
  font-size: 1.2rem;
}
.feature p {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  margin-bottom: 16px;
}
.feature a {
  font-weight: 500;
  color: var(--vp-c-brand);
  text-decoration: none;
}
</style>