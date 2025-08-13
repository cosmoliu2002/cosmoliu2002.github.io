---
layout: page  # 保持页面布局（根据主题调整，如`layout: archive`等）
title: 归档  
permalink: /archives/  # 固定访问路径
---

<!-- 可选：添加内联样式优化显示效果 -->
<style>
.archives {
  max-width: 800px;
  margin: 2rem auto;
}
/* 年份标题 */
.archives h2 {
  margin: 2rem 0 1rem;
  font-size: 1.8rem;
  font-weight: 700;
}
/* 文章列表 */
.archives ul {
  list-style: none;
  padding-left: 1.5rem;
  margin-bottom: 2rem;
}
.archives li {
  margin: 0.5rem 0;
  line-height: 1.6;
}
/* 日期样式 */
.post-date {
  color: #666;
  margin-right: 1.5rem;
}
</style>

<div class="archives">
  <h2>共计 {{ site.posts.size }} 篇文章</h2>  <!-- 显示文章总数 -->
  {% for post in site.posts %}  <!-- 遍历所有文章（默认按发布时间倒序） -->
    {% assign current_year = post.date | date: "%Y" %}  <!-- 获取当前文章年份 -->
    
    <!-- 分组逻辑：当年份变化时，闭合上一个列表，开启新列表并显示年份标题 -->
    {% if forloop.first %}  <!-- 第一篇文章，直接开启年份和列表 -->
      <h2>{{ current_year }}</h2>
      <ul>
    {% else %}  <!-- 非第一篇，比较上一篇的年份 -->
      {% assign prev_year = site.posts[forloop.index0 - 1].date | date: "%Y" %}
      {% if current_year != prev_year %}  <!-- 年份不同时，闭合旧列表，开启新列表 -->
        </ul>
        <h2>{{ current_year }}</h2>
        <ul>
      {% endif %}
    {% endif %}
    
    <!-- 显示单篇文章：月-日 + 标题链接 -->
    <li>
      <span class="post-date">{{ post.date | date: "%m-%d" }}</span>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
    
    {% if forloop.last %}  <!-- 最后一篇文章，闭合列表 -->
      </ul>
    {% endif %}
  {% endfor %}
</div>
