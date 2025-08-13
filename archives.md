---
layout: page
title: 归档  
permalink: /archives/
---

<style>
.archives {
  max-width: 800px;
  margin: 2rem auto;
  padding: 0 1rem;
}

.archives-header {
  margin: 2rem 0 3rem;
  font-size: 1.2rem;
  color: #666;
}

/* 年份分组容器 */
.year-group {
  margin-bottom: 3rem;
}

/* 年份标题 */
.year-title {
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #eee;
}

/* 文章列表 */
.post-list {
  list-style: none;
  padding-left: 0;
}

.post-item {
  margin: 0.8rem 0;
  padding-left: 1rem;
  border-left: 2px solid #f0f0f0;
  transition: border-color 0.3s;
}

.post-item:hover {
  border-left-color: #666;
}

.post-date {
  color: #888;
  margin-right: 1.2rem;
  font-family: monospace;
}

.post-link {
  color: #333;
  text-decoration: none;
  transition: color 0.3s;
}

.post-link:hover {
  color: #0066cc;
  text-decoration: underline;
}
</style>

<div class="archives">
  <div class="archives-header">共计 {{ site.posts.size }} 篇文章</div>
  
  {% assign current_year = "" %} <!-- 初始化当前年份变量 -->
  
  {% for post in site.posts %}
    {% assign post_year = post.date | date: "%Y" %} <!-- 获取当前文章年份 -->
    
    <!-- 当年份变化时，创建新的年份分组 -->
    {% if post_year != current_year %}
      {% assign current_year = post_year %} <!-- 更新当前年份 -->
      
      <!-- 闭合上一个年份的列表（如果不是第一个分组） -->
      {% if forloop.index0 != 0 %}
        </ul>
      {% endif %}
      
      <!-- 新的年份分组 -->
      <div class="year-group">
        <h2 class="year-title">{{ current_year }}</h2>
        <ul class="post-list">
    {% endif %}
    
    <!-- 文章条目 -->
    <li class="post-item">
      <span class="post-date">{{ post.date | date: "%m-%d" }}</span>
      <a href="{{ post.url }}" class="post-link">{{ post.title }}</a>
    </li>
    
    <!-- 最后一篇文章时闭合列表 -->
    {% if forloop.last %}
      </ul>
    {% endif %}
  {% endfor %}
</div>
