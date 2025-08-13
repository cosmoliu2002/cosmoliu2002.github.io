---
layout: page  # 使用页面布局
title: 归档  # 页面标题
permalink: /archives/  # 页面访问路径（固定为/archives）
---

<div class="archives-container">
  <h2>共计 {{ site.posts.size }} 篇文章</h2>  <!-- 显示文章总数 -->
  
  {% for post in site.posts %}  <!-- 循环遍历所有博客文章 -->
    {% assign current_year = post.date | date: "%Y" %}  <!-- 获取文章发布年份 -->
    {% assign current_month = post.date | date: "%m" %}  <!-- 获取文章发布月份 -->
    
    <!-- 按月份分组显示（参考示例格式） -->
    {% if forloop.first %}
      <h3>{{ current_year }}-{{ current_month }}</h3>
      <ul>
    {% else %}
      {% assign prev_year = site.posts[forloop.index0 - 1].date | date: "%Y" %}
      {% assign prev_month = site.posts[forloop.index0 - 1].date | date: "%m" %}
      {% if current_year != prev_year or current_month != prev_month %}
        </ul>
        <h3>{{ current_year }}-{{ current_month }}</h3>
        <ul>
      {% endif %}
    {% endif %}
    
    <!-- 显示单篇文章：日期 + 标题链接 -->
    <li>
      <span class="post-date">{{ post.date | date: "%m-%d" }}</span>  <!-- 显示月-日 -->
      <a href="{{ post.url }}" class="post-title">{{ post.title }}</a>  <!-- 文章标题链接 -->
    </li>
    
    {% if forloop.last %}
      </ul>
    {% endif %}
  {% endfor %}
</div>
