---
title: Hexo文章url设置方法
date: 2025-08-14 18:15:28
comments: true
categories:
  - [Hexo]
id: hexo-url-setting
excerpt: ""
tags:
  - Hexo
---

hexo文章的url在博客根目录的_config.yml中进行配置，默认配置如下：
```yml
permalink: :year/:month/:day/:title/ #年/月/日/文章路径
```

这里的:title为source/_post下的文章相对路径，但是这样很容易造成url中文乱码，和不同浏览器因为字符集的问题导致url失效。
因此建议尽量不使用默认配置，推荐使用如下两种方案:

1. 自定义url

修改`_config.yml`配置为：

```yml
permalink: :year/:month/:day/:id/ #这里我们把title替换为id
```

同时在文章头部的Front-matter中添加`id`信息：
```yml
---
title: 测试
date: 2025-08-14 13:54:28
categories: 类别1
id: blog1 # 此处的blog001对应url中的id部分
tags:
	- Python
	- Shell
---
```

2. 使用hash值

如果不想每次都要自定义文章id，可以直接修改`_config.yml`为：
```yml
permalink: :year/:month/:day/:hash/ #这里我们把title替换为hash
```
这样每次生成文章url，会自动生成hash值，保证不重复且不会因为编码出错。

最后附上hexo的url自定义规则：
|  变量   | 描述  |
|  ----  | ----  |
|:year	| 文章的发表年份（4 位数）|
|:month	| 文章的发表月份（2 位数）|
|:i_month |	文章的发表月份（去掉开头的零）|
|:day |	文章的发表日期 (2 位数)|
|:i_day |	文章的发表日期（去掉开头的零）|
|:hour |	文章发表时的小时 (2 位数)|
|:minute |	文章发表时的分钟 (2 位数)|
|:second |	文章发表时的秒钟 (2 位数)|
|:title |	文件名称 (relative to “source/_posts/“ folder)|
|:name |	文件名称|
|:post_title |	文章标题|
|:id |	文章 ID (not persistent across cache reset)|
|:category |	分类。如果文章没有分类，则是 default_category 配置信息|
|:hash |	SHA1 hash of filename (same as :title) and date (12-hexadecimal)|


