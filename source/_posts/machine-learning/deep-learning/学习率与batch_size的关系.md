---
title: 学习率与 Batch Size 的关系
date: 2025-09-01 14:47:42
comments: true
id: lr-bs-relationship
excerpt: ""
categories:
  - [Machine Learning, Deep Learning]
tags:
  - 学习率
  - batch size
---

> 参考文章
> [当Batch Size增大时，学习率该如何随之变化？](https://kexue.fm/archives/10542)
> [Adam的epsilon如何影响学习率的Scaling Law？](https://kexue.fm/archives/10563)
> [重新思考学习率与Batch Size的关系（一）：现状](https://kexue.fm/archives/11260)

当Batch Size增大时，每个Batch的梯度将会更准，因此每次更新梯度的幅度可以更大，也就是增大学习率，以求更快达到终点，缩短训练时间。那么学习率增大多少才是最合适的？



# 平方根缩放

## One weird trick for parallelizing convolutional neural networks

结论：Batch Size扩大到 $n$ 倍，则学习率扩大到 $\sqrt{n}$ 倍。

$$\frac{\eta^2}{B} = \text{常数} \implies \eta \propto \sqrt{B}$$

# 线性缩放

线性缩放即 $\eta \propto B$ 在实践中的表现往往更好。

# 单调有界

OpenAI的《An Empirical Model of Large-Batch Training》通过损失函数的二阶近似来分析SGD的最优学习率，得出“**学习率随着Batch Size的增加而单调递增但有上界**”的结论。

# 实践分析

$$B \ll \mathcal{B}_{\text{noise}}$$

$$1 + \mathcal{B}_{\text{noise}} / B \approx \mathcal{B}_{\text{noise}} / B$$

所以$\eta^* \approx \eta_{\text{max}} \frac{B}{\mathcal{B}_{\text{noise}}} \propto B$，即线性缩放，这再次体现了线性缩放只是小Batch Size时的局部近似；当$B > \mathcal{B}_{\text{noise}}$时，$\eta^*$逐渐趋于饱和值$\eta_{\text{max}}$，这意味着训练成本的增加远大于训练效率的提升。所以，$\mathcal{B}_{\text{noise}}$相当于一个分水岭，当Batch Size超过这个数值时，就没必要继续投入算力去增大Batch Size了。

# 数据效率

$\left( \frac{S}{S_{\text{min}}} - 1 \right) \left( \frac{E}{E_{\text{min}}} - 1 \right) = 1$

其中 $S$ 为总训练步数，$S_{min}$ 为最少训练步数，$E$ 为总训练数据量，$E_{min}$ 为最少训练数据量。上述公式就是训练数据量和训练步数之间的缩放规律，表明数据量越小，那么应该缩小Batch Size，让训练步数更多，才能更有机会达到更优的解。

# Adam 自适应版

上述结论都是基于 SGD 进行推导的，现在讨论对于 Adam 等自适应学习率优化器的结论。

这部分内容由《Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling》完成。

在Batch Size本身较小时，Adam确实适用于平方根缩放定律。

# 涌现行为

有可能涌现出“Batch Size足够大时学习率反而应该减小”的Surge现象。

当Batch Size超过这个$\mathcal{B}_{\text{noise}}$后，最佳学习率不应该增大反而要减小！