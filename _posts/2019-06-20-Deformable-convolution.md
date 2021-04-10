---
layout: post
title: 目标检测刷分利器 deformable convolution
tags: [object detection, instance segmentation, deep learning, computer vision]
description: Deformable convolution 是卷积神经网络中一种特殊的卷积运算，这篇文章会介绍这种特殊的卷积层以及它的“打开方式”。
date: 2019-06-20
feature_image: 
usemathjax: true
---
Deformable convolution 最早是在 MSRA 2017 年的论文 [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211) 中提出的。通过对传统 CNN 的卷积操作进行改进，赋予其对于形状不规则的 pattern 更强的特征提取能力。2018 年的 [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168) 有提出了进一步的改进。虽然两篇论文中还提出了很多其他的创新点，但 deformable convolution 凭借其简单好用，成为目标检测领域刷分的标配。再开始介绍之前，我们先来回顾一些关于常规卷积层的一些概念。

## 数学表达

### Convolution
一个常规的 $$3\times3$$ 卷积操作，可以表示成如下数学表示：

\\[
\mathfrak{R} = \{(-1, -1), (-1, 0), ..., (0, 1), (1, 1)\} \\
y(\boldsymbol{p_0})=\sum_{\boldsymbol{p_n} \in \mathfrak{R}} w(\boldsymbol{p_n}) x(\boldsymbol{p_0} + \boldsymbol{p_n})
\\]

### Deformable convolution
而一个 deformable convolution 的数学表达式是：

\\[
y(\boldsymbol{p_0})=\sum_{\boldsymbol{p_n} \in \mathfrak{R}} w(\boldsymbol{p_n}) x(\boldsymbol{p_0} + \boldsymbol{p_n} + \Delta\boldsymbol{p_n})
\\]

附加的 offset 向量是一个可以通过学习获得的参数。因为这个向量并不一定是整数，$$x(\boldsymbol{p_0} + \boldsymbol{p_n} + \Delta\boldsymbol{p_n})$$ 的值并不能直接从特征图上获得，但是可以经过双线性插值得到。在 $$\boldsymbol{p}$$ 点的双线性差值可以表示为

\\[
x(\boldsymbol{p})=\sum_{\boldsymbol{q}} (\max(0, 1-|q_x-p_x|)x(q_x) + \max(0, 1-|q_y-p_y|)x(q_y))
\\]

式中的 $$q$$ 可以表示特征上的任意一点，但是距离超过 1 的都不会对 $$p$$ 造成任何影响。

### Modulated deformable convolution
Offset 为方方正正的卷积增加了一个形变量，但是不同的 offset 所起到的作用其实并不是一样的，于是 Deformable ConvNets v2 的论文加入了一个新的可学习的参数$$m$$，用来调制卷积核的信号幅度。

\\[
y(\boldsymbol{p_0})=\sum_{\boldsymbol{p_n} \in \mathfrak{R}} w(\boldsymbol{p_n}) x(\boldsymbol{p_0} + \boldsymbol{p_n} + \Delta\boldsymbol{p_n}) \Delta m_n
\\]

## 网络结构

### Convolution & group convolution

在切入正题之前，先来回顾一下常规的 convolution 以及 group convolution。

一个卷积运算，如果输入 channel 数 $$c_1$$，输出 channel 数 $$c_2$$，卷积核为 $$k_h \times k_w$$，当 ``group=1`` 时，它的权重是 $$(c_2, c_1, k_h, k_w)$$ 的张量。

<img src="/images/2019-06-20-Deformable-convolution/groupfilter1.png" width="600px"/>

如果 ``group=g``，此时输入在 channel 方向被等分成 g 份，每一份的 channel 数是 $$c_1 / n$$。那么就可以对每一份输入进行一个 $$(c_2/g, c_1/g, k_h, k_w)$$ 的卷积，就可以得到 channel 数为 $$c_2/g$$ 的输出。将 g 份输出拼接到一起，又重新得到 channel=$$c_2$$的输出。然而这时权重数量是 $$(g, c_2/g, c_1/g, k_h, k_w)$$，仅是原来的 $$1/g$$。

<img src="/images/2019-06-20-Deformable-convolution/groupfilter2.png" width="600px"/>

### Modulated deformable convolution

因为 modulated deformable convolution 只比 deformable convolution 多出一个参数，在原理上是一样的，所以这里就写在一起了。

我们希望对于特征上的每一个点，都可以有一组 $$\Delta\boldsymbol{p_n}$$ 和 $$\Delta m_n$$，每一组的数量与卷积核尺寸相同。例如对于一个 $$3\times3$$ 的卷积核，$$\Delta\boldsymbol{p_n}$$ 尺寸是 $$(2, 3, 3)$$，$$\Delta m_n$$ 尺寸是 $$(1, 3, 3)$$。

通过一次卷积可以得到这些参数。如图所示，输入的特征图经过一次卷积，输出 channel 数为 $$3 \times k_h \times k_w$$ 的特征。提取前 2/3 的 channel 作为 $$\Delta\boldsymbol{p_n}$$，后 1/3 channel 作为 $$\Delta m_n$$，当做输入传给 deformable convolution 进行计算。

这个卷积的参数，包括 ``kernel_size``, ``stride``, ``padding``, ``dilation`` 都与 deformable convolution 的参数保持一致，但是 weight 和 bias 都初始化为 0。也就是说，初始情况下，offset 为 0，mask 为 1，等价于一个常规卷积。

<img src="/images/2019-06-20-Deformable-convolution/deformable_conv.png" width="600px"/>

到这里，我们发现在 deformable convolution 的过程中，所有 channel 使用的同一套 offset 和 mask。可不可以使用不同的呢？这里就引入了 deformable group 这个概念。 它与 group convolution 的 group 有些相似，但并不是一回事。当 deformable group 大于 1 时，输入特征会在 channel 方向被等分，每一份内使用相同的 offset 和 mask。这时，生成 offset 和 mask 卷积的 filter number 也就需要变成 $$3 \times k_h \times k_w \times group_{deformable}$$

<img src="/images/2019-06-20-Deformable-convolution/deformable_conv_group.png" width="600px"/>

对于 deformable convolution 的实现，还涉及到了 `im2col` 的改进 `im2col_step`。关于 `im2col` 和 `im2col_step`, 可以参考最后的 5 6 两篇形象的介绍。

## ResNet 中的使用

在论文中，deformable convolution 可以植入到 ResNet stage2 - stage5 的最后几个 residual unit 中。对于 bottleneck，用在中间的 $$3\times3$$ 卷积上，而对于 basic block，则用在第一个 $$3\times3$$ 卷积上。

例如 ResNet-101 中，可以在将 stage5 的最后三个 bottleneck 全部用 deformable convolution 改造。


## 参考资料
1. [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)
2. [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)
3. 官方代码 https://github.com/msracver/Deformable-ConvNets
4. Group convolution https://blog.yani.io/filter-group-tutorial/
5. im2col https://blog.csdn.net/mrhiuser/article/details/52672824
6. im2col_step https://zhuanlan.zhihu.com/p/58185157
