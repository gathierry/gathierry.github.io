---
layout: post
title: 形象的转置卷积
tags: [deep learning]
description: 形象理解转置卷积里各个参数的作用
date: 2018-08-13
feature_image: images/2018-08-13-Transposed-convolution/title.gif
---

关于转置卷积的用处，以及极其背后的原理，不是这里要讨论的重点，可以参考 https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0

假设我们对转置卷积已经有了初步的了解，那么要如何使用呢？

<!--more-->

转置卷积包含三个参数 ```kernal_size```, ```stride``` 和 ```padding```

先来看一个最简单的例子
	
	input size = 4x4
	kernel = 3
	stride = 1
	padding = 0

<img src="/images/2018-08-13-Transposed-convolution/deconv1.png" width="600px"/>

这里面实际的操作是：

1. 卷积核与输入的第一行元素相乘，所得结果构成输出的第 1-3 列
2. 卷积核与输入的第二行元素相乘，所得结果构成输出的第 2-4 列,两次输出重合的元素直接相加。如果 ```stride = 2``` 则输出第 3-5 列，也就是说，stride 是在输出尺度上的间隔。
3. 以此类推，得到后序的几列，行方向也是同样的道理

在这个例子中，```output size = 6x6```
最后再说 ```padding```。这里的 padding 并不是向外 padding，而是向内的。如果 ```padding = 1```，那么输入矩阵的最外一圈全部被忽略，卷积只作用于

	[x22, x23]
	[x32, x33]

输出矩阵尺寸为 ```4x4```

由此可以推出
```output size = stride * (input - 1) + kernel  - 2 * padding```
	