---
layout: post
title: 深度学习中的优化器
tags: [deep learning]
description: 深度学习中常见的优化器，包括 SGD, momentum SGD, Nesterov accelerated gradient, Adagrad, AdaDelta, RMSprop, Adam
date: 2018-08-17
feature_image: 
usemathjax: true
---

深度学习中常见的优化器，包括 SGD, momentum SGD, Nesterov accelerated gradient, Adagrad, AdaDelta, RMSprop, Adam.

<!--more-->

### SGD

$$\theta \leftarrow \theta - \eta g_t$$

### Momentum SGD

$$ V(t) = \gamma V(t-1) + \eta g_t \\
\theta \leftarrow \theta - V(t)$$

$$\gamma$$ 取值通常是 0.9

也就是说，参数每次的变化量，是上一次变化量，与当前梯度的加权平均。
<img src="/images/2018-08-17-Optimizer-in-DL/momentum.png" width="600px"/>

### Nesterov accelerated gradient

$$ V(t) = \gamma V(t-1) + \eta \nabla f(\theta - \gamma V(t-1)) \\
\theta \leftarrow \theta - V(t)$$

在动量的基础上进一步修改，并不直接计算梯度，而是按照计算根据动量移动之后的梯度。
<img src="/images/2018-08-17-Optimizer-in-DL/nesterov.png" width="600px"/>

### Adagrad

后面的算法对每个参数进行分别的优化

$$ n_t = n_{t-1} + g_t^2 \\
\theta_i \leftarrow \theta_i - \frac{\eta}{\sqrt{n_t + \epsilon}} g_t $$

这里其实是通过记录每一次迭代中梯度的平方，来降低学习速率。之前梯度下降的越多，学习速率就会随之下降。

### AdaDelta

Adadelta是对Adagrad的扩展。

$$ n_t = \nu n_{t-1} + (1-\nu)g_t^2 \\
\theta_i \leftarrow \theta_i - \frac{\sqrt{\sum_{r=1}^{t-1}\Delta \theta_r}}{\sqrt{n_t + \epsilon}} g_t $$

分子是之前迭代量的累加结果，AdaDelta 不需要预设全局学习速率。

### RMSprop

RMSprop可以算作Adadelta的一个特例，$$\nu=0.5$$

$$ n_t = 0.5 n_{t-1} + 0.5g_t^2 \\
\theta_i \leftarrow \theta_i - \frac{\eta}{\sqrt{n_t + \epsilon}} g_t $$

### Adam (Adaptive moment)

Adam 计算一阶矩和二阶矩

一阶矩：$$s_t=\rho_1 s_{t-1} + (1-\rho_1)g_t$$

二阶矩：$$r_t=\rho_2 r_{t-1} + (1-\rho_2)g_t^2$$

偏置校正 

$$
\widehat{s} = \frac{s}{1-\rho_1^t} \\
\widehat{r} = \frac{r}{1-\rho_2^t}
$$

参数更新

$$\theta_i = \theta_i - \eta \frac{\widehat{s}}{\sqrt{\widehat{r}}+\epsilon}$$

常用的默认值是 $$\rho_1=0.9, \rho_2=0.999$$

*参考资料  
https://zhuanlan.zhihu.com/p/22252270  
https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f*
