---
layout: post
title: 最优化理论笔记
tags: [data analysis, machine learning]
description: 这部分内容包括了最优化的形式，KTT条件，拉格朗日算子及其对偶形式
date: 2018-08-017
feature_image: 
---

最优化问题都可以表达成以下形式：

$$\begin{align}
\min_x &f(x)  \\
s.t.\ g_j(x) &\leq 0 \quad (j=1,2,...,m) \\
h_k(x) & =0 \quad (k=1,2,...,l)
\end{align}$$

当 $$f(x), g_j(x)$$ 是凸函数，$$h_k(x)$$ 是仿射函数时，优化问题是**凸优化问题**。

- 等式约束优化  
  可以构建拉格朗日方程
  
  $$L(x, \lambda)=f(x)+\sum_{k=1}^l \lambda_k h_k(x)$$
  
  通过计算
  
  $$\begin{cases}
  \frac{\partial L}{\partial x_i} = 0 \\
  \frac{\partial L}{\partial \lambda_k} = 0
  \end{cases}$$
  
  得到可能的极值点。

- 不等式约束优化  
  对于不等式 $$ g_j(x) \leq 0 $$，加入松弛项，使得 $$g_j(x)+r_j^2=0$$
  此时可以构建拉格朗日函数
  
  $$L(x, \mu, r)=f(x)+\sum_{j=1}^m \mu_j (g_j(x)+r_j^2)$$
  
  通过计算
  
  $$\begin{cases}
  \frac{\partial L}{\partial x_i} = 0 \\
  \frac{\partial L}{\partial \mu_j} = g_j(x)+r_j^2 = 0 \\
  \frac{\partial L}{\partial r_j} = \mu_j r_j = 0 \\
  \mu_j \geq 0
  \end{cases}$$
  
  得到可能的极值点。
  
  若 $$\mu_j=0$$，说明该约束没有起作用，$$g_j(x)+r_j^2$$ 可以忽略。  
  若 $$\mu_j \neq 0$$，则 $$r_j=0$$，那么根据第二个方程 $$g_j(x)=0$$。可得
  
  $$\begin{cases}
  \frac{\partial L}{\partial x_i} = 0 \\
  \mu_j g_j(x)=0 \\
  \mu_j \geq 0
  \end{cases}$$
  
- All in one  
  当同时存在等式约束和不等式约束时，有广义拉格朗日函数
  
  $$L=f(x)+\sum_{j=1}^m \mu_j g_j(x) + \sum_{k=1}^l \lambda_k h_k(x)$$
  
  存在极值点的必要条件是
  
  $$\begin{cases}
  \frac{\partial L}{\partial x_i} = 0 \\
  \lambda_k \neq 0 \\
  \mu_j g_j(x)=0 \\
  \mu_j \geq 0 \\
  g_j \leq 0 \\
  h_k = 0
  \end{cases}$$
  
  这就是 **KTT 条件**。
  

对于拉格朗日函数，如果不满足约束条件，则存在 $$\max_{\mu, \lambda} L = +\infty$$，反之 $$\max_{\mu, \lambda} L = f(x)$$。原问题就变成了

$$p^* = \min_x \max_{\mu, \lambda} L$$

原问题存在对偶问题

$$d^* = \max_{\mu, \lambda} \min_x L$$

如果优化问题是**凸优化**，不等式约束严格可行，即 $$c_i(x)<0$$，则存在 $$x^*$$ 是原始问题的解，$$\lambda^*, \mu^*$$ 是对偶问题的解，且 $$p^*=d^*=L(x^*,\alpha^*,\beta^*)$$。  
此时 $$x^*$$ 是原始问题的解，$$\lambda^*, \mu^*$$ 是对偶问题的解，充要条件是它们满足 KTT 条件。

*参考资料 https://zhuanlan.zhihu.com/p/26514613*
