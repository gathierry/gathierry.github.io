---
layout: post
title: 深度学习面试你必须知道这些答案
tags: [deep learning]
description: 本文节选了网上分享的一部分深度学习相关的面试题，以及搜索到的答案。
date: 2018-05-21
feature_image: images/2018-05-21-Anwsers-for-DL-Interview/title.jpg
usemathjax: true
---
本文节选了网上分享的一部分深度学习相关的面试题，并将搜集到的答案随之附上。  
参考了来自 _https://cloud.tencent.com/developer/article/1064505_ 的内容。

<!--more-->

#### Q1. 列举常见的一些范数及其应用场景，如 $$L^0$$，$$L^1$$，$$L^2$$，$$L^\infty$$，Frobenius 范数
- 范数用于测量一个向量。比如 $$\boldsymbol{x}$$ 的 $$L^p$$ 范数记为
$$||\boldsymbol{x}||_p=(\sum_i|x_i|^p)^{\frac{1}{p}}$$ 其中 $$p\in\mathbb{R}$$, $$p\geq1$$
- $$L^\infty$$范数，
\\[||\boldsymbol{x}||_{\infty}=max_i |x_i|\\]
- $$L^0$$范数表示向量中非零元素的个数  
- Frobenius范数用于测量一个矩阵
\\[
||\boldsymbol{A_F}||=\sqrt{\sum_{i,j}A^2_{i,j}}
\\]  
- $$L^1$$，$$L^2$$范数常用在正则化中，加入正则项会改变目标函数的优化路径。$$L^2$$也被成为“岭回归”，能让学习算法“感知”到具有较高方差的输入 $$\boldsymbol{x}$$ 因此相对增加输出方差的特征的权重将会收缩。$$L^1$$也被成为“LASSO回归”，会产生更稀疏的解，使得部分特征参数为0，表明相应特征可以被安全忽略。

#### Q2. 简单介绍一下贝叶斯概率与频率派概率，以及在统计中对于真实参数的假设
直接与事件发生的频率相联系（如抽取扑克牌），被称为频率派概率；而涉及到确定性水平（如病人诊断为阳性），被称为贝叶斯概率。

#### Q3. 概率密度的万能近似器
- 高斯混合模型是概率密度的万能近似器，任何平滑的概率密度都可以用具有足够多组件的高
斯混合模型以任意精度来逼近。  
- 一个高斯混合模型包含了一组参数不同的高斯分布，参数包括均值 $$\mu^(i)$$，协方差矩阵 $$\Sigma^(i)$$，以及每个组件 $$i$$ 的先验概率 $$\alpha_i=P(c=i)$$。

#### Q4. 简单介绍一下 sigmoid，relu，softplus，tanh，RBF 及其应用场景
- sigmoid
\\[
\sigma(x)=\frac{1}{1+\exp(-x)}
\\]
\\[
\sigma(x)’=\sigma(x)(1-\sigma(x))
\\]

- softplus
\\[
\zeta(x)=\log(1+\exp(x))
\\]
<img src="/images/2018-05-21-Anwsers-for-DL-Interview/softplus.png" width="400px"/>

#### Q5. Jacobian，Hessian 矩阵及其在深度学习中的重要性
仅使用梯度信息的优化算法被称为一阶优化算法，如梯度下降。使用 Hessian 矩阵的优化算法被称为二阶最优化算法，如牛顿法。

#### Q6. KL 散度在信息论中度量的是那个直观量
- 如果我们对于同一个随机变量 有两个单独的概率分布 $$P(x)$$ 和 $$Q(x)$$，我们可以使用 KL 散度来衡量这两个分布的差异。
\\[
D_{KL}(P||Q)=\mathbb{E}_{x \sim P}[\log{P(x)} - \log{Q(x)}]
\\]
- KL 散度有很多有用的性质，最重要的是它是非负的。KL 散度为 0 当且仅当 P 和 Q 在离散型变量的情况下是相同的分布，或者在连续型变量的情况下是“几乎处处”相同的。  
它并不是真的距离因为它不是对称的。$$D_{KL}(P||Q)\neq D_{KL}(Q||P)$$  
- 一个和 KL 散度密切联系的量是交叉熵
$$
H(P,Q)=H(P)+D_{KL}(P||Q)=\mathbb{E}_{x \sim P}[\log{P(x)}] + D_{KL}(P||Q) = -\mathbb{E}_{x \sim P}[\log{Q(x)}]
$$
- 深度学习里的 Cross entropy loss 中 Q 表示预测值，P 表示真实值，Cross entropy 与 KL 散度相差一个常数。

#### Q7. 数值计算中的计算上溢与下溢问题，如 softmax 中的处理方式
- 当接近零的数被四舍五入为零时发生**下溢**。许多函数在其参数为零而不是一个很小的正数时才会表现出质的不同。
- 当大量级的数被近似为 $$\infty$$ 或 $$-\infty$$ 时发生**上溢**。
- 对于softmax函数$$
softmax(\boldsymbol{x})_i=\frac{\exp(x_i)}{\sum_{j=1}^n\exp(x_j)}
$$
当所有$$x_i$$都等于一个很小的负数时，分母为0；当所有$$x_i$$都是非常大的正数时，会发生上溢。解决方法，是令$$
softmax(\boldsymbol{x})_i=\frac{\exp(x_i-\max_k x_k)}{\sum_{j=1}^n\exp(x_j-\max_k x_k)}
$$

#### Q8. 与矩阵的特征值相关联的条件数（病态条件）指什么，与梯度爆炸与梯度弥散的关系
考虑函数$$f(\boldsymbol{x})=\boldsymbol{A}^{-1}\boldsymbol{x}$$当$$\boldsymbol{A}$$具有特征值分解时，其条件数为$$max_{i,j}|\frac{\lambda_i}{\lambda_j}|$$，即分子是最大的特征值，分母是最小的特征值，当该数很大时，矩阵求逆对输入的误差特别敏感，输入中的舍入误差可能导致输出的巨大变化。

#### Q9. 在基于梯度的优化问题中，如何判断一个梯度为 0 的零界点为局部极大值、全局极小值还是鞍点，Hessian 矩阵的条件数与梯度下降法的关系
- 梯度为 0 时，当 Hessian 矩阵是正定的，是一个局部极小点。当 Hessian 矩阵是负定的 时，是一个局部极大点。这就是所谓的二阶导数测试。如果 Hessian 的特征值中至少一个是正的且至少一个是负的，那么是某个横截面 的局部极大点，却是另一个横截面的局部极小点。
- 可以使用 Hessian 矩阵的信息来指导搜索。牛顿法基于一个二阶泰勒展开来近似 $$x_0$$ 附近的 $$f(x)$$
\\[
f(x) = f(x_0) + (x - x_0)^T\nabla_xf(x_0)+\frac{1}{2}(x - x_0)^TH(f)(x - x_0)
\\]
计算 $$f'(x)=0$$ 可得到函数的临界点
\\[
x^*=x_0 - H(f)(x_0)^{-1} \nabla_xf(x_0)
\\]
当 f 是一个正定二次函数时，牛顿法只要应用上面的一次式就能直接跳到函数的最小点。如果 f 不是一个真正二次但能在局部近似为正定二次，牛顿法则需要多次迭代上式。迭代地更新近似函数和跳到近似函数的最小点可以比梯度下降更快地到达临界点。这在接近局部极小点时是一个特别有用的性质，但是在鞍点附近 是有害的。当附近的临界点是最小点(Hessian 正定)时牛顿法才适用，这时梯度下降不会被吸引到鞍点(除非梯度指向鞍点)。

#### Q10. KTT 方法与约束优化问题，活跃约束的定义
部分参考自 (http://www.cnblogs.com/mo-wang/p/4775548.html)  
对于一个优化问题
\\[
\min{f(x)}
\\]

- 如果约束是等式
\\[
h_k(x)=0 \quad k=1,2,...,l
\\]
可以使用拉格朗日乘子法，定义拉格朗日函数
\\[
F(x,\lambda)=f(x)+\sum_{k=1}^{l}\lambda_kh_k(x)
\\]
通过解方程组
\\[
\frac{\partial F}{\partial x_i}=0 \\
...\\
\frac{\partial F}{\partial \lambda_k}=0
\\]
计算出最优解。

- 如果约束是不等式
\\[
h_j(x)=0 \quad j=1,2,...,p \\
g_k(x)\leq 0 \quad k=1,2,...,q
\\]
此时的广义拉格朗日函数可以定义为
\\[
L(x,\lambda, \mu)=f(x) + \sum_{j=1}^{p}\lambda_jh_j(x) + \sum_{k=1}^{q}\mu_kg_k(x)
\\]
KTT 条件是最优点的必要条件，不一定是充分条件：
  - L 梯度为 0
  - h(x)=0
  -  不等式约束显示的“互补松弛性”：$$\sum_{k=1}^{q}\mu_kg_k(x)=0$$
如果 $$g_k(x^*)=0$$ 那么称这个约束是**活跃的**。

#### Q11. 模型容量，表示容量，有效容量，最优容量概念

- 通过调整模型的容量 (capacity)，我们可以控制模型是否偏向于过拟合或者欠拟合。通俗地，模型的容量是指其拟合各种函数的能力。容量低的模型可能很难拟合训练集。容量高的模型可能会过拟合。
- 学习算法可以从哪些函数族中选择函数。这被称为模型的表示容量 (representational capacity)。
- 额外的限制因素，比如优化算法的不完美，意味着学习算法的有效容量 (effective capacity)可能小于模型族的表示容量。

#### Q12. 正则化中的权重衰减与加入先验知识在某些条件下的等价性

- 我们可以加入权重衰减 (weight decay) 来修改线性回归的训练标准。带权重衰减的线性回归最小化训练集上的均方误差和正则项的和，其偏好于平方 L2 范数较小的权重。
- 在进行最大后验估计 (MAP) 时，
  \\[
  \theta_{MAP}=argmax_\theta \log{p(x|\theta)}+\log{p(\theta)}
  \\]
  如果先验分布是 $$\mathcal{N}(w; 0, \frac{1}{\lambda}I^2)$$，那么对数先验项正比于熟悉的权重衰减惩罚 $$\lambda w^T w$$。

#### Q13. 高斯分布的广泛应用的缘由

- 我们想要建模的很多分布的真实情况是比较接近正态分布的。 中心极限定理说明很多独立随机变量的和近似服从正态分布。这意味着在实际中，很多复杂系统都可以被成功地建模成正态分布的噪声，即使系统可以被分解成一些更结构化的部分。
- 在具有相同方差的所有可能的概率分布中，正态分布在实数上具有最大的不确定性。因此，我们可以认为正态分布是对模型加入的先验知识量最少的分布。正态分布拥有最大的熵，我们通过这个假定来保证了最小可能量的结构。

#### Q14. 最大似然估计中最小化 KL 散度与最小化分布之间的交叉熵的关系

- 一种解释最大似然估计的观点是将它看作最小化训练集上的经验分布和模型分布之间的差异，两者之间的差异程度可以通过 KL 散度度量。
- 最小化 KL 散度其实就是在最小化分布之间的交叉熵。
- 虽然最优 $$\theta$$ 在最大化似然或是最小化 KL 散度时是相同的，但目标函数值是不一样的。在软件中，我们通常将两者都称为最小化代价函数。因此最大化似然变成了最小化负对数似然(NLL)，或者等价的是最小化交叉熵。

#### Q15. 在线性回归问题，具有高斯先验权重的 MAP 贝叶斯推断与权重衰减的关系，与正则化的关系

同 Q13

#### Q16. 稀疏表示，低维表示，独立表示

- 低维表示尝试将 信息尽可能压缩在一个较小的表示中，比如 PCA。
- 稀疏表示将数据集嵌入到输入项大多数为零的表示中，L1 惩罚可以诱导稀疏的参数。稀疏表示通常用于需要增加表示维数的情况，使得大部分为零的表示不会丢失很多信息。这会使得表示的整体结构倾向于将数据分布在表示空间的坐标轴上。
- 独立表示试图分开数据分布中变化的来源，使得表示的维度是统计独立的。

#### Q17. 列举一些无法基于梯度的优化来最小化的代价函数及其具有的特点

维数灾难

#### Q18. 在深度神经网络中，引入了隐藏层，放弃了训练问题的凸性，其意义何在

一些隐藏单元可能并不是在所有的输入点上都是可微的，在实践中，梯度下降对这些机器学习模型仍然表现得足够好。部分原因是神经网络训练算法通常不会达到代价函数的局部最小值，而是仅仅显著地减小它的值。

#### Q19. 函数在某个区间的饱和与平滑性对基于梯度的学习的影响

最广泛使用的隐式“先验”是平滑先验，或局部不变性先验。这个先验表明我们学习的函数不应在小区域内发生很大的变化。许多简单算法完全依赖于此先验达到良好的泛化，其结果是不能推广去解决人工智能级别任务中的统计挑战。

#### Q20. 梯度爆炸的一些解决办法

参考 _https://blog.csdn.net/qq_25737169/article/details/78847691_

- 梯度爆炸会导致结果不收敛。
- 梯度爆炸问题可以通过梯度截断来缓解(执行梯度下降步骤之前设置梯度的阈值)。
- 较大的权重也会产生使得激活函数饱和的值，导致饱和单元的梯度完全丢失。通过正则化和 Batch Normalization可以解决。
- 激活函数如 Relu 使导数变为 1，也可以解决梯度爆炸的问题。

_对于梯度消失，除了以上几种方案，Residual block 的 shortcut 可以用来解决_

#### Q21. MLP 的万能近似性质

万能近似定理 (universal approximation theorem)(Hornik et al., 1989; Cybenko, 1989) 表明

- 一个前馈神经网络如果具有线性输出层和至少一层具有任何一种 “挤压” 性质的激活函数(e.g. sigmoid) 的隐藏层，只要给予网络足够数量的隐藏单元，它可以以任意的精度来近似任何从一个有限维空间到另一个有限维空间的 Borel 可测函数。
- 前馈网络的导数也可以任意好地来近似函数的导数 (Hornik et al., 1990)。
- 万能近似定理也已经被证明对于更广泛类别的激活函数也是适用的，其中就包括现在常用的整流线性单元 (Leshno et al., 1993)。
- 万能近似定理说明了，存在一个足够大的网络能够达到我们所希望的任意精度，但是定理并没有说这个网络有多大，可能大的不可实现。

*Borel 可测的概念超出了本书的范畴；对于我们想要实现的目标，只需要知道定义在 $$\mathbb{R}^n$$ 的有界闭集上的任意连续函数是 Borel 可测的， 因此可以用神经网络来近似。神经网络也可以近似从任何有限维离散空间映射到另一个的任意函数。*

#### Q22. 在前馈网络中，深度与宽度的关系及表示能力的差异

更深的模型往往表现更好。这不仅仅是因为模型更大。Goodfellow et al. (2014d) 的实验表明，增加卷积网络层中参数的数量，但是不增加它们的深度，在提升测试集性能方面几乎没有效果。  
浅层模型在参数数量达到 2000 万时就过拟合，而深层模型在参数数量超过 6000 万时仍然表现良好。这表明，使用深层模型表达出了对模型可以学习的函数空间的有用偏好。具体来说，它表达了一种信念，即该函数应该由许多更简单的函数复合在一起而得到。这可能导致学习由更简单的表示所组成的表示或者学习具有顺序依赖步骤的程序。

#### Q23. 为什么交叉熵损失可以提高具有 sigmoid 和 softmax 输出的模型的性能，而使用均方误差损失则会存在很多问题。分段线性隐藏层代替 sigmoid 的利弊

- 均方误差在 20 世纪 80 年代和 90 年代流行，但逐渐被交叉熵损失替代，并且最大似然原理的想法在统计学界和机器学习界之间广泛传播。使用交叉熵损失大大提高了具有 sigmoid 和 softmax 输出的模型的性能，而当使用均方误差损失时会存在饱和和学习缓慢的问题。
- 对于小的数据集，Jarrett et al. (2009b) 观察到，使用整流非线性甚至比学习隐藏层的权重值更加重要。随机的权重足以通过整流网络传播有用的信息，允许在顶部的分类器层学习如何将不同的特征向量映射到类标识。  
  当有更多数据可用时，学习开始提取足够的有用知识来超越随机选择参数的性能。Glorot et al. (2011a) 说明，__在深度整流网络中的学习比在激活函数具有曲率或两侧饱和的深度网络中的学习更容易__。  
  整流线性单元还表明神经科学继续对深度学习算法的发展产生影响。Glorot et al. (2011a) 从生物学考虑整流线性单元的导出。半整流非线性旨在描述生物神经元的这些性质:
  1. 对于某些输入，生物神经元是完全不活跃的。
  2. 对于某些输入，生物神经元的输出和它的输入成比例。
  3. 大多数时间， 生物神经元是在它们不活跃的状态下进行操作，即稀疏激活。

#### Q24. 表示学习的发展的初衷？并介绍其典型例子：自编码器

- 对于许多任务来说，我们很难知道应该提取哪些特征，解决这个问题的途径之一是使用机器学习来发掘表示本身，而不仅仅把表示映射到输出。这种方法我们称之为**表示学习**。
- 表示学习算法的典型例子是自编码器。自编码器由一个编码器函数和一个解码器函数组合而成。编码器函数将输入数据转换为一种不同的表示，而解码器函数则将这个新的表示转换到原来的形式。我们期望当输入数据经过编码器和解码器之后尽可能多地保留信息，同时希望新的表示有各种好的特性，这也是自编码器的训练目标。

#### Q25. 在做正则化过程中，为什么只对权重做正则惩罚，而不对偏置做权重惩罚

我们通常只对权重做惩罚而不对偏置做正则惩罚。

- 精确拟合偏置所需的数据通常比拟合权重少得多。
- 每个权重会指定两个变量如何相互作用。我们需要在各种条件下观察这两个变量才能良好地拟合权重。而每个偏置仅控制一个单变量。这意味着，我们不对其进行正则化也不会导致太大的方差。
- **正则化偏置参数可能会导致明显的欠拟合**。

#### Q26. 在深度学习神经网络中，所有的层中考虑使用相同的权重衰减的利弊

有时希望对网络的每一层使用单独的惩罚，并分配不同的系数。但是寻找合适的多个超参数的代价很大，因此为了减少搜索空间，我们会在所有层使用相同的权重衰减。

#### Q27. 正则化过程中，权重衰减与 Hessian 矩阵中特征值的一些关系，以及与梯度弥散，梯度爆炸的关系

令 $$w^*$$ 为未正则化的目标函数取得最小训练误差时的权重向量，即 $$w^*=\arg\min_wJ(w)$$，在 $$w^*$$ 的邻域对目标函数做二次近似。

$$
\hat{J}(\theta)=J(w^*)+\frac{1}{2}(w-w^*)^\top H(w-w^*)
$$

因为 $$w^*$$ 是 $$J$$ 的一个最优点，此时一阶项梯度 $$H(w-w^*)$$ 为零，而 $$H$$ 是半正定的。  
如果加入正则项，梯度变为
$$
\alpha w + H(w-w^*) = 0 \\
w = (H+\alpha I)^{-1}Hw^*
$$
当 $$\alpha$$ 趋于 0 时，$$w$$ 会趋向 $$w^*$$。
将 $$H$$ 分解成对角矩阵 $$\Lambda$$ 和一组特征向量的标准正交基 $$Q$$，上式化为

$$
\begin{align}
w & = (Q\Lambda Q^\top + \alpha I)^{-1}Q \Lambda Q^\top w^* \\
  & = (Q(\Lambda+\alpha I)Q^\top)^{-1}Q\Lambda Q^\top w^* \\
  & = Q(\Lambda+\alpha I))^{-1}\Lambda Q^\top w^*
\end{align}
$$

随着 $$\alpha$$ 增加，我们会根据 $$\frac{\lambda_i}{\lambda_i+\alpha}$$ 因子缩放与 H 第 i 个特征向量对齐的 $$w^*$$ 的分量。沿着特征值较大的方向（$$\lambda_i >> \alpha$$）正则化的影响较小。而 $$\lambda_i << \alpha$$ 的分量将会收缩到几乎为零。
<img src="/images/2018-05-21-Anwsers-for-DL-Interview/wdecay.png" width="400px"/>

#### Q28. L1／L2 正则化与高斯先验／对数先验的 MAP 贝叶斯推断的关系

- L2 正则化相当于权重是高斯先验的 MAP 贝叶斯推断
- L1 正则化相当于权重是对数先验的 MAP 贝叶斯推断，即权重先验是各项同性的拉普拉斯分布

#### Q29. 什么是欠约束，为什么大多数的正则化可以使欠约束下的欠定问题在迭代过程中收敛

一个例子是应用于线性可分问题的逻辑回归。如果权重向量 w 能够实现完美分类， 那么 2w 也会以更高似然实现完美分类。类似随机梯度下降的迭代优化算法将持续增加的大小，理论上永远不会停止。在实践中，数值实现的梯度下降最终会达到导致数值溢出的超大权重，而正则化可以抑制这一问题。

#### Q30. 为什么考虑在模型训练时对输入 (隐藏单元／权重) 添加方差较小的噪声，与正则化的关系

- 神经网络被证明对噪声不是非常健壮 (Tang and Eliasmith, 2010)。改善神经网络健壮性的方法之一是简单地将随机噪声添加到 输入再进行训练。向隐藏单元施加噪声也是可行的，这可以被看作在多个抽象层上进行的数据集增强。
- 最小化带权重噪声的目标函数等同于最小化附加正则化项的目标函数。这种形式的正则化鼓励参数进入权重小扰动对输出相对影响较小的参数空间区域。换句话说，它推动模型进入对权重小的变化相对不敏感的区域，找到的点不只是极小点，还是由平坦区域所包围的极小点。

#### Q31. 共享参数的概念及在深度学习中的广泛影响 

- 参数范数惩罚是正则化参数使其彼此接近的一种方式，而更流行的方法是使用约束：强迫某些参数相等，这种正则化方法通常被称为**参数共享**。
- 参数共享的一个显著优点是，只有参数的子集需要被存储在内存中，可以显著减少模型所占用的内存。
- 对于卷积神经网络，参数共享的特殊形式使得神经网络层具有对平移**等变**的性质。

#### Q32. Dropout 与 Bagging 集成方法的关系，以及 Dropout 带来的意义与其强大的原因

- Bagging（bootstrap aggregating）是通过结合几个模型降低泛化误差的技术 (Breiman, 1994)。主要想法是分别训练几个不同的模型，然后让所有模型表决测试样例的输出。其奏效的原因是不同的模型通常不会在测试集上产生完全相同的误差。在误差完全相关的情况下，平方误差的期望不会减少，但在误差完全不相关的情况下，平方误差期望减少为原来的  1/k，k是模型的数量。
- Dropout 可以被看做是集成了大型神经网络的子网络。
- 只有极少的训练样本可用时，Dropout不会很有效。
- 在推断时，每一个输出都要乘以权值被保留的概率。通过掩码 $$\mu$$ 定义每个子模型的概率分布
  \\[p(y|x, \mu)\\]
  输出取算术平均值 \\[\sum_\mu p(\mu)p(y|x, \mu)\\]

#### Q33. 批量梯度下降法更新过程中，批量的大小与各种更新的稳定性关系  

- 更大的批量会计算更精确的梯度估计，但是回报却是小于线性的。n 个样本均值的标准差是 $$\sigma/\sqrt(n)$$ ，$$\sigma$$ 是样本值真实标准差。
- 可能是由于小批量在学习过程中加入了噪声，它们会有一些正则化效果(Wilson and Martinez, 2003)。泛化误差通常在批量大小为1 时最好。因为梯度估计的高方差，小批量训练需要较小的学习率以保持稳定性，会进一步增加训练时间。
- 仅基于梯度 $$g$$ 的更新方法通常相对鲁棒，并能使用较小的批量获得成功。使用 Hessian 矩阵的二阶方法通常需要更大的批量。这些大批量需要最小化估计二阶项 $$H^{-1}g$$ 的波动。
- 小批量是随机抽取的这点也很重要。从一组样本中计算出梯度期望的无偏估计要求这些样本是独立的。


#### Q34. 如何避免深度学习中的病态，鞍点，梯度爆炸，梯度弥散 

- 牛顿法在解决带有病态条件的Hessian 矩阵的凸优化问题时，是一个非常优秀的工具，但是牛顿法运用到神经网络时需要很大的改动。
- 克服鞍点同样需要修改牛顿法（加动量）
- 关于梯度爆炸和梯度弥散，参考 Q20

#### Q35. SGD 以及学习率的选择方法，带动量的 SGD 对于 Hessian 矩阵病态条件及随机梯度方差的影响 

- 通常最好是检测最早的几轮迭代，选择一个比在效果上表现最佳的学习率更大的学习率，但又不能太大导致严重的震荡。
- 随机梯度下降学习过程有时会很慢。动量方法 (Polyak, 1964) 旨在加速学习，特别是处理高曲率、小但一致的梯度，或是带噪声的梯度。动量算法积累了之前梯度指数级衰减的移动平均，并且继续沿该方向移动。
- 一个病态条件的二次目标函数看起来像一个长而窄的山谷或具有陡峭边的峡谷。动量正确地纵向穿过峡谷，而普通的梯度步骤则会浪费时间在峡谷的窄轴上来回移动。
- 动量算法引入了变量 $$\boldsymbol{v}$$ 充当速度角色，相比于 SGD 中直接 $$\boldsymbol{\theta} = \boldsymbol{\theta} - \epsilon \boldsymbol{g}$$，带动量的 SGD 先更新速度 $$\boldsymbol{v}=\alpha \boldsymbol{v} - \epsilon \boldsymbol{g}$$，再更新参数 $$\boldsymbol{\theta}=\boldsymbol{\theta}+\boldsymbol{v}$$。当许多连续的梯度指向相同的方向时，步长最大。如果动量算法总是观测到梯度 $$\boldsymbol{g}$$，那么它会在方向$$-\boldsymbol{g}$$ 上不停加速，直到达到最终速度，其中步长大小为 
  \\[\frac{\epsilon ||\boldsymbol{g}||}{1-\alpha}\\]
  因此将动量的超参数视为 $$\frac{1}{1-\alpha}$$ 有助于理解。例如，$$\alpha=0.9$$ 对应着最大速度10 倍于梯度下降算法。在实践中，$$\alpha$$ 的一般取值为0.5，0.9 和0.99。之前梯度对现在方向产生影响，抑制了随机梯度的方差。
- Nesterov 动量，在计算梯度 $$\frac{1}{m}\nabla _\theta \sum_iL(f(x, \theta), y)$$ 之前更新参数，变成 $$\frac{1}{m}\nabla _\theta \sum_iL(f(x, \theta+\alpha v), y)$$。在凸批量梯度的情况下，Nesterov 动量将额外误差收敛率从O(1/k)（k 步后）改进到O(1/k2)。可惜，在随机梯度的情况下，Nesterov 动量没有改进收敛率。

#### Q36. 初始化权重过程中，权重大小在各种网络结构中的影响，以及一些初始化的方法；偏置的初始化

- 更大的初始权重具有更强的破坏对称性的作用，有助于避免冗余的单元，也有助于避免在每层线性成分的前向或反向传播中丢失信号——矩阵中更大的值在矩阵乘法中有更大的输出。如果初始权重太大，那么会在前向传播或反向传播中产生爆炸的值。在循环网络中，很大的权重也可能导致混沌 (chaos)(对于输入中很小的扰动非常敏感，导致确定性前向传播过程表现随机)。较大的权重也会产生使得激活函数饱和的值，导致饱和单元的梯度完全丢失。这些竞争因素决定了权重的理想初始大小。
- Xavier 均匀初始化 $$W_{i,j}=\mathcal{U}(-\sqrt{\frac{6}{m+n}}, \sqrt{\frac{6}{m+n}})$$。 m, n 分别是输入和输出的单元数。Xavier 初始化目的是使每一层输出的方差尽量相等，但推导时只考虑了激活函数是线性函数的情况。
- MSRA 初始化  $$W_{i,j}=\mathcal{N}(0, \sqrt{\frac{2}{m}})$$
- 设置偏置为零通常在大多数权重初始化方案中是可行的。存在一些我们可能设置偏置为非零值的情况:
  - 如果偏置是作为输出单元，那么初始化偏置以获取正确的输出边缘统计通常是有利的。
  - 有时，我们可能想要选择偏置以避免初始化引起太大饱和。
  - 有时，一个单元会控制其他单元能否参与到等式中。例如 LSTM

#### Q37. 自适应学习率算法: AdaGrad，RMSProp，Adam 等算法的做法

#### Q38. 二阶近似方法: 牛顿法，共轭梯度，BFGS 等的做法

#### Q39. Hessian 的标准化对于高阶优化算法的意义

#### Q40. 卷积网络中的平移等变性的原因，常见的一些卷积形式

对于卷积，参数共享的特殊形式使得神经网络层具有对平移等变的性质。卷积包含的参数有：kernel size, stride, padding, dilation
