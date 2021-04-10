---
layout: post
title: Deep Learning读书笔记 -- 第二章 线性代数
categories: [data science]
tags: [deep learning]
description: 范数、特征分解和奇异值分解
---

## 范数
范数是满足下列性质的任意函数：

- $$f(\boldsymbol{x})=0\Rightarrow \boldsymbol{x}=0$$
- $$f(\boldsymbol{x}+\boldsymbol{y}) \le f(\boldsymbol{x})+f(\boldsymbol{y})$$
- $$\forall \alpha \in \mathbb{R}, f(\alpha\boldsymbol{x})=|\alpha|f(\boldsymbol{x})$$

常用范数有：

- $$L^2$$范数
- $$L^1$$范数
- $$L^\infty$$范数，也叫**最大范数**，表示向量中具有最大幅值元素的绝对值  
- Frobenius范数，用来衡量矩阵大小
\\[||\boldsymbol{A_F}||=\sqrt{\sum_{i,j}A^2_{i,j}}\\]

## 特征分解
每个实对称矩阵都可以分解成实特征向量和实特征值:

$$ A=Q\Lambda Q^T $$

Q是正交矩阵

## 奇异值分解（SVD）
$$ A=UDV^T $$

A是m\*n矩阵，U是m\*m矩阵，D是m\*n矩阵，V是n\*n矩阵。  
U和V都是正交矩阵，D是对角矩阵，对角线上的元素被称为**奇异值**。  
U的列向量是$$AA^T$$的特征向量。  
V的列向量是$$A^TA$$的特征向量。  
A的非零奇异值是$$AA^T$$特征值的平方根，也是$$A^TA$$特征值的平方根。  
SVD最有用的一个性质可能是拓展矩阵求逆到非方矩阵上。

## Moore-Penrose伪逆
对于方程 $$Ax=y$$，A可能不是方阵。A的伪逆
\\[
A^+=VD^+U^T
\\]
符号含义与SVD相同。D的伪逆，是非零元素去倒数之后再转置得到的。  
当A列数多于行数时，解$$x=A^+y$$是所有可行解中L2范数最小的一个。  
当A列数小于行数时，没有解，通过伪逆解出的x使得$$||Ax-y||$$最小。