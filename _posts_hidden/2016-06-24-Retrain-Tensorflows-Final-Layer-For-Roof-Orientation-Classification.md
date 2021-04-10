---
layout: post
title: Retrain Tensorflow's final layer 用于建筑物卫星图的屋顶朝向识别
tags: [machine learning, deep learning, tensorflow, DSG2016, computer vision]
description: 题目出自于Data Science Game 2016 初赛，本文简要记录了使用Tensorflow Inception V3的过程。
date: 2016-06-24
feature_image: 
---
(challenge site : https://inclass.kaggle.com/c/data-science-game-2016-online-selection)

## 题目概述
输入为建筑物屋顶的卫星图（类似Google地图卫星图），输出为1，2，3或4，分别代表南北朝向、东西朝向、平屋顶以及其他。训练集8000张图片，测试集13999张图片。图片大小不等，大多在70-200px。

## 准备工作
首先要clone github上的[tensorflow项目](https://github.com/tensorflow/tensorflow)。同时安装[bazel](http://www.bazel.io)。

## 训练Inception V3
这里我们事实上只训练了Inception网络的最后一层。Inception原本是用于识别图片中物体种类ImageNet，有1000个输出分别代表1000个种类。而这里我们只需要4个输出，所以要对最后一层进行改动。

以下操作都在tensorflow的根目录下执行。

如果是第一次使用，需要先```./configure```一下。
之后build retrain工具。目前大多数CPU都支持AVX指令集，通过如下命令构建速度更快
	
	bazel build -c opt --copt=-mavx tensorflow/examples/image_retraining:retrain

之后使用训练集对网络重新训练

	bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /path/to/image/directory

这里需要注意的是，图片保存必须以如下形式:

- image_dir
	- class1
		- img11
		- img12
	- class2
		- img21
		- img22
	- class3
		- img31
		- img32
	- class4
		- img41
		- img42

不同种类的图片保存在不同文件夹下。
	
## 预测分类

训练结束之后会在```/tmp/```目录下发现```output_graph.pb```和 ```output_labels.txt```两个文件。前者储存了网络的信息而后者是输出标签的列表。

有了这两个文件，我们就可以对测试集中的图片进行分类了。

仍然是先build

	bazel build tensorflow/examples/label_image:label_image

之后进行分类

	bazel-bin/tensorflow/examples/label_image/label_image \
	--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
	--output_layer=final_result \
	--image=path/to/image/file

如果想要将训练好的网络用于其他地方，只需要复制```output_graph.pb```和 ```output_labels.txt```两个文件到要用的机器上，再重新build label_image 之后分类即可。




## Reference

https://www.tensorflow.org/versions/r0.8/how_tos/image_retraining/index.html





