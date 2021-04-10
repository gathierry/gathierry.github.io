---
layout: post
title: 拆解 MaskRCNN-Benchmark
tags: [object detection, instance segmentation, deep learning, computer vision]
description:
date: 2019-02-17
feature_image: images/2019-02-17-maskrcnn-benchmark/title.png
---
# 写在前面

[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) 是继 Detectron 之后 FAIR 用 Pytorch1.0 最新的实现，在效率上超越了 Detectron。通过阅读、实验代码，可以对 Mask RCNN 这一多任务模型的诸多细节有更深刻的认识。

**这篇文章是阅读代码时做的分析，便于理解工程的结构和思路，并不会去解释 Mask RCNN 的论文，也不会逐行分析每句代码的作用。**

## 安装方法

项目的安装方法在[安装文档](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md)写的很详细，这里就不多啰嗦了。

文档中要求使用 nighlty release 版本的 PyTorch 1.0，事实上 stable 版也是可以的，不过一些运算的速度可能会降低。

在最后用 ```setup.py``` 安装项目的时候，Mac 用户要多往下看一行，使用最后一行被注释掉的安装命令

	MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

## 代码结构

程序的训练/测试入口是```tools/train_net.py```和```tools/test_net.py```。

```configs```文件夹中包含了不同网络配置的参数，这里我们用```configs/e2e_mask_rcnn_R_50_FPN_1x.yaml```为例。

```maskrcnn_benchmark```目录下包含了数据处理、网络运算等核心代码，也是接下来的重点所在。

## 网络流程

粗略梳理一下网络结构，可以发现，整个流程大概可以用这个框图表示，红色部分表示训练过程，蓝色部分表示测试过程，黑色表示共用部分。
<img src="/images/2019-02-17-maskrcnn-benchmark/maskrcnn-benchmark.png" width="1000px"/>

在后序文章中将会结合代码，依次分析其中的细节。

# 数据整理和加载

既然是监督学习，网络模型首先就要定义输入。对于目标检测和实例分割任务而言，输入的除了图像，还需要每幅图中检测框的坐标，以及物体分割的 mask。这样的输入数据格式显然比单纯的分类问题要复杂很多，同时还需要考虑到输入图片的分辨率、长宽比不一致。在实现时，因为深度学习框架的输入必须是张量的形式，这就要求开发者对数据进行足够的预处理，才能放入网络。

## 检测、分割标签数据的整理

一张图中可能会存在多个检测框以及分割实例，而且这个数量是随不同图像变化的。

maskrcnn-benchmark 中，在```maskrcnn_benchmark/structures/bounding_box.py```中定义了一个```BoxList```类。每一个```BoxList```对象即是一幅图像的标签。这个类内部包含了```Tensor```类型的成员变量```bbox```用来存储一张图中检测框的信息，同时还有一个字典类型的成员变量```extra_fields```用来存储分割以及其他可能会用到的数据。

如果我们看一下整套代码，就会发现```extra_fields```所存过的变量，包括

- 分割标签```mask```
- 物体类别```labels```
- 匹配检测框与 ground truth 框的```matched_idx```
- RPN 预测的置信度```objectness```
- RCNN 预测的置信度```score```
- anchor 是否超出图片边界```visibility```

除了```mask```以外，其他变量都比较简单，对于每个框都都可以用一个标量表示。

分割标签的数据更加不规则，每个数据表示一个多边形，是由长度不等的 list 表示的。每张图片会有多个 mask，而每个 mask 也可能分成多个 polygons（比如物体被遮挡分成两部分）。代码中通过```maskrcnn_benchmark/structures/segmentation_mask.py```整理分割的标签数据，并且作为一个```extra_field```存在一个```BoxList```对象中。

这样一层包装，有效的将标签信息传入到网络模型中。

## DataLoader

PyTorch 的数据加载，是基于 [```torch.utils.data.DataLoader```](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。```DataLoder```的主要参数有

- ```dataset```
- ```batch_sampler```
- ```collate_fn```

```dataset```提供每次迭代产生的单个数据，```collate_fn```将产生的数据组合成 batch，而每个 batch 中数据的取样规则，通过```batch_sampler```来控制。

### Dataset

有了图片和标签，就可以构造[```torch.utils.data.Dataset```](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)。```Dataset```主要的输出是```(image, boxlist)```。一些预处理以及数据增强，都在这里实现。PyTorch 提供了图像预处理以及数据增强的接口[```torchvision.transforms```](https://pytorch.org/docs/stable/torchvision/transforms.html)。但是这里当我们对图片进行 resize，flip 之类的操作时，需要对检测框、分割框进行相应的调整，所以在项目目录```maskrcnn_benchmark\data\transforms\transforms.py```中重写了定制化的接口。

### Batch Collator

```collate_fn```函数在```dataset```取值之后调用。假设```batch_size=2```，那么```collate_fn```的输入就是长度为 2 的 list，每个元素是一组```dataset```的输出，比如```(image, boxlist, idx)```。

当 mini-batch > 1 时，同一个 batch 中的图片大小必须是一致的。然而输入图像有时无法满足这一要求。这时我们就需要在```collate_fn```中将 batch 中的所有图片 padding 到该 batch 最大图片的尺寸。代码中在```maskrcnn_benchmark\structrues\image_list.py```构造了```ImageList```类，将 padding 后的图像组成一个 tensor，同时储存了原本的图像尺寸信息。

### Batch Sampler

PyTorch 中提供了[```torch.utils.data.BatchSampler```](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler)的接口，包含基本的实现。然而在```maskrcnn_benchmark\data\samplers\```进行了扩展，这里只说```grouped_batch_sampler.py```。

```GroupedBatchSampler```通过参数```group_id```的指导，将```group_id```相同的数据放入同一个 batch。maskrcnn 中这个```group_id```是由图片长宽比大于/小于 1 来区分的。这样做的结果就是避免横版图片和竖版图片放进同一个 batch，从而导致 ```collate_fn```时为了补齐尺寸差异而大量填零。

<img src="/images/2019-02-17-maskrcnn-benchmark/dataloader.png" width="600px"/>

**经过以上步骤，数据就以```ImageList```和```BoxList```的形式传入了模型。然而它们的本质仍然是```tensor```类型，这也是为什么经过包装的数据可以被运算框架接受并且进行高效运算。**

# Backbone 和 FPN

项目中的 backbone 用的是经过 ImageNet 预训练的 ResNet 模型。但是存在两个区别：

## 1. 固定 Batch Normalization  

在代码中，有一个重新实现的 BN 层```maskrcnn_benchmark/layers/batch_norm.py```，这里的参数全部是固定的初始值，不会随着训练而更新。固定 BN 的原因，在[这里](https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)也给出了解释，是因为在 batch size per GPU 比较小的情况下，如果不采用多 GPU 同步，BN 达不到应有的效果，所以干脆使用固定的原 pretrained 模型的值。

## 2. 参数初始化

另一个和 torchvision 的 resnet 实现不同的地方是参数初始化的方法。在 maskrcnn-benchmark 的 ResNet 和 FPN 中，所有的初始化都使用的 [```nn.init.kaiming_uniform_(conv.weight, a=1)```](https://pytorch.org/docs/0.3.1/nn.html#torch.nn.init.kaiming_uniform`)

# RPN

在 Mask RCNN 中，预设 anchor 数量从 Faster RCNN 的 9 个， 提升到了 15 个。这里```ratio=[0.5,1,2]```，而```scale```在每一层 feature map 上（```stride=4,8,16,32,64```）的尺度都是 8，映射到原图尺度上就是```scale=[32,64,128,256,512]```。

假设图像尺寸为 $$(H, W)$$，特征图 $$stride=\{S1, S2, S3, S4, S5\}$$，则特征图尺寸为 $$(H/S_i, W/S_i)$$。在每层特征上，$$scale=s$$, $$ratio=\{r1, r2, r3\}$$。Batch size 记为 $$B$$，anchor 数量记为 $$A$$。

## RPN Head

RPN 由一个简单的 head 紧接 FPN 每一个 level 的特征之后，它们**共享** head 的权重。设 FPN 输出的特征图为 P2, P3, P4, P5, P6, 每个特征图都经过同一个 3x3 卷积，然后两个 1x1 卷积分支得到两个输出，$$(B, A, H/S_i, W/S_i)$$ 和 $$(B, 4\times A, H/S_i, W/S_i)$$，表示前景的 score 和四个坐标的 regression 结果。

这里的卷积层初始化使用的并不是 xavier 或者 kaiming，而是
	

	torch.nn.init.normal_(l.weight, std=0.01)
	torch.nn.init.constant_(l.bias, 0)

详细代码在```maskrcnn_benchmark/modeling/rpn/rpn.py```

## Anchor Generator

RPN head 可以输出 box score 和 box regression，然而如果要想转化成原图尺度下的 bounding box，还需要根据相应的 anchor 进行转化。即

$$BoundingBoxes = BoxCoder(regression, anchor)$$

对于一个 $$(H/S_i, W/S_i)$$ 的特征图，需要生成 $$A \times H/S_i \times W/S_i$$ 个 anchor，这些 anchor 可以储存成```BoxList```对象。

### 1. 生成 anchor cells

每个 anchor cell 都是一个组合 $$(\Delta x_1, \Delta y_1, \Delta x_2, \Delta y_2)$$。feature map 上的一个像素，对应原图上 $$S_i \times S_i$$ 个像素。以 $$(0, 0, S_i-1, S_i-1)$$ 为基础进行放缩，可以生成 $$A$$ 个 anchor cells。每个特征图对应各自的 anchor cells。

### 2. 原图上取网格点

对每一个特征图，在原图尺度上以 $$S_i$$ 等间隔的取点 $$(x_a^i, y_a^i)$$ 经过变换可以得到一系列 anchors $$(x_a^i + \Delta x_1, y_a^i + \Delta y_1, x_a^i + \Delta x_2, y_a^i + \Delta y_2)$$。因为 anchor cell 假设以 $$(0, 0, S_i-1, S_i-1)$$ 为基础，所以在特征图上取点可以从 0 开始

	torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)

最终每个特征图都在原图上生成了 $$H/S_i \times W/S_i$$ 个等距且距离为 $$S_i$$ 的格点。

### 3. 生成 anchors

每一个格点与相应特征图上所有的 anchor cell 组合，生成一个 anchor，最终生成形状为 $$(A \sum_i (H/S_i \times W/S_i), 4)$$ 的 anchor boxes。

### 4. 包装进 BoxList

这些 anchor boxes 被包装成```BoxList```的对象。这里对每一个 anchor 进行了检查，如果存在边界超出图片尺寸的 anchor，会通过```add_field("visibility", 0)```来标记，在后续操作中进行忽略。

最终输出的 anchor 形式为 $$(B, 5, A \times H/S_i \times W/S_i, 4)$$。前两个维度以嵌套```list```形式储存，后两个维度以```BoxList```形式储存。

## RPN 后处理

RPN 生成了大量的 box，每个 anchor 都对应一个 box，显然不能全部输出到后面的网络中去，这就需要一些的后处理步骤选取最有意义的 proposals。这一步涉及到三个常数，```PRE_NMS_TOP_N```，```POST_NMS_TOP_N```，```FPN_POST_NMS_TOP_N```。后面会挨个涉及到。

### Box Coder

有了 anchor，我们已经可以复原出 RPN head 的输出所对应的 box 坐标了。这个转换的公式通过```maskrcnn_benchmark/modeling/box_coder.py```的```BoxCoder```实现。

Encode 过程是从真实 box 转化成输出格式。

- $$t_x = \frac{x-x_a}{w_a} \times W_x$$
- $$t_y = \frac{y-y_a}{h_a} \times W_y$$
- $$t_w = \log{\frac{w}{w_a}} \times W_w$$
- $$t_h = \log{\frac{h}{h_a}} \times W_h$$

Decode 则是相反的过程。

- $$x = \frac{t_x w_a}{W_x} + x_a$$
- $$y = \frac{t_y h_a}{W_y} + y_a$$
- $$w = w_a \exp{\frac{t_w}{W_w}}$$
- $$h = h_a \exp{\frac{t_h}{W_h}}$$

与论文中稍有不同的是加入了一项权重项。代码中 $$W=(1,1,1,1)$$，另外通过 ```bbox_xform_clip=math.log(1000. / 16)``` 限制了 decode 时 ```exp``` 指数的上限，避免出现过大的值。

**对回归值的编码，是为了计算 loss 能有效收敛。而其他对框的操作，如计算 IoU 则要解码后进行。**

### 过滤冗余的输出

<img src="/images/2019-02-17-maskrcnn-benchmark/proposal.png" width="1000px"/>

对于每一个 feature map 的输出，根据 score 排序，保留至多前```PRE_NMS_TOP_N```个 box。需要注意的是这里是对整个 batch 内部进行排序，而不是单张图片内部进行排序。之后通过```clip_to_image```去除超出图片边界的框，通过```remove_small_boxes```去掉过小的框（这里阈值为 0，即不会删框）。然后进行 NMS，保留至多```POST_NMS_TOP_N```个框。

因为有五个特征输出，所以经过上一步会得到至多```5*POST_NMS_TOP_N```个 proposals。通过 score 统一排序，保留前```FPN_POST_NMS_TOP_N```个。这一步训练和测试稍有区别，训练时是整个 batch 排序，而测试时是每张图片分别排序。不过从源代码的注释上来看，这两个阶段应该会统一成后一种。

最后，如果是训练，则把 ground truth 也加入到 proposal 中，丰富正样本。

## RPN Loss

RPN 的 loss 分为两部分，分类的交叉熵 loss 和回归的 smoothL1 loss。然而并不是所有的预测结果都被计入 loss 当中。首先要区分出预测结果中哪些是正确的，哪些是错误的，哪些是忽略不计的。之后在正确分类和误分类中按固定比例采样，计算 loss，而不是一股脑全部算进去。计算 loss 除了需要 RPN 的两个输出，还需要 anchors 和 targets。

### Matcher

在```maskrcnn_benchmark/modeling/matcher.py```里实现的```Matcher```，就是通过对预测结果和真实结果进行两两对比，划分正确样本、错误样本和忽略的样本。```Matcher```在初始化时需要定两个阈值，```RPN.FG_IOU_THRESHOLD```和```RPN.BG_IOU_THRESHOLD```。

使用时，```Matcher```的输入是一个尺寸为```(target#, prediction#)```的矩阵。矩阵里的值是两两配对的IoU。通过求最大值找到对于每一个 prediction 最靠谱的 target，得到两个```(prediction#, )```的向量——最大值和相应的 target index。如果最大值小于```RPN.BG_IOU_THRESHOLD```，index 设成 -1，如果最大值在```RPN.BG_IOU_THRESHOLD```和```RPN.FG_IOU_THRESHOLD```之间，index 设成 -2，其余保持不变，仍然是原来的 target index，将这个结果记为```matches```。

代码中还设定了```allow_low_quality_matches=True```。就是将与每个 target IoU 最大（一个或多个，可能并列）的 prediction 也设为正样本。先求出每个 target 对应的最大的 overlap，一个长度为```(target#, )```的向量，之后在原始的输入的```(target#, prediction#)```矩阵中在每一行找到等于最大 overlap 的 prediction 的 index，在```matches```中将响应的 prediction 对应的 target index 恢复成 -1，-2 以外有效的 target。

### FG/BG Sampler

通过```Matcher```，每一个 prediction 都可以被归入负样本（0），正样本（1），忽略样本（-1）中的一个。为了均衡样本，可以通过预设参数```BATCH_SIZE_PER_IMAGE```，```POSITIVE_FRACTION```强制调整正负样本的比例。当正样本小于```BATCH_SIZE_PER_IMAGE * POSITIVE_FRACTION```时，负样本数量就是```BATCH_SIZE_PER_IMAGE - num_pos```。当正样本数量大于这个限制时，随机选取，剩下的则在负样本中随机选取。

### 求 loss

最后就可以整理采样的样本计算 loss 了。这里 target 需要先通过```BoxCoder``` encode 转化成网络输出的形式，再计算 loss。

代码在```maskrcnn_benchmark/layers/smooth_l1_loss.py```重新实现并扩展了 smoothL1 loss，加入了参数```beta```来控制 smoothL1 中二阶曲线的范围。

# RCNN

通过 FPN 和 RPN head，我们已经得到了从原图中提取的五个不同分辨率的 feature 以及 一系列 proposals。可以通过 RoI align 从 feature map 中提取 proposed RoI 的特征。

## RCNN head

项目中实现了三种 feature extractor，这里以```FPN2MLPFeatureExtractor```为例。RoI align 的输出结果经过 FC 转化成```ROI_BOX_HEAD.MLP_HEAD_DIM=1024```维，之后再经过一个 FC 保持维度不变。这个输出连接出两个全连接分支，一个输出```num_class```个神经元，一个输出```4*num_class```个神经元，分别作为分类和回归的结果。

## RCNN 后处理（推断）

如果是推断过程，还需要进行后处理。后处理包含三个任务：

1. 将 score 小于固定阈值```ROI_HEADS.SCORE_THRESH=0.05```的框删掉。
2. 对每一类的框进行 NMS，阈值取```ROI_HEADS.NMS=0.5```。
3. 将所有类别输出框的数量，通过 score 排序限制在```ROI_HEADS.DETECTIONS_PER_IMG=100```以内。

然而在此之前，需要将网络输出转化成 bounding box 的形式。这次的参照不再是 anchor，而是 RPN 给出的 proposal，```BoxCoder```的权重也变成了```ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)```

$$BoundingBoxes = BoxCoder(regression, proposal)$$


## 采样与计算 Loss（训练）

而如果是训练过程，就不需要后处理，但是需要另外两个步骤。

### 1. 采样

在 RoI align 之前，先对 proposal 进行采样。这里的采样过程和 RPN 求 loss 时的采样过程有些不同。这里同样用到了```Matcher```和```BalancedPositiveNegativeSampler```，但是参数上有很大不同。

|                              | RPN  | RCNN |
| ---------------------------- | ---- | ---- |
| FG\_IOU\_THRESHOLD           | 0.7  | 0.5  |
| BG\_IOU\_THRESHOLD           | 0.3  | 0.5  |
| allow\_low\_quality\_matches | T    | F    |
| BATCH\_SIZE\_PER\_IMAGE      | 256  | 512  |
| POSITIVE\_FRACTION           | 0.5  | 0.25 |

proposals 经过```Matcher```与 GT 框匹配，然后再经过```Sampler```采样，输出新的 proposals 进行 RoI align。
	

### 2. 求 loss。

首先用```BoxCoder```以 proposal 为参照将 GT 框编码。这里的权重也是```ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)```。损失函数和 RPN 一样仍然是交叉熵 loss 和 target 类的 smoothL1 loss，这里的 ```beta=1```，与 RPN 中不同。

# Mask

RCNN 的检测结果，会作为 proposals 输入到 mask head 中。

## Mask head

如果是训练过程，输出的检测结果同时包含了正/负样本，需要先滤去负样本，只保留正样本。

对于 mask 分支，可以选择是否与 rcnn 共享 feature extractor。如果共享，就是直接使用 rcnn feature extractor 的输出，也就是 RoI align 后两次全连接的特征。不过在官方实现中，是不 share 的，mask 分支经过 RoI align 后再通过 4 个```kernel=256```的```conv3x3 + Relu```。提取的特征经过一个转置卷积上采样到原来边长的两倍，再经过```conv1x1```输出```num_class```个 map 作为 mask。

## 计算 Loss

训练阶段，要将 target 中的 mask 转化成与输出一样的形式比对求交叉熵 loss。

annotation 中给出的 mask 是多边形的形式```[x0,y0,x1,y1,x2,y2,...]```，以图片左上角为参考系原点。先将多边形的坐标转换成以 box 左上角为参照系原点的坐标，之后```resize```到 mask 输出的大小（代码中是```28x28```），最后通过 coco 提供的解码工具把多边形转换成```28x28```的```tensor```。

## Mask 后处理

在推断阶段，mask 要同样要进行后处理。
与求 loss 时相反，这一次是要将输出的 mask 恢复到原图里。先将 mask bilinear resize 到 box 长宽的比例，再根据 box 左上角的点还原出 mask 在原图中的位置，最后通过阈值将 mask prob 中低于阈值的部分过滤掉。