---
layout: post
title: tf.keras RNN 接口总结
tags: [deep learning, tensorflow]
description: tf.keras 中 RNN 相关接口的总结，包括LSTM，GRU，LSTMCell，GRUCell，RNN，Bidirection
date: 2021-05-09
feature_image: images/2021-05-09-tf-rnn-layers/RNN-vs-LSTM-vs-GRU.png
usemathjax: true
---
在 tensorflow.keras.layers API 下面，与时间序列有关的层包括，`LSTM`, `LSTMCell`, `GRU`, `GRUCell`, `RNN`, `Bidirectional`. 因为类的名称比较相似，容易混淆，这篇文章总结了它们的在不同参数条件下，输入输出的数据形式。文中所有代码基于 `tensorflow==2.4.1`

<!--more-->

## Setup

一个常见的时间序列输入张量的形状，是`(N, T, C)`. 另外我们用`H` 表示下文中 `LSTM` 和`GRU`的`hidden_size`.

```python
N = 2  # batch size
T = 3  # time frame
C = 4  # channel
H = 1  # hidden size
x = tf.random.uniform([N, T, C], dtype=tf.float32)
```

## LSTM

`tf.keras.layers.LSTM`包含了两个影响输出的参数`return_state`和`return_sequences`.  `return_state`表示是否返回最终的隐藏状态；`return_sequences`表示是否只返回最后一个时间的输出。

```python
lstm_layer = tf.keras.layers.LSTM(units=H, return_state=False, return_sequences=False)
pprint(lstm_layer(x))  # (N, H)
"""
tf.Tensor(
[[-0.24242838]
 [-0.23425369]], shape=(2, 1), dtype=float32)
"""
```

```python
lstm_layer = tf.keras.layers.LSTM(units=H, return_state=True, return_sequences=False)
print(lstm_layer(x))  # [output (N, H), state h (N, H), state c (N, H)]
# obviously output == state h
"""
[
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.01009315],
       [-0.03920766]], dtype=float32)>,
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.01009315],
       [-0.03920766]], dtype=float32)>,
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.02487712],
       [-0.08066361]], dtype=float32)>
]
"""
```

```python
lstm_layer = tf.keras.layers.LSTM(units=H, return_state=False, return_sequences=True)
print(lstm_layer(x))  # (N, T, H)
"""
tf.Tensor(
[[[0.19904119]
  [0.22823204]
  [0.28809297]]

 [[0.16896814]
  [0.19136839]
  [0.2848598 ]]], shape=(2, 3, 1), dtype=float32)
"""
```

```python
lstm_layer = tf.keras.layers.LSTM(units=H, return_state=True, return_sequences=True)
print(lstm_layer(x))  # [output seq (N, T, H), state h (N, H), state c (N, H)]
# state h == output[:, -1, :]
"""
[
<tf.Tensor: shape=(2, 3, 1), dtype=float32, numpy=
array([[[-0.13393377],
        [-0.15039259],
        [-0.2113396 ]],

       [[-0.09126788],
        [-0.16637455],
        [-0.1658781 ]]], dtype=float32)>,
 <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.2113396],
       [-0.1658781]], dtype=float32)>,
 <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.31270757],
       [-0.2806935 ]], dtype=float32)>
]
"""
```

## LSTMCell

`LSTMCell` 与`LSTM`的区别在于，后者的输入是一个`(N, T, C)`的序列，而前者的输入仅仅是一个时间点的值`(N, C)`，也就是说，`LSTMCell`表示的是`LSTM`的一步。所以输入的张量中还要加入上一时间点的状态。

```python
lstmcell_layer = tf.keras.layers.LSTMCell(units=H)
state_h = tf.random.uniform([N, H])
state_c = tf.random.uniform([N, H])
pprint(lstmcell_layer(x[:, 0, :], states=[state_h, state_c]))
# (output (N, H), [state h (N, H), state c (N, H)])
# output == state h
"""
(
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.02514206],
       [ 0.16296071]], dtype=float32)>,
 [
 <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.02514206],
       [ 0.16296071]], dtype=float32)>,
  <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.07196014],
       [ 0.4330987 ]], dtype=float32)>
 ]
)
"""
```

用`tf.keras.layers.RNN`包装后的`LSTMCell` 与`tf.keras.layers.LSTM`等价，但是[前者无法使用cuDNN的加速](https://www.tensorflow.org/guide/keras/rnn#using_cudnn_kernels_when_available)。

## GRU

`GRU`与`LSTM`的参数很相似，但是不同于LSTM的两个内部状态，GRU只包含一个状态，且这个状态就是每一步的输出。

```python
gru_layer = tf.keras.layers.GRU(units=H, return_state=False, return_sequences=False)
pprint(gru_layer(x))  # (N, H)
"""
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[0.2544468 ],
       [0.19708085]], dtype=float32)>
"""
```

```python
gru_layer = tf.keras.layers.GRU(units=H, return_state=True, return_sequences=False)
pprint(gru_layer(x))  # [output (N, H), state (N, H)]
# output == state
"""
[
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.24980411],
       [-0.1053919 ]], dtype=float32)>,
 <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.24980411],
       [-0.1053919 ]], dtype=float32)>
]
"""
```

```python
gru_layer = tf.keras.layers.GRU(units=H, return_state=False, return_sequences=True)
pprint(gru_layer(x))  # [N, T, H]
"""
<tf.Tensor: shape=(2, 3, 1), dtype=float32, numpy=
array([[[0.24226524],
        [0.50201494],
        [0.6352898 ]],

       [[0.26033005],
        [0.41407382],
        [0.60243475]]], dtype=float32)>
"""
```

```python
gru_layer = tf.keras.layers.GRU(units=H, return_state=True, return_sequences=True)
pprint(gru_layer(x))  # [output (N, T, H), state (N, H)]
# output[:, -1, :] == state
"""
[
<tf.Tensor: shape=(2, 3, 1), dtype=float32, numpy=
array([[[-0.2569086 ],
        [-0.25784296],
        [-0.18923259]],

       [[-0.1484761 ],
        [-0.33333457],
        [-0.32789811]]], dtype=float32)>,
 <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-0.18923259],
       [-0.32789811]], dtype=float32)>
]
"""
```

## GRUCell

与`LSTMCell`相似，但只需要输入一个状态张量。

```python
grucell_layer = tf.keras.layers.GRUCell(units=H)
state_h = tf.random.uniform([N, H])
pprint(grucell_layer(x[:, 0, :], states=state_h))
# (output (N, H), state h (N, H))
# output == state h
"""
(
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[0.78630126],
       [0.6398365 ]], dtype=float32)>,
 <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[0.78630126],
       [0.6398365 ]], dtype=float32)>
)
"""
```

## Bidirectional

用`tf.keras.layers.Bidirectinonal`包裹的`LSTM`,`GRU`,`RNN ` ，在默认情况下(`merge_mode='concat'`)，输出的 channel 数量由 `H` 变为`2H`.


## 参考资料
1. https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
2. https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell
3. https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
4. https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell
5. https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
6. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
7. https://www.tensorflow.org/guide/keras/rnn