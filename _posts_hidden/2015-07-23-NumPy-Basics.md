---
layout: post
title: NumPy Tutorial
tags: [python, numpy]
description: NumPy's fundamental data structure ndarray
date: 2015-07-23
feature_image: 
---
## Multidimensional data structure - ndarray
	In [3]: import numpy as np

### Initiation
- vector column  

		In [4]: vector = [1, 2, 3]
		In [5]: arr_vec = np.array(vector)
		In [6]: arr_vec
		Out[6]: array([1, 2, 3])

- matrix

		In [10]: matrix = [[1, 2, 3], [4, 5, 6]]
		In [11]: arr_mat = np.array(matrix)
		In [12]: arr_mat
		Out[12]: 
		array([[1, 2, 3],
		       [4, 5, 6]])

- others

		In [13]: np.zeros(3)
		Out[13]: array([ 0.,  0.,  0.])
		In [14]: np.ones((2, 3))
		Out[14]: 
		array([[ 1.,  1.,  1.],
		       [ 1.,  1.,  1.]])
		In [15]: np.eye(2)
		Out[15]: 
		array([[ 1.,  0.],
		       [ 0.,  1.]])

### Scalar operations (+ - * / **)
Instead of the multiplication between matrix, * is the multiplication between 2 elements of the same index

	In [16]: arr_mat
	Out[16]: 
	array([[1, 2, 3],
	       [4, 5, 6]])
	In [17]: arr_mat * arr_mat
	Out[17]: 
	array([[ 1,  4,  9],
	       [16, 25, 36]])

### Indexing and slicing
- vector

		In [19]: arr = np.arange(10)
		In [20]: arr
		Out[20]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		In [21]: arr[3]
		Out[21]: 3
		In [22]: arr[0:3]
		# 0:3 means 0, 1, 2
		Out[22]: array([0, 1, 2])
		In [23]: arr[0:3] = 10
		In [24]: arr
		Out[24]: array([10, 10, 10,  3,  4,  5,  6,  7,  8,  9])
  
  A slice is part of the array. If the value of a slice is changed, so will be the value in the original array

		In [28]: slice = arr[5:7]
		In [29]: slice[1] = 0
		In [30]: arr
		Out[30]: array([10, 10, 10,  3,  4,  5,  0,  7,  8,  9])
  
  But it could be copied if followed by ```arr[5:7].copy()```. Howerver, the code below won't change the array either. We'll talk about this later in the *Fancy indexing* section.
  
		In [36]: slice = arr[0]
		In [37]: slice = 99
		In [38]: arr
		Out[38]: array([10, 10, 10,  3,  4,  5,  0,  7,  8,  9])

- matrix

		In [44]: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
		In [45]: arr2d = np.array(matrix)
		In [46]: arr2d[2]
		Out[46]: array([7, 8, 9])
		In [47]: arr2d[0][2]
		Out[47]: 3
		In [56]: arr2d[:2, 1:]
		# [0:2] and [1:max+1]
		Out[56]: 
		array([[2, 3],
		       [5, 6]])

### Fancy indexing
	In [61]: new_arr = arr2d[[0, 2]]
	In [62]: new_arr
	Out[62]: 
	array([[1, 2, 3],
	       [7, 8, 9]])

Fancy indexing is different from the slicing. The fancy indexing is a copy of the value while the slicing is a copy of the reference.

### Transpose

	In [63]: new_arr.T
	Out[63]: 
	array([[1, 7],
	       [2, 8],
	       [3, 9]])

### where

    In [5]: xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    In [6]: yarr = xarr + 1
    In [8]: cond = np.array([True, False, True, True, False])
    In [9]: result = np.where(cond, xarr, yarr)
    In [10]: result
    Out[10]: array([ 1.1,  2.2,  1.3,  1.4,  2.5])

```xarr``` and ```yarr``` could also be 2 scalar variables. With ```where``` the matrix in the condition will be tested. Replace the elements passed with xarr (or the value at the same position if xarr is a matrix). Replace the ones failed with elements in yarr.

### Tricks for the boolean array

    % use sum to count the positive number
    In [16]: arr = np.random.randn(100)
    In [17]: (arr > 0).sum()
    Out[17]: 50

```random``` is the lib for the generation of randomn numbers

### sort

```arr.sort(0)``` sort each column
```arr.sort(1)``` sort each row