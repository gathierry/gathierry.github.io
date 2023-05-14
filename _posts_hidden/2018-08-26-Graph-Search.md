---
layout: post
title: 图的搜索
tags: [python, leetcode, graph]
description: 图中的一些基本算法，包括广度优先搜索、深度优先搜索。
date: 2018-08-26
feature_image: 
---

首先看一下有向图和无向图中节点的定义

    class DirectedGraphNode:
        def __init__(self, x):
            self.label = x
            self.neighbors = []
            
    class UndirectedGraphNode:
        def __init__(self, x):
            self.label = x
            self.neighbors = [] 

<!--more-->

## 深度优先搜索 (DFS)

### 非递归实现

使用栈来储存节点，每次弹出一个节点，将其所有的相邻节点全部压入栈内。弹出栈时打印该节点。  
如果在有向图中，压入栈之前要考虑该节点是否已经被经过。

    def dfs(graph):
        """
        @param: graph: A list of Directed graph node
        @return: Any order for the given graph.
        """
        queue = []
        res = []
        node = graph[0]
        queue.append(node)
        while queue:
            v = queue.pop()
            res.append(v)
            for w in v.neighbors:
                if w not in res and w not in queue:
                    queue.append(w)
        return res

### 递归实现

    def dfs_recursive(graph, vertex, path=[]):
         path.append(vertex)
         for neighbor in vertex:
             if neighbor not in path:
                 path = dfs_recursive(graph, neighbor, path)
         return path

## 广度优先搜索（BFS）

### 非递归实现

广度优先搜索使用了一个队列，每次从头部退出一个节点时，将与它相邻的节点全部加入队列。在加入队列的同时，打印当前节点。

    def bfs(graph):
        """
        @param: graph: A list of Directed graph node
        @return: Any order for the given graph.
        """
        queue = []
        res = []
        node = graph[0]
        queue.append(node)
        res.append(node)
        while queue:
            v = queue.pop(0)
            for w in v.neighbors:
                if w not in res:
                    queue.append(w)
                    res.append(w)
        return res
