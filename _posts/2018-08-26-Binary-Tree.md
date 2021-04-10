---
layout: post
title: 二叉树的遍历
tags: [python, leetcode, binary tree]
description: 二叉树遍历，包括前序、中序、后序遍历，递归与非递归实现。
date: 2018-08-26
feature_image: 
---

## 层级遍历（BFS）

二叉树的层级遍历可以看作是广度优先搜索。构建一个队列，每遍历当前一层时，都将下一层的子节点放进队列。下面的代码为了按行输出各级节点，所以使用了临时内存空间。
    
    def levelOrder(root):
        """
        @param root: A Tree
        @return: Level order a list of lists of integer
        """
        if root is None:
            return []
        res = []
        level = [root]
        while len(level) > 0:
            res.append([node.val for node in level])
            tmp = []
            for node in level:
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
            level = tmp
        return res

### 锯齿形层级遍历

    def zigzagLevelOrder(self, root):
        if root is None:
            return []
        res = []
        level = [root]
        l = 0
        while len(level) > 0:
            tmp = []
            rr = []
            while len(level) > 0:
                node = level.pop()
                rr.append(node.val)
                if l % 2:
                    if node.right:
                        tmp.append(node.right)
                    if node.left:
                        tmp.append(node.left)
                else:
                    if node.left:
                        tmp.append(node.left)
                    if node.right:
                        tmp.append(node.right)
            res.append(rr)
            level = tmp
            l += 1
        return res

## 前序遍历（DFS）

前序遍历的顺序是**根节点->左子节点->右子节点**。所以前序遍历的第一个点一定是根节点。

### 非递归实现

用栈作为容器，先将左子树加入栈中，一直到底，再弹出每个点加入右子树，一旦出现左子树就继续到底。

    def preorderTraversal(root):
        """
        @param root: A Tree
        @return: Preorder in ArrayList which contains node values.
        """
        st = []
        res = []
        tn = root
        while tn or len(st) > 0:
            while tn:
                st.append(tn)
                res.append(tn.val)
                tn = tn.left
            top = st.pop()
            tn = top.right
        return res

另一种实现方法

    def preorderTraversal(root):
        res = []
        if root is None:
            return res
        st = [root]
        while st:
            tn = st.pop()
            res.append(tn.val)
            if tn.right:
                st.append(tn.right)
            if tn.left:
                st.append(tn.left)
        return res

### 递归实现

    def preorderTraversalRecur(root, res):
        res.append(root.val)
        if root.left:
            preorderTraversalRecur(root.left, res)
        if root.right:
            preorderTraversalRecur(root.right, res)
        
    def preorderTraversal(root):
        res = []
        if root:
            preorderTraversalRecur(root, res)
        return res

## 中序遍历（DFS）

中序遍历的顺序是**左子节点->根节点->右子节点**。中序遍历的一个特点是，对于二叉搜索树，中序遍历一定是单调递增的。

### 非递归实现

与前序遍历相同，只是打印结果的位置不同。

    def inorderTraversal(root):
        st = []
        res = []
        tn = root
        while tn or len(st) > 0:
            while tn:
                st.append(tn)
                tn = tn.left
            top = st.pop()
            res.append(top.val)
            tn = top.right
        return res

### 递归实现

    def inorderTraversalRecur(root, res):
        if root.left:
            inorderTraversalRecur(root.left, res)
        res.append(root.val)
        if root.right:
            inorderTraversalRecur(root.right, res)
        
    def inorderTraversal(root):
        res = []
        if root:
            inorderTraversalRecur(root, res)
        return res

## 后序遍历（DFS）

### 非递归实现

第一步与前两种方式相同，一直到左子树的最底端。

    def postorderTraversal(root):
        st = []
        res = []
        tn = root
        lastVisit = None
        # go to the bottom of left tree
        while tn:
            st.append(tn)
            tn = tn.left
        while len(st) > 0:
            top = st.pop()
            # if no right child or right child is visitied, then visit current node
            if top.right is None or top.right is lastVisit:
                res.append(top.val)
                lastVisit = top
            else:
                st.append(top)
                tn = top.right
                while tn:
                    st.append(tn)
                    tn = tn.left
        return res

另一种方法使用两个栈，找出后序遍历的逆序，存在栈中。

    def postorderTraversal(root):
        if root == None:
            return
        st1 = []
        st2 = []
        tn = root
        st1.append(tn)
        while st1:
            tn = st1.pop()
            if tn.left:
                st1.append(node.left)
            if tn.right:
                st1.append(node.right)
            st2.append(tn)
        res = []
        while st2:
            res.append(st2.pop().val)
        return res

### 递归实现

    def postorderTraversalRecur(root, res):
        if root.left:
            postorderTraversalRecur(root.left, res)
        if root.right:
            postorderTraversalRecur(root.right, res)
        res.append(root.val)
        
    def postorderTraversal(root):
        res = []
        if root:
            postorderTraversalRecur(root, res)
        return res