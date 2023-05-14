---
layout: post
title: Fortran Notes
tags: [fortran]
description: Fotran学习笔记，参考自http://micro.ustc.edu.cn/Fortran/ZJDing/
date: 2015-04-21
feature_image: images/2015-04-21-Fortran-Notes/title.JPG
---
Fortran语言由IBM在1957年开发出来，是第一个被正式采用并流传至今的高级语言，直至今天，在数值计算领域仍然被广泛应用。历史上陆续有 Fortran77, Fortran90, Fortran 95、Fortran 2003、Fortran 2008 等标准。

<!--more-->

# 1.3 Fortran程序的基本组成

## 1.3.3 程序组成

### 程序单位

- PROGRAM
- SUBROUTINE/FUNCTION/BLOCK DATA/MODULE
MODULE通过USE与主程序相联系
BLOCK DATA不推荐使用，功能被MODULE，USE取代
- CONTAINS

### 程序体和语句顺序

1. 单位起始语句：PROGRAM(整个程序唯一), FUNCTION, SUBROUTINE, MODULE
2. USE
3. 说明部分：PARAMETER & DATA - 定义：派生类型，接口快，变量类型，语句函数
4. 执行部分：DATA - 执行结果
5. CONTAINS
6. 内部模块或内部过程
7. END

# 2.3 选择结构

## 2.3.1 IF构造

```
IF (...) THEN
    ...
ELSE IF (...) THEN
    ...
ELSE
    ...
END IF
```

## 2.3.2 CASE构造

```
SELECT CASE (...)
    CASE (...)
        ...
    CASE (...)
        ...
    CASE DEFAULT
        ...
END SELECT
```

- 整型：CASE (值1，值2) / (下界 :)  / ( : 上界) / (下界 : 上界)
- 字符型：CASE (字符)
- 逻辑型：CASE (.TRUE.) / (.FALSE.)

# 3.1 单纯循环

## 3.1.2 有变量DO构造

```
DO 循环变量=初值，终值（, 步长）
END DO
```

- 隐DO循环
I=m1,m2,m3形式。它不是独立语句，只是用作为读写语句的输入输出表中一个组成部分，用来控制重复读写的次数

# 3.2 条件循环

## 3.2.1 无变量DO构造

- EXIT [DO构造名] = break
- CYCLE [DO构造名] = continue

## 3.2.2 DO WHILE构造

```
DO WHILE (...)
END DO
```

# 4.1 数据类型和属性

*类型 (种别)( ,属性说明表) :: 变量名表(=初值)* 

Eg: 
```REAL(KIND=2), DIMENSION(1:10):: X,Y```

## 4.1.1 类型，赋值
*DATA 变量名表/初值表/*

## 4.1.2 种别

种别值确定了数据的大小范围和精度 

- 种别值：KIND=k
- 种别函数：KIND(X), SELECTED_REAL_KIND(n, m) ...

## 4.1.3 属性

- PARAMETER = final
- DIMENSION （详见数组部分）

# 4.2 非数值型数据

## 4.2.2 字符型数据

- CHARACTER(LEN = 整型字符长度表达式, KIND = 种别值)(, 属性说明) :: 变量名表(=初始值)
- Substring : A(start:end)
- 拼接字符：str1//str2
- 字符比较：== 或 /=
- 函数：
LEN(str)，LEN_TRIM(str) 不记尾部空格
INDEX(string, substring)，VERIFY(string, map)返回一个正整数，指明string第一个不在map中的字符
TRIM 去除尾部空格

# 4.3 派生数据类型
- 定义
TYPE[,访问属性说明(PUBLIC/PRIVATE)::] 派生类型名 
成员1类型说明
……
成员n类型说明
END TYPE [派生类型名]
Eg：

```
TYPE STUDENT_TYPE
    INTEGER(4) :: number
    TYPE(CLASS_TYPE) :: class
    CHARACTER(LEN=10) :: name, address
    TYPE(SCORES_TYPE) :: scores
END TYPE STUDENT_TYPE
```

- 初始化
STUDENT_TYPE(n, class1, "Ming", "ming@me.com", scores1)
- 访问属性 %
student1%class

# 5.1 数组类型与定义

## 5.1.1 定义数组
REAL a(x, y, z)  rank=3，size=xyz，shape=(x, y, z)
下界缺省值为1

# 5.2 数组赋值与运算

## 5.2.2 运算
- 基本：+,-,*./,**
- 函数：
    - MATMUL(A, B)
    - DOT_PRODUCT(A, B)
    - SUM
    - SIZE
    - TRANSPOSE

# 6.1 程序单元结构

## 6.1.2 主程序
```
PROGRAM 程序名
    说明部分
    执行部分
CONTAINS
    内部过程
END PROGRAM 程序名
```

## 6.1.3 过程

- 外部过程
    - FUNCTION
    ```
    FUNCTION
        说明部分
        执行部分
    CONTAINS
        内部过程
    END FUNCTION 函数名
    ```
    - SUBROUTINE
    ```
    SUBROUTINE
        说明部分
        执行部分
    CONTAINS
        内部过程
    END FUNCTION 子程序名
    ```
- 内部过程
同上但不再含有CONTAINS部分

# 6.2 过程

##6.2.2 外部过程

- 子程序 SUBROUTINE
    ```
    SUBROUTINE 子程序名(PARA1, PARA2, ...)

    END SOUBROUTINE 子程序名
    ```
    ```CALL 子程序名(PARA1, PARA2, ...)```调用子程序

- 函数 FUNCTION
不同之处在于函数返回一个值，且通常不改变哑元的值。
```
FUNCTION 函数名(PARA1, PARA2, ...)
END FUNCTION 函数名
```
```函数名(PARA1, PARA2, ...)```调用

## 6.2.5 过程接口

- INTERFACE

```
INTERFACE [类属说明]
    [接口体]…(只包括说明部分)
    [模块过程语句]…（MODULE PROCEDURE 过程名表）
END INTERFACE [类属说明]
```

Eg.

```
INTERFACE
    SUBROUTINE EXT1(X,Y,Z)
        REAL,DIMENSI0N(100,100)::X,Y,Z
    END SUBROUTINE EXT1
    
    SUBROUTINE EXT2(X,Z)
        REAL X
        COMPLEX(KIND=4) Z(2000)
    END SUBROUTINE EXT2

    FUNCTION EXT3(P,Q)
        LOGICAL EXT3
        INTEGER P(1000)
        LOGICAL Q(1000)
    END FUNCTION EXT3
END INTERFACE
```

# 6.3 模块

## 6.3.2 模块的用法

- 定义模块

```
MODULE 模块名
    类型说明部分
    [CONTAINS
        内部过程
        …
        [内部过程]]
END MODULE [模块名]
```

**其中只有说明部分，没有执行部分**

- 引用模块
```USE 模块名1, 模块名2, … 模块名n```
```USE模块名, ONLY : [,仅用名列表]```
ONLY只取模块中的一部分变量与本程序共享

## 6.3.3 模块的应用
- 全局数据
将所有全局数据整理到一个模块中以供使用
- 过程共享
把一些过程放在一个模块中统一说明，通过USE MODULE PROCEDURE 调用
Eg.

```
PROGRAM CHANGE_KIND
    USE Module1
    INTERFACE DEFAULT
        MODULE PROCEDURE Sub1, Sub2
    END INTERFACE
...
END PROGRAM
```

```
MODULE Module1
    CONTAINS
        FUNCTION Sub1(y)
            REAL(8) y
            sub1 = REAL(y)
        END FUNCTION
        FUNCTION Sub2(z)
            INTEGER(2) z
            sub2 = INT(z)
        END FUNCTION
END MODULE
```

- 公共派生类型
可用来封装一些派生类型，供其它程序单元公用

