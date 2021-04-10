---
layout: post
title: Hadoop笔记
tags: [Hadoop, machine learning, Java]
description: 这篇文章简单介绍了Hadoop，并且记录了在OSX El Capitan上安装配置Hadoop的过程。
date: 2016-04-12
feature_image:
---
## MapRduce
MapReduce是一个分布式计算的软件框架。它将单个作业分成小份，输入数据切片分发到计算机集群上的每个节点，每个节点只在本地数据上做运算，这个阶段称作map，对应的运算代码成为mapper。每个mapper的输出通过某种方式组合，再被分成小份发送到各个节点进行下一步处理，这个过程称作reduce，相应的代码叫做reducer。reducer输出的结果就是最终结果。

## Hadoop
Hadoop是MapReduce框架的一个开源实现。除了分布式计算之外，Hadoop也自带分布式文件系统——[HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html)，由一个Namenode和多个Datanode组成。

## 在OSX El Capitan上的安装与配置Hadoop 2.7.2

### 使用homebrew安装Hadoop
在终端中输入```brew install hadoop```命令   
Hadoop会被自动安装在```/usr/local/Cellar/hadoop/2.7.2```目录下

### 配置Java
Hadoop是一个Java的开源项目，所以需要计算机上有Java环境。通过下面的命令找到JVM的目录

    $/usr/libexec/java_home 
    /Library/Java/JavaVirtualMachines/jdk1.8.0_77.jdk/Contents/Home

并设置环境变量

    export JAVA_HOME={your java home directory}

### 配置SSH
首先，在System Preferences > Sharing 里开启Remote Login选项。  
使用```ssh localhost```命令检测是否可以连接。  
如果无法连接，在终端输入

    $ssh-keygen -t dsa -P '' -f ~/.ssh/id_dsa
    $cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys #设置免密码登录

附[```ssh-keygen```命令用法](http://www.delafond.org/traducmanfr/man/man1/ssh-keygen.1.html)

现在就可以通过```ssh localhost```发起连接了，如需退出连接，按```ctrl+D```或输入```exit```。

### 配置Hadoop
配置文件在```/usr/local/Cellar/hadoop/2.7.2/libexec/etc/hadoop```目录下。

#### hdfs-site.xml:设置副本数为1

    <configuration>
        <property>
            <name>dfs.replication</name>
            <value>1</value>
        </property>
    </configuration>

#### core-site.xml:文件访问系统端口

    <configuration>
      <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
      </property>
    </configuration>

#### mapred-site.xml:设置MapReduce使用的框架为[Yarn](https://hadoop.apache.org/docs/r2.7.1/hadoop-yarn/hadoop-yarn-site/YARN.html)

    <configuration>
      <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
      </property>
    </configuration>

#### yarn-site.xml

    <configuration>
        <property>
            <name>yarn.nodemanager.aux-services</name>
            <value>mapreduce_shuffle</value>
        </property>
    </configuration>

## 运行Hadoop
通过以下配置简化Hadoop的启动和停止过程。

    export PATH=$PATH:/usr/local/Cellar/hadoop/2.7.2/bin:/usr/local/Cellar/hadoop/2.7.2/sbin
    alias hstart="start-dfs.sh;start-yarn.sh"
    alias hstop="stop-yarn.sh stop-dfs.sh"

格式化文件系统，

    $hdfs namenode -format

通过```hstart```启动 NameNode daemon, DataNode daemon（UI界面[http://localhost:50070/](http://localhost:50070/)）和 ResourceManager daemon, NodeManager daemon（UI界面[http://localhost:8088/](http://localhost:8088/)）

建立用户空间

    $hdfs dfs -mkdir /user
    $hdfs dfs -mkdir /user/{username}

附[```dfs```命令用法](http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/FileSystemShell.html)

查看进程```jps```
    
    #进程号 | 进程名
    7873 DataNode
    8257 Jps
    8114 ResourceManager
    7780 NameNode
    8212 NodeManager
    7992 SecondaryNameNode

### test case
把本地 etc/hadoop 下的一些文件上传到 HDFS的 input 中：

    $hdfs dfs -put /usr/local/Cellar/hadoop/2.7.2/libexec/etc/hadoop input

可以在``` /user/{username}/input```中找到上传的文件。

运行Hadoop提供的例子：在上传的数据中使用 MapReduce 运行 grep， 计算以dfs开头的单词出现的次数，结果保存到 output 中：

    $hadoop jar /usr/local/Cellar/hadoop/2.7.2/libexec/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.2.jar grep input output 'dfs[a-z.]+'

查看运行结果：

    $hdfs dfs -cat output/part-r-00000

文件系统可以通过[http://localhost:50070/explorer.html#/](http://localhost:50070/explorer.html#/)查看。

删除刚才生成的文件:

    $hdfs dfs -rm -r /user/{username}/input
    $hdfs dfs -rm -r /user/{username}/output

停止Hadoop ```hstop```。

至此我们完成了Hadoop的安装和配置，并且运行了一个小例子。后面的文章中会记录一些在本地Hadoop上进行的数据挖掘方面的实验。



## 编写mapper和reducer

这里我们用python实现一个计算均值和方差的程序。

*mrMeanMapper.py*

    import sys
    from numpy import mat, mean, power
    def read_input(file):
        for line in file:
            yield line.rstrip()
    
    input = read_input(sys.stdin)
    input = [float(line) for line in input]
    numInputs = len(input)
    input = mat(input)
    sqInput = power(input, 2)
    print "%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput))
    
    print >> sys.stderr, "report: still alive"

*mrMeanReducer.py*

    import sys
    from numpy import mat, mean, power
    def read_input(file):
        for line in file:
            yield line.rstrip()
    
    input = read_input(sys.stdin)
    mapperOut = [line.split('\t') for line in input]
    cumVal = 0.0
    cumSumSq = 0.0
    cumN = 0.0
    for instance in mapperOut:
        nj = float(instance[0])
        cumN += nj
        cumVal += nj * float(instance[1])
        cumSumSq += nj * float(instance[2])
    mean = cumVal / cumN
    varSum = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean) / cumN
    print "%d\t%f\t%f" % (cumN, mean, varSum)
    print >> sys.stderr, "report: still alive"

*inputFile.txt* 是一个多行的文本文件，每一行有一个0到1之间的有理数。

通过

    $cat inputFile.txt | python mrMeanMapper.py | python mrMeanReducer.py

测试，如果正常输出

    report: still alive
    100	0.509570	0.084777
    report: still alive

则说明程序可以正常运行。这里被打印的inputFile.txt作为输入传入mapper，同理mapper的输出作为输入传入reducer。


## 启动本地Hadoop

如上一篇笔记所描述的那样，我们通过```ssh localhost```和```hstart```在本地启动Hadoop。

## Hadoop实验

下面就可以在Hadoop上进行实验了。

首先将inputFile.txt上传到HDFS中，存成名为*mrmean-i*的文件：

    $hadoop fs -copyFromLocal inputFile.txt mrmean-i

然后通过streaming，在Hadoop上执行python脚本：

    $hadoop jar /usr/local/Cellar/hadoop/2.7.2/libexec/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar -input mrmean-i -output mrmean-o -file mrMeanMapper.py mrMeanReducer.py -mapper "python mrMeanMapper.py" -reducer "python mrMeanReducer.py"

这里要注意选项file不能少，输出文件在自动生成的文件夹*mrmean-o*中。通过以下命令可以查看和下载：

    $hadoop fs -cat mrmean-o/part-00000
    $hadoop fs -copyToLocal mrmean-o/part-00000

最后执行```hstop```, ```exit```关闭Hadoop并断开ssh连接。

到这里我们就完成了一个简单的mapReduce程序。后面的学习中，将要考虑如何把机器学习的各种算法使用的分布式系统上来。




*参考资料：*  

- http://hustlijian.github.io/tutorial/2015/06/19/Hadoop入门使用.html  
- http://zhongyaonan.com/hadoop-tutorial/setting-up-hadoop-2-6-on-mac-osx-yosemite.html  
- 机器学习实战 by Peter Harrington
