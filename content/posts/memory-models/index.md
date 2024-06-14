+++
title = 'Memory Models读后感'
date = 2022-10-09T10:15:19+08:00
author = "Skyan"
tags = ["programming"]
ShowToc = true
ShowBreadCrumbs = true
+++

读了Russ Cox关于Memory Models的三篇综述文章：
* [Hardware Memory Models](https://research.swtch.com/hwmm)
* [Programming Language Memory Models](https://research.swtch.com/plmm)
* [Updating the Go Memory Model](https://research.swtch.com/gomm)

读完之后终于搞明白Memory Models的大体脉络。

首先在一个单核单线程的程序中，是没有内存模型问题的。随着多核多线程的引入，程序开始实现并行化提升性能，同时编译器、操作系统以及硬件针对并行化做了大量的优化，这就出现了很多新的并发数据访问问题，就需要设计规范的内存模型来解决。

内存模型分为硬件内存模型和编程语言内存模型两个层面。

其中硬件内存模型分为以Intel CPU为代表的x86-TSO模型和以ARM/POWER为代表的Relaxed Memory Model模型。前者模型中，多核对写入主存的数据提供全局序，后者模型中，多核之间不存在全局序的概念。为了协调不同硬件，又有DRF-SC(Data-Race-Free Sequential Consistency)模型，该模型是一种能让大部分硬件都能接受实现的顺序内存访问模型，这样DRF-SC可以作为硬件内存模型的统一标准。

编程语言内存模型的定义分为两个流派：

1. 以Java，Javascript为代表的happen-before流派。特点是通过happen-before语义来定义语言的哪些方面如何实现数据的同步。该流派规范定义最早，也最为经典。Java9以后增加了弱同步，类似C++的弱同步(acquire/release)。
2. 以C/C++，Rust，Swift等为代表的强-弱-无顺序流派。特点是分为三类同步语义，强代表前后顺序一致，和Java的happen before语义一致。弱代表在部分协调条件下的一致。无代表没有同步。

按照作者的介绍，这两个流派的内存模型都存在缺陷。Java的内存模型定义并不严格，还是有非因果、非一致的情况发生。C++的内存模型有很多未定义的情况，尤其是弱同步和无同步语义其实存在很多坑，甚至在不同硬件下运行结果可能还不一致。

Go语言采取中间流派，即只通过happen-before来定义同步语义，只支持顺序一致，不支持弱同步，不支持无同步，严格定义，同时约束编译器的优化，保证最终结果可预测，即使有未定义的情况发生，也能约束结果在有限的范围内。

最终结论是，截止目前2022年，内存模型还处于不断探索和研究的方向，Go的这个尝试也是建立在无数前人理论的基础之上，未来有待更加规范定义的模型出现。
