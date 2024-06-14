+++
title = '并行编程之内存管理(总结)'
date = 2022-06-14T19:25:10+08:00
author = "Skyan"
tags = ["programming", "parallel programming"]
ShowToc = true
ShowBreadCrumbs = true
+++


这篇总结并行编程的三种常见的内存管理方法，三种方法如下：
* Reference Counting：[并行编程之内存管理(一)]({{< ref "parallel-programming-1" >}})
* Hazard Pointer：[并行编程之内存管理(二)]({{< ref "parallel-programming-2" >}})
* RCU：[并行编程之内存管理(三)]({{< ref "parallel-programming-3" >}})

三种方法各有利弊，优缺点对比如下：
| | 引用计数(Reference Counting) | Hazard Pointer | RCU |
| --- | --- | --- | --- |
| 读性能 | 低，不可扩展 | 高，可扩展 | 高，可扩展 |
| 可保护的对象数量 | 可扩展 | 不可扩展 | 可扩展 |
| 保护周期 | 支持长周期 | 支持长周期 | 用户必须限制保护周期长度 |
| 遍历是否需要重试 | 如果和删除冲突需要重试 | 如果和删除冲突需要重试 | 不需要 |
| 存在性保证 | 复杂 | 保证 | 保证 |
| 读竞争 | 高 | 无 | 无 |
| 对象遍历开销 | CAS原子操作，内存屏障，cache missing | 内存屏障 | 无 |
| 读遍历机制 | 无锁 | 无锁 | 有限制的无等待(wait free) |
| 读引用获取 | 可能失败 | 可能失败 | 不会失败 |
| 内存开销 | 有限 | 有限 | 无限 |
| 内存回收机制 | 无锁 | 无锁 | 阻塞 |
| 自动回收 | 是 | 部分 | 部分 |

已经有很多C++项目，包括开源和大厂内部项目，都开始采用Hazard Pointer和RCU来实现并发数据结构，而且C++标准委员会也已经在讨论将这两个组件加入到C++26标准中。

时代在变，并行编程技术也在突飞猛进。多线程多核(multi-core)技术，甚至甚多核(many-core)技术都在飞速发展，加上各种并行编程范式的应用，可以预见到在一段时期内，并行编程将面临百花齐放，技术爆发的局面。
