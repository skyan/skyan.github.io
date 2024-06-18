+++
title = 'X下云'
date = 2023-11-22T20:02:00+08:00
author = "Skyan"
tags = ["cloud computing"]
ShowToc = true
ShowBreadCrumbs = true
+++

## 背景
2022 年 10 月 27 日马斯克以 440 亿美元完成 Twitter私有化交易并改名X.com，整整一年过去后，X的工程架构也发生巨大的变化，今年10月27日X工程团队发表了一个推文，总结一年的巨大变化，我觉得这个帖子很有代表性，给所有做engineering的同学新的启发。

## 原文

> https://twitter.com/XEng/status/1717754398410240018

This has been a year full of engineering excellence that sometimes can go unnoticed. Besides all the visible changes you see on our app, here are some of the most important improvements we have made under the hood.
* Consolidated the tech stacks for For you, Following, Search, Profiles, Lists, Communities and Explore around a singular product framework.
* Completely rebuilt the For you serving and ranking systems from the ground up, resulting in a decrease 90% reduction in lines of code from 700K to 70K, a 50% decrease in our compute footprint, and an 80% increase in the throughput of posts scored per request.
* Unified the For you and video personalization and ranking models, which significantly improved video recommendation quality.
* Refactored the API middleware layer of our tech stack and in doing so simplified the architecture by  removing more than 100K lines of code and thousands of unused internal endpoints and eliminating unadopted client services.
* Reduced post metadata sourcing latency by 50%, and global API timeout errors by 90%.
* Blocked bots and content scrapers at a rate +37% greater than 2022. On average, we prevent more than 1M bots signup attacks each day and we’ve reduced DM spam by 95%.
* Shutdown the Sacramento data center and re-provisioned the 5,200 racks and 148,000 servers, which generated more than $100M in annual savings. In total, we freed up 48 MW of capacity and tore down 60k lbs. of network ladder rack before re-provisioning it to other data centers.
* Optimized our usage of cloud service providers and began doing much more on-prem. This shift has reduced our monthly cloud costs by 60%. Among the changes we made was a shift of all media/blob artifacts out of the cloud, which reduced our overall cloud data storage size by 60%, and separately, we succeeded in reducing cloud data processing costs by 75%.
* Built on-prem GPU Supercompute clusters and designed, developed, and delivered 43.2Tbps of new network fabric architecture to support the clusters.
* Scaled network backbone capacity and redundancy, which resulted in $13.9M/year in savings.
* Started automated peak traffic failover tests to validate the scalability and availability of the entire platform continuously.

-----

## 中文版
这是工程的卓越一年，但有时却以不被关注的方式进行。除了你在我们的应用程序上看到的所有可见的变化之外，以下是我们在幕后做出的一些最重要的改进。
* 围绕单一的产品框架整合了For you、Following、Search、Profiles、Lists、Communities和Explore的技术堆栈。
* 从头开始完全重建了“For you”和排名系统，使代码行从700K减少了90%，减少到70K，我们的计算开销减少了50%，每个请求的帖子吞吐量增加了80%。
* 统一了For you和视频个性化和排名模型，显著提高了视频推荐质量。
* 重构了我们技术堆栈中的API中间件层，并通过删除超过10万行代码和数千个未使用的内部接口以及清除未使用的客户端服务。
* 将帖子元数据源延迟减少50%，全局API超时错误减少90%。
* 被阻止的机器人和内容抓取器的比率比2022年高出37%。平均而言，我们每天防止超过100万个机器人注册攻击，DM垃圾邮件减少了95%。
* 关闭萨克拉门托数据中心，重新配置5200个机架和14.8万台服务器，每年节省超过1亿美元。在将其重新配置到其他数据中心之前，我们总共释放了48兆瓦的容量，并拆除了6万磅的网络梯形架。
* 优化了我们对云服务提供商的使用，并开始进行更多的私有化部署。这一转变使我们每月的云成本降低了60%。我们所做的更改之一是将所有媒体/blob工件移出云，这将我们的整体云数据存储大小减少了60%，另外，我们还成功地将云数据处理成本减少了75%。
* 基于私有化的GPU超级计算集群，设计、开发并交付了43.2Mbps的新网络结构架构来支持集群。
* 扩展了网络主干网容量和冗余，每年节省1390万美元。
* 已启动自动化峰值流量故障切换测试，以不断验证整个平台的可扩展性和可用性。


## 评价
十年前，上云是一个非常流行的词，没想到十年后，下云反而成了一个时髦的词汇。X所做的事情再一次证明，与其屎上雕花，还不如拆了重建。不搞云原生的政治正确，还是更关注真实需求。遇到这样一个有魄力的老板，悲哉？幸哉？
