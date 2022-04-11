---
title: 【研二上】Yelp数据集相关
date: 2021-09-29
category: 机器学习
tag: 周报
---

### Yelp

美国最大的点评网站，成立于2004年，从2005年便开始应用过滤系统移除可疑或虚假的评论，门店主页显示系统推荐的评论，同时也可以通过页面底部的链接查看过滤/不推荐的评论：

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210927093754046.png" alt="image-20210927093754046" style="zoom: 50%;" />

> *[What Yelp fake review filter might be doing?](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiwj-vJo-rOAhUH2BoKHX7IA0UQFggcMAA&url=https%3A%2F%2Fwww.aaai.org%2Focs%2Findex.php%2FICWSM%2FICWSM13%2Fpaper%2Fdownload%2F6006%2F6380&usg=AFQjCNEaT55wVWuin3v07jJBFjybxOFT4g)* A. Mukherjee, V. Venkataraman, B. Liu, and N. S. Glance, ICWSM, 2013.

文章首次在真实的商业网站Yelp中实施虚假评论检测工作。

这篇文章主要探究了Yelp点评网站的评论审查系统如何过滤虚假评论。因为Yelp的算法是商业机密的，只能通过现有的方法来训练Yelp评论数据，而现有方法主要对AMT(一个众包平台)的伪虚假评论检测，实验发现在Yelp真实评论上，基于行为特征的检测很有效，但基于语言特征的检测效果不佳。

作者首先使用了使用**语言分词模型**，发现远低于模型在伪虚假评论上的准确率，原因是Yelp上的虚假评论和真实评论具有几乎相同的单词分布，这可能是由于这些虚假评论有意对抗过滤系统的审查机制，让它们看起来像是真实评论，但在语言心理学上分析漏洞明显。

然后对评论者的**异常行为进行分析**，发现行为特征检测效果远好于语言特征。例如：评论数、好评占比、内容长度、**评分偏差程度**、内容相似度等。且这些特征数据是容易获得的，对诸如账户IP地址、点击行为等特征无法进行尝试。因此，这是通用的SPAM评论检测方法。

### YelpCHI

Yelp上的评论太多，YelpCHI是收集自Chicago酒店和餐厅的 67,395 条评论的数据集。评论包括门店和用户信息、时间戳、评级和纯文本评论。该数据集包含38,063 名评论者对 201 家酒店和餐厅的评论。原数据集来自论文：

> Rayana, S., & Akoglu, L. (2015). *Collective Opinion Spam Detection:Bridging Review Networks and Metadata. Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD ’15.* doi:10.1145/2783258.2783370

阅读参考：https://www.cnblogs.com/C-W-K/p/13846305.html

文章提出一个无监督框架，它利用来自所有元数据（文本、时间戳、评级)以及关系数据(网络）的信息，并在一个统一的框架下集体利用它们来发现可疑的用户和评论。

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210928105037985.png" alt="image-20210928105037985" style="zoom: 80%;" />

这个框架和GNN做的事情是一样的，利用图网络、节点特征（元数据）、节点标签得到节点嵌入。论文结果中同样得出，基于语言特征的检测效果远不如基于行为特征。特别是**评分偏差程度**和评分极端化特征。

 

使用该数据集的论文：

* Graph-Consis网络（上周看的）

> Liu, Z., Dou, Y., Yu, P. S., Deng, Y., & Peng, H. (2020). *Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval.* doi:10.1145/3397271.3401253

* CARE-GNN网络

> Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters. Proceedings of the 29th ACM International Conference on Information & Knowledge Management.* doi:10.1145/3340531.3411903

* 数据集中review特征的提取

> Zhang, S., Yin, H., Chen, T., Hung, Q. V. N., Huang, Z., & Cui, L. (2020). GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Detection. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. doi:10.1145/3397271.3401165