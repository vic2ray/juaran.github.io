---
title: 【研二上】Resnet变体模型解释
date: 2021-12-29
category: 机器学习
tag: 周报
---



上周论文**Residual networks behave like ensembles of relatively shallow networks**(2016)首次用实验验证了残差连接属于Ensemble模型的猜想。后续的论文研究了不同的ResNet改进版本，进一步探索了极深网络，也尝试解释残差连接的有效性：

### **Resnet变体**

- FractalNet 分形网络，2017年，引用765

> Gustav Larsson, Michael Maire, & Gregory Shakhnarovich (2017). FractalNet: Ultra-Deep Neural Networks without Residuals.. International Conference on Learning Representations.

受残差网络中跳过连接(skip connection)的启发，论文提出了跳过路径(drop path)，对连接路径进行随机丢弃，用正则方法提高模型表单能力

1. 论文的实验说明了路径长度才是训练深度网络的需要的基本组件，而不单单是残差块
2. 分形网络和残差网络都有很大的网络深度，但是在训练的时候都具有更短的有效的梯度传播路径
3. 本质是将ResNet以一种规则展开，侧面证明了ResNet的Ensemble的本质

------

- ResNet with stochastic depth 随机深度残差网络，2016年，引用1440

> Huang G, Sun Y, Liu Z, et al. Deep networks with stochastic depth[C]//European conference on computer vision. Springer, Cham, 2016: 646-661.

清华大学黄高(也是DenseNet的提出者)在EECV会议上提出了Stochastic Depth（随机深度网络）。这个网络主要是针对ResNet训练时做了一些优化，即随机丢掉一些层，优化了速度和性能。

作者的解释是，不激活一部分block事实上实现了一种隐性的**模型融合**（**Implicit model ensemble**），由于训练时模型的深度随机，预测时模型的深度确定，事实上是在测试时把不同深度的模型融合了起来。

------

- **DenseNet**，2017年，引用21144

> Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 4700-4708.

densenet 论文核心思想：对前每一层都加一个单独的shortcut，使得任意两层网络都可以直接“沟通”（Resnet中只有残差块内的网络能够连接）。**这样使得浅层和深层的特征域可以更自由的组合**，使模型的结果具有更好的鲁棒性。

------

以上论文是使用不同方式在深度神经网络上提升性能的尝试，除此之外还有Wide resnet等，正是ResNet开创了极深度网络的先河。从中我们可以得出一些结论：

1. Ensemble是Resnet及其改进模型的成功的重要原因
2. 这意味着极深网络只是看上去很深，实际上用短路连接、丢弃路径等方式绕开了深度网络难训练的问题，训练的仍是浅层网络
3. 提升Ensemble中的基础网络能够提升整个网络性能

参考：

[FractalNet: Ultra-Deep Neural Networks without Residuals](https://www.cnblogs.com/liaohuiqiang/p/9218445.html)

https://cloud.tencent.com/developer/article/1582021

[Deep Networks with Stochastic Depth](https://zhuanlan.zhihu.com/p/31200098)

[极深网络（ResNet/DenseNet）: Skip Connection为何有效及其它](https://blog.csdn.net/malefactor/article/details/67637785)

[DenseNet：比ResNet更优的CNN模型](https://zhuanlan.zhihu.com/p/37189203)

### **总结目前工作**

1. **实验**：首先，我们使用残差网络在内部威胁数据集上进行了实验，实验结果表明其检测性能优于基于统计的方法、图聚类方法、传统机器学习方法，以及图神经网络模型
2. **理论**：其次，我们探究了残差模型有效性的理论解释，其多个变体模型验证了resnet是多个浅层网络集成(融合)的猜想。我们希望利用残差连接的这一性质，融合多域特征以学习到不同的内部威胁行为

### **理论缺陷**

但我发现另一个欠缺的理论解释是，**为什么可以使用CNN进行内部威胁检测？**目前还没有研究使用CNN在内部威胁数据集上建模。我们不能简单地将表格数据视为图像，因此需要找到在表格数据上进行卷积的实践。

与语音和图像数据相比，数据的重要信息以特征的顺序嵌入，而大多数表格型数据（tabluar data，或称结构化数据，一个样本对应一维特征向量）是异构数据，这些特征之间的相关性比图像或语音数据中的空间或语义关系弱。

一种做法是使用1D CNN，但是，一维卷积层也需要特征之间的空间局部相关性，即卷积核期望连续的列在空间上是相关的（例如，时序数据），而大部分的表格数据集往往无空间关联。

**我们将单维的日志数据特征重构成二维序列(域-特征)，直觉上增强了不同域的上下文联系，以期通过卷积融合异构数据，这样的做法是否具有说服力？**

### **寻找实践**

在Kaggle的竞赛中，有团队使用了论文DeepInsight中的方法，将非图像数据转为图像形式并进行卷积，在MoA预测中取得第一名。这个竞赛需要由药物特征预测药物机理，类似威胁检测任务。引用论文:

> Sharma A, Vans E, Shigemizu D, et al. DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture[J]. Scientific reports, 2019, 9(1): 1-7.

<img src="https://storage.googleapis.com/kaggle-markpeng/MoA/deepinsight_architecture.png" alt="https://storage.googleapis.com/kaggle-markpeng/MoA/deepinsight_architecture.png" style="zoom: 25%;" />

------

第二名团队通过重新排序(reshape)这些特征后进行一维卷积，网络中也添加了跳过连接，这种方法和我们的做法相似。作者的想法是：

- CNN 结构在特征提取方面表现良好，但在表格数据中很少使用，因为正确的特征排序是未知的
- 一个简单的想法是将数据直接reshape成多通道图像格式，通过反向传播使用FC层学习正确排序

引用论文：Arık S O, Pfister T. Tabnet: Attentive interpretable tabular learning[J]. arXiv, 2020.

<img src="https://miro.medium.com/max/875/1*MqIpbeDxONYe3moaam9s9w.png" alt="https://miro.medium.com/max/875/1*MqIpbeDxONYe3moaam9s9w.png" style="zoom:67%;" />

参考：

Deep Neural Networks and Tabular Data: A Survey 深度神经网络和表格数据综述

[异构表格数据的挑战，深度神经网络如何解？](https://www.jiqizhixin.com/articles/2021-11-13-4)

[如何用深度学习处理结构化数据？](https://www.jiqizhixin.com/articles/2017-12-04-7)

[表格数据集上的卷积神经网络](https://medium.com/spikelab/convolutional-neural-networks-on-tabular-datasets-part-1-4abdd67795b6)

https://www.kaggle.com/c/lish-moa

[Mechanisms of Action (MoA) Prediction top解法总结](https://zhuanlan.zhihu.com/p/349911032)

https://www.kaggle.com/c/lish-moa/discussion/202256