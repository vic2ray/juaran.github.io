---
title: 【研二上】LSTM-CNN内部威胁检测模型
date: 2021-11-20
category: 机器学习
tag: 周报
---



### **LSTM-CNN模型**

> Yuan, F., Cao, Y., Shang, Y., Liu, Y., Tan, J., & Fang, B. (2018). Insider Threat Detection with Deep Neural Network. Computational Science – ICCS 2018, 43–54.
>
> doi:10.1007/978-3-319-93698-7_4

- 首先，与自然语言建模类似，我们使用长短期记忆(LSTM)通过用户动作来学习用户行为的语言，并提取抽象的时间特征
- 其次，将提取出的特征转换为固定大小的特征矩阵，卷积神经网络(CNN)利用这些固定大小的特征矩阵来检测内部威胁

这篇文章受之前文章中聚合用户一天内所有行为的启发，观察到直接聚合所有行为很可能会掩盖其中的异常行为，因此使用LSTM来建模用户行为序列，考虑到了对时间的建模。正常行为序列是普遍的，偏离这些正常动作序列则被视为异常行为

方法概述：

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211121105547749.png" alt="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211121105547749.png" style="zoom: 50%;" />

类似于自然语言建模，将user action视为word，动作序列组成的user behavior即sentence，输入LSTM提取用户行为的特征。具体地，每个用户`k`共有`J`天的行为序列构成序列矩阵`S`，LSTM训练得到行为特征提取器，输出隐含高级特征表示的固定大小的特征矩阵用于CNN分类

以上方法在数据预处理阶段工作量较大，需要将所有用户每天的操作日志转换为序列并编码向量表示，与当前工作直接提取日数据特征不同

实验结果：

论文对比了不同LSTM隐层数量、不同CNN卷积核大小的模型ROC曲线，最优模型达到AUC=**0.9449**，但是感觉实验结果不太靠谱（接近完美分类器），其他模型最优结果仅为0.8左右。猜测是因为训练和测试数据集没有按照标签比例划分，导致测试集内很少异常样本

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211123194707598.png" alt="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211123194707598.png" style="zoom:67%;" />