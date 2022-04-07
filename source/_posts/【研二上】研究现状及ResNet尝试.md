---

title: 回顾研究现状及ResNet尝试
category: 内部威胁检测
date: 2021-11-14

---



回顾综述论文：内部威胁检测的深度学习：回顾、挑战和机遇（2020.5）

> Deep Learning for Insider Threat Detection: Review, Challenges and Opportunities

### 研究现状

由于数据集的极度不平衡性质，大部分提出的方法采用**无监督学习**范式进行内部威胁检测。对于检测粒度，大多数论文侧重于检测**恶意子序列**（如24小时内的活动）或恶意会话

目前的主要文献所采用的的深度学习架构有：DNN、RNN、CNN、GNN

<!-- more -->

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211114120805630.png" alt="image-20211114120805630" style="zoom:60%;" />

* Deep Feed-forward Neural Network

学姐论文中使用到的DBN属于这一类型。发现这里竟然引用到了：

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211114122147038.png" alt="image-20211114122147038" style="zoom:80%;" />

* Recurrent Neural Network

递归神经网络(RNN)主要用于对序列数据进行建模，通过自循环连接保持隐藏状态，对序列中的信息进行编码，使用最广泛的是LSMT和GRU。这类方法将用户活动数据建模为序列数据，其基本思想是训练一个RNN模型来预测用户的下一个活动或活动周期。只要预测结果和用户的实际活动没有明显的差异，我们就认为用户遵循正常的行为

粗读了其中的监督学习方法论文：[79] 深度神经网络威胁检测（中科院网安研究院，2018）

> Yuan, F., Cao, Y., Shang, Y., Liu, Y., Tan, J., & Fang, B. (2018). *Insider Threat Detection with Deep Neural Network. Computational Science – ICCS 2018, 43–54.* doi:10.1007/978-3-319-93698-7_4

具体地，首先使用LSTM从序列化日志数据中提取用户行为特征，**然后将特征转换为固定大小的特征矩阵，输入CNN进行分类检测**。**实验使用CERT r4.2数据集，划分70%训练集和30%测试集，分类粒度为日数据**。评估指方法使用AUC，主要评估FPR和TPR(recall)指标

* Convolutional Neural Network

最近一项关于内部威胁检测的研究提出了一种基于CNN的、通过分析鼠标生物行为特征的用户认证方法。该方法将计算机用户的鼠标行为表示为一幅图像。如果发生ID盗窃攻击，用户的鼠标行为将与合法用户不一致。因此，将CNN模型应用于基于鼠标行为生成的图像，以识别潜在的内部威胁

这篇文章使用CNN通过识别异常的鼠标图像来检测威胁，并不是使用CERT日志数据集。也就是说，还没有文章使用CNN在CERT数据集进行卷积分类，我们以下的工作基于此空缺

* Graph Neural Network

见之前实验和总结，不再赘述

### Method

由于数据的隐私性，目前大部分基于深度学习的威胁检测研究依然使用耐吉卡梅隆大学CMU提供的CERT r4.2日志数据，其模拟生成了1000个用户连续17个月的计算机设备操作记录。大部分研究在用户日数据级别上进行分类，且无监督学习（聚类）方式较多。但数据集中包含insider标签用于监督训练，严格检测出insider更加具有模型评估和现实应用意义，因此我们的工作采用监督学习的方式

据我们所知，CNN在视觉图像领域取得了巨大成功，但是内部威胁检测领域，仅有少部分文章将CNN应用于异常数据检测。一方面，CNN仅接受固定大小的二维矩阵输入，适合空域图像等感知型数据，而内部威胁数据特征往往采用不同域特征的行组合；另一方面，CNN虽然能够自动学习特征，但卷积后的特征可解释性较差，分类结果不能给人类专家带来可信的评判依据。因此，我们所作的贡献如下：

* 首次尝试将内部威胁数据特征表示为二维特征矩阵。具体地，每一行代表一个域的特征
* 尝试使用CNN卷积不同域的特征。具体地，我们使用ResNet，以避免反向传播时梯度消失
* 在以上工作基础上，尝试加入多种注意力机制（SE、CBAM、BAM），提高模型检测能力

### Experiments And Results

数据集介绍及实验参数、评估方法略。每个模型100轮训练。实验使用的深度神经网络模型如下：

* 34层网络
  * resnet34
  * se_resnet34_channel 压缩激发(Squeeze-and-Excitation)结构的通道注意力模块（SE模块）
  * cbam_resnet34 卷积块注意力模块(Convolutional Block Attention Module)
    * channel 通道维度
    * spatial 空间维度
    * joint 联合维度
  * bam_restnet34 瓶颈注意模块(Bottleneck Attention Module)
    * channel
    * spatial
    * joint
* 50层网络
  * resnet50
  * ......

#### 80% train data

| Model                 | Pr       | Dr       | F1       |
| --------------------- | -------- | -------- | -------- |
| resnet34              | 0.95     | 0.57     | 0.71     |
| se_resnet34_channel   | 0.93     | 0.58     | 0.71     |
| bam_resnet34_channel  | **0.98** | **0.62** | **0.76** |
| bam_resnet34_spatial  | 0.97     | 0.48     | 0.64     |
| bam_resnet34_joint    | 0.95     | 0.59     | 0.73     |
| cbam_resnet34_channel | 0.95     | 0.58     | 0.72     |
| cbam_resnet34_spatial | 0.96     | 0.61     | 0.75     |
| cbam_resnet34_joint   | 0.96     | 0.59     | 0.73     |
| resnet50              | 0.98     | 0.58     | 0.73     |
| se_resnet50_channel   | 0.97     | 0.61     | 0.75     |
| bam_resnet50_channel  | 0.96     | 0.60     | 0.74     |
| bam_resnet50_spatial  | 0.94     | 0.62     | 0.75     |
| bam_resnet50_joint    | **0.98** | 0.61     | 0.75     |
| cbam_resnet50_channel | 0.93     | **0.64** | **0.76** |
| cbam_resnet50_spatial | 0.94     | 0.61     | 0.74     |
| cbam_resnet50_joint   | 0.94     | 0.62     | 0.75     |

以上结果：

* 50层模型效果整体比34层模型在检测率上高出1~6%，但精确度稍有下降2~4%，说明随着网络层数加深，能够捕获的特征越多，代价是损失部分精度
* 加入了注意力模块的模型比普通模型表现更优，提升最高6%的检测率
* 关注不同维度的注意力模型的表达能力也有所不同，但效果接近

#### 20% train data

| Model                     | Pr       | Dr       | F1       |
| ------------------------- | -------- | -------- | -------- |
| resnet34                  | 0.82     | 0.39     | 0.51     |
| se_resnet34_channel       | 0.78     | 0.45     | 0.57     |
| bam_resnet34_channel      | 0.75     | 0.42     | 0.54     |
| bam_resnet34_spatial      | 0.66     | 0.35     | 0.45     |
| bam_resnet34_joint        | 0.83     | 0.36     | 0.50     |
| cbam_resnet34_channel.txt | 0.84     | 0.45     | 0.59     |
| cbam_resnet34_spatial.txt | **0.88** | 0.47     | **0.61** |
| cbam_resnet34_joint       | 0.85     | **0.48** | 0.61     |
| resnet50                  | **0.92** | 0.48     | 0.63     |
| se_resnet50_channel       | 0.89     | 0.50     | 0.64     |
| bam_resnet50_channel      | 0.83     | 0.44     | 0.58     |
| bam_resnet50_spatial      | 0.89     | 0.46     | 0.61     |
| bam_resnet50_joint        | 0.90     | **0.51** | **0.65** |
| cbam_resnet50_channel     | 0.92     | 0.46     | 0.61     |
| cbam_resnet50_spatial     | 0.90     | 0.43     | 0.58     |
| cbam_resnet50_joint       | 0.84     | 0.50     | 0.63     |

以上结果：

* 仅使用少量数据的情况下，以上深度学习模型比传统模型检测效果更优（决策树等）

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211116202354887.png" alt="image-20211116202354887" style="zoom:67%;" />

* 在少样本的情况下，50层网络比34层网络在精度、检测率上均有明显提升（最高10%以上）
* 在少样本的情况下，注意力模块带来最高11%的检测率提升

 