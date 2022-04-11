---
title: 【研二上】与GCN模型实验对比
date: 2021-11-23
category: 机器学习
tag: 周报
---



### 对比GCN

GCN论文基于1000个用户进行检测，使用40个威胁人员训练，30个威胁人员测试

实验部分对比了GCN与四种机器学习和深度学习方法：GCN达到**93%**的检测精度(Pr)，其他方法约85%；GCN达到**83.3%**的召回率，其他方法约70%；

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211122113316307.png" alt="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211122113316307.png" style="zoom:67%;" />

我们采取**用户日数据**进行实验。数据处理后，得到1000个员工330,452实例数据，其中包含**70**个恶意员工的**966**条恶意实例（一个员工可能在多个工作日内发生异常行为）

对了进行对比，我们需要将基于日实例数据的评估结果转换为基于用户的评估结果：如果一个正常用户的至少有一个数据实例被归类为“恶意”，则被错误分类；而如果他们的至少有一个恶意数据实例被系统归类为“恶意”，则被正确识别为恶意内部人员

使用上周训练好的模型，将基于实例的模型结果转换成基于用户的检测结果，具体地，根据实例index取得对应user id，然后将预测结果与实际结果取交集和差集得到混淆矩阵系数，计算得到恶意用户检测召回率和检测精度。转换如下：

```python
import numpy as np

# 取测试集数据在原数据中索引
test_insider_idx = X_test_idx[ np.where(np.array( test_labels )>0)[0] ]
pred_insider_idx = X_test_idx[ np.where(np.array( test_preds ) >0 )[0] ]
# 取得威胁人员的 user id
test_insider_user = np.unique( data_user.iloc[test_insider_idx, :].values.T[0] )
pred_insider_user = np.unique( data_user.iloc[pred_insider_idx, :].values.T[0] )
# print(test_insider_user)
# print(pred_insider_user)

# 结果取交集，即预测对的 --> TP
true_positive_users = np.intersect1d(test_insider_user, pred_insider_user)
# 结果取差集，即预测错的 --> FP, FN
false_positive_users = np.setdiff1d(pred_insider_user, test_insider_user)
false_negtive_users = np.setdiff1d(test_insider_user, pred_insider_user)

# recall = tp / tp + fn
true_positive_rate = true_positive_users.shape[0] / (true_positive_users.shape[0] + false_negtive_users.shape[0])
print("Recall", true_positive_rate)
# precision = tp / tp + fp
false_positive_rate = true_positive_users.shape[0] / (false_positive_users.shape[0] + true_positive_users.shape[0])
print("Precision", false_positive_rate)
```

ResNet模型基于用户检测结果：

| Model                 | Pr    | Dr    |
| --------------------- | ----- | ----- |
| resnet34              | 0.91  | 0.77  |
| se_resnet34_channel   | 0.82  | 0.73  |
| bam_resnet34_channel  | 0.956 | 0.846 |
| cbam_resnet34_spatial | 0.87  | 0.75  |
| resnet50              | 0.95  | 0.73  |

实验结果表明，注意力模型结果高于GCN的精确率(95%)和召回率(83.3%)