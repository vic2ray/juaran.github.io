---
title: GraphConsis代码阅读
date: 2021-10-04
category: 图神经网络
---

* 论文：*Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection.*
* 代码：https://github.com/safe-graph/DGFraud-TF2/tree/main/algorithms/GraphConsis

<!-- more -->

### 1. 数据集

下载地址：https://github.com/safe-graph/DGFraud-TF2/blob/main/dataset/Yelpchi.zip

数据结构：

| name     | size        | meaning          |
| -------- | ----------- | ---------------- |
| features | 45954x32    | 节点32维特征     |
| label    | 1x45954     | 节点二分类标签   |
| net_rsr  | 45954x45954 | review-product图 |
| net_rtr  | 45954x45954 | review-time图    |
| net_rur  | 45954x45954 | review-user图    |

补充说明：

1. 32维特征向量包括：15 review features, 9 user features, and 8 product features
2. *R-U-R* connects reviews posted by the same user
3. *R-S-R* connects reviews under the same product with the same rating
4. *R-T-R* connects two reviews under the same product posted in the same month

### 2. 数据加载和划分

* 图数据一般采用稀疏矩阵存储，使用`scipy.io`读取标签、特征、图
* 使用`sklearn`的`train_test_split`快速划分按标签比例分布的数据集

``` python
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
 
 
data_path = '../YelpChi.mat'
train_size = 0.8
 
def load_data_yelp():
    # 特征、标签、图加载
    data = scipy.io.loadmat(data_path)
    truelabels, features = data['label'], data['features'].astype(float)  # 特征是[0,1]浮点数
    # print(truelabels.nonzero()[1].size)  # 6677 fraud review: 14.5%
    rownetworks = [data['net_rur'], data['net_rsr'], data['net_rtr']]        # 45954 x 45954 x 3
 
    # 划分数据集
    y = truelabels.tolist()[0] 
    X = np.arange(len(y))  # [0, 1, 2, ..., 45953]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                     train_size=train_size, 
                     stratify=y,     # 按照标签的比例生成相同标签比例的数据集
                    random_state=1)  # 相同的随机状态得到相同的随机数结果
    # ==>划分得到80%训练集，20%测试集，且0,1标签的比例相同
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=1)
    # ==>划分训练集中的20%最为验证集
 
    split_ids = [X_train, X_val, X_test]  # ==> 节点划分索引
 
    return rownetworks, features, np.array(y), split_ids
 
adj_list, features, y, split_ids = load_data_yelp()
 
```

### 3. 数据预处理

* 特征按行归一化，以加快收敛速度
* 邻接矩阵转为字典key-value形式存储图

``` python
# GCN代码中的特征按行归一化，加快收敛
def preprocess_feature(features):
    features = scipy.sparse.lil_matrix(features)
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    features = r_mat_inv.dot(features)
 
    return features
 
# 标签和特征处理
label = np.array([y]).T  # 标签数组转置：(45954, ) => (45954, 1)
features = np.array(preprocess_feature(features).todense())  # 特征归一化
 
# 图处理：邻接矩阵表示 ==> 字典表示
neigh_dicts = []        # 最终得到 [{10: [], ...}, {0: [], ...}, {3: [], ...}]      注意，不是所有节点都有邻居！
for net in adj_list:        #  r-u-r, r-s-r, r-t-r 三个关系图
    # print(net.size)       # rur边数量98630 rsr边6805486 rtr边1147232
    neigh_dict = {}         # 图节点信息：{ 0: [2,6701], 1: [], ..., 45953: []  }
    for i in range(len(y)):     # 所有节点
        neigh_dict[i] = []      # 节点的邻居节点列表初始化空
    nodes1 = net.nonzero()[0]       # 非零元素行索引，即起始节点u
    nodes2 = net.nonzero()[1]       # 非零元素列索引，即连接节点v
    for node1, node2 in zip(nodes1, nodes2):        # (u, v)
        neigh_dict[node1].append(node2)         # u -> v
    neigh_dicts.append({k: np.array(v, dtype=np.int64) for k, v in neigh_dict.items()})
 
# neigh_dicts[1][3]  # 第二个图的节点3有[ 5 6703 6704]三个邻节点
```

### 4. 邻居采样和创建计算子图

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211002120254463.png" alt="image-20211002120254463" style="zoom: 80%;" />

#### 4.1 采样和混淆矩阵

1. 首先计算节点`u`和邻居`v`的差异分数
2. 若两个节点无一致性则不采样，得到节点的邻接矩阵
3. 对采样后的邻接矩阵**去除空列**，得到`[u,v]`混淆邻接矩阵
4. 返回混淆矩阵和矩阵行列信息：
   * dstsrc    参与计算的所有目标节点u和邻节点v列表
   * dstsrc2src  混淆矩阵的行含义，即v节点是哪些
   * dstsrc2dst 混淆矩阵的列含义，即u节点是哪些
   * dif_mat      混淆矩阵包含了u->v的连接信息，且边权重为度的平均

``` python
import tensorflow as tf
 
 
eps = 0.001   # 节点特征差异超参数
 
# 计算节点与其邻节点的特征差异分数：consistent score
def calc_consistency_score(n, ns):      # 计算与邻节点的差异大小
    # Equation 3 in the paper
    consis = tf.exp(-tf.pow(tf.norm(tf.tile([features[n]], [len(ns), 1]) -
                                    features[ns], axis=1), 2))
    consis = tf.where(consis > eps, consis, 0)  # 低于阈值则返回0
    return consis
 
# 按照差异分数进行邻节点采样
def sample(n, ns, sample_size):
    if len(ns) == 0:
        return []  # 没有邻节点
    consis_score = calc_consistency_score(n, ns)
    # Equation 4 in the paper
    prob = consis_score / tf.reduce_sum(consis_score)     # 按照consis_score计算邻节点采样权重
 
    # prop=0的邻节点不被选择。得到一个采样后的 邻节点 列表
    return np.random.choice(ns, min(len(ns), sample_size), replace=False, p=prob)
 
# sample(3, neigh_dicts[1][3], 5)  # 对原图采样：array([6703,    5, 6704])
 
# 计算 u-> v 的混淆矩阵
def compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size):
    # 1. 计算采样后的图邻接矩阵
    rows = []           # 新的邻接矩阵行
    for n in dst_nodes:
        ns = sample(n, neigh_dict[n], sample_size)    # 采样邻节点
        row = np.zeros(len(neigh_dict), dtype=np.float32)  # 邻接矩阵一行节点u
        row[ns] = 1   # 节点 u 的邻节点 v
        rows.append(row)
    adj_mat_full = np.stack(rows)   # 堆叠一个batch_size的n个节点，得到采样后的邻接矩阵: batch_size x 45954
    # print(adj_mat_full.shape)   # (batch_size, 45954)
 
    # 2. 去除空列得到混淆矩阵
    nonzero_cols_mask = np.any(adj_mat_full.astype(np.bool), axis=0)  # 取非空列v节点索引，去掉大部分空列
    adj_mat = adj_mat_full[:, nonzero_cols_mask]    # 去除空列的 u->v 邻接矩阵
    adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)  # 统计 u 的度
    dif_mat = np.nan_to_num(adj_mat / adj_mat_sum)        # 平均度作为v节点的权重  ==> 混淆矩阵
    # print(dif_mat)  行: u : 0, 1, 2 三个节点
    """  列: v : [   0    2    4 6702] 四列非空
    [[0.  0.5 0.  0.5] 
     [0.  0.  1.  0. ]
     [0.5 0.  0.  0.5]]
     """
 
    # 3. 混淆矩阵的行列节点信息
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]  # v 节点索引
    # print(nonzero_cols_mask.nonzero()[0])     # [   0    2    4 6702]  ==> 计算图中的源节点 v
    # print(src_nodes)                  # v 节点 [   0    2    4 6702]
    dstsrc = np.union1d(dst_nodes, src_nodes)    # 排序返回 [u,v]
    # print(dstsrc)                   # u,v 所有节点 [   0    1    2    4 6702]
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)
    # print(dstsrc2src)               # [0 2 3 4] v 源节点在 dstsrc 中索引
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)
    # print(dstsrc2dst)               # [0 1 2]  u 目标节点在 dstsrc 中索引
 
    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat
 
# compute_diffusion_matrix(np.array([0]), neigh_dicts[1], 5)
"""
(array([   0,    2, 6702]),   待计算节点
 array([1, 2]),               v: 邻节点索引，混淆矩阵的行
 array([0]),                  u: 目标节点索引，混淆矩阵的列
 array([[0.5, 0.5]], dtype=float32))  [u,v] 混淆矩阵
"""
```

#### 4.2 创建计算图节点的minibatch

1. 每一个输入网络中的batch包含三个minibatch，对应三种类型的图
2. 每个minibatch包含了**两层**邻节点到目标节点的**混淆矩阵**代表图节点计算信息
3. 按照batch_size对训练集分批产生网络的输入(mini_batch)和输出(mini_batch_labels)

``` python
from collections import namedtuple
 
 
batch_size = 512  # 批大小
sample_sizes = [5, 5]  # 每一层采样大小
 
# 训练节点迭代生成minibatch
def generate_training_minibatch(nodes_for_train):
    nodes = np.copy(nodes_for_train)
    ix = 0    # 分批切片索引
    np.random.shuffle(nodes)  # 再次打乱节点顺序
    # 前batch整数倍个批次
    while len(nodes) > ix + batch_size:
        mini_batch_nodes = nodes[ix: ix+batch_size]  # 取一个batch
        mini_batch_labels = labels[mini_batch_nodes]  # 取对应标签
        # 创建batch
        mini_batch = build_batch(mini_batch_nodes)
 
        ix = ix + batch_size
        yield mini_batch, mini_batch_labels
    # 最后不足批部分
    last_batch_nodes = nodes[ix: -1]
    last_batch_labels = labels[last_batch_nodes]
    mini_batch = build_batch(last_batch_nodes)
    yield mini_batch, last_batch_labels
 
# 产生训练batch，即创建节点计算图
def build_batch(nodes):
    batch = []
    for neigh_dict in neigh_dicts:  # 三个关系图都要计算节点子图
        # 目标节点u，和两层的[u,v]混淆矩阵
        dst_nodes, dstsrc2dsts, dstsrc2srcs, dif_mats = [nodes], [], [], []
        for sample_size in reversed(sample_sizes):  # 采样层数=网络层数
            ds, d2s, d2d, dm = compute_diffusion_matrix(dst_nodes.pop(), neigh_dict, sample_size)
            # print("ds: ",ds, "d2s: ",d2s, "d2d: ",d2d, "dm: ",dm)
            # 添加邻节点信息
            dst_nodes.append(ds)
            dstsrc2srcs.append(d2s)  # v
            dstsrc2dsts.append(d2d)  # u
            dif_mats.append(dm)      # dif_mat
        src_nodes = dst_nodes.pop()  # 所有u,v
        # 具名元组，存放layer层节点树列表
        MiniBatchFields = ["src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"]
        MiniBatch = namedtuple("MiniBatch", MiniBatchFields)       
        batch.append(MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats))
 
    return batch
 
# batch = build_batch(np.array([0]))
""" 节点 0 的计算图batch：
ds:  [0] d2s:  [] d2d:  [0] dm:  []  第一张图没有邻居
ds:  [0] d2s:  [] d2d:  [0] dm:  []
第二张图：0 - 2 - 6702
ds:  [   0    2 6702] d2s:  [1 2] d2d:  [0] dm:  [[0.5 0.5]]  ==> layer 1
ds:  [   0    2 6702] d2s:  [0 1 2] d2d:  [0 1 2] dm:  [[0.  0.5 0.5]  ==> layer 2
 [0.5 0.  0.5]
[0.5 0.5 0. ]]
ds:  [0] d2s:  [] d2d:  [0] dm:  []  第三张图没有邻居
ds:  [0] d2s:  [] d2d:  [0] dm:  []
"""
```

### 5. 网络模型

1. 首先定义聚合函数，按照GraphSAGE的聚合公式：

$$
h^{(l)}_{v} = \sigma(W^{(l)} · CONCAT(h^{(l-1)}_v, MEAN({h_u^{l-1}, \forall u \in N(v)})))
$$

2. 在GraphSAGE聚合函数基础上，增加关系注意力参数
3. 网络层：包含两个聚合线性层和一个输出分类层
4. 前向传播：依次聚合batch中各个节点的每一层信息

``` python
from keras import Model, layers
 
 
init_fn = tf.keras.initializers.GlorotUniform
 
# GraphSAGE网络模型聚合函数
class SageMeanAggregator(layers.Layer):
    def __init__(self, src_dim, dst_dim, activ=True, **kwargs):
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
        self.w = self.add_weight(name=kwargs["name"] + "_weight",
                                 shape=(src_dim * 2, dst_dim),    # 特征维度拼接后 x 2
                                 dtype=tf.float32,
                                 initializer=GlorotUniform,
                                 trainable=True)
    def call(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat):
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)    # 从features中抽取m个目标节点u特征：m x 32
        src_features = tf.gather(dstsrc_features, dstsrc2src)    # 从features中抽取n个邻节点v特征：n x 32
        aggregated_features = tf.matmul(dif_mat, src_features)   # 聚合邻居特征：mxn x nx32 = m x 32
        concatenated_features = tf.concat([aggregated_features, dst_features],  # 拼接特征：m x 64 ==> 64维特征！
                                          1)
        x = tf.matmul(concatenated_features, self.w)    # W*x: src_dimx64 x 64xdsr_dim ==> src_dim x dsr_dim
        return self.activ_fn(x)             # 激活函数。最终得到h(l)嵌入
 
# GranphConsis聚合函数继承GraphSAGE，增加了关系注意力
class ConsisMeanAggregator(SageMeanAggregator):
    def __init__(self, src_dim, dst_dim, **kwargs):
         super().__init__(src_dim, dst_dim, activ=False, **kwargs)
    def __call__(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat,
                 relation_vec, attention_vec):
        # Equation 5,6 in the paper        增加可训练的关系注意力参数
        x = super().__call__(dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat)
        relation_features = tf.tile([relation_vec], [x.shape[0], 1])
        alpha = tf.matmul(tf.concat([x, relation_features], 1), attention_vec)
        alpha = tf.tile(alpha, [1, x.shape[-1]])      # 注意力机制函数alpha
        x = tf.multiply(alpha, x)
        return x
 
# GraphConsis网络模型
class GraphConsis(Model):
    def __init__(self, features_dim: int, internal_dim: int, num_layers: int,
                 num_classes: int, num_relations: int) -> None:
        """
        :param int features_dim: input dimension                输入特征维度
        :param int internal_dim: hidden layer dimension         隐含层数
        :param int num_layers: number of sample layer           网络层数
        :param int num_classes: number of node classes          二分类
        :param int num_relations: number of relations           三种关系
        """
        super().__init__()
        self.seq_layers = []          # 两个Linear layer + 一个Dense layer
        self.attention_vec = tf.Variable(tf.random.uniform(
            [2 * internal_dim, 1], dtype=tf.float32))       # 注意力参数，维度与隐层维数相关
        self.relation_vectors = tf.Variable(tf.random.uniform(      # 不同关系聚合参数
            [num_relations, internal_dim], dtype=tf.float32))
        for i in range(1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            input_dim = internal_dim if i > 1 else features_dim
            aggregator_layer = ConsisMeanAggregator(input_dim, internal_dim, name=layer_name)
            self.seq_layers.append(aggregator_layer)            # 添加线性聚合层
 
        self.classifier = tf.keras.layers.Dense(num_classes,    # 最后一层维度 --> 2，softmax激活分类
                                                activation=tf.nn.softmax,
                                                use_bias=False,
                                                kernel_initializer=init_fn,
                                                name="classifier")
    def call(self, minibatchs: namedtuple, features: tf.Tensor) -> tf.Tensor:
        xs = []   # 一个minibatch的节点的嵌入列表
        for i, minibatch in enumerate(minibatchs):      # 遍历每个节点
            # 节点初始特征作为h(0)嵌入
            x = tf.gather(tf.Variable(features, dtype=float), tf.squeeze(minibatch.src_nodes))   
            # 聚合节点邻居信息得到最终层嵌入x
            for aggregator_layer in self.seq_layers:
                x = aggregator_layer(x,     # 输入特征（上一层的嵌入）
                                     minibatch.dstsrc2srcs.pop(),    # 邻节点信息
                                     minibatch.dstsrc2dsts.pop(),    # 目标节点信息
                                     minibatch.dif_mats.pop(),       # 混淆矩阵
                                     tf.nn.embedding_lookup(
                                         self.relation_vectors, i),     # 关系注意力参数
                                     self.attention_vec     # 注意力参数
                                     )
            xs.append(x)
 
        return self.classifier(tf.nn.l2_normalize(tf.reduce_sum(
            tf.stack(xs, 1), axis=1, keepdims=False), 1))    # 执行分类
```

### 6. 网络训练

``` python
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score
 
import os           # 禁用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
 
num_classes = 2
lr = 0.5
nhid = 128    # 128个隐层。输入32 --> 拼接后64
epochs = 1
batch_size = 512
 
train_nodes = split_ids[0]
val_nodes = split_ids[1]
test_nodes = split_ids[2]
 
# 图网络模型建立
model = GraphConsis(features.shape[-1], nhid, len(sample_sizes), num_classes, len(neigh_dicts))
# 梯度下降优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
# 损失函数定义
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
 
for epoch in range(epochs):        # 训练轮次
    print(f"Epoch {epoch:d}: training...")
    # 迭代产生每一批的节点(batch, lables)
    minibatch_generator = generate_training_minibatch(train_nodes)
    batchs = len(train_nodes) / batch_size     # 每一轮分为多少批
    for inputs, inputs_labels in tqdm(minibatch_generator, total=batchs):       # tqdm显示批次进度
 
        with tf.GradientTape() as tape:
            predicted = model(inputs, features)     # 输出预测值
            loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)      # 损失值
            acc = accuracy_score(inputs_labels,     # 准确率
                                 predicted.numpy().argmax(axis=1))
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        print(f" loss: {loss.numpy():.4f}, acc: {acc:.4f}")
 
    # validation
    print("Validating...")
    val_results = model(build_batch(val_nodes, neigh_dicts,
                                    args.sample_sizes, features), features)
    loss = loss_fn(tf.convert_to_tensor(labels[val_nodes]), val_results)
    val_acc = accuracy_score(labels[val_nodes],
                             val_results.numpy().argmax(axis=1))
    print(f" Epoch: {epoch:d}, "
          f"loss: {loss.numpy():.4f}, "
          f"acc: {val_acc:.4f}")
 
# testing
print("Testing...")
results = model(build_batch(test_nodes, neigh_dicts, sample_sizes, features), features)
test_acc = accuracy_score(labels[test_nodes],results.numpy().argmax(axis=1))
print(f"Test acc: {test_acc:.4f}")
 
# 计算 AUC
fpr, tpr, thresholds = metrics.roc_curve(labels[test_nodes], results.numpy().argmax(axis=1), pos_label=1)
auc = metrics.auc(fpr, tpr)
print(f"AUC: {auc}")
```

### 7. 实验结果

经过1、个epoch训练和验证，Accuracy看起来在85%左右：

>  98%|█████████████████████████████████▍| 226/229.765625 [04:37<00:04,  1.24s/it]
>  loss: 0.3349, acc: 0.8984
>  99%|█████████████████████████████████▋| 228/229.765625 [04:39<00:02,  1.24s/it]
>  loss: 0.4782, acc: 0.8203
> 100%|██████████████████████████████████| 230/229.765625 [04:41<00:00,  1.16s/it]
>  loss: 0.3758, acc: 0.8763
> Validating...
>  Epoch: 1, loss: 0.4136, acc: 0.8548

但是测试时发现accuracy虽然也有85%，但是AUC只有**0.5**：

> Testing...
> Test acc: 0.8547
> AUC: 0.5

其中测试集共有9191条数据，`label=1`的数据有1335条（14.5%），也就是说模型预测全部结果为`label=0`，相当于全部盲猜0也有85%的准确率（样本分布不均衡造成），因此Accuracy基本不能作为模型评判标准，应该关注**AUC**（论文实验结果：0.7428）和**F1**等。~~但实际训练时模型参数未调好一直没有收敛。~~ 10轮训练后测试结果：

> Testing...
> Test acc: 0.8666
> AUC: 0.7161637947474008

使用GPU运行时发现GPU内存占用率高，但是GPU利用率低，是因为大部分计算消耗在创建minibatch上：从大图中提取当前训练批次的节点的混淆矩阵，导致验证或测试时内存溢出。



参考文章：[sklearn的train_test_split()各函数参数含义解释（非常全）](https://www.cnblogs.com/Yanjy-OnlyOne/p/11288098.html)

