# GNN、GCN 与 M3ANET 学习笔记

## 1. 什么是 GNN？

GNN（Graph Neural Network，图神经网络）是一类专门处理图结构数据的神经网络。

一张图可以表示为：

$$G=(V,E)$$

- $V$：节点集合。
- $E$：边集合。
- 节点特征：描述每个节点自身的信息。
- 边特征：描述节点之间的关系。

GNN 的核心思想是：每个节点收集邻居的信息，然后更新自己的特征。

$$m_i=\operatorname{AGG}\left(\{h_j:j\in\mathcal N(i)\}\right)$$

$$h_i'=\operatorname{UPDATE}(h_i,m_i)$$

其中：

- $h_i$：节点 $i$ 当前的特征向量。
- $\mathcal N(i)$：节点 $i$ 的邻居集合。
- $m_i$：节点 $i$ 从邻居收集到的信息。
- $\operatorname{AGG}$：求和、平均、最大值或注意力等聚合操作。

## 2. 图数据如何表示？

### 2.1 节点特征矩阵

假设图中有 $N$ 个节点，每个节点有 $F$ 个特征：

$$H\in\mathbb R^{N\times F}$$

- $N$ 行：一行对应一个节点。
- $F$ 列：一列对应一种特征。

例如：

$$H=\begin{bmatrix}1&2\\3&0\\2&1\end{bmatrix}$$

那么：

$$h_1=[1,2],\quad h_2=[3,0],\quad h_3=[2,1]$$

因此：

$$h_i=\text{节点 }i\text{ 的特征向量}$$

在 M3ANET 中，节点是 EEG 电极，$h_i$ 是第 $i$ 个电极的一段 EEG 信号或经过处理后的特征。

### 2.2 边特征矩阵

假设有 $M$ 条边，每条边有 $D$ 个特征：

$$E_{\mathrm{attr}}\in\mathbb R^{M\times D}$$

这表示：

- $M$ 行：一行对应一条边。
- $D$ 列：一列对应一种边特征。

### 2.3 邻接矩阵

邻接矩阵 $A$ 表示节点之间是否相连：

$$A_{ij}=\begin{cases}1,&i,j\text{ 相连}\\0,&i,j\text{ 不相连}\end{cases}$$

对于无向图，$A$ 通常是对称矩阵，即 $A_{ij}=A_{ji}$。

### 2.4 全局特征

全局特征描述整张图，而不是某个节点或某条边。例如实验条件、温度、压力或整张图的类别。

## 3. 信息聚合

如果节点 $v$ 连接了三条边，边特征分别为 $e_1,e_2,e_3$，使用求和聚合时：

$$m_v=e_1+e_2+e_3$$

一般写成：

$$m_v=\rho_{E\rightarrow V}\left(\{e_{uv}:u\in\mathcal N(v)\}\right)$$

常见聚合方式：

- Sum：能够保留邻居数量信息。
- Mean：减弱节点度数的影响。
- Max：保留每个维度最突出的信息。
- Attention：学习不同邻居的重要程度。

聚合必须具有排列不变性，因为图中的邻居没有天然顺序：

$$e_1+e_2+e_3=e_3+e_1+e_2$$

## 4. 什么是 GCN？

GCN（Graph Convolutional Network，图卷积网络）是 GNN 的一种。

它可以理解为：根据图的连接关系，让每个节点融合自己和邻居的特征。

- CNN：一个像素收集周围像素的信息。
- GCN：一个节点收集图中相邻节点的信息。

经典的一层 GCN 为：

$$H^{(l+1)}=\sigma\left(\hat D^{-\frac12}\hat A\hat D^{-\frac12}H^{(l)}\Theta^{(l)}\right)$$

令：

$$\hat A_{\mathrm{norm}}=\hat D^{-\frac12}\hat A\hat D^{-\frac12}$$

则可以简写为：

$$H^{(l+1)}=\sigma\left(\hat A_{\mathrm{norm}}H^{(l)}\Theta^{(l)}\right)$$

一层 GCN 完成三件事：邻居聚合、特征变换和非线性激活。

## 5. 为什么要加入自连接？

考虑下面的图：

```text
节点2 —— 节点1 —— 节点3
```

它的邻接矩阵为：

$$A=\begin{bmatrix}0&1&1\\1&0&0\\1&0&0\end{bmatrix}$$

对角线是 0，表示节点默认不与自己相连。

单位矩阵为：

$$I=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}$$

加入自连接：

$$\hat A=A+I$$

得到：

$$\hat A=\begin{bmatrix}1&1&1\\1&1&0\\1&0&1\end{bmatrix}$$

这样每个节点更新时会同时使用自己的特征和邻居的特征。

例如，节点 2 会聚合：

$$h_1+h_2$$

## 6. 度和度矩阵

节点的度是它连接的节点数量。

原始图中：

- 节点 1 的度为 2。
- 节点 2 的度为 1。
- 节点 3 的度为 1。

原始度矩阵为：

$$D=\begin{bmatrix}2&0&0\\0&1&0\\0&0&1\end{bmatrix}$$

加入自连接后：

$$\hat d_1=3,\quad\hat d_2=2,\quad\hat d_3=2$$

所以：

$$\hat D=\begin{bmatrix}3&0&0\\0&2&0\\0&0&2\end{bmatrix}$$

计算规则为：

$$\hat D_{ii}=\sum_j\hat A_{ij}$$

也就是对 $\hat A$ 每一行求和，再把结果放到度矩阵的对角线上。

## 7. 度矩阵为什么写成矩阵？

度数原本可以写成一个列表：

$$[3,2,2]$$

把它写成对角矩阵后，可以通过一次矩阵乘法分别缩放所有节点，并且不会混合不同节点的信息。

$$\hat D^{-1}=\begin{bmatrix}\frac13&0&0\\0&\frac12&0\\0&0&\frac12\end{bmatrix}$$

例如：

$$X=\begin{bmatrix}12\\8\\10\end{bmatrix}$$

那么：

$$\hat D^{-1}X=\begin{bmatrix}12/3\\8/2\\10/2\end{bmatrix}=\begin{bmatrix}4\\4\\5\end{bmatrix}$$

因此：

- $\hat A$ 负责把自己和邻居的信息加起来。
- $\hat D^{-1}$ 负责按照每个节点的度分别缩放。

## 8. 为什么前后各乘一个 $\hat D^{-1/2}$？

经典 GCN 使用对称归一化：

$$\hat A_{\mathrm{norm}}=\hat D^{-\frac12}\hat A\hat D^{-\frac12}$$

对于：

$$\hat D=\begin{bmatrix}3&0&0\\0&2&0\\0&0&2\end{bmatrix}$$

有：

$$\hat D^{-\frac12}=\begin{bmatrix}\frac1{\sqrt3}&0&0\\0&\frac1{\sqrt2}&0\\0&0&\frac1{\sqrt2}\end{bmatrix}$$

右边的 $\hat D^{-1/2}$ 根据发送节点的度数缩放信息；左边的 $\hat D^{-1/2}$ 根据接收节点的度数缩放信息。

节点 $j$ 传给节点 $i$ 的权重为：

$$w_{ij}=\frac1{\sqrt{\hat d_i\hat d_j}}$$

本例中的归一化邻接矩阵为：

$$\hat A_{\mathrm{norm}}=\begin{bmatrix}\frac13&\frac1{\sqrt6}&\frac1{\sqrt6}\\\frac1{\sqrt6}&\frac12&0\\\frac1{\sqrt6}&0&\frac12\end{bmatrix}$$

节点 1 和节点 2 的度分别是 3 和 2，因此二者之间的信息传递权重为：

$$w_{12}=w_{21}=\frac1{\sqrt{3\times2}}=\frac1{\sqrt6}$$

这种归一化的作用是：

- 同时考虑发送节点和接收节点的度。
- 避免高度节点产生或接收过强的信息。
- 保持无向图的传播矩阵对称。
- 让多层 GCN 的数值和训练更加稳定。

注意：对称归一化不是简单求平均，所以每行权重之和不一定等于 1。

## 9. 完整的一层 GCN

$$H^{(l+1)}=\sigma\left(\hat A_{\mathrm{norm}}H^{(l)}\Theta^{(l)}\right)$$

各部分含义：

| 符号 | 含义 |
|---|---|
| $H^{(l)}$ | 第 $l$ 层所有节点的特征 |
| $\hat A_{\mathrm{norm}}$ | 决定节点怎样归一化地交换信息 |
| $\Theta^{(l)}$ | 可学习的特征变换参数 |
| $\sigma$ | ReLU 等非线性激活函数 |
| $H^{(l+1)}$ | 更新后的节点特征 |

第一步是聚合邻居：

$$H_{\mathrm{agg}}=\hat A_{\mathrm{norm}}H^{(l)}$$

第二步是特征变换：

$$Z=H_{\mathrm{agg}}\Theta^{(l)}$$

如果：

$$H_{\mathrm{agg}}\in\mathbb R^{N\times F_l}$$

$$\Theta^{(l)}\in\mathbb R^{F_l\times F_{l+1}}$$

那么：

$$Z\in\mathbb R^{N\times F_{l+1}}$$

节点数量不变，但每个节点的特征维度可以改变。

第三步是激活：

$$H^{(l+1)}=\sigma(Z)$$

常用的 ReLU 为：

$$\operatorname{ReLU}(x)=\max(0,x)$$

## 10. 多层 GCN

考虑：

```text
A —— B —— C —— D
```

- 一层 GCN：融合一跳邻居信息。
- 两层 GCN：间接融合两跳邻居信息。
- 三层 GCN：间接融合三跳邻居信息。

例如：

$$H^{(1)}=\sigma\left(\hat A_{\mathrm{norm}}H^{(0)}\Theta^{(0)}\right)$$

$$H^{(2)}=\sigma\left(\hat A_{\mathrm{norm}}H^{(1)}\Theta^{(1)}\right)$$

不同层一般使用不同的参数矩阵。层数越多，节点能够获得的信息范围越大，但太深时可能出现过平滑，使不同节点的特征越来越相似。

## 11. GCN 如何训练？

训练过程为：

```text
输入图和节点特征
        ↓
多层GCN
        ↓
得到预测结果
        ↓
与真实答案计算损失
        ↓
反向传播更新参数
```

基础 GCN 中：

- 邻接矩阵 $A$ 通常由图结构给定。
- 度矩阵 $D$ 根据 $A$ 计算。
- 参数矩阵 $\Theta$ 通过训练学习。

M3ANET 中的电极邻接矩阵 $A$ 被定义为可学习参数，所以模型也能学习电极之间的连接强度。

## 12. 节点表示与图级嵌入

GCN 输出通常仍然包含所有节点：

$$H^{(L)}\in\mathbb R^{N\times F}$$

如果需要表示整张图，可以对所有节点进行池化：

$$h_G=\operatorname{Pool}\left(h_1^{(L)},\ldots,h_N^{(L)}\right)$$

例如平均池化：

$$h_G=\frac1N\sum_{i=1}^{N}h_i^{(L)}$$

得到：

$$h_G\in\mathbb R^F$$

这就是图级嵌入。

M3ANET 中没有紧接着用简单的求和或平均把所有电极压成一个向量，而是保留多电极结构，再使用 Conv1D 和后续网络投影。因此，论文中的“图级嵌入”可以理解为包含整张电极图空间关系的 EEG 表示。

## 13. M3ANET 为什么使用 GCN？

M3ANET 把多通道 EEG 建模成图：

- 节点：EEG 电极。
- 节点特征：每个电极的一段 EEG。
- 边：电极之间的空间或功能关系。
- 整张图：一个 EEG 片段。

GCN 的作用是根据电极连接关系，对多通道 EEG 进行空间信息融合。

GCN 与 Conv1D 的分工是：

- GCN 处理电极之间的空间关系。
- Conv1D 处理 EEG 随时间的变化。

## 14. 什么是图拉普拉斯矩阵？

归一化图拉普拉斯矩阵为：

$$L=I-D^{-\frac12}AD^{-\frac12}$$

可以这样理解：

- $D^{-1/2}AD^{-1/2}$ 描述相邻节点如何传播信息。
- $L$ 描述节点与周围节点之间的差异。

如果某个电极与相邻电极的信号相似，拉普拉斯作用后的差异较小；如果它们差异很大，结果也会较大。

## 15. 为什么使用 Chebyshev 多项式？

传统谱图卷积需要对拉普拉斯矩阵进行特征分解：

$$L=U\Lambda U^T$$

然后在图频域中进行卷积。这种计算比较昂贵。

Chebyshev 图卷积使用多项式近似卷积核：

$$Y=\sum_{k=0}^{K-1}T_k(\tilde L)X\Theta_k$$

前几项为：

$$T_0(\tilde L)=I$$

$$T_1(\tilde L)=\tilde L$$

$$T_2(\tilde L)=2\tilde L^2-I$$

更高阶通过递推计算：

$$T_k(\tilde L)=2\tilde L T_{k-1}(\tilde L)-T_{k-2}(\tilde L)$$

这样只需要矩阵乘法，不需要进行昂贵的拉普拉斯特征分解。

不同阶数可以粗略理解为：

- $T_0$：节点自身信息。
- $T_1$：一跳邻域信息。
- $T_2$：最多涉及两跳邻域的信息。

更准确地说，$k$ 阶多项式具有不超过 $k$ 跳的感受野，并可能包含更近距离的信息组合。

## 16. M3ANET 中的三个 Chebyshev 分量

M3ANET 默认使用：

```python
k_adj = 3
```

代码生成：

$$T_0(\tilde L),\quad T_1(\tilde L),\quad T_2(\tilde L)$$

分别计算：

$$Y_0=T_0(\tilde L)X\Theta_0$$

$$Y_1=T_1(\tilde L)X\Theta_1$$

$$Y_2=T_2(\tilde L)X\Theta_2$$

最后相加并激活：

$$Y=\operatorname{ReLU}(Y_0+Y_1+Y_2)$$

它更接近三个阶数的并行组合：

```text
                ┌→ 0阶图卷积 ─┐
EEG节点特征 ────┼→ 1阶图卷积 ─┼→ 相加 → ReLU
                └→ 2阶图卷积 ─┘
```

而不是严格的串联结构：

```text
GCN1 → GCN2 → GCN3
```

因此，结合源码，更准确的说法是：M3ANET 使用三个 Chebyshev 阶数或三个图卷积分支，同时提取自身、局部和更大范围的电极关系。

## 17. M3ANET 的四个主要模块

### 17.1 EEG 编码器

```text
多通道EEG
   ↓
Chebyshev图卷积
   ↓
Conv1D
   ↓
三个ResBlock
   ↓
EEG embedding
```

作用：

- 使用 GCN 提取电极之间的空间关系。
- 使用 Conv1D 和 ResBlock 提取 EEG 时间特征。
- 得到与听觉注意有关的 EEG 表示。

### 17.2 语音编码器

输入是多人混合语音，包含多尺度 Conv1D 和 GroupMamba。

作用：

- 小卷积核提取短时、局部语音信息。
- 大卷积核提取较长时间范围的信息。
- GroupMamba 建模长距离语音依赖。
- 最终得到多尺度语音表示。

### 17.3 模态对齐模块

声音到达大脑并产生 EEG 需要经过生理过程：

```text
声音 → 耳朵 → 听觉神经 → 大脑皮层 → EEG
```

所以 EEG 与声音可能存在时间延迟。

M3ANET 使用 InfoNCE 对比学习：

- 正样本：时间上正确对应的 EEG 和语音。
- 负样本：时间上不对应的 EEG 和语音。
- 拉近正确配对，推远错误配对。

作用是让 EEG 特征和对应语音特征在时间上对齐。

### 17.4 说话人提取器

它主要包含 CMCA、DPRNN、掩码估计和转置一维卷积解码器。

首先融合 EEG 与语音特征：

$$Y=\operatorname{CMCA}(E,X)$$

然后预测掩码：

$$M=\operatorname{DPRNN}(Y)$$

用掩码过滤混合语音特征：

$$X_{\mathrm{target}}=M\odot X_{\mathrm{mixture}}$$

最后通过解码器恢复目标说话人的语音。

## 18. M3ANET 完整流程

```text
多通道EEG
  ↓
学习电极连接关系
  ↓
Chebyshev GCN提取脑区空间信息
  ↓
Conv1D和ResBlock提取时间信息
  ↓
EEG特征
                    \
                     → 对齐并融合 → 预测掩码 → 提取目标语音
                    /
混合语音
  ↓
多尺度Conv1D
  ↓
GroupMamba提取长距离语音信息
  ↓
语音特征
```

M3ANET 的核心思路是：

$$\text{EEG告诉模型关注谁，混合语音提供有哪些声音，网络最终提取被关注的声音。}$$

## 19. 核心公式速查

加入自连接：

$$\hat A=A+I$$

度矩阵：

$$\hat D_{ii}=\sum_j\hat A_{ij}$$

对称归一化：

$$\hat A_{\mathrm{norm}}=\hat D^{-\frac12}\hat A\hat D^{-\frac12}$$

基础 GCN：

$$H^{(l+1)}=\sigma\left(\hat A_{\mathrm{norm}}H^{(l)}\Theta^{(l)}\right)$$

Chebyshev 图卷积：

$$Y=\sum_{k=0}^{K-1}T_k(\tilde L)X\Theta_k$$

