# GCN 全过程与 M3ANET Chebyshev 分量

## 1. GCN 到底在做什么？

GCN（Graph Convolutional Network，图卷积网络）的核心作用是：

> 根据图的连接关系，让每个节点收集自己和邻居的信息，再用可学习参数把这些信息加工成新的节点特征。

完整流程如下：

```text
准备节点特征
      ↓
建立节点连接图
      ↓
加入自连接
      ↓
归一化连接权重
      ↓
聚合自己和邻居的信息
      ↓
使用可学习参数变换特征
      ↓
经过激活函数
      ↓
重复若干层或输出任务结果
```

## 2. 准备节点特征

假设有三个 EEG 电极：

```text
电极2 —— 电极1 —— 电极3
```

每个电极是一个图节点。假设当前每个电极只有一个特征：

$$h_1=6,\quad h_2=2,\quad h_3=4$$

节点特征矩阵为：

$$H^{(0)}=\begin{bmatrix}6\\2\\4\end{bmatrix}$$

其中，上标 $0$ 表示进入第一层 GCN 前的原始特征。

一般情况下：

$$H^{(0)}\in\mathbb R^{N\times F}$$

- $N$：节点数量，在 EEG 中是电极数量。
- $F$：每个节点的特征维度。
- $h_i$：第 $i$ 个节点的特征向量。

## 3. 建立邻接矩阵

根据连接关系，邻接矩阵为：

$$A=\begin{bmatrix}0&1&1\\1&0&0\\1&0&0\end{bmatrix}$$

邻接矩阵回答：每个节点应该向哪些节点收集信息？

## 4. 加入自连接

为了让节点在聚合邻居信息时保留自己的信息，需要加入单位矩阵：

$$\hat A=A+I$$

得到：

$$\hat A=\begin{bmatrix}1&1&1\\1&1&0\\1&0&1\end{bmatrix}$$

现在：

- 节点 1 收集节点 1、2、3 的信息。
- 节点 2 收集节点 1、2 的信息。
- 节点 3 收集节点 1、3 的信息。

## 5. 计算度矩阵

对 $\hat A$ 每一行求和：

$$\hat d_1=3,\quad\hat d_2=2,\quad\hat d_3=2$$

构造度矩阵：

$$\hat D=\begin{bmatrix}3&0&0\\0&2&0\\0&0&2\end{bmatrix}$$

度矩阵记录每个节点聚合多少个节点的信息。

## 6. 对连接权重进行归一化

如果直接把邻居信息相加，邻居很多的节点可能得到过大的数值。因此，GCN 使用对称归一化：

$$\hat A_{\mathrm{norm}}=\hat D^{-\frac12}\hat A\hat D^{-\frac12}$$

本例中：

$$\hat A_{\mathrm{norm}}=\begin{bmatrix}\frac13&\frac1{\sqrt6}&\frac1{\sqrt6}\\\frac1{\sqrt6}&\frac12&0\\\frac1{\sqrt6}&0&\frac12\end{bmatrix}$$

近似为：

$$\hat A_{\mathrm{norm}}\approx\begin{bmatrix}0.333&0.408&0.408\\0.408&0.5&0\\0.408&0&0.5\end{bmatrix}$$

节点 $j$ 传给节点 $i$ 的权重为：

$$w_{ij}=\frac1{\sqrt{\hat d_i\hat d_j}}$$

它同时考虑发送节点和接收节点的度数。

## 7. 聚合自己和邻居的信息

计算：

$$H_{\mathrm{agg}}=\hat A_{\mathrm{norm}}H^{(0)}$$

节点 1：

$$h_{1,\mathrm{agg}}=0.333h_1+0.408h_2+0.408h_3\approx4.449$$

节点 2：

$$h_{2,\mathrm{agg}}=0.408h_1+0.5h_2\approx3.449$$

节点 3：

$$h_{3,\mathrm{agg}}=0.408h_1+0.5h_3\approx4.449$$

因此：

$$H_{\mathrm{agg}}\approx\begin{bmatrix}4.449\\3.449\\4.449\end{bmatrix}$$

这一步完成根据图结构融合自己和邻居的信息。

## 8. 使用可学习参数变换特征

为了让模型学习与任务相关的特征，还要乘一个可学习参数矩阵：

$$Z=\hat A_{\mathrm{norm}}H^{(l)}\Theta^{(l)}$$

如果：

$$H^{(l)}\in\mathbb R^{N\times F_l}$$

$$\Theta^{(l)}\in\mathbb R^{F_l\times F_{l+1}}$$

那么：

$$Z\in\mathbb R^{N\times F_{l+1}}$$

节点数量 $N$ 不变，但每个节点的特征维度可以变化。

- $\hat A_{\mathrm{norm}}$ 决定哪些节点之间传递信息，以及传播权重。
- $\Theta^{(l)}$ 学习如何组合和转换聚合后的特征。

## 9. 经过激活函数

完整的一层 GCN 为：

$$H^{(l+1)}=\sigma\left(\hat A_{\mathrm{norm}}H^{(l)}\Theta^{(l)}\right)$$

常用激活函数是 ReLU：

$$\operatorname{ReLU}(x)=\max(0,x)$$

所以，一层 GCN 完成：

```text
邻居聚合 + 可学习特征变换 + 非线性激活
```

## 10. 多层 GCN

考虑：

```text
A —— B —— C —— D
```

- 一层 GCN：节点大约获得一跳邻居的信息。
- 两层 GCN：节点大约获得两跳邻居的信息。
- 三层 GCN：节点大约获得三跳邻居的信息。

每层通常使用不同的可学习参数：

$$H^{(1)}=\sigma\left(\hat A_{\mathrm{norm}}H^{(0)}\Theta^{(0)}\right)$$

$$H^{(2)}=\sigma\left(\hat A_{\mathrm{norm}}H^{(1)}\Theta^{(1)}\right)$$

层数过多可能导致过平滑，即不同节点的特征越来越相似。

## 11. GCN 最后输出什么？

### 节点级任务

直接使用所有节点最终的特征：

$$H^{(L)}=\begin{bmatrix}h_1^{(L)}\\h_2^{(L)}\\\vdots\\h_N^{(L)}\end{bmatrix}$$

### 图级任务

对所有节点进行池化：

$$h_G=\operatorname{Pool}\left(h_1^{(L)},\ldots,h_N^{(L)}\right)$$

例如平均池化：

$$h_G=\frac1N\sum_{i=1}^{N}h_i^{(L)}$$

### 边级任务

组合两个节点的最终特征，预测它们之间的关系：

$$\hat y_{ij}=f\left(h_i^{(L)},h_j^{(L)}\right)$$

## 12. GCN 如何训练？

```text
输入图和节点特征
        ↓
GCN进行邻居聚合和特征变换
        ↓
产生预测结果
        ↓
与真实答案计算损失
        ↓
反向传播更新参数
```

在标准 GCN 中：

- 邻接矩阵 $A$ 通常固定。
- 度矩阵 $D$ 根据 $A$ 计算。
- 参数 $\Theta$ 通过训练学习。

在 M3ANET 中，电极邻接矩阵 $A$ 也是可学习参数，因此模型还会学习电极之间的连接强度。

---

# M3ANET 中三个 Chebyshev 分量的输入与输出

## 13. 它不是三个串联层

M3ANET 代码中的三个 Chebyshev 图卷积分量不是：

```text
X → 第1层 → 第2层 → 第3层 → 输出
```

而是：

```text
                 ┌→ 0阶分量 Y₀ ─┐
同一个输入 X ────┼→ 1阶分量 Y₁ ─┼→ 相加 → ReLU → Y
                 └→ 2阶分量 Y₂ ─┘
```

三个分量都接收同一个原始 EEG 输入 $X$。

## 14. 总输入

输入 Chebynet 的 EEG 为：

$$X\in\mathbb R^{B\times N\times T}$$

- $B$：batch size。
- $N$：EEG 电极数量。
- $T$：每个电极的一段 EEG 信号长度。

M3ANET 代码注释中的示例为：

$$X\in\mathbb R^{8\times128\times29184}$$

即 8 个 EEG 样本，每个样本有 128 个电极，每个电极有 29184 个时间采样点。

每个电极节点的特征为：

$$h_i\in\mathbb R^{29184}$$

## 15. 0 阶分量

$$T_0(\tilde L)=I$$

计算：

$$Y_0=\Theta_0T_0(\tilde L)X=\Theta_0IX$$

主要处理电极自身的信息，并通过参数 $\Theta_0$ 进行可学习变换。

输入输出形状：

$$[B,N,T]\rightarrow[B,N,T]$$

代码示例：

$$[8,128,29184]\rightarrow[8,128,29184]$$

## 16. 1 阶分量

$$T_1(\tilde L)=\tilde L$$

计算：

$$Y_1=\Theta_1T_1(\tilde L)X=\Theta_1\tilde LX$$

它主要提取当前电极与直接相关电极之间的信息。

输入仍然是原始 $X$，而不是 $Y_0$。

输入输出形状：

$$[B,N,T]\rightarrow[B,N,T]$$

代码示例：

$$[8,128,29184]\rightarrow[8,128,29184]$$

## 17. 2 阶分量

$$T_2(\tilde L)=2\tilde L^2-I$$

计算：

$$Y_2=\Theta_2T_2(\tilde L)X$$

由于包含 $\tilde L^2$，它最多可以涉及两跳范围的电极信息。

输入仍然是原始 $X$，而不是 $Y_1$。

输入输出形状：

$$[B,N,T]\rightarrow[B,N,T]$$

代码示例：

$$[8,128,29184]\rightarrow[8,128,29184]$$

## 18. 三个分量合并

三个输出形状相同：

$$Y_0,Y_1,Y_2\in\mathbb R^{B\times N\times T}$$

因此可以逐元素相加，再经过 ReLU：

$$Y=\operatorname{ReLU}(Y_0+Y_1+Y_2)$$

最终输出形状不变：

$$Y\in\mathbb R^{B\times N\times T}$$

其含义是每个电极的新特征同时融合了：

- 电极自身的信息。
- 直接相关电极的信息。
- 更大范围电极的信息。

## 19. 三个分量汇总

| 分量 | 输入 | 图算子 | 主要范围 | 输出 |
|---|---|---|---|---|
| 0 阶 | $X$ | $T_0(\tilde L)=I$ | 节点自身 | $Y_0$ |
| 1 阶 | $X$ | $T_1(\tilde L)=\tilde L$ | 一跳邻域 | $Y_1$ |
| 2 阶 | $X$ | $T_2(\tilde L)=2\tilde L^2-I$ | 最多两跳邻域 | $Y_2$ |

三个分量的输入输出形状都是：

$$[B,N,T]\rightarrow[B,N,T]$$

## 20. Chebynet 后面的处理

Chebyshev 图卷积的输出为：

$$Y\in\mathbb R^{B\times128\times T}$$

接下来进入一维卷积：

```python
Conv1d(
    in_channels=128,
    out_channels=64,
    kernel_size=8,
    stride=4
)
```

形状变为：

$$[B,128,T]\rightarrow[B,64,T']$$

之后再经过三个 ResBlock，继续提取并压缩 EEG 时间特征，最终得到 EEG embedding。

完整流程为：

```text
[B, 128, T]
      ↓
0阶、1阶、2阶Chebyshev分量并行提取
      ↓
相加 + ReLU
      ↓
[B, 128, T]
      ↓
Conv1D
      ↓
[B, 64, T']
      ↓
三个ResBlock
      ↓
最终EEG embedding
```

## 21. 最终总结

标准 GCN：

$$H\xrightarrow{\text{图结构聚合}}\hat A_{\mathrm{norm}}H\xrightarrow{\text{参数变换}}\hat A_{\mathrm{norm}}H\Theta\xrightarrow{\text{激活}}H'$$

M3ANET 的 Chebyshev 图卷积：

$$Y=\operatorname{ReLU}\left(T_0(\tilde L)X\Theta_0+T_1(\tilde L)X\Theta_1+T_2(\tilde L)X\Theta_2\right)$$

核心区别是：M3ANET 的三个 Chebyshev 分量都以同一个原始 EEG 特征 $X$ 为输入，分别提取不同邻域范围的信息，输出形状相同，最后进行相加。

