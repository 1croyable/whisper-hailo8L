编码器前处理：

<img src="D:\WXhistoire\xwechat_files\wxid_1rispibg30f722_ae69\temp\RWTemp\2025-10\9e20f478899dc29eb19741386f9343c8\65756467ba976702174650b9d1f76156.jpg" alt="65756467ba976702174650b9d1f76156" style="zoom: 33%;" />

---

进入残差块层

Norm层，
$$
\text{LayerNorm}(x_i)
= \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \cdot \gamma + \beta
$$

$$
Q=XW_Q+b_Q,\quad K, V一样\\
X:(T, d_{model}),W_Q:(d_{model},d_{model}),Q:(T,d_{model})
$$

评分，每个头独立进行：
$$
S^{(h)} = \frac{Q^{(h)} (K^{(h)})^\top}{d_k^{\frac14}}
$$
让每一行的评分归一化
$$
S^{(h)}_{norm}=softmax(S^{(h)})
$$
加权求和
$$
O^{(h)} = S^{(h)}_{norm} V^{(h)}
$$
再通过一个线性层：
$$
\text{Output} = O W_O + b_O
$$
Norm层，
$$
\text{LayerNorm}(x_i)
= \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \cdot \gamma + \beta
$$
前馈神经网络
$$
线性层\to GELU\to线性层
$$
以上循环n次后出来

补一个norm层后输出

---

解码器数学原理不变，多余的计算如$S^{(h)}$的计算额外加了个掩码矩阵$M$，这种算子开发板肯定支持。所有的操作做完后，得到的$X$矩阵（一个语义向量矩阵）形状是$(batch, T, d_{model})$，应用embedding矩阵去查表：
$$
\text{logits} = X W_E^\top
$$
行归一化后每一行表示出现某个词汇的概率
$$
logits_{last}=softmax(logits)
$$
