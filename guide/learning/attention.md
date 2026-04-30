---
title: Self-Attention 自注意力机制
outline: [2, 3]
next: false
---

# Self-Attention 机制：原理与从零实现

Self-Attention（自注意力）是一种让序列中**每个位置都能“看见”其他位置，并根据相关性动态汇总信息**的机制。

例如下面这句话：

> 我 喜欢 苹果 因为 它 很 甜

当模型处理“它”时，Self-Attention 可以让“它”更多关注“苹果”，从而帮助模型理解“它”指代的是“苹果”。

![注意力机制](/images/scaleddot-product.png)

---

## 1. Self-Attention 在做什么？

假设输入是一串 token：

$$
x_1, x_2, \dots, x_n
$$

每个 token 都会被表示成一个向量：

$$
x_i \in \mathbb{R}^d
$$

Self-Attention 的目标是：对每个位置 `i`，根据它和其他位置 `j` 的相关性，生成一个新的表示。

也就是：

> 新的 `x_i` = 对所有位置的信息做一次加权求和。

这里的权重不是人工设定的，而是模型在训练过程中自己学习出来的。

---

## 2. Q、K、V 是什么？

Self-Attention 会把每个输入向量映射成三个向量：

| 名称 | 全称 | 作用 |
| --- | --- | --- |
| Q | Query | 当前 token 用来“查询”其他 token 的向量 |
| K | Key | 每个 token 用来被匹配的“索引”向量 |
| V | Value | 每个 token 真正携带的信息向量 |

可以把它类比成检索系统：

| Attention 概念 | 检索系统类比 |
| --- | --- |
| Query | 搜索词 |
| Key | 文档标签 |
| Value | 文档内容 |

对于输入矩阵：

$$
X \in \mathbb{R}^{\text{seq\_len} \times d_{model}}
$$

通过三个线性层可以得到：

$$
Q = XW_q
$$

$$
K = XW_k
$$

$$
V = XW_v
$$

接下来，模型会用 `Q` 和 `K` 计算相关性，再用这个相关性去加权 `V`。

---

## 3. 核心公式

Self-Attention 的核心公式如下：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

可以拆成四步理解：

1. `QK^T`：每个 token 的 Query 和所有 token 的 Key 做点积，得到注意力分数。
2. `/ sqrt(d_k)`：对分数进行缩放，避免点积结果过大导致 softmax 过于尖锐。
3. `softmax(...)`：把分数转换成概率权重。
4. `softmax(...) V`：用权重对 Value 做加权求和，得到新的 token 表示。

::: tip 为什么要除以 $\sqrt{d_k}$？
当向量维度较大时，点积结果的数值容易变大。过大的分数会让 softmax 输出过于接近 one-hot，导致梯度不稳定。因此需要用 $\sqrt{d_k}$ 进行缩放。
:::

---

## 4. 一个极简例子

假设输入有 3 个 token：

$$
X = [x_1, x_2, x_3]
$$

每个 token 都会产生自己的 `Q`、`K`、`V`。

例如第 2 个 token 会拿自己的 query `q2` 去和所有 key 比较：

$$
score_{21} = q_2 \cdot k_1
$$

$$
score_{22} = q_2 \cdot k_2
$$

$$
score_{23} = q_2 \cdot k_3
$$

经过 softmax 后，假设得到权重：

$$
[0.1, 0.7, 0.2]
$$

那么第 2 个 token 的新表示就是：

$$
new\_x_2 = 0.1v_1 + 0.7v_2 + 0.2v_3
$$

也就是说，第 2 个 token 最关注自己，其次关注第 3 个 token，较少关注第 1 个 token。

---

## 5. 用 NumPy 从零实现 Self-Attention

下面是一个最小实现，不依赖深度学习框架。

```python
import numpy as np


def softmax(x, axis=-1):
    """稳定版 softmax，避免数值溢出。"""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(X, W_q, W_k, W_v):
    """
    Args:
        X:   [seq_len, d_model]
        W_q: [d_model, d_k]
        W_k: [d_model, d_k]
        W_v: [d_model, d_v]

    Returns:
        output: [seq_len, d_v]
        attention_weights: [seq_len, seq_len]
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V

    return output, attention_weights
```

测试代码：

```python
np.random.seed(42)

seq_len = 4
d_model = 8
d_k = 8
d_v = 8

X = np.random.randn(seq_len, d_model)

W_q = np.random.randn(d_model, d_k)
W_k = np.random.randn(d_model, d_k)
W_v = np.random.randn(d_model, d_v)

output, attn = self_attention(X, W_q, W_k, W_v)

print("output shape:", output.shape)
print("attention shape:", attn.shape)
print("attention weights:")
print(attn)
```

输出形状应该是：

```text
output shape: (4, 8)
attention shape: (4, 4)
```

其中：

```text
attention_weights[i][j]
```

表示第 `i` 个 token 对第 `j` 个 token 的关注程度。

---

## 6. 加上 Batch 维度

实际模型里通常会有 batch 维度：

$$
X \in \mathbb{R}^{\text{batch\_size} \times \text{seq\_len} \times d_{model}}
$$

对应实现如下：

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def batched_self_attention(X, W_q, W_k, W_v):
    """
    Args:
        X:   [batch_size, seq_len, d_model]
        W_q: [d_model, d_k]
        W_k: [d_model, d_k]
        W_v: [d_model, d_v]

    Returns:
        output: [batch_size, seq_len, d_v]
        attention_weights: [batch_size, seq_len, seq_len]
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    d_k = Q.shape[-1]
    scores = Q @ np.transpose(K, (0, 2, 1)) / np.sqrt(d_k)

    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V

    return output, attention_weights
```

测试代码：

```python
np.random.seed(42)

batch_size = 2
seq_len = 4
d_model = 8
d_k = 8
d_v = 8

X = np.random.randn(batch_size, seq_len, d_model)

W_q = np.random.randn(d_model, d_k)
W_k = np.random.randn(d_model, d_k)
W_v = np.random.randn(d_model, d_v)

output, attn = batched_self_attention(X, W_q, W_k, W_v)

print(output.shape)  # (2, 4, 8)
print(attn.shape)    # (2, 4, 4)
```

---

## 7. 用 PyTorch 实现 Self-Attention 层

这个版本更接近真实神经网络中的写法。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_v]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
```

使用示例：

```python
batch_size = 2
seq_len = 5
d_model = 16
d_k = 16
d_v = 16

x = torch.randn(batch_size, seq_len, d_model)

attn = SelfAttention(d_model, d_k, d_v)

output, weights = attn(x)

print(output.shape)   # torch.Size([2, 5, 16])
print(weights.shape)  # torch.Size([2, 5, 5])
```

---

## 8. Mask：为什么不能看到未来？

在语言模型中，生成文本时不能让当前位置看到未来 token。

例如预测第 3 个词时，模型不能提前看到第 4、第 5 个词。

因此需要加入 **causal mask**。

允许看到的范围是：

```text
x1
x1 x2
x1 x2 x3
x1 x2 x3 x4
```

不允许看到未来：

```text
x3 不能看 x4、x5
```

PyTorch 实现如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        _, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
```

::: warning 注意
Causal Mask 常用于自回归语言模型，例如 GPT 类模型。它的作用是保证模型在预测当前 token 时，只能使用当前及之前的信息。
:::

---

## 9. Multi-Head Attention 多头注意力机制

![多头注意力机制](/images/multihead.png)

我们发现，与其使用一个单独的注意力函数，并让其中的 queries、keys 和 values 都具有 $d_{\text{model}}$ 维度，不如将 queries、keys 和 values 分别通过不同的、学习得到的线性投影，投影 $h$ 次，分别映射到 $d_k$、$d_k$ 和 $d_v$ 维空间中。

在这些经过投影后的 queries、keys 和 values 的每一个版本上，我们并行地执行注意力函数，从而得到 $d_v$ 维的输出值。然后将这些输出值拼接起来，并再次进行一次投影，得到最终的输出，如图 2 所示。

> 为了说明为什么点积会变大，假设 $q$ 和 $k$ 的各个分量都是相互独立的随机变量，均值为 0，方差为 1。那么它们的点积：
>
> $$
> q \cdot k = \sum_{i=1}^{d_k} q_i k_i
> $$
>
> 其均值为 0，方差为 $d_k$。

多头注意力机制允许模型在不同位置上，从不同的表示子空间中共同关注信息。而如果只使用单个注意力头，这种能力会因为平均化而受到限制。

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

其中：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

其中，投影矩阵为：

$$
W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}
$$

$$
W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}
$$

$$
W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

以及：

$$
W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}
$$

在本文中，我们使用 $h = 8$ 个并行的注意力层，也就是 8 个注意力头。对于每个注意力头，我们设置：

$$
d_k = d_v = d_{\text{model}} / h = 64
$$

由于每个注意力头的维度被降低，因此总计算成本与使用完整维度的单头注意力机制相近。

---

## 10. 最小 Multi-Head Self-Attention 实现

下面是一个最小可运行版本：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            causal: 是否启用 causal mask

        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # [batch_size, seq_len, d_model]
        # -> [batch_size, num_heads, seq_len, d_head]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_head ** 0.5

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()

            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # [batch_size, num_heads, seq_len, d_head]
        # -> [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)

        output = self.W_o(output)

        return output, attention_weights
```

测试代码：

```python
x = torch.randn(2, 5, 32)

attn = MultiHeadSelfAttention(d_model=32, num_heads=4)

output, weights = attn(x, causal=True)

print(output.shape)   # torch.Size([2, 5, 32])
print(weights.shape)  # torch.Size([2, 4, 5, 5])
```

---

## 11. Shape 总结

实现 Self-Attention 时，最重要的是理解下面几个 shape：

| 变量 | Shape | 含义 |
| --- | --- | --- |
| `X` | `[batch_size, seq_len, d_model]` | 输入 token 表示 |
| `Q / K / V` | `[batch_size, seq_len, d_k]` | 查询、索引、信息向量 |
| `scores` | `[batch_size, seq_len, seq_len]` | token 两两之间的相关性分数 |
| `attention_weights` | `[batch_size, seq_len, seq_len]` | softmax 后的注意力权重 |
| `output` | `[batch_size, seq_len, d_v]` | 加权求和后的新 token 表示 |

如果是 Multi-Head Attention，则常见 shape 是：

| 变量 | Shape |
| --- | --- |
| `Q / K / V` | `[batch_size, num_heads, seq_len, d_head]` |
| `attention_weights` | `[batch_size, num_heads, seq_len, seq_len]` |
| `output` | `[batch_size, seq_len, d_model]` |

---

## 12. 总结

Self-Attention 的本质是：

> 每个 token 根据自己和其他 token 的相关性，动态决定应该从哪些 token 中吸收多少信息。

最核心的流程是：

```text
输入 X
↓
生成 Q, K, V
↓
QK^T 计算相关性
↓
softmax 得到注意力权重
↓
加权求和 V
↓
得到新的 token 表示
```

只要理解了 `Q`、`K`、`V`、`scores` 和 `attention_weights` 之间的 shape 变化，就能比较顺利地读懂 Transformer 中的 Attention 实现。

----

<div style="margin-top: 50px; display: flex; justify-content: flex-end;">
<a href="/LLM-Study-Notes_by-1ndigoRiVow/guide/learning/??" style="
background-color: var(--vp-c-brand-1);
color: white;
padding: 12px 24px;
border-radius: 20px;
text-decoration: none;
font-weight: 600;
display: flex;
align-items: center;
gap: 8px;
transition: transform 0.2s;
" onmouseover="this.style.transform='translateX(5px)'" onmouseout="this.style.transform='translateX(0)'">
下一章：
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
</a>
</div>

```