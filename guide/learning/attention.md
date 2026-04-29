---
outline: [2, 3]
---

# Self-Attention机制

Self-attention（自注意力）是一种让序列中**每个位置都能“看见”其他位置，并根据相关性动态汇总信息**的机制。

比如一句话：

> 我 喜欢 苹果 因为 它 很 甜

当模型处理“它”时，self-attention 可以让“它”更多关注“苹果”，从而理解“它”指代什么。

---

## 1. Self-attention 在做什么？

假设输入是一串 token $x_i$，
每个 token 都会变成一个向量 $$x_i ∈ R^d$$

Self-attention 的目标是：
对每个位置 `i`，根据它和其他位置 `j` 的相关性，生成一个新的表示：
$新的 x_i = 对所有位置的信息加权求和$

权重不是固定的，而是模型自己学出来的。

---

## 2. Q、K、V

Self-attention 会把每个输入向量映射成三个向量：

- Query   Q：查询
- Key     K：索引
- Value   V：携带的信息

可以类比成检索系统：

- Query：搜索词
- Key：每篇文档的标签
- Value：文档内容

对于输入矩阵：
$$X shape = [seq_len, d_model]$$

通过三个线性层得到：

- $$Q = XWq$$
- $$K = XWk$$
- $$V = XWv$$

然后用 Q 和 K 计算相关性，再用相关性去加权 V。

---

## 3. 核心公式

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

分解一下：

$QK^T$ 表示每个 token 的 Query 和所有 token 的 Key 做点积，得到注意力分数。

$/ sqrt(d_k)$ 是为了防止点积结果太大，导致 softmax 过于尖锐。

$softmax(...)$ 把分数变成概率权重。

$softmax(...) V$ 用权重对 Value 加权求和，得到新的 token 表示。

---

## 4. 一个极简例子

假设输入有 3 个 token：

$$X = [
  x1,
  x2,
  x3
]$$

每个 token 都会产生自己的 Q、K、V。

比如第 2 个 token 会拿自己的 query `q2` 去和所有 key 比较：

$$
score_21 = q2 · k1
score_22 = q2 · k2
score_23 = q2 · k3
$$

softmax 后得到：

$$[0.1, 0.7, 0.2]$$


那么第 2 个 token 的新表示就是：

$$new_x2 = 0.1 * v1 + 0.7 * v2 + 0.2 * v3$$

也就是说，第 2 个 token 最关注自己，其次关注第 3 个 token，较少关注第 1 个 token。

---

# 5. 从零用 NumPy 实现 Self-Attention

下面是最小实现，不依赖深度学习框架。

```python
import numpy as np


def softmax(x, axis=-1):
    # 防止数值溢出
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(X, W_q, W_k, W_v):
    """
    X:   [seq_len, d_model]
    W_q: [d_model, d_k]
    W_k: [d_model, d_k]
    W_v: [d_model, d_v]

    return:
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

测试一下：

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

其中 `attention_weights[i][j]` 表示：
第 `i` 个 token 对第 `j` 个 token 的关注程度。

---

# 6. 加上 batch 维度

实际模型里通常有 batch：

$$X shape = [batch_size, seq_len, d_model]$$

实现如下：

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def batched_self_attention(X, W_q, W_k, W_v):
    """
    X:   [batch_size, seq_len, d_model]
    W_q: [d_model, d_k]
    W_k: [d_model, d_k]
    W_v: [d_model, d_v]

    return:
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

测试：

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

print(output.shape)  # [2, 4, 8]
print(attn.shape)    # [2, 4, 4]
```

---

# 7. 用 PyTorch 从零实现一个 SelfAttention 层

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
        x: [batch_size, seq_len, d_model]
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

使用：

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

# 8. Mask 是什么？

在语言模型里，生成文本时不能让当前位置看到未来 token。

比如预测第 3 个词时，不能看到第 4、第 5 个词。

所以要加 causal mask：

```text
允许看到：
x1
x1 x2
x1 x2 x3
x1 x2 x3 x4
```

不允许看到未来：

```text
x3 不能看 x4、x5
```

PyTorch 实现：

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = x.shape

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

---

# 9. Multi-Head Attention 是什么？

单头 attention 只有一套 Q、K、V。

Multi-head attention 就是同时做多组 self-attention：

- head 1 学语法关系
- head 2 学指代关系
- head 3 学局部上下文
- head 4 学长距离依赖

然后把多个 head 的结果拼起来，再过一个线性层。

简单理解：

- Self-Attention：一个视角看句子
- Multi-Head Attention：多个视角同时看句子

---

## 10. 最小 Multi-Head Self-Attention 实现

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
        x: [batch_size, seq_len, d_model]
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

测试：

```python
x = torch.randn(2, 5, 32)

attn = MultiHeadSelfAttention(d_model=32, num_heads=4)

output, weights = attn(x, causal=True)

print(output.shape)   # [2, 5, 32]
print(weights.shape)  # [2, 4, 5, 5]
```

---

## 11. 总结一句话

Self-attention 的本质是：

> 每个 token 根据自己和其他 token 的相关性，动态决定应该从哪些 token 中吸收多少信息。

最核心的流程就是：

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

实现时最重要的是理解四个 shape：

$$X:      [batch_size, seq_len, d_model]
Q/K/V:  [batch_size, seq_len, d_k]
scores: [batch_size, seq_len, seq_len]
output: [batch_size, seq_len, d_v]$$