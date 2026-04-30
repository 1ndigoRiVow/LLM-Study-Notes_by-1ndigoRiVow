---
title: Token词元
outline: [2, 3]
next: false
---

# Token

在 NLP / 大模型里，**token 可以理解成“模型读写文本的最小单位”**。

它不一定等于一个字，也不一定等于一个词，而是模型分词器切出来的一小段文本。

---

## 1. 最直观理解

一句话：

```text
我喜欢苹果
```

可能会被切成：

```text
我 / 喜欢 / 苹果
```

这里每一段就是一个 token。

英文句子：

```text
I like apples
```

可能会被切成：

```text
I / like / apples
```

但也可能是：

```text
I / like / apple / s
```

因为模型可能把 `apples` 拆成 `apple` 和 `s`。

---

## 2. token 不等于单词

比如：

```text
unbelievable
```

模型可能不会把它当成一个完整单词，而是切成：

```text
un / believable
```

或者：

```text
un / believe / able
```

再比如一个不常见的人名、代码变量名、网址，通常会被拆得更碎：

```text
getUserProfileById
```

可能被切成：

```text
get / User / Profile / By / Id
```

甚至更碎。

---

## 3. 中文里的 token

中文没有天然空格，所以 token 可能是一个字，也可能是一个词：

```text
自注意力机制
```

可能被切成：

```text
自 / 注意力 / 机制
```

也可能是：

```text
自 / 注 / 意 / 力 / 机制
```

具体怎么切，取决于模型使用的 tokenizer。

---

## 4. 为什么要有 token？

因为模型不能直接理解文字。

它真正处理的是数字。

流程大概是：

```text
文本
↓
切成 token
↓
每个 token 转成编号
↓
编号转成向量 embedding
↓
送进神经网络
```

比如：

```text
我 喜欢 苹果
```

可能先变成 token：

```text
["我", "喜欢", "苹果"]
```

再变成编号：

```text
[1256, 8452, 3921]
```

再变成向量：

```text
[0.12, -0.33, ...]
[0.44,  0.08, ...]
[-0.19, 0.27, ...]
```

模型真正计算的是这些向量。

---

## 5. 和 self-attention 的关系

前面说 self-attention 时提到：

```text
x1, x2, x3, ..., xn
```

这里的每个 `x_i`，通常就是一个 token 的向量表示。

比如句子：

```text
我 喜欢 苹果
```

对应：

```text
x1 = “我” 的向量
x2 = “喜欢” 的向量
x3 = “苹果” 的向量
```

self-attention 做的事情就是：

> 每个 token 看其他 token，然后决定要从它们那里吸收多少信息。

比如处理“它”这个 token 时，它可能会更多关注前面的“苹果”。

---

## 6. token 为什么重要？

因为大模型的很多限制都是按 token 算的。

比如：

```text
上下文长度 = 模型一次最多能读多少 token
输出长度 = 模型一次最多能生成多少 token
费用 = API 常常按 token 计费
```

一般粗略估算：

```text
英文：1 token ≈ 0.75 个英文单词
中文：1 token 常常接近 1 个汉字或 1 个短词
```

但这只是近似，不同模型会不一样。

---

一句话总结：

> **token 是模型处理语言时的基本文本单位。文本会先被切成 token，再转成数字和向量，最后才能被 self-attention 等结构处理。**

----

<div style="margin-top: 50px; display: flex; justify-content: flex-end;">
<a href="/LLM-Study-Notes_by-1ndigoRiVow/guide/learning/attention" style="
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
下一章：Self-Attention
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
</a>
</div>

```