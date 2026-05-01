---
title: GMM
outline: [2, 3]
next: false
---

# 动态面板模型、差分 GMM 与系统 GMM

## 1. 动态面板模型

动态面板模型通常写成：

$$
Y_{it} = \alpha Y_{i,t-1} + \beta X_{it} + \mu_i + \varepsilon_{it}
$$

其中：

- $Y_{it}$：个体 $i$ 在时期 $t$ 的被解释变量
- $Y_{i,t-1}$：被解释变量的一期滞后项
- $X_{it}$：解释变量
- $\mu_i$：个体固定效应，individual effect
- $\varepsilon_{it}$：随机扰动项，也叫 idiosyncratic error
- $\alpha$：动态持续性参数
- $\beta$：解释变量 $X$ 对 $Y$ 的影响

这个模型之所以叫“动态面板模型”，是因为它在右边加入了被解释变量的滞后项：

$$
Y_{i,t-1}
$$

这意味着当前的 $Y$ 不仅受到当前 $X$ 的影响，也受到过去 $Y$ 的影响。

例如：

- 企业绩效可能受到上一年绩效影响
- 城市经济增长可能受到上一年增长水平影响
- 居民消费可能受到上一期消费习惯影响
- 企业创新可能具有路径依赖

---

## 2. 动态面板中的内生性问题

在动态面板模型中，一个核心问题是：

$$
Y_{i,t-1}
$$

通常是内生的。

原模型为：

$$
Y_{it} = \alpha Y_{i,t-1} + \beta X_{it} + \mu_i + \varepsilon_{it}
$$

其中 $\mu_i$ 表示个体不随时间变化的固定特征，例如：

- 企业管理能力
- 城市地理禀赋
- 国家制度环境
- 个人先天能力

由于：

$$
Y_{i,t-1}
$$

本身也受到 $\mu_i$ 的影响，因此：

$$
Cov(Y_{i,t-1}, \mu_i) \neq 0
$$

如果直接使用混合 OLS，估计会有偏。

固定效应模型可以消除 $\mu_i$，但在动态面板中，即使用固定效应，仍然会出现偏误。

这是因为固定效应变换之后，滞后因变量仍然会和变换后的误差项相关。

例如，对模型做去均值处理后：

$$
Y_{i,t-1} - \bar Y_i
$$

会和：

$$
\varepsilon_{it} - \bar \varepsilon_i
$$

相关。

这种偏误被称为：

$$
Nickell\ Bias
$$

也就是 Nickell 偏误。

当时间维度 $T$ 较小时，Nickell 偏误比较明显；当 $T$ 趋于无穷大时，偏误会逐渐减小。

因此，动态面板模型通常不适合直接使用普通 OLS 或标准固定效应估计。

---

## 3. 差分 GMM：Arellano-Bond 方法

差分 GMM，Difference GMM，主要由 Arellano and Bond 在 1991 年提出。

它的基本思路是：

> 先对模型做一阶差分，消除个体固定效应 $\mu_i$，再用变量自身的滞后项作为工具变量。

原模型为：

$$
Y_{it} = \alpha Y_{i,t-1} + \beta X_{it} + \mu_i + \varepsilon_{it}
$$

对两边做一阶差分：

$$
\Delta Y_{it}
=
\alpha \Delta Y_{i,t-1}
+
\beta \Delta X_{it}
+
\Delta \varepsilon_{it}
$$

其中：

$$
\Delta Y_{it} = Y_{it} - Y_{i,t-1}
$$

$$
\Delta Y_{i,t-1} = Y_{i,t-1} - Y_{i,t-2}
$$

$$
\Delta \varepsilon_{it} = \varepsilon_{it} - \varepsilon_{i,t-1}
$$

一阶差分后，个体效应被消掉：

$$
\Delta \mu_i = \mu_i - \mu_i = 0
$$

但是新的问题出现了。

因为：

$$
\Delta Y_{i,t-1}
=
Y_{i,t-1} - Y_{i,t-2}
$$

而：

$$
\Delta \varepsilon_{it}
=
\varepsilon_{it} - \varepsilon_{i,t-1}
$$

其中 $Y_{i,t-1}$ 包含 $\varepsilon_{i,t-1}$，所以：

$$
Cov(\Delta Y_{i,t-1}, \Delta \varepsilon_{it}) \neq 0
$$

因此，差分后的滞后因变量仍然是内生的。

---

## 4. 差分 GMM 的工具变量

差分 GMM 的解决办法是：

> 使用 $Y$ 的更早期水平值作为 $\Delta Y_{i,t-1}$ 的工具变量。

也就是说，用：

$$
Y_{i,t-2}, Y_{i,t-3}, \dots
$$

作为：

$$
\Delta Y_{i,t-1}
$$

的工具变量。

注意，这里使用的是：

$$
Y_{i,t-2}, Y_{i,t-3}, \dots
$$

而不是：

$$
\Delta Y_{i,t-2}, \Delta Y_{i,t-3}, \dots
$$

原因是，如果误差项 $\varepsilon_{it}$ 不存在序列相关，那么更早期的 $Y$ 与当前差分误差项 $\Delta \varepsilon_{it}$ 不相关，但又和 $\Delta Y_{i,t-1}$ 相关。

因此，它们可以作为工具变量。

差分 GMM 的矩条件可以写成：

$$
E[Y_{i,t-s} \Delta \varepsilon_{it}] = 0
$$

其中：

$$
s \geq 2
$$

也就是说，对于 $t$ 期的差分方程，可以使用至少滞后两期的水平变量作为工具变量。

---

## 5. 系统 GMM：Blundell-Bond 方法

系统 GMM，System GMM，是 Blundell and Bond 在 1998 年提出的方法。

它是在差分 GMM 的基础上进一步扩展而来的。

差分 GMM 只估计差分方程：

$$
\Delta Y_{it}
=
\alpha \Delta Y_{i,t-1}
+
\beta \Delta X_{it}
+
\Delta \varepsilon_{it}
$$

而系统 GMM 同时估计两个方程：

1. 差分方程
2. 水平方程

因此它被称为“系统”GMM。

---

## 6. 系统 GMM 的两个方程

### 6.1 差分方程

差分方程为：

$$
\Delta Y_{it}
=
\alpha \Delta Y_{i,t-1}
+
\beta \Delta X_{it}
+
\Delta \varepsilon_{it}
$$

在差分方程中，使用水平值的滞后项作为工具变量：

$$
Y_{i,t-2}, Y_{i,t-3}, \dots
$$

用于工具化：

$$
\Delta Y_{i,t-1}
$$

对应的矩条件为：

$$
E[Z_d' \Delta \varepsilon] = 0
$$

其中 $Z_d$ 是差分方程中的工具变量矩阵。

---

### 6.2 水平方程

系统 GMM 还会保留原来的水平方程：

$$
Y_{it}
=
\alpha Y_{i,t-1}
+
\beta X_{it}
+
\mu_i
+
\varepsilon_{it}
$$

在水平方程中，使用差分值的滞后项作为工具变量。

例如，用：

$$
\Delta Y_{i,t-1}
$$

作为：

$$
Y_{i,t-1}
$$

的工具变量。

对应的矩条件为：

$$
E[Z_l'(\mu_i + \varepsilon_{it})] = 0
$$

其中 $Z_l$ 是水平方程中的工具变量矩阵。

简单来说：

| 方程 | 被估计的形式 | 使用的工具变量 |
|---|---|---|
| 差分方程 | $\Delta Y_{it}$ 方程 | 水平值滞后项，如 $Y_{i,t-2}$ |
| 水平方程 | $Y_{it}$ 方程 | 差分值滞后项，如 $\Delta Y_{i,t-1}$ |

---

## 7. 为什么需要系统 GMM？

差分 GMM 有一个重要问题：

> 当 $Y$ 具有很强的持续性时，滞后水平值会成为弱工具变量。

也就是说，如果：

$$
\alpha \approx 1
$$

那么 $Y$ 很接近随机游走，变量高度持续。

这时：

$$
Y_{i,t-2}
$$

虽然是 $\Delta Y_{i,t-1}$ 的工具变量，但它们之间的相关性可能不够强。

结果就是：

- 工具变量偏弱
- 估计效率较低
- 有限样本偏误较大

系统 GMM 通过额外加入水平方程，并使用差分滞后项作为水平方程的工具变量，可以提供更多矩条件，从而提高估计效率。

因此，系统 GMM 的优势主要包括：

1. 提高估计效率，尤其当 $\alpha$ 接近 1 时
2. 缓解差分 GMM 中弱工具变量问题
3. 减少有限样本偏误
4. 允许使用更多矩条件

---

## 8. 系统 GMM 的适用场景

系统 GMM 通常适用于以下情况：

### 8.1 面板数据结构是大 $N$、小 $T$

也就是：

$$
N > T
$$

通常要求个体数比较多，时间期数比较少。

经验上经常是：

$$
N 较大,\quad T < 10 \text{ 或 } T < 20
$$

但这不是绝对标准。

例如：

- 300 家企业，8 年数据
- 200 个城市，10 年数据
- 100 个国家，15 年数据

---

### 8.2 模型中包含滞后因变量

例如：

$$
Y_{it} = \alpha Y_{i,t-1} + \beta X_{it} + \mu_i + \varepsilon_{it}
$$

如果模型右边有 $Y_{i,t-1}$，就很容易出现动态面板偏误。

---

### 8.3 解释变量可能存在内生性

除了滞后因变量之外，$X_{it}$ 本身也可能内生。

例如：

- 企业创新与企业绩效互相影响
- 金融发展与经济增长互相影响
- 环境规制与企业生产率互相影响
- 数字经济与产业升级互相影响

在这种情况下，系统 GMM 可以使用变量自身的滞后项作为工具变量。

---

## 9. 系统 GMM 的识别条件

系统 GMM 要成立，需要满足一些关键条件。

---

### 9.1 误差项不存在二阶自相关

差分 GMM 和系统 GMM 通常允许差分误差存在一阶自相关。

因为：

$$
\Delta \varepsilon_{it}
=
\varepsilon_{it}
-
\varepsilon_{i,t-1}
$$

所以一阶自相关往往是自然出现的。

因此，AR(1) 显著通常不是问题。

真正关键的是 AR(2) 检验。

如果存在二阶自相关，说明更早期的滞后变量可能与当前误差项相关，工具变量就可能失效。

所以通常希望：

$$
AR(1) 显著
$$

但：

$$
AR(2) 不显著
$$

---

### 9.2 工具变量整体有效

工具变量必须满足外生性，也就是不能和误差项相关。

常见检验包括：

- Sargan 检验
- Hansen 检验

它们的原假设通常是：

$$
H_0: 工具变量整体有效
$$

因此，一般希望检验结果不显著。

如果 Hansen 或 Sargan 检验显著，说明工具变量可能不满足外生性。

不过需要注意：

> Hansen 检验不显著并不等于工具变量一定有效，只能说明没有足够证据拒绝工具变量有效。

---

### 9.3 工具变量数量不能过多

系统 GMM 很容易生成大量工具变量。

如果工具变量数量太多，会导致：

- 过拟合内生变量
- Hansen 检验失真
- p 值虚高
- 估计结果不可靠

因此，实证中通常需要控制工具变量数量。

常见做法包括：

- 限制滞后阶数
- 使用 `collapse` 选项
- 合并工具变量矩阵
- 保证工具变量数量小于个体数量 $N$

---

## 10. 常见检验总结

| 检验 | 作用 | 理想结果 |
|---|---|---|
| AR(1) 检验 | 检验差分误差是否存在一阶自相关 | 通常显著 |
| AR(2) 检验 | 检验是否存在二阶自相关 | 不显著 |
| Sargan 检验 | 检验工具变量整体有效性 | 不显著 |
| Hansen 检验 | 稳健的过度识别检验 | 不显著 |
| Difference-in-Hansen 检验 | 检验额外工具变量集合是否有效 | 不显著 |
| 工具变量数量 | 判断是否存在工具变量过多问题 | 不宜过多，最好小于个体数 |

---

## 11. 差分 GMM 与系统 GMM 的区别

| 方法 | 核心做法 | 工具变量 | 优点 | 问题 |
|---|---|---|---|---|
| 差分 GMM | 对模型一阶差分，消除个体效应 | 用水平值滞后项工具化差分变量 | 消除固定效应，处理动态内生性 | 当变量持续性强时，工具变量可能偏弱 |
| 系统 GMM | 同时估计差分方程和水平方程 | 差分方程用水平滞后，水平方程用差分滞后 | 效率更高，有限样本表现更好 | 工具变量容易过多，对设定要求更强 |

---

## 12. 一句话总结

动态面板模型中，由于加入了滞后因变量 $Y_{i,t-1}$，普通 OLS 和标准固定效应估计通常会产生偏误。

差分 GMM 通过一阶差分消除个体效应，并使用水平值的滞后项作为工具变量。

系统 GMM 则在差分 GMM 的基础上加入水平方程，同时使用“水平滞后工具变量”和“差分滞后工具变量”，从而提高估计效率，尤其适合大 $N$、小 $T$、变量高度持续的动态面板数据。

---

## 13. Stata代码实现

```Stata
// 声明面板
xtset id year
xtdescribe

// 生成滞后项（用于理解）
gen y_l1 = L.y

// 变量定义
global y "y"                   // 因变量
global x "x1 x2"               // 外生变量
global endogenous_x "x3"       // 内生变量
global predet_x "x4"           // 前定变量（与当期ε相关，与过去ε不相关）

/*=========================================
差分GMM估计
=========================================*/
// 基本差分GMM
xtabond2 $y L.$y $x, gmm($y, lag(2 4)) iv($x) robust

/*
参数说明：
- L.y: 因变量一阶滞后
- gmm(y, lag(2 4)): 将y的2-4期滞后作为GMM型工具变量
- iv(x): x作为标准IV（外生变量）
- robust: 异方差稳健标准误
*/

/*=========================================
系统GMM估计（推荐）
=========================================*/
// 标准系统GMM
xtabond2 $y L.$y $x, gmm($y, lag(2 4)) iv($x) robust twostep

// 带内生变量和前定变量的系统GMM
xtabond2 $y L.$y $x, ///
         gmm($y, lag(2 4)) ///
         gmm($endogenous_x, lag(2 4)) ///
         iv($x) ///
         robust twostep

// 更完整的设定
xtabond2 $y L.$y $x $predet_x, ///
         gmm($y, lag(2 4)) ///
         gmm($endogenous_x, lag(2 3)) ///
         iv($x) ///
         iv($predet_x, lag(1 1)) ///
         robust twostep

// 正交分解（small T large N时推荐）
xtabond2 $y L.$y $x, gmm($y, lag(2 4)) iv($x) robust orthogonal

/*=========================================
检验
=========================================*/
xtabond2 $y L.$y $x, gmm($y, lag(2 4)) iv($x) robust twostep

/* 准则1：自相关检验（Arellano-Bond） */
estat abond
/*
输出解读：
- AR(1)：一阶自相关，预期p < 0.05（存在一阶自相关）
- AR(2)：二阶自相关，要求p > 0.05（不存在二阶自相关）
- AR(3)及以上：也应不显著

如果AR(2) p < 0.05，说明ε_it存在序列相关，GMM不一致
*/

/* 准则2：过度识别检验（Sargan/Hansen J） */
estat sargan
estat hansenc
/*
判断标准：
- p > 0.05：接受工具变量有效的原假设（通过检验）
- p < 0.05：工具变量无效，可能存在过度识别问题

注意：
- Sargan检验要求同方差假设
- Hansen检验允许异方差（更稳健，与robust选项搭配）
- p值过于接近1（如0.999）也可能有问题（工具变量过多）
*/

/* 准则3：工具变量有效性 */
estat overid
/*
结合以下判断：
- 工具变量个数不应超过个体数
- Hansen J p值在0.1-0.9之间较为理想
*/

/* 准则4：系数合理性 */
/*
检查：
- 滞后项系数应在0-1之间（平稳性）
- 系数符号和大小应符合经济直觉
- 系统GMM系数应在差分GMM和OLS之间
  OLS向上偏误 → 系统GMM → 固定效应向下偏误
*/

/*=========================================
系统GMM的模型设定优化
=========================================*/
// 限制工具变量数量（避免过多工具变量）
xtabond2 $y L.$y $x, gmm($y, lag(2 2)) iv($x) robust twostep

// collapse选项：压缩工具变量矩阵
xtabond2 $y L.$y $x, gmm($y, lag(2 4) collapse) iv($x) robust twostep

// 使用level选项包含水平方程
xtabond2 $y L.$y $x, gmm($y, lag(2 4)) iv($x) robust twostep

// 一阶差分GMM（不使用水平方程）
xtabond2 $y L.$y $x, gmm($y, lag(2 4)) iv($x) robust twostep nolevel

/*=========================================
不同估计方法的比较
=========================================*/
// OLS（有向上偏误）
reg $y L.$y $x i.year, robust
estimates store ols

// 固定效应（有向下偏误）
xtreg $y L.$y $x i.year, fe robust
estimates store fe

// 差分GMM
xtabond2 $y L.$y $x, gmm($y, lag(2 4)) iv($x) robust twostep nolevel
estimates store diff_gmm

// 系统GMM
xtabond2 $y L.$y $x, gmm($y, lag(2 4)) iv($x) robust twostep
estimates store sys_gmm

// 比较结果
esttab ols fe diff_gmm sys_gmm using "gmm_compare.rtf", replace ///
       cells(b(star fmt(4)) se(par fmt(4))) ///
       star(* 0.10 ** 0.05 *** 0.01) ///
       stats(N ar1 ar2 hansen, fmt(0 3 3 3)) ///
       title("动态面板模型比较")
/*
预期结果排序：
OLS系数 > 系统GMM > 固定效应系数
如果系统GMM系数不在两者之间，说明模型设定有问题
*/

/*=========================================
系统GMM结果输出
=========================================*/
// 完整结果输出
xtabond2 $y L.$y $x, gmm($y, lag(2 4)) iv($x) robust twostep

// 保存结果
estimates store sysgmm
```

----

<div style="margin-top: 50px; display: flex; justify-content: flex-end;">
<a href="/LLM-Study-Notes_by-1ndigoRiVow/guide/econ/??" style="
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
