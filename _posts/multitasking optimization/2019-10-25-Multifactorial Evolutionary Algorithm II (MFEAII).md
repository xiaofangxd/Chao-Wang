---
title: Multifactorial Evolutionary Algorithm II (MFEAII)
commentable: flase
Edit: 2019-10-25
mathjax: true
mermaid: true
tags: reading
categories: multitasking optimization
description: This is the third in a series of multitasking optimization algorithms-Multifactorial Evolutionary Algorithm II (MFEAII).
---
## 摘要

		此篇博客主要介绍了MFEA理论推导及其改进算法MFEA-II。在多任务优化的情景下，如果任务之间存在潜在关系，那么高质量的解在这些任务之间的转移可以显著提高算法的性能。然而有的时候缺乏关于任务间协同作用的任何先验知识（黑盒优化），主要是负迁移导致算法的性能受损，因此负向任务间交互的易感性会阻碍整体的收敛行为。为了减轻这种负迁移的情况，MFEA-II在MFEA-I的基础上利用在线学习以及不同任务在多任务设置中的相似性（和差异）增强了优化进程。此算法具有原则性的理论分析，其试图让任务之间有害交互的趋势最小化。本博客首先阐述了进化多任务优化的背景，然后介绍了分布估计算法收敛性的理论分析，其次介绍了Kavitesh Kumar Bali等人利用理论分析结果设计了新的算法MFEA-II。

## 一.背景

		首先阐述一下迁移学习和多任务学习的区别。迁移学习利用来自各种源任务的可用数据池来提高相关目标任务的学习效率。例如目前有一个任务B是分辨出一张图片里面是否有猫，假设我们之前做过一个任务A是分辨出一张图片里面是否有狗，那么我们可以利用任务A的数据提高任务B的学习效率，具体来说假设我们任务A的学习模型是一个深度卷积神经网络，此时我们任务两个分类任务很大程度上是相关的，因此我们可以直接将A模型拿来当做B的学习模型，并且可以直接利用A模型的参数，这样可以提高B任务的学习效率。迁移学习主要针对的是目标任务，而多任务学习则是通过在不同任务之间共享信息，同时学习所有任务，以提高整体泛化性能。具体来说，多任务优化的目的是通过促进全方位的知识转移，实现更大的协同搜索，从而促进多个问题的同时高效解决。
	
		之前大多数的多任务学习主要用于回归和预测等领域，而从多任务贝叶斯优化[1]开始，越来越多的学者才在优化领域里开始研究知识迁移。多任务贝叶斯优化是支持自动机器学习超参数优化的一个突出进展。利用多任务高斯过程模型自适应学习任务间依赖关系，改进了贝叶斯优化算法的整体搜索，取得了较好的效果。然而，贝叶斯优化通常局限于相当低或中等维数的问题。这是因为，随着搜索空间维数的增加，保证良好的空间覆盖率(用于学习精确的高斯过程模型)所需的数据点数量呈指数级增长(通常称为冷启动问题)。此外，贝叶斯优化的应用主要局限于连续搜索空间，不直接应用于组合搜索空间，其中不定核可能难以处理。相比之下，进化算法(EAs)在填补这些空白方面一直很有前景。EAs在适应不同的数据表示(无论是连续还是组合)方面提供了巨大的灵活性，并且可以很好地扩展到更高的维度。因此产生了很多多任务进化优化算法，并解决了许多实际问题以及理论问题。但由于缺乏任务间的协同作用，负迁移会导致算法性能受损，因此MFEA-II考虑到这一点，使用数据驱动的在线学习潜在的相似性问题。
## 二.多任务进化优化——MFEA系列

		多任务进化优化顾名思义是利用进化算法去优化多个任务，Yew-Soon Ong等学者将其提出的MFEA建模成了分布估计算法，对其构造和采样了概率混合分布（结合来自不同任务的搜索分布）作为在多任务设置中初始化知识的交换方法。为了通过混合概率分布研究迁移的内部原理，其在之前的理论分析上又提出来新的算法MFEA-II。

### 2.1 混合模型概述

		混合模型必须要求定义一个公共的搜索空间，因此下面先介绍一下统一搜索空间(unified search space)。
	
		在多任务优化中，由于每个优化任务都有自己的搜索空间，因此我们必须建立一个统一的搜索空间以至于可以进行知识迁移。不是一般性，考虑同时求解K个最大化任务$\left\{T_{1}, T_{2}, \ldots, T_{K}\right\}$，每个任务对应的搜索维度为$D_{1}, D_{2}, \ldots, D_{K}$，在这种情况下，我们可以定义一个维度为$D_{u n i f i e d}=\max \left\{D_{1}, D_{2}, \ldots, D_{K}\right\}$的统一空间$X$。这样定义主要有以下两点好处：第一，当同时使用多维搜索空间解决多个任务时，它有助于规避与维数诅咒相关的挑战。第二，它被认为是一种基于种群搜索的有效方法，可以促进有用的遗传物质的发现和从一个任务到另一个任务的隐性转移。这里$X$的范围限制为$[0,1] D_{u n i f i e d}$，它作为一个连续的统一空间，所有候选解都映射到其中(编码)，对于各种离散/组合问题，可以设计不同的编码/解码过程。
	
		接下来我们就可以正式定义混合模型的知识迁移啦！不失一般性的，我们定义$P^{k}$为第k个子任务对应的子种群。定义$f_{k}^{*}=f_{k}\left(x^{*}\right)$为第k个任务的全局最大值。假设对于任务$T_k$而言，x是$X$中的候选解之一，那么定义以下假设：对于任意的$f_{k}^{\prime}<f_{k}^{*}$，集合$H=\left\{x | x \in X, f_{k}(x)>f_{k}^{\prime}\right\}$的Borel measure是正的。在进化多任务算法里，我们可以将与每个任务相联系的子种群在一个时间步$t>0$内源于的潜在概率分布定义为$p^{1}(x, t), p^{2}(x, t), \ldots, p^{K}(x, t)$。因此，在促进任务间交互的过程中，第t次迭代时为第k个任务生成的后代种群被认为是从下面的混合分布中得到的
$$
q^{k}(x, t)=\alpha_{k} \cdot p^{k}(x, t)+\sum_{j \neq k} \alpha_{j} \cdot p^{j}(x, t)
$$
有限的$q^{k}(x, t)$是所有K个可用分布的线性组合，混合系数为$\alpha^{\prime} s\geq 0$。其满足$\alpha_{k}+\sum_{j \neq k} \alpha_{j}=1$。

		通过上述建模我们可以认为多任务进化优化中的知识转移是(9)中从混合概率分布里采样出来的候选点。混合的程度是由系数$\alpha^{\prime} s$控制的，可想而知在缺乏对任务间关系的先验知识的情况下，混合概率分布有可能会错误地造成以负迁移。然而MFEA中并没有考虑，因此原作者在其基础上又提出了MFEA-II进行改进。以下我们还是先分析一下混合模型在多任务下的全局收敛性。主要是要证明基于概率建模的EAs的渐进全局收敛性保持不变。
	
		为了便于数学推导，必须要假设每个子种群很大，即$N \rightarrow \infty$。虽然其不符合实际，但它被认为是一种合理的简化。有这么条件，我们就可以通过Glivenko-Cantelli theorem（随着样本的增加，经验分布函数将随着样本的增加而收敛于其真实的分布函数）推得，子种群$P^{k}$的经验概率密度更接近于真实的潜在分布$p^{k}(x)$。假设初始化的所有子种群中，$p^{k}(x, t=0)$是正的且连续。如果$\lim _{t \rightarrow \infty} \mathbb{E}\left[f_{k}(x)\right]=\lim _{t \rightarrow \infty} \int_{X} f_{k}(x) \cdot p^{k}(x, t) \cdot d x=f_{k}^{*}, \forall k$，那么在多任务环境下每个子任务渐进收敛到全局最优。通俗点说就是随着优化过程的进行，种群分布必须逐渐集中在统一空间中与全局最优对应的点上。在进化多任务算法里，我们通常采用$(\mu, \lambda)$的选择策略，即对于每个任务，从$\lambda$中选择$\mu$个个体($\mu<\lambda$)作为下一代的父代。那么选出来的父代定义为$P_{c}^{k}$。首先，让我们定义一个参数$\theta=\mu / \lambda,0<\theta<1$。那么在此选择方式下，混合概率分布为
$$
p^{k}(x, t+1)=\left\{\begin{array}{ll}{\frac{q^{k}(x, t)}{\theta}} & {\text { if } f_{k}(x) \geq \beta^{k}(t+1)} \\ {0} & {\text { otherwise }}\end{array}\right.
$$
这里$\beta^{k}(t+1)$是一个实数，其满足
$$
\int_{f_{k}(x) \geq \beta^{k}(t+1)} q^{k}(x, t) \cdot d x=\theta
$$
那么给出以下定理。

**定理1**：假设对于所有的k，$p^{k}(x, t=0)$是正的且连续并且$N \rightarrow \infty$。在多任务处理过程中，如果$\alpha_{k}>\theta$，那么各任务渐近收敛到全局最优解。

**证明**：假设第k个任务对应的目标值集合为$F^{k}(t)$。那么集合最小值是$\beta^{k}(t)=\inf F^{k}(t)$。换句话说
$$
p^{k}(x, t)=0 \Leftrightarrow f_{k}(x)<\beta^{k}(t)
$$
我们可以将原问题转化为证明$\beta^{k}(t)$收敛即可。$\beta^{k}(t)<\beta^{k}(t+1)$使人有一种直觉，即种群的扩散一定在缩小，逐渐扩大到统一空间中最有希望的区域。因此以下证明$\beta^{k}(t)<\beta^{k}(t+1)$。

根据(9)式，得
$$
\int_{f_{k}(x) \geq \beta^{k}(t)} q^{k}(x, t) \cdot d x \geq \int_{f_{k}(x) \geq \beta^{k}(t)} \alpha_{k} \cdot p^{k}(x, t) \cdot d x
$$
根据(12)可知$\int_{f_{k}(x) \geq \beta^{k}(t)} p^{k}(x, t) \cdot d x=1$，根据(13)得，
$$
\int_{f_{k}(x) \geq \beta^{k}(t)} q^{k}(x, t) \cdot d x \geq \alpha_{k}
$$
如果$\alpha_{k}>\theta$，根据(14)(11)得，
$$
\int_{f_{k}(x) \geq \beta^{k}(t)} q^{k}(x, t) \cdot d x>\int_{f_{k}(x) \geq \beta^{k}(t+1)} q^{k}(x, t) \cdot d x
$$
根据（15），由于左边比右边值大，因此易知，$\beta^{k}(t)<\beta^{k}(t+1)$。又因为$\beta^{k}$不可能超过$f_{k}^{*}$，那么意味着存在一个极限$\lim _{t \rightarrow \infty} \beta^{k}(t)=f_{k}^{\prime}$使得$f_{k}^{\prime} \leq f_{k}^{*}$。因此以下利用反证法证明$f_{k}^{\prime}=f_{k}^{*}$。即假设$f_{k}^{\prime}<f_{k}^{*}$。根据公式(9)(10)得，
$$
p^{k}(x, t) \geq p^{k}(x, 0)\left[\frac{\alpha_{k}}{\theta}\right]^{t} \forall x : f_{k}(x)>f_{k}^{\prime}
$$
令$H=\left\{x | x \in \boldsymbol{X}, f_{k}(x)>f_{k}^{\prime}\right\}$，因为对所有的$x \in \boldsymbol{X}$，$p^{k}(x, 0)>0$并且$\frac{\alpha_{k}}{\theta}>1$，那么
$$
\lim _{t \rightarrow \infty} p^{k}(x, t)=+\infty, \forall x \in H
$$
那么根据Fatou’s lemma可知，
$$
\lim _{t \rightarrow \infty} \int_{H} p^{k}(x, t) \cdot d x=+\infty
$$
由于$p^{k}(x, t)$是概率密度函数，求和应该为1，因此(18)矛盾，所以$f_{k}^{\prime}=f_{k}^{*}$。即
$$
\lim _{t \rightarrow \infty} \beta^{k}(t)=f_{k}^{*}
$$

### 2.2 MFEA的简介和理论分析

		以上简要介绍了混合模型的基本知识，下面介绍一下MFEA并对其进行理论分析。MFEA事实上是一种基于交叉和变异的进化算法，那上面的混合模型岂不是白分析了？其实在相对强的以父为中心的进化算子约束下，基于交叉和变异的EAs算法与基于随机采样的优化算法有相似之处。其实就是以父代为中心的操作符倾向于使后代更接近父代，这种情况发生的概率更高。常见的例子包括模拟二进制交叉和多项式变异以及小方差的高斯变异等等。在这种情况下，父代和子代的经验密度分布认为相近是十分合理的。因此上面3.1的分析就可以直接拿过来分析MFEA了。
	
		MFEA的基本思想就是使用一个种群P去解决K个优化子任务，每个任务都被视为影响种群进化的因素。与第k个任务关联的子种群表示为$P_k$。对于每个个体而言，定义了技能因子为其擅长任务的编号，标量适应度定义为某个个体在所有与它具有相同技能因子的个体里函数值排名的倒数。MFEA的算法伪代码如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019102520004013.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNDM0NDMw,size_16,color_FFFFFF,t_70)

		根据算法流程易知，MFEA中的知识转移程度由标为随机匹配概率(rmp)的标量(用户定义)参数控制。在MFEA的任意t代，种群$P(t)$的概率密度函数$p(x, t)$是K个概率密度函数$p^{k}(x, t)$的混合。第K个概率密度函数对应第K个子种群$P^{k}(t) \subset P(t)$。*：*在进化算子的适当假设下，$p_{c}(x, t) \approx p(x, t)$，任务之间的交叉分布$p_{c}^{k}(x, t)$不一定等于$p^{k}(x, t)$。有以下定理：

**定理2**：在以父为中心的进化算子的假设下，在MFEA中第k个任务的子代混合分布$p_{c}^{k}(x, t)$可以表示为
$$
p_{c}^{k}(x, t)=\left[1-\frac{0.5 \cdot(K-1) \cdot r m p}{K}\right] \cdot p^{k}(x, t)+\sum_{j \neq k} \frac{0.5 \cdot r m p}{K} \cdot p^{j}(x, t)
$$
**证明**：让后代的解$x_a$被分配一个技能因子k，在以父为中心的进化算子的假设下，$x_a$是从$p^{1}(x, t), p^{2}(x, t), \ldots, \ldots, o r, p^{k}(x, t), \ldots, o r, p^{K}(x, t)$中得到的。有贝叶斯定理可得，
$$
P\left(x_{a} \sim p^{k}(x, t) | \tau_{a}=k\right)=\frac{P\left(\tau_{a}=k | x_{a} \sim p^{k}(x, t)\right) \cdot P\left(x_{a} \sim p^{k}(x, t)\right)}{P\left(\tau_{a}=k\right)}
$$
考虑在MFEA中统一分配资源给K个优化任务，我们有
$$
P\left(x_{a} \sim p^{k}(x, t)\right)=P\left(\tau_{a}=k\right)=\frac{1}{K}
$$
根据上面两式得，
$$
P\left(x_{a} \sim p^{k}(x, t) | \tau_{a}=k\right)=P\left(\tau_{a}=k | x_{a} \sim p^{k}(x, t)\right)
$$
根据算法流程，可以知道$P\left(\tau_{a}=k | x_{a} \sim p^{k}(x, t)\right)$有种3情况：

第一种情况，两个个个体具有相同的技能因子时，$x_a$子代继承同样的技能因子，概率为
$$
P(\text {Scenario}-1)=\frac{1}{K}*1=\frac{1}{K}
$$
第二种情况，满足rmp要求，$x_a$子代继承其技能因子的概率为
$$
P(\text {Scenario}-2)=\frac{0.5 \cdot(K-1) \cdot r m p}{K}
$$
第三种情况，$x_a$是一个很小的变异，那么概率为
$$
P(\text {Scenario}-3)=\frac{(K-1) \cdot (1-r m p)}{K}
$$
那么
$$
P\left(\tau_{a}=k | x_{a} \sim p^{k}(x, t)\right)=\sum_{i=1}^{3} P(S c e n a r i o-i)
$$
因此
$$
P\left(x_{a} \sim p^{k}(x, t) | \tau_{a}=k\right)=1-\frac{0.5 \cdot(K-1) \cdot r m p}{K}
$$

$$
P\left(x_{a} \sim p^{j}(x, t) \forall j \neq k | \tau_{a}=k\right)=\sum_{j \neq k} \frac{0.5 \cdot r m p}{K}
$$

结合上面两式得
$$
p_{c}^{k}(x, t)=\left[1-\frac{0.5 \cdot(K-1) \cdot r m p}{K}\right] \cdot p^{k}(x, t)+\sum_{j \neq k} \frac{0.5 \cdot r m p}{K} \cdot p^{j}(x, t)
$$
证毕。

**定理3**：在大种群的假设下($N \rightarrow \infty$)，如果$\theta(=\mu / \lambda) \leq 0.5$，使用以父为中心的交叉以及$(\mu, \lambda)$选择机制，那么MFEA算法在所有任务上具有全局的收敛性。

**证明**：根据定理2可知，$\alpha_{k}=1-\frac{0.5 \cdot(K-1) \cdot r m p}{K}$。因为$0 \leq r m p \leq 1$，所以$0.5<\alpha_{k} \leq 1$。因此$\alpha_{k}>\theta$。那么根据定理1得证。

		从算法1可以看出，MFEA的性能依赖于rmp的选择。由于缺乏关于任务间关系的先验知识，因此需要调优适当的rmp。从本质上讲，不好的rmp可能会导致任务间的消极交互，或者潜在的知识交换损失。原作者还在文章里证明了在任务之间没有互补性的情况下，单任务可能比多任务处理具有更快的收敛速度，MFEA收敛速度的降低是依赖于rmp的。因此为了更好的探究算法里潜在的知识迁移，作者提出了一种在线rmp估计算法MFEA-II，从理论上保证最小化不同优化任务之间的负面(有害)交互。

### 2.3 MFEA-II的理论分析与算法介绍

		从MFEA的收敛速度分析中，易推断多任务处理中的负迁移可以通过强制$p_{c}^{k}(x, t)$和$p^{k}(x, t)$越近越好。根据定理2可知，让rmp为0即可，但这样就没有信息迁移了，因此，主要目标是提出一个促进知识转移的策略(rmp > 0)，同时减轻负面的任务间交互。为此原作者提出了一种新的基于数据驱动的在线学习方式来估计rmp的值，从而推断出一个最优的混合分布。这里rmp不再是一个标量值，而是K*K大小的矩阵：
$$
R M P=\left[\begin{array}{cccc}{r m p_{1,1} } & {r m p_{1,2}} & {\cdot} & {\cdot} \\ {r m p_{2,1}} & {r m p_{2,2}} & {\cdot} & {\cdot} \\ {\cdot} & {\cdot} & {\cdot} & {\cdot} \\ {\cdot} & {\cdot} & {\cdot} & {\cdot} & {\cdot}\end{array}\right]
$$
这里$r m p_{j, k}=r m p_{k, j}$用来捕捉两个任务之间的协同作用。易知$r m p_{j, j}=1, \forall j$。这种增强提供了一个明显的优势，在许多实际场景中，任务之间的互补性在不同的任务对之间可能不是一致的，原作者提出的方法能够捕获不一致的任务间协同。

		假设在第t代，$g^{k}(x, t)$是任意第k个任务的一个真实分布$p^{k}(x, t)$的概率评估模型，那么$g^{k}(x, t)$是根据子种群$P^{k}(t)$的数据集构建的。将真实的密度函数替换为已学习的概率模型，并使用RMP矩阵的元素代替标量RMP，可以将式(20)重写为
$$
g_{c}^{k}(x, t)=\left[1-\frac{0.5}{K} \cdot \sum_{k \neq j} r m p_{k, j}\right] \cdot g^{k}(x, t)+\frac{0.5}{K} \sum_{j \neq k} r m p_{k, j} \cdot g^{j}(x, t)
$$
这里$g_{c}^{k}(x, t)$是一种近似子代种群$p_{c}^{k}(x, t)$分布的概率混合模型。为了使$p_{c}^{k}(x, t)$和$p^{k}(x, t)$越近越好，等价于学习RMP矩阵使得子代概率模型$g_{c}^{k}(x, t)$精确地建模父代分布$p^{k}(x, t)$。采用KL来度量其两者。即最小化以下式子
$$
\min _{R M P} \sum_{k=1}^{K} K L\left(p^{k}(x, t) \| g_{c}^{k}(x, t)\right)
$$
即
$$
\min _{R M P} \sum_{k=1}^{K} \int_{\boldsymbol{X}} p^{k}(x, t) \cdot\left[\log p^{k}(x, t)-\log g_{c}^{k}(x, t)\right] \cdot d x
$$
最小化上式即最大化以下式子（因为第一项和RMP没关系）
$$
\max _{R M P} \sum_{k=1}^{K} \int_{X} p^{k}(x, t) \cdot \log g_{c}^{k}(x, t) \cdot d x
$$
又因为$\mathbb{E}\left[\log g_{c}^{k}(x, t)\right]=\int_{\boldsymbol{X}} p^{k}(x, t) \cdot \log g_{c}^{k}(x, t) \cdot d x$，那么上式可修改为
$$
\max _{R M P} \sum_{k=1}^{K} \mathbb{E}\left[\log g_{c}^{k}(x, t)\right]
$$
假设$\theta(=\mu / \lambda)=0.5$，那么$\mathbb{E}\left[\log g_{c}^{k}(x, t)\right] = \frac{1}{N/2} \sum_{i=1}^{N / 2} \log g_{c}^{k}\left(x_{i k}, t\right)$，修改上式为
$$
\max _{R M P} \sum_{k=1}^{K} \frac{1}{N/2} \sum_{i=1}^{N / 2} \log g_{c}^{k}\left(x_{i k}, t\right)
$$
即假设对于所有$k \in\{1,2, \ldots, K\}$，在每个子种群$P^{k}(t)$上建立概率模型$g^{k}(x, t)$。那么学习RMP矩阵就是最大化下列似然函数:
$$
\max _{R M P} \sum_{k=1}^{K} \sum_{i=1}^{N / 2} \log g_{c}^{k}\left(x_{i k}, t\right)
$$
这里$x_{ik}$是第i个采样个体在数据集$P^{k}(t)$上。

		通过以上方法进行RMP矩阵的数据驱动学习，可以直接使多任务设置中的各种任务相互作用，同时有助于抑制负迁移。由于实际中$P^{k}(t)$数据较少，如果一个复杂的模型很容易过拟合，事实上在经典的迁移/多任务学习文献中，通常通过内部交叉验证步骤或引入噪声来克服，以防止过度拟合。MFEA-II的作者建议使用一个简单快速的模型去防止过拟合（例如单变量的边际分布）。
	
		因此MFEA-II的框架如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191025200105482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNDM0NDMw,size_16,color_FFFFFF,t_70)

图左:现有MFEA的总体框架。注意MFEA使用了一个离线标量rmp赋值。图右:MFEA-II与添加在线RMP矩阵学习模块。

		其中RMP的在线学习的伪代码如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191025200240517.png)
这里要求$\sum_{k=1}^{K} \sum_{i=1}^{N / 2} \log g_{c}^{k}\left(x_{i k}, t\right)$是向上凸的，因此利用经典的优化器可以在较小的计算开销下求解问题。

MFEA-II中的任务间交叉的伪代码如下

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191025200111723.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNDM0NDMw,size_16,color_FFFFFF,t_70)
		综上所述，MFEA-II在MFEA的基础上加入了在线评估rmp技术，使得直接自动的设置各种任务的相互作用，同时有助于抑制负迁移。这里就不列出它的代码了，感兴趣的同学去我主页上下载吧。
## 三.参考文献
[1]A. Gupta, Y.-S. Ong, and L. Feng, “Multifactorial evolution: Toward evolutionary multitasking,” IEEE Trans. Evol. Comput., vol. 20, no. 3, pp.343–357, 2016.
[2]K. K. Bali, Y. Ong, A. Gupta and P. S. Tan, “Multifactorial Evolutionary Algorithm with Online Transfer Parameter Estimation: MFEA-II,” in IEEE Transactions on Evolutionary Computation, doi: 10.1109/TEVC.2019.2906927, 2019.