# 概率分布 - torch.distributions

的`分布 `包中包含参数化概率分布和采样函数。这使得随机计算图形和优化随机梯度估计的建设。这个软件包通常遵循[
TensorFlow分布](https://arxiv.org/abs/1711.10604)包裹的设计。

这是不可能直接通过随机样本backpropagate。不过，也有用于创建可通过被backpropagated替代的功能主要有两种方法。这些是得分函数估计器/似然比估计器/加固和pathwise衍生物估计。加固这通常被视为在强化学习政策梯度法的基础上，与pathwise衍生估计是在变的自动编码重新参数伎俩常见。虽然得分函数仅需要的值样本
F  （ × ） F（X） F  （ × ） 时，pathwise衍生物需要衍生物 F  ' （ × ） F'（x）的 F  ' （ × ）
。接下来的章节中增强学习的榜样讨论这两个。欲了解更多详情，请参阅[梯度估计使用随机计算图形[HTG97。](https://arxiv.org/abs/1506.05254)

## 得分函数

当概率密度函数是可微分的相对于它的参数，我们只需要`样品（） `和`log_prob（） `实施加固：

Δθ=αr∂log⁡p(a∣πθ(s))∂θ\Delta\theta = \alpha r \frac{\partial\log
p(a|\pi^\theta(s))}{\partial\theta}Δθ=αr∂θ∂logp(a∣πθ(s))​

其中 θ \ THETA  θ 为参数， α \阿尔法 α 是学习速率， R  R  R  是奖励和 p  （ 一 |  π θ （ S  ） ） P（A
| \ PI ^ \ THETA（S）） P  （ 一 |  π θ （ S  ） ） 是服用概率操作 一个 一 一 在状态 S  S  S  给定的策略
π θ \ PI ^ \ THETA  π θ 。

在实践中，我们将采样来自一个网络的输出的动作，在一个环境中应用该动作，然后用`log_prob
`构造的等效损失函数。注意，我们使用一个负的，因为优化使用梯度下降，而上面的规则假定梯度上升。有了明确的政策，实施加固将如下代码：

    
    
    probs = policy_network(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()
    

## Pathwise衍生物

实现这些随机/策略梯度的另一种方法是使用从`R样品的重新参数化特技（）
`的方法，其中所述参数化的随机变量可以通过的一个参数确定的函数构造一个无参数的随机变量。因此，重新参数化样本变为微的。用于实现pathwise衍生物将如下所示的代码：

    
    
    params = policy_network(state)
    m = Normal(*params)
    # Any distribution with .has_rsample == True could work based on the application
    action = m.rsample()
    next_state, reward = env.step(action)  # Assuming that reward is differentiable
    loss = -reward
    loss.backward()
    

## 发行

_class_`torch.distributions.distribution.``Distribution`(
_batch_shape=torch.Size([])_ , _event_shape=torch.Size([])_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/distribution.html#Distribution)

    

碱：[ `对象 `](https://docs.python.org/3/library/functions.html#object "\(in
Python v3.7\)")

分布是概率分布的抽象基类。

_property_`arg_constraints`

    

返回从参数名字典来 `应该由这种分配的每个参数满足约束 `对象。参数数量不属于张量不必出现在这个字典。

_property_`batch_shape`

    

返回在其参数是成批的形状。

`cdf`( _value_
)[[source]](_modules/torch/distributions/distribution.html#Distribution.cdf)

    

返回在值评价的累积密度/质量函数。

Parameters

    

**值** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） -

`entropy`()[[source]](_modules/torch/distributions/distribution.html#Distribution.entropy)

    

返回分布的熵，批处理过batch_shape。

Returns

    

形状batch_shape的张量。

`enumerate_support`( _expand=True_
)[[source]](_modules/torch/distributions/distribution.html#Distribution.enumerate_support)

    

包含由离散分布支持的所有值返回张量。其结果将枚举尺寸0，所以结果的形状将是（基数）+ batch_shape + event_shape （其中
event_shape =（）[HTG3用于单变量分布）。

请注意，此枚举在所有分批张量在锁步 [0,0]，[1,1]，...] [HTG1。与扩大=假，枚举沿着昏暗0发生，但与剩余批次尺寸是单尺寸，
[[0]，[1]，.. 。

来遍历充分笛卡尔乘积使用 itertools.product（m.enumerate_support（））。

Parameters

    

**展开** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in
Python v3.7\)")） - 是否在批量变暗以匹配分配的 batch_shape 扩展支持。

Returns

    

张量循环访问尺寸0。

_property_`event_shape`

    

返回单个样品（无配料）的形状。

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/distribution.html#Distribution.expand)

    

返回一个新的分配实例（或填充由派生类提供的现有实例）具有扩展为 batch_shape 批次的尺寸。此方法调用[ `展开 `
](tensors.html#torch.Tensor.expand
"torch.Tensor.expand")上分布的参数。因此，这不适用于扩大分销实例分配新的内存。此外，此不赘述任何ARGS检查或 __init__.py
参数广播中，首先创建一个实例时。

Parameters

    

  * **batch_shape** （ _torch.Size_ ） - 所需的扩展的大小。

  * **_instance** \- 由需要重写 .expand 子提供了新的实例。

Returns

    

有一批新的尺寸分布例如扩大到的batch_size [HTG1。

`icdf`( _value_
)[[source]](_modules/torch/distributions/distribution.html#Distribution.icdf)

    

返回在值评估了逆累积密度/质量函数。

Parameters

    

**value** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) –

`log_prob`( _value_
)[[source]](_modules/torch/distributions/distribution.html#Distribution.log_prob)

    

返回在值评估的概率密度/质量函数的对数。

Parameters

    

**value** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) –

_property_`mean`

    

返回分布的均值。

`perplexity`()[[source]](_modules/torch/distributions/distribution.html#Distribution.perplexity)

    

返回分布的困惑，分批在batch_shape。

Returns

    

Tensor of shape batch_shape.

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/distribution.html#Distribution.rsample)

    

生成sample_shape形重新参数化样本或sample_shape形批量重新参数化的样本，如果分布参数是成批的。

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/distribution.html#Distribution.sample)

    

生成样本的sample_shape形样品或sample_shape形批次如果分布参数是成批的。

`sample_n`( _n_
)[[source]](_modules/torch/distributions/distribution.html#Distribution.sample_n)

    

生成n个样本或样品的n个批次如果分布参数是成批的。

_property_`stddev`

    

返回分布的标准偏差。

_property_`support`

    

返回表示此发行版的支持 `约束 `对象。

_property_`variance`

    

返回分布的方差。

##  ExponentialFamily 

_class_`torch.distributions.exp_family.``ExponentialFamily`(
_batch_shape=torch.Size([])_ , _event_shape=torch.Size([])_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/exp_family.html#ExponentialFamily)

    

碱： `torch.distributions.distribution.Distribution`

ExponentialFamily为属于一个指数族，其概率质量的概率分布的抽象基类/密度函数具有下面定义的形式

pF(x;θ)=exp⁡(⟨t(x),θ⟩−F(θ)+k(x))p_{F}(x; \theta) = \exp(\langle t(x),
\theta\rangle - F(\theta) + k(x))pF​(x;θ)=exp(⟨t(x),θ⟩−F(θ)+k(x))

其中 θ \ THETA  θ 表示自然的参数， T  （ × ） T（X） T  （ × ） 表示充分统计量， F  （ θ ）  F（\ THETA）
F  （ θ ） 是日志归一化本功能离子对于给定的家庭和 K  （ × ） K（x）的 K  （ × ） 是载波度量。

注意

这个类是分布之间的媒介类和属于一个指数家庭主要是检查
.entropy（）和分析KL散方法的正确性分布。我们使用这个类来计算熵和使用AD框架KL信息量和布雷格曼分歧（礼貌：弗兰克·尼尔森和理查德·诺克，熵和指数家庭交叉熵）。

`entropy`()[[source]](_modules/torch/distributions/exp_family.html#ExponentialFamily.entropy)

    

方法来计算使用日志归一化的布雷格曼发散熵。

## 伯努利

_class_`torch.distributions.bernoulli.``Bernoulli`( _probs=None_ ,
_logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/bernoulli.html#Bernoulli)

    

碱： `torch.distributions.exp_family.ExponentialFamily`

创建一个伯努利分布由 `probs参数 `或 `logits`（但不是两者） 。

样品是二进制（0或1）。他们采取值 1 的概率 P 和 0 的概率 1 - P 。

例：

    
    
    >>> m = Bernoulli(torch.tensor([0.3]))
    >>> m.sample()  # 30% chance 1; 70% chance 0
    tensor([ 0.])
    

Parameters

    

  * **probs** （ _号码_ _，_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 抽样的概率 1 

  * **logits** （ _号码_ _，_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 采样的对数比值 1 

`arg_constraints`_= {'logits': Real(), 'probs': Interval(lower_bound=0.0,
upper_bound=1.0)}_

    

`entropy`()[[source]](_modules/torch/distributions/bernoulli.html#Bernoulli.entropy)

    

`enumerate_support`( _expand=True_
)[[source]](_modules/torch/distributions/bernoulli.html#Bernoulli.enumerate_support)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/bernoulli.html#Bernoulli.expand)

    

`has_enumerate_support`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/bernoulli.html#Bernoulli.log_prob)

    

`logits`[[source]](_modules/torch/distributions/bernoulli.html#Bernoulli.logits)

    

_property_`mean`

    

_property_`param_shape`

    

`probs`[[source]](_modules/torch/distributions/bernoulli.html#Bernoulli.probs)

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/bernoulli.html#Bernoulli.sample)

    

`support`_= Boolean()_

    

_property_`variance`

    

## 贝塔

_class_`torch.distributions.beta.``Beta`( _concentration1_ , _concentration0_
, _validate_args=None_
)[[source]](_modules/torch/distributions/beta.html#Beta)

    

Bases: `torch.distributions.exp_family.ExponentialFamily`

β分布由 `参数concentration1`和 `concentration0`。

Example:

    
    
    >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
    tensor([ 0.1046])
    

Parameters

    

  * **concentration1** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的分布的第一浓度参数（常称为α）

  * **concentration0** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分配的第二浓度参数（通常被称为测试版）

`arg_constraints`_= {'concentration0': GreaterThan(lower_bound=0.0),
'concentration1': GreaterThan(lower_bound=0.0)}_

    

_property_`concentration0`

    

_property_`concentration1`

    

`entropy`()[[source]](_modules/torch/distributions/beta.html#Beta.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/beta.html#Beta.expand)

    

`has_rsample`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/beta.html#Beta.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=()_
)[[source]](_modules/torch/distributions/beta.html#Beta.rsample)

    

`support`_= Interval(lower_bound=0.0, upper_bound=1.0)_

    

_property_`variance`

    

## 二项式

_class_`torch.distributions.binomial.``Binomial`( _total_count=1_ ,
_probs=None_ , _logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/binomial.html#Binomial)

    

Bases: `torch.distributions.distribution.Distribution`

创建一个二项分布由`TOTAL_COUNT`和参数为 `probs`或 `logits`（但不是两者）。 `TOTAL_COUNT
`必须broadcastable与 `probs`/`logits`。

Example:

    
    
    >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
    >>> x = m.sample()
    tensor([   0.,   22.,   71.,  100.])
    
    >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
    >>> x = m.sample()
    tensor([[ 4.,  5.],
            [ 7.,  6.]])
    

Parameters

    

  * **TOTAL_COUNT** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 数目的伯努利试验的

  * **probs** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 事件概率

  * **logits** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 事件日志赔率

`arg_constraints`_= {'logits': Real(), 'probs': Interval(lower_bound=0.0,
upper_bound=1.0), 'total_count': IntegerGreaterThan(lower_bound=0)}_

    

`enumerate_support`( _expand=True_
)[[source]](_modules/torch/distributions/binomial.html#Binomial.enumerate_support)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/binomial.html#Binomial.expand)

    

`has_enumerate_support`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/binomial.html#Binomial.log_prob)

    

`logits`[[source]](_modules/torch/distributions/binomial.html#Binomial.logits)

    

_property_`mean`

    

_property_`param_shape`

    

`probs`[[source]](_modules/torch/distributions/binomial.html#Binomial.probs)

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/binomial.html#Binomial.sample)

    

_property_`support`

    

_property_`variance`

    

## 范畴

_class_`torch.distributions.categorical.``Categorical`( _probs=None_ ,
_logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/categorical.html#Categorical)

    

Bases: `torch.distributions.distribution.Distribution`

创建由任一 `probs`或 `logits`（但不是两者参数化的分类分配）。

Note

它相当于分布[ `torch.multinomial（） `](torch.html#torch.multinomial
"torch.multinomial")样本。

样品是整数，从 { 0  ， ...  ， K  \-  1  }  \ {0，\ ldots，K- 1 \\}  { 0  ， ...  ， K  \-
1  }  其中 K 是`probs.size（-1） `。

如果 `probs`是1D与长度 -  K ，每一个元素是该索引处采样的类的相对概率。

如果 `probs`是2D，它被处理为批量相对概率向量。

Note

`probs`必须是非负的，有限的，并且有一个非零和，并且将被归一化总和为1。

参见：[ `torch.multinomial（） `](torch.html#torch.multinomial
"torch.multinomial")

Example:

    
    
    >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
    >>> m.sample()  # equal probability of 0, 1, 2, 3
    tensor(3)
    

Parameters

    

  * **probs** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 事件概率

  * **logits** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 事件日志赔率

`arg_constraints`_= {'logits': Real(), 'probs': Simplex()}_

    

`entropy`()[[source]](_modules/torch/distributions/categorical.html#Categorical.entropy)

    

`enumerate_support`( _expand=True_
)[[source]](_modules/torch/distributions/categorical.html#Categorical.enumerate_support)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/categorical.html#Categorical.expand)

    

`has_enumerate_support`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/categorical.html#Categorical.log_prob)

    

`logits`[[source]](_modules/torch/distributions/categorical.html#Categorical.logits)

    

_property_`mean`

    

_property_`param_shape`

    

`probs`[[source]](_modules/torch/distributions/categorical.html#Categorical.probs)

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/categorical.html#Categorical.sample)

    

_property_`support`

    

_property_`variance`

    

## 柯西

_class_`torch.distributions.cauchy.``Cauchy`( _loc_ , _scale_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/cauchy.html#Cauchy)

    

Bases: `torch.distributions.distribution.Distribution`

从柯西（洛仑兹）分布的样品。独立正态分布的随机变量的装置的比的分布0 如下柯西分布。

Example:

    
    
    >>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
    >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
    tensor([ 2.3214])
    

Parameters

    

  * **LOC** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 模式或分布的中值。

  * **规模** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 半峰半宽。

`arg_constraints`_= {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}_

    

`cdf`( _value_
)[[source]](_modules/torch/distributions/cauchy.html#Cauchy.cdf)

    

`entropy`()[[source]](_modules/torch/distributions/cauchy.html#Cauchy.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/cauchy.html#Cauchy.expand)

    

`has_rsample`_= True_

    

`icdf`( _value_
)[[source]](_modules/torch/distributions/cauchy.html#Cauchy.icdf)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/cauchy.html#Cauchy.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/cauchy.html#Cauchy.rsample)

    

`support`_= Real()_

    

_property_`variance`

    

## χ2 

_class_`torch.distributions.chi2.``Chi2`( _df_ , _validate_args=None_
)[[source]](_modules/torch/distributions/chi2.html#Chi2)

    

碱： `torch.distributions.gamma.Gamma`

创建由形状参数 `DF`参数化的χ2分布。这是完全等同于`伽玛（阿尔法= 0.5 * df，则 的β= 0.5） `

Example:

    
    
    >>> m = Chi2(torch.tensor([1.0]))
    >>> m.sample()  # Chi2 distributed with shape df=1
    tensor([ 0.1046])
    

Parameters

    

**DF** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in
Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） -
分布的形状参数

`arg_constraints`_= {'df': GreaterThan(lower_bound=0.0)}_

    

_property_`df`

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/chi2.html#Chi2.expand)

    

## 狄利克雷

_class_`torch.distributions.dirichlet.``Dirichlet`( _concentration_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/dirichlet.html#Dirichlet)

    

Bases: `torch.distributions.exp_family.ExponentialFamily`

创建由浓度`浓度 `参数化的狄利克雷分布。

Example:

    
    
    >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
    >>> m.sample()  # Dirichlet distributed with concentrarion concentration
    tensor([ 0.1046,  0.8954])
    

Parameters

    

**浓度** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分布的浓度参数（通常称为α）

`arg_constraints`_= {'concentration': GreaterThan(lower_bound=0.0)}_

    

`entropy`()[[source]](_modules/torch/distributions/dirichlet.html#Dirichlet.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/dirichlet.html#Dirichlet.expand)

    

`has_rsample`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/dirichlet.html#Dirichlet.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=()_
)[[source]](_modules/torch/distributions/dirichlet.html#Dirichlet.rsample)

    

`support`_= Simplex()_

    

_property_`variance`

    

## 指数

_class_`torch.distributions.exponential.``Exponential`( _rate_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/exponential.html#Exponential)

    

Bases: `torch.distributions.exp_family.ExponentialFamily`

创建由`速率 `参数化的指数分布。

Example:

    
    
    >>> m = Exponential(torch.tensor([1.0]))
    >>> m.sample()  # Exponential distributed with rate=1
    tensor([ 0.1046])
    

Parameters

    

**速率** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in
Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 率= 1
/刻度分配

`arg_constraints`_= {'rate': GreaterThan(lower_bound=0.0)}_

    

`cdf`( _value_
)[[source]](_modules/torch/distributions/exponential.html#Exponential.cdf)

    

`entropy`()[[source]](_modules/torch/distributions/exponential.html#Exponential.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/exponential.html#Exponential.expand)

    

`has_rsample`_= True_

    

`icdf`( _value_
)[[source]](_modules/torch/distributions/exponential.html#Exponential.icdf)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/exponential.html#Exponential.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/exponential.html#Exponential.rsample)

    

_property_`stddev`

    

`support`_= GreaterThan(lower_bound=0.0)_

    

_property_`variance`

    

##  FisherSnedecor 

_class_`torch.distributions.fishersnedecor.``FisherSnedecor`( _df1_ , _df2_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/fishersnedecor.html#FisherSnedecor)

    

Bases: `torch.distributions.distribution.Distribution`

创建由`DF1`和`DF2`参数化的费雪分布。

Example:

    
    
    >>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))
    >>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
    tensor([ 0.2453])
    

Parameters

    

  * **DF1** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 自由度参数1的

  * **DF2** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 自由度参数2的

`arg_constraints`_= {'df1': GreaterThan(lower_bound=0.0), 'df2':
GreaterThan(lower_bound=0.0)}_

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/fishersnedecor.html#FisherSnedecor.expand)

    

`has_rsample`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/fishersnedecor.html#FisherSnedecor.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/fishersnedecor.html#FisherSnedecor.rsample)

    

`support`_= GreaterThan(lower_bound=0.0)_

    

_property_`variance`

    

## 伽玛

_class_`torch.distributions.gamma.``Gamma`( _concentration_ , _rate_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/gamma.html#Gamma)

    

Bases: `torch.distributions.exp_family.ExponentialFamily`

创建由形状`浓度参数 `和`速率 `一个Gamma分布。

Example:

    
    
    >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
    >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
    tensor([ 0.1046])
    

Parameters

    

  * **浓度** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分布的形状参数（通常称为α）

  * **速率** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 率= 1 /刻度分布（通常被称为测试版）

`arg_constraints`_= {'concentration': GreaterThan(lower_bound=0.0), 'rate':
GreaterThan(lower_bound=0.0)}_

    

`entropy`()[[source]](_modules/torch/distributions/gamma.html#Gamma.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/gamma.html#Gamma.expand)

    

`has_rsample`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/gamma.html#Gamma.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/gamma.html#Gamma.rsample)

    

`support`_= GreaterThan(lower_bound=0.0)_

    

_property_`variance`

    

## 几何

_class_`torch.distributions.geometric.``Geometric`( _probs=None_ ,
_logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/geometric.html#Geometric)

    

Bases: `torch.distributions.distribution.Distribution`

创建一个几何分布由参数 `probs`，其中 `probs`是的概率伯努利试验的成功。它代表在 K  \+  1  的概率 K + 1  K
\+  1  伯努利试验，第一个 K  K  ķ  试验看到一个成功之前失败。

样品是一个非负整数[0， INF  ⁡ \ INF  在 F  ） 。

Example:

    
    
    >>> m = Geometric(torch.tensor([0.3]))
    >>> m.sample()  # underlying Bernoulli has 30% chance 1; 70% chance 0
    tensor([ 2.])
    

Parameters

    

  * **probs** （ _号码_ _，_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 取样 1 的概率。必须在范围（0，1]

  * **logits** （ _号码_ _，_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 采样的对数比值 1  。

`arg_constraints`_= {'logits': Real(), 'probs': Interval(lower_bound=0.0,
upper_bound=1.0)}_

    

`entropy`()[[source]](_modules/torch/distributions/geometric.html#Geometric.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/geometric.html#Geometric.expand)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/geometric.html#Geometric.log_prob)

    

`logits`[[source]](_modules/torch/distributions/geometric.html#Geometric.logits)

    

_property_`mean`

    

`probs`[[source]](_modules/torch/distributions/geometric.html#Geometric.probs)

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/geometric.html#Geometric.sample)

    

`support`_= IntegerGreaterThan(lower_bound=0)_

    

_property_`variance`

    

## 冈贝尔

_class_`torch.distributions.gumbel.``Gumbel`( _loc_ , _scale_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/gumbel.html#Gumbel)

    

碱： `torch.distributions.transformed_distribution.TransformedDistribution`

从Gumbel分布样本。

例子：

    
    
    >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
    >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
    tensor([ 1.0124])
    

Parameters

    

  * **LOC** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分布的位置参数

  * **规模** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分布的尺度参数

`arg_constraints`_= {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}_

    

`entropy`()[[source]](_modules/torch/distributions/gumbel.html#Gumbel.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/gumbel.html#Gumbel.expand)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/gumbel.html#Gumbel.log_prob)

    

_property_`mean`

    

_property_`stddev`

    

`support`_= Real()_

    

_property_`variance`

    

##  HalfCauchy 

_class_`torch.distributions.half_cauchy.``HalfCauchy`( _scale_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/half_cauchy.html#HalfCauchy)

    

Bases: `torch.distributions.transformed_distribution.TransformedDistribution`

创建由规模其中参数化的半正态分布：

    
    
    X ~ Cauchy(0, scale)
    Y = |X| ~ HalfCauchy(scale)
    

Example:

    
    
    >>> m = HalfCauchy(torch.tensor([1.0]))
    >>> m.sample()  # half-cauchy distributed with scale=1
    tensor([ 2.3214])
    

Parameters

    

**规模** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in
Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） -
全柯西分布的尺度

`arg_constraints`_= {'scale': GreaterThan(lower_bound=0.0)}_

    

`cdf`( _value_
)[[source]](_modules/torch/distributions/half_cauchy.html#HalfCauchy.cdf)

    

`entropy`()[[source]](_modules/torch/distributions/half_cauchy.html#HalfCauchy.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/half_cauchy.html#HalfCauchy.expand)

    

`has_rsample`_= True_

    

`icdf`( _prob_
)[[source]](_modules/torch/distributions/half_cauchy.html#HalfCauchy.icdf)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/half_cauchy.html#HalfCauchy.log_prob)

    

_property_`mean`

    

_property_`scale`

    

`support`_= GreaterThan(lower_bound=0.0)_

    

_property_`variance`

    

##  HalfNormal 

_class_`torch.distributions.half_normal.``HalfNormal`( _scale_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/half_normal.html#HalfNormal)

    

Bases: `torch.distributions.transformed_distribution.TransformedDistribution`

Creates a half-normal distribution parameterized by scale where:

    
    
    X ~ Normal(0, scale)
    Y = |X| ~ HalfNormal(scale)
    

Example:

    
    
    >>> m = HalfNormal(torch.tensor([1.0]))
    >>> m.sample()  # half-normal distributed with scale=1
    tensor([ 0.1046])
    

Parameters

    

**规模** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in
Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） -
全正态分布的规模

`arg_constraints`_= {'scale': GreaterThan(lower_bound=0.0)}_

    

`cdf`( _value_
)[[source]](_modules/torch/distributions/half_normal.html#HalfNormal.cdf)

    

`entropy`()[[source]](_modules/torch/distributions/half_normal.html#HalfNormal.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/half_normal.html#HalfNormal.expand)

    

`has_rsample`_= True_

    

`icdf`( _prob_
)[[source]](_modules/torch/distributions/half_normal.html#HalfNormal.icdf)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/half_normal.html#HalfNormal.log_prob)

    

_property_`mean`

    

_property_`scale`

    

`support`_= GreaterThan(lower_bound=0.0)_

    

_property_`variance`

    

## 独立

_class_`torch.distributions.independent.``Independent`( _base_distribution_ ,
_reinterpreted_batch_ndims_ , _validate_args=None_
)[[source]](_modules/torch/distributions/independent.html#Independent)

    

Bases: `torch.distributions.distribution.Distribution`

重新诠释一些分布作为事件DIMS的一批DIMS的。

这是用于改变 `log_prob（） `结果的形状主要是有用的。例如，要创建一个具有相同形状的对角线正态分布为多元正态分布（这样它们可以互换），您可以：

    
    
    >>> loc = torch.zeros(3)
    >>> scale = torch.ones(3)
    >>> mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
    >>> [mvn.batch_shape, mvn.event_shape]
    [torch.Size(()), torch.Size((3,))]
    >>> normal = Normal(loc, scale)
    >>> [normal.batch_shape, normal.event_shape]
    [torch.Size((3,)), torch.Size(())]
    >>> diagn = Independent(normal, 1)
    >>> [diagn.batch_shape, diagn.event_shape]
    [torch.Size(()), torch.Size((3,))]
    

Parameters

    

  * **base_distribution** （ _torch.distributions.distribution.Distribution_ ） - 碱分布

  * **reinterpreted_batch_ndims** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 批次的数量变暗以重新解释作为事件变暗

`arg_constraints`_= {}_

    

`entropy`()[[source]](_modules/torch/distributions/independent.html#Independent.entropy)

    

`enumerate_support`( _expand=True_
)[[source]](_modules/torch/distributions/independent.html#Independent.enumerate_support)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/independent.html#Independent.expand)

    

_property_`has_enumerate_support`

    

_property_`has_rsample`

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/independent.html#Independent.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/independent.html#Independent.rsample)

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/independent.html#Independent.sample)

    

_property_`support`

    

_property_`variance`

    

## 拉普拉斯

_class_`torch.distributions.laplace.``Laplace`( _loc_ , _scale_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/laplace.html#Laplace)

    

Bases: `torch.distributions.distribution.Distribution`

创建由`LOC`和参数化的拉普拉斯分布：ATTR：”缩放”。

Example:

    
    
    >>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
    >>> m.sample()  # Laplace distributed with loc=0, scale=1
    tensor([ 0.1046])
    

Parameters

    

  * **LOC** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分布的平均

  * **规模** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分布的尺度

`arg_constraints`_= {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}_

    

`cdf`( _value_
)[[source]](_modules/torch/distributions/laplace.html#Laplace.cdf)

    

`entropy`()[[source]](_modules/torch/distributions/laplace.html#Laplace.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/laplace.html#Laplace.expand)

    

`has_rsample`_= True_

    

`icdf`( _value_
)[[source]](_modules/torch/distributions/laplace.html#Laplace.icdf)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/laplace.html#Laplace.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/laplace.html#Laplace.rsample)

    

_property_`stddev`

    

`support`_= Real()_

    

_property_`variance`

    

## 对数正态分布

_class_`torch.distributions.log_normal.``LogNormal`( _loc_ , _scale_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/log_normal.html#LogNormal)

    

Bases: `torch.distributions.transformed_distribution.TransformedDistribution`

创建一个日志正态分布由参数 `LOC`和 `规模 `其中：

    
    
    X ~ Normal(loc, scale)
    Y = exp(X) ~ LogNormal(loc, scale)
    

Example:

    
    
    >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
    >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
    tensor([ 0.1046])
    

Parameters

    

  * **LOC** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 平均log分布的

  * **规模** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 日志中的分布的标准偏差

`arg_constraints`_= {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}_

    

`entropy`()[[source]](_modules/torch/distributions/log_normal.html#LogNormal.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/log_normal.html#LogNormal.expand)

    

`has_rsample`_= True_

    

_property_`loc`

    

_property_`mean`

    

_property_`scale`

    

`support`_= GreaterThan(lower_bound=0.0)_

    

_property_`variance`

    

##  LowRankMultivariateNormal 

_class_`torch.distributions.lowrank_multivariate_normal.``LowRankMultivariateNormal`(
_loc_ , _cov_factor_ , _cov_diag_ , _validate_args=None_
)[[source]](_modules/torch/distributions/lowrank_multivariate_normal.html#LowRankMultivariateNormal)

    

Bases: `torch.distributions.distribution.Distribution`

创建具有由`cov_factor`和参数化的低秩形式协方差矩阵多元正态分布`cov_diag`：

    
    
    covariance_matrix = cov_factor @ cov_factor.T + cov_diag
    

例

    
    
    >>> m = LowRankMultivariateNormal(torch.zeros(2), torch.tensor([1, 0]), torch.tensor([1, 1]))
    >>> m.sample()  # normally distributed with mean=`[0,0]`, cov_factor=`[1,0]`, cov_diag=`[1,1]`
    tensor([-0.2102, -0.5429])
    

Parameters

    

  * **LOC** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 与形状分布的平均值 batch_shape + event_shape 

  * **cov_factor** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 具有形状 batch_shape + event_shape +（秩）协方差矩阵的低秩形式因子部分

  * **cov_diag** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 具有形状协方差矩阵的低秩的形式对角部分 batch_shape + event_shape 

Note

用于行列式和协方差矩阵的逆的计算，避免当 cov_factor.shape [1] & LT ; & LT ; cov_factor.shape [0]
由于[
Woodbury的矩阵身份](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)和[矩阵行列式引理](https://en.wikipedia.org/wiki/Matrix_determinant_lemma)。由于这些公式，我们只需要计算小尺寸“电容”矩阵的行列式和逆：

    
    
    capacitance = I + cov_factor.T @ inv(cov_diag) @ cov_factor
    

`arg_constraints`_= {'cov_diag': GreaterThan(lower_bound=0.0), 'cov_factor':
Real(), 'loc': Real()}_

    

`covariance_matrix`[[source]](_modules/torch/distributions/lowrank_multivariate_normal.html#LowRankMultivariateNormal.covariance_matrix)

    

`entropy`()[[source]](_modules/torch/distributions/lowrank_multivariate_normal.html#LowRankMultivariateNormal.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/lowrank_multivariate_normal.html#LowRankMultivariateNormal.expand)

    

`has_rsample`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/lowrank_multivariate_normal.html#LowRankMultivariateNormal.log_prob)

    

_property_`mean`

    

`precision_matrix`[[source]](_modules/torch/distributions/lowrank_multivariate_normal.html#LowRankMultivariateNormal.precision_matrix)

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/lowrank_multivariate_normal.html#LowRankMultivariateNormal.rsample)

    

`scale_tril`[[source]](_modules/torch/distributions/lowrank_multivariate_normal.html#LowRankMultivariateNormal.scale_tril)

    

`support`_= Real()_

    

`variance`[[source]](_modules/torch/distributions/lowrank_multivariate_normal.html#LowRankMultivariateNormal.variance)

    

## 多项式

_class_`torch.distributions.multinomial.``Multinomial`( _total_count=1_ ,
_probs=None_ , _logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/multinomial.html#Multinomial)

    

Bases: `torch.distributions.distribution.Distribution`

创建一个多项分布由`TOTAL_COUNT`和参数为 `probs`或 `logits`（但不是两者）。 probs 索引超过类别 `
最内尺寸。所有其他尺寸价格指数比批次。`

注意，`TOTAL_COUNT`不必如果只 `log_prob（） `被称为指定（见下面例子）

Note

`probs`必须是非负的，有限的，并且有一个非零和，并且将被归一化总和为1。

  * `样品（） `需要一个单一的共享 TOTAL_COUNT 所有参数和样品。

  * `log_prob（） `允许为每个参数和样品不同 TOTAL_COUNT 。

Example:

    
    
    >>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
    >>> x = m.sample()  # equal probability of 0, 1, 2, 3
    tensor([ 21.,  24.,  30.,  25.])
    
    >>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)
    tensor([-4.1338])
    

Parameters

    

  * **TOTAL_COUNT** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的实验中

  * **probs** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities

  * **logits** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 事件数概率

`arg_constraints`_= {'logits': Real(), 'probs': Simplex()}_

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/multinomial.html#Multinomial.expand)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/multinomial.html#Multinomial.log_prob)

    

_property_`logits`

    

_property_`mean`

    

_property_`param_shape`

    

_property_`probs`

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/multinomial.html#Multinomial.sample)

    

_property_`support`

    

_property_`variance`

    

##  MultivariateNormal 

_class_`torch.distributions.multivariate_normal.``MultivariateNormal`( _loc_ ,
_covariance_matrix=None_ , _precision_matrix=None_ , _scale_tril=None_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/multivariate_normal.html#MultivariateNormal)

    

Bases: `torch.distributions.distribution.Distribution`

创建一个多变量正态分布（也称为高斯分布）由一个平均向量和协方差矩阵参数分布。

多元正态分布可以在正定协方差矩阵 Σ \ mathbf {方面来参数化\西格玛}  Σ 或正定精度矩阵 Σ \-  1  \ mathbf {\西格玛} ^
{ - 1}  Σ \-  1  或下三角矩阵 [HTG8 5]  L  \ mathbf {L}  L  具有正值对角项，使得 Σ =  L  L  ⊤
\ mathbf {\西格玛} = \ mathbf {L} \ mathbf {L} ^ \顶 Σ  =  L  L  ⊤
。这个三角矩阵可以通过例如获得协方差的Cholesky分解。

Example

    
    
    >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
    >>> m.sample()  # normally distributed with mean=`[0,0]`and covariance_matrix=`I`
    tensor([-0.2102, -0.5429])
    

Parameters

    

  * **LOC** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分布的平均

  * **covariance_matrix** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 正定协方差矩阵

  * **precision_matrix** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 正定矩阵精度

  * **scale_tril** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 协方差的下三角因子，具有正值的对角

Note

只有 `covariance_matrix`或 `precision_matrix`或 `酮 scale_tril`可以被指定。

使用 `scale_tril`将更有效率：所有计算内部是基于 `scale_tril`。如果 `covariance_matrix`
或 `precision_matrix`被传递，相反，它只是用来计算相应的使用Cholesky分解下三角矩阵。

`arg_constraints`_= {'covariance_matrix': PositiveDefinite(), 'loc':
RealVector(), 'precision_matrix': PositiveDefinite(), 'scale_tril':
LowerCholesky()}_

    

`covariance_matrix`[[source]](_modules/torch/distributions/multivariate_normal.html#MultivariateNormal.covariance_matrix)

    

`entropy`()[[source]](_modules/torch/distributions/multivariate_normal.html#MultivariateNormal.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/multivariate_normal.html#MultivariateNormal.expand)

    

`has_rsample`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/multivariate_normal.html#MultivariateNormal.log_prob)

    

_property_`mean`

    

`precision_matrix`[[source]](_modules/torch/distributions/multivariate_normal.html#MultivariateNormal.precision_matrix)

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/multivariate_normal.html#MultivariateNormal.rsample)

    

`scale_tril`[[source]](_modules/torch/distributions/multivariate_normal.html#MultivariateNormal.scale_tril)

    

`support`_= Real()_

    

_property_`variance`

    

##  NegativeBinomial 

_class_`torch.distributions.negative_binomial.``NegativeBinomial`(
_total_count_ , _probs=None_ , _logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/negative_binomial.html#NegativeBinomial)

    

Bases: `torch.distributions.distribution.Distribution`

创建一个负二项分布，即`TOTAL_COUNT`得以实现故障之前需要独立同伯努利试验数目的分布。每个伯努利试验的成功的概率是 `probs`。

Parameters

    

  * **TOTAL_COUNT** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 负伯努利的非负数试验停止，虽然分布仍然是成立的实值计

  * **probs** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 在半开区间[0成功的事件概率，1）

  * **logits** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 事件日志把握成功的概率

`arg_constraints`_= {'logits': Real(), 'probs':
HalfOpenInterval(lower_bound=0.0, upper_bound=1.0), 'total_count':
GreaterThanEq(lower_bound=0)}_

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/negative_binomial.html#NegativeBinomial.expand)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/negative_binomial.html#NegativeBinomial.log_prob)

    

`logits`[[source]](_modules/torch/distributions/negative_binomial.html#NegativeBinomial.logits)

    

_property_`mean`

    

_property_`param_shape`

    

`probs`[[source]](_modules/torch/distributions/negative_binomial.html#NegativeBinomial.probs)

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/negative_binomial.html#NegativeBinomial.sample)

    

`support`_= IntegerGreaterThan(lower_bound=0)_

    

_property_`variance`

    

## 正常

_class_`torch.distributions.normal.``Normal`( _loc_ , _scale_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/normal.html#Normal)

    

Bases: `torch.distributions.exp_family.ExponentialFamily`

创建普通的（也称为高斯分布）由`LOC`和`规模 `参数化分布。

Example:

    
    
    >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    >>> m.sample()  # normally distributed with loc=0 and scale=1
    tensor([ 0.1046])
    

Parameters

    

  * **LOC** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的分布的平均值（通常称为作为亩）

  * **规模** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的分布的标准偏差（通常称为西格马）

`arg_constraints`_= {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}_

    

`cdf`( _value_
)[[source]](_modules/torch/distributions/normal.html#Normal.cdf)

    

`entropy`()[[source]](_modules/torch/distributions/normal.html#Normal.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/normal.html#Normal.expand)

    

`has_rsample`_= True_

    

`icdf`( _value_
)[[source]](_modules/torch/distributions/normal.html#Normal.icdf)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/normal.html#Normal.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/normal.html#Normal.rsample)

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/normal.html#Normal.sample)

    

_property_`stddev`

    

`support`_= Real()_

    

_property_`variance`

    

##  OneHotCategorical 

_class_`torch.distributions.one_hot_categorical.``OneHotCategorical`(
_probs=None_ , _logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/one_hot_categorical.html#OneHotCategorical)

    

Bases: `torch.distributions.distribution.Distribution`

创建由 `参数化的独热分类分布probs`或 `logits`。

样品独热编码大小的矢量`probs.size（-1） `。

Note

`probs`必须是非负的，有限的，并且有一个非零和，并且将被归一化总和为1。

参见：`torch.distributions.Categorical（） [HTG3用于probs的 `规范 `和 `logits`。`

Example:

    
    
    >>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
    >>> m.sample()  # equal probability of 0, 1, 2, 3
    tensor([ 0.,  0.,  0.,  1.])
    

Parameters

    

  * **probs** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities

  * **logits** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event log probabilities

`arg_constraints`_= {'logits': Real(), 'probs': Simplex()}_

    

`entropy`()[[source]](_modules/torch/distributions/one_hot_categorical.html#OneHotCategorical.entropy)

    

`enumerate_support`( _expand=True_
)[[source]](_modules/torch/distributions/one_hot_categorical.html#OneHotCategorical.enumerate_support)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/one_hot_categorical.html#OneHotCategorical.expand)

    

`has_enumerate_support`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/one_hot_categorical.html#OneHotCategorical.log_prob)

    

_property_`logits`

    

_property_`mean`

    

_property_`param_shape`

    

_property_`probs`

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/one_hot_categorical.html#OneHotCategorical.sample)

    

`support`_= Simplex()_

    

_property_`variance`

    

## 帕累托

_class_`torch.distributions.pareto.``Pareto`( _scale_ , _alpha_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/pareto.html#Pareto)

    

Bases: `torch.distributions.transformed_distribution.TransformedDistribution`

从帕累托1型分布的样品。

Example:

    
    
    >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
    >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
    tensor([ 1.5623])
    

Parameters

    

  * **scale** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _or_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Scale parameter of the distribution

  * **阿尔法** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分布的形状参数

`arg_constraints`_= {'alpha': GreaterThan(lower_bound=0.0), 'scale':
GreaterThan(lower_bound=0.0)}_

    

`entropy`()[[source]](_modules/torch/distributions/pareto.html#Pareto.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/pareto.html#Pareto.expand)

    

_property_`mean`

    

_property_`support`

    

_property_`variance`

    

## 泊松

_class_`torch.distributions.poisson.``Poisson`( _rate_ , _validate_args=None_
)[[source]](_modules/torch/distributions/poisson.html#Poisson)

    

Bases: `torch.distributions.exp_family.ExponentialFamily`

创建由`速率 `，速率参数参数化的泊松分布。

样品为非负整数，由给定PMF

rateke−ratek!\mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!} ratekk!e−rate​

Example:

    
    
    >>> m = Poisson(torch.tensor([4]))
    >>> m.sample()
    tensor([ 3.])
    

Parameters

    

**速率** （ _号码_ _，_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 速率参数

`arg_constraints`_= {'rate': GreaterThan(lower_bound=0.0)}_

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/poisson.html#Poisson.expand)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/poisson.html#Poisson.log_prob)

    

_property_`mean`

    

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/poisson.html#Poisson.sample)

    

`support`_= IntegerGreaterThan(lower_bound=0)_

    

_property_`variance`

    

##  RelaxedBernoulli 

_class_`torch.distributions.relaxed_bernoulli.``RelaxedBernoulli`(
_temperature_ , _probs=None_ , _logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/relaxed_bernoulli.html#RelaxedBernoulli)

    

Bases: `torch.distributions.transformed_distribution.TransformedDistribution`

创建RelaxedBernoulli分布，由 `温度 `参数化的，并且或者 `probs`或 `logits`
（但不是两者）。这是伯努利分布的松弛版本，所以这些值是（0，1），并且具有reparametrizable样品。

Example:

    
    
    >>> m = RelaxedBernoulli(torch.tensor([2.2]),
                             torch.tensor([0.1, 0.2, 0.3, 0.99]))
    >>> m.sample()
    tensor([ 0.2951,  0.3442,  0.8918,  0.9021])
    

Parameters

    

  * **温度** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 松弛温度

  * **probs** ( _Number_ _,_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the probability of sampling 1

  * **logits** ( _Number_ _,_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log-odds of sampling 1

`arg_constraints`_= {'logits': Real(), 'probs': Interval(lower_bound=0.0,
upper_bound=1.0)}_

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/relaxed_bernoulli.html#RelaxedBernoulli.expand)

    

`has_rsample`_= True_

    

_property_`logits`

    

_property_`probs`

    

`support`_= Interval(lower_bound=0.0, upper_bound=1.0)_

    

_property_`temperature`

    

##  LogitRelaxedBernoulli 

_class_`torch.distributions.relaxed_bernoulli.``LogitRelaxedBernoulli`(
_temperature_ , _probs=None_ , _logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/relaxed_bernoulli.html#LogitRelaxedBernoulli)

    

Bases: `torch.distributions.distribution.Distribution`

创建LogitRelaxedBernoulli分布由参数 `probs`或 `logits`（但不是两者）
，这是一个RelaxedBernoulli分布的分对数。

样品是在（0，1）的值的logits。见[1]的更多细节。

Parameters

    

  * **temperature** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – relaxation temperature

  * **probs** ( _Number_ _,_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the probability of sampling 1

  * **logits** ( _Number_ _,_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log-odds of sampling 1

[1]的具体分布：离散随机变量的连续松弛（麦迪逊等人，2017）

[2]范畴重新参数化与冈贝尔-使用SoftMax（Jang等，2017）

`arg_constraints`_= {'logits': Real(), 'probs': Interval(lower_bound=0.0,
upper_bound=1.0)}_

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/relaxed_bernoulli.html#LogitRelaxedBernoulli.expand)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/relaxed_bernoulli.html#LogitRelaxedBernoulli.log_prob)

    

`logits`[[source]](_modules/torch/distributions/relaxed_bernoulli.html#LogitRelaxedBernoulli.logits)

    

_property_`param_shape`

    

`probs`[[source]](_modules/torch/distributions/relaxed_bernoulli.html#LogitRelaxedBernoulli.probs)

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/relaxed_bernoulli.html#LogitRelaxedBernoulli.rsample)

    

`support`_= Real()_

    

##  RelaxedOneHotCategorical 

_class_`torch.distributions.relaxed_categorical.``RelaxedOneHotCategorical`(
_temperature_ , _probs=None_ , _logits=None_ , _validate_args=None_
)[[source]](_modules/torch/distributions/relaxed_categorical.html#RelaxedOneHotCategorical)

    

Bases: `torch.distributions.transformed_distribution.TransformedDistribution`

创建RelaxedOneHotCategorical分布通过参数化 `温度 `，并且或者 `probs`或 `logits`。这是`
OneHotCategorical`分布宽松的版，所以它的样品都在单一，且reparametrizable。

Example:

    
    
    >>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
                                     torch.tensor([0.1, 0.2, 0.3, 0.4]))
    >>> m.sample()
    tensor([ 0.1294,  0.2324,  0.3859,  0.2523])
    

Parameters

    

  * **temperature** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – relaxation temperature

  * **probs** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities

  * **logits** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 每个事件的对数概率。

`arg_constraints`_= {'logits': Real(), 'probs': Simplex()}_

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/relaxed_categorical.html#RelaxedOneHotCategorical.expand)

    

`has_rsample`_= True_

    

_property_`logits`

    

_property_`probs`

    

`support`_= Simplex()_

    

_property_`temperature`

    

## 学生

_class_`torch.distributions.studentT.``StudentT`( _df_ , _loc=0.0_ ,
_scale=1.0_ , _validate_args=None_
)[[source]](_modules/torch/distributions/studentT.html#StudentT)

    

Bases: `torch.distributions.distribution.Distribution`

创建一个学生T分布的自由度参数`DF`，意思是`LOC`和规模`量表 `。

Example:

    
    
    >>> m = StudentT(torch.tensor([2.0]))
    >>> m.sample()  # Student's t-distributed with degrees of freedom=2
    tensor([ 0.1046])
    

Parameters

    

  * **DF** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 自由度

  * **loc** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _or_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution

  * **scale** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _or_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – scale of the distribution

`arg_constraints`_= {'df': GreaterThan(lower_bound=0.0), 'loc': Real(),
'scale': GreaterThan(lower_bound=0.0)}_

    

`entropy`()[[source]](_modules/torch/distributions/studentT.html#StudentT.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/studentT.html#StudentT.expand)

    

`has_rsample`_= True_

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/studentT.html#StudentT.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/studentT.html#StudentT.rsample)

    

`support`_= Real()_

    

_property_`variance`

    

##  TransformedDistribution 

_class_`torch.distributions.transformed_distribution.``TransformedDistribution`(
_base_distribution_ , _transforms_ , _validate_args=None_
)[[source]](_modules/torch/distributions/transformed_distribution.html#TransformedDistribution)

    

Bases: `torch.distributions.distribution.Distribution`

分发类，它适用变换的序列的碱分布的扩展。令f是施加变换的组合物：

    
    
    X ~ BaseDistribution
    Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
    log p(Y) = log p(X) + log |det (dX/dY)|
    

请注意，`.event_shape`的 `TransformedDistribution`
是其碱分布及其变换的最大形状，因为变换可以介绍事件之间的相关性。

为 `的使用的示例TransformedDistribution`将是：

    
    
    # Building a Logistic Distribution
    # X ~ Uniform(0, 1)
    # f = a + b * logit(X)
    # Y ~ f(X) ~ Logistic(a, b)
    base_distribution = Uniform(0, 1)
    transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
    logistic = TransformedDistribution(base_distribution, transforms)
    

对于更多的例子，请看的 `冈贝尔 `所述实施方式中， `HalfCauchy``HalfNormal`， `对数正态分布 `， `
帕累托 `， `威布尔 `， `RelaxedBernoulli`和 `RelaxedOneHotCategorical`

`arg_constraints`_= {}_

    

`cdf`( _value_
)[[source]](_modules/torch/distributions/transformed_distribution.html#TransformedDistribution.cdf)

    

通过反转变换（S）和计算基础分布的分数计算的累积分布函数。

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/transformed_distribution.html#TransformedDistribution.expand)

    

_property_`has_rsample`

    

`icdf`( _value_
)[[source]](_modules/torch/distributions/transformed_distribution.html#TransformedDistribution.icdf)

    

计算使用变换（S）的倒数累积分布函数以及计算基分布的分数。

`log_prob`( _value_
)[[source]](_modules/torch/distributions/transformed_distribution.html#TransformedDistribution.log_prob)

    

分数通过反转变换（S）和使用所述碱分布的得分得分和日志腹肌DET雅可比样品。

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/transformed_distribution.html#TransformedDistribution.rsample)

    

生成sample_shape形重新参数化样本或sample_shape形批量重新参数化的样本，如果分布参数是成批的。从基地分配样品的第一和适用变换（）为列表中的每个变换。

`sample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/transformed_distribution.html#TransformedDistribution.sample)

    

生成样本的sample_shape形样品或sample_shape形批次如果分布参数是成批的。从基地分配样品的第一和适用变换（）为列表中的每个变换。

_property_`support`

    

## 统一

_class_`torch.distributions.uniform.``Uniform`( _low_ , _high_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/uniform.html#Uniform)

    

Bases: `torch.distributions.distribution.Distribution`

生成均匀地从半开区间`分布的随机样品[低， 高） `。

Example:

    
    
    >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
    >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
    tensor([ 2.3418])
    

Parameters

    

  * **低** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 较低范围（含）。

  * **高** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 上限范围（不包括）。

`arg_constraints`_= {'high': Dependent(), 'low': Dependent()}_

    

`cdf`( _value_
)[[source]](_modules/torch/distributions/uniform.html#Uniform.cdf)

    

`entropy`()[[source]](_modules/torch/distributions/uniform.html#Uniform.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/uniform.html#Uniform.expand)

    

`has_rsample`_= True_

    

`icdf`( _value_
)[[source]](_modules/torch/distributions/uniform.html#Uniform.icdf)

    

`log_prob`( _value_
)[[source]](_modules/torch/distributions/uniform.html#Uniform.log_prob)

    

_property_`mean`

    

`rsample`( _sample_shape=torch.Size([])_
)[[source]](_modules/torch/distributions/uniform.html#Uniform.rsample)

    

_property_`stddev`

    

_property_`support`

    

_property_`variance`

    

## 威布尔

_class_`torch.distributions.weibull.``Weibull`( _scale_ , _concentration_ ,
_validate_args=None_
)[[source]](_modules/torch/distributions/weibull.html#Weibull)

    

Bases: `torch.distributions.transformed_distribution.TransformedDistribution`

从两参数Weibull分布样本。

Example

    
    
    >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
    >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
    tensor([ 0.4784])
    

Parameters

    

  * **规模** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分布（拉姆达）的尺度参数。

  * **浓度** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分配的浓度参数（k /形状）。

`arg_constraints`_= {'concentration': GreaterThan(lower_bound=0.0), 'scale':
GreaterThan(lower_bound=0.0)}_

    

`entropy`()[[source]](_modules/torch/distributions/weibull.html#Weibull.entropy)

    

`expand`( _batch_shape_ , __instance=None_
)[[source]](_modules/torch/distributions/weibull.html#Weibull.expand)

    

_property_`mean`

    

`support`_= GreaterThan(lower_bound=0.0)_

    

_property_`variance`

    

##  KL散度

`torch.distributions.kl.``kl_divergence`( _p_ , _q_
)[[source]](_modules/torch/distributions/kl.html#kl_divergence)

    

计算相对熵 K  L  （ P  ∥ q  ） KL（p \ | q）的 K  L  （ p  ∥ q  ） [HTG47两个分布之间。

KL(p∥q)=∫p(x)log⁡p(x)q(x) dxKL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)}
\,dxKL(p∥q)=∫p(x)logq(x)p(x)​dx

Parameters

    

  * **P** （ _发行_ ） - A `发行 `对象。

  * **Q** （ _发行_ ） - A `发行 `对象。

Returns

    

一批形状 batch_shape 的KL分歧的。

Return type

    

[张量](tensors.html#torch.Tensor "torch.Tensor")

Raises

    

[ **NotImplementedError**
](https://docs.python.org/3/library/exceptions.html#NotImplementedError "\(in
Python v3.7\)") \- 如果分布类型还没有被通过 `注册 register_kl（） `。

`torch.distributions.kl.``register_kl`( _type_p_ , _type_q_
)[[source]](_modules/torch/distributions/kl.html#register_kl)

    

装饰器注册到成对函数`kl_divergence（） `。用法：

    
    
    @register_kl(Normal, Normal)
    def kl_normal_normal(p, q):
        # insert implementation here
    

查找返回由子类下令最具体的（类型，类型）相匹配。如果匹配是不明确的，一个 RuntimeWarning 升高。例如为了解决模棱两可的情况：

    
    
    @register_kl(BaseP, DerivedQ)
    def kl_version1(p, q): ...
    @register_kl(DerivedP, BaseQ)
    def kl_version2(p, q): ...
    

要注册第三最特定的实现，例如：

    
    
    register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.
    

Parameters

    

  * **type_p** （[ _输入_ ](https://docs.python.org/3/library/functions.html#type "\(in Python v3.7\)")） - 的`发行 `的子类。

  * **type_q** （[ _输入_ ](https://docs.python.org/3/library/functions.html#type "\(in Python v3.7\)")） - 的`发行 `的子类。

## 变换

_class_`torch.distributions.transforms.``Transform`( _cache_size=0_
)[[source]](_modules/torch/distributions/transforms.html#Transform)

    

抽象类与可计算日志DET雅可比可翻转变换。它们主要在`torch.distributions.TransformedDistribution`使用。

缓存是tranforms其逆要么是昂贵或数值不稳定有用。请注意，你必须要好好memoized值占用，因为autograd图可被逆转。例如，虽然有或无缓存了以下工作：

    
    
    y = t(x)
    t.log_abs_det_jacobian(x, y).backward()  # x will receive gradients.
    

然而，由于依赖逆转缓存时，以下将错误：

    
    
    y = t(x)
    z = t.inv(y)
    grad(z.sum(), [y])  # error because z is x
    

派生类应该实现的一个或两个的`_call（） `或`_inverse（） `。该设定派生类双射=真还应当执行 `
log_abs_det_jacobian（） `。

Parameters

    

**CACHE_SIZE** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int
"\(in Python v3.7\)")） - 高速缓存的大小。如果是零，没有缓存完成。如果为一，最新的单值缓存。只有0和1的支持。

Variables

    

  * **〜Transform.domain** （ `约束 `） - 表示有效输入此变换的约束。

  * **〜Transform.codomain** （ `约束 `） - 表示有效输出给此变换哪些约束被输入到逆变换。

  * **〜Transform.bijective** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 是否此变换是双射。变换`T`是双射当且仅当`T.INV（T（X）） ==  × `和`T（T.INV（Y）） ==  Y`为每一个`×`在陪域的域和`Y`。未双射变换应至少保持较弱伪逆特性`T（T.INV（T（X）） ==  T（X） `和`T.INV（吨（T.INV（Y））） ==  T.INV（Y） `。

  * **〜Transform.sign** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 对于双射变换单变量，这应该是+1或-1取决于是否变换是单调递增或递减。

  * **〜Transform.event_dim** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 这是在变换`event_shape`相关一起维数。这应该是0为逐点变换，1为共同作用于载体，2变换的变换，关于矩阵联合行动，等等。

_property_`inv`

    

返回逆 `变换 `的此变换。这应该满足`t.inv.inv  是 T`。

_property_`sign`

    

返回雅可比的行列式的符号，如果适用。一般而言，这不仅使为双射变换感。

`log_abs_det_jacobian`( _x_ , _y_
)[[source]](_modules/torch/distributions/transforms.html#Transform.log_abs_det_jacobian)

    

计算日志DET雅可比登录| DY / DX | 给定的输入和输出。

_class_`torch.distributions.transforms.``ComposeTransform`( _parts_
)[[source]](_modules/torch/distributions/transforms.html#ComposeTransform)

    

构成在一个链中的多个变换。所组成的变换是负责缓存。

Parameters

    

**份** （变换的 `列表 `） - 变换的列表组成。

_class_`torch.distributions.transforms.``ExpTransform`( _cache_size=0_
)[[source]](_modules/torch/distributions/transforms.html#ExpTransform)

    

通过变换映射 Y  =  EXP  ⁡ （ × ） Y = \ EXP（X） Y  =  EXP  （ × ） 。

_class_`torch.distributions.transforms.``PowerTransform`( _exponent_ ,
_cache_size=0_
)[[source]](_modules/torch/distributions/transforms.html#PowerTransform)

    

通过映射 Y  =  × 指数变换 Y = X ^ {\文本{指数}}  Y  =  × 指数 。

_class_`torch.distributions.transforms.``SigmoidTransform`( _cache_size=0_
)[[source]](_modules/torch/distributions/transforms.html#SigmoidTransform)

    

通过变换映射 Y  =  1  1  \+  实验值 ⁡ （ \-  × ） Y = \压裂{1} {1 + \ EXP（-x）}  Y  =  1  \+
EXP  （ \-  × ） 1  [ H T G102]  和 × =  分对数 （ Y  ） ×= \文本{分对数}（Y） × =  分对数 （ Y
） 。

_class_`torch.distributions.transforms.``AbsTransform`( _cache_size=0_
)[[source]](_modules/torch/distributions/transforms.html#AbsTransform)

    

通过变换映射 Y  =  |  × |  Y = | X |  Y  =  |  × |  。

_class_`torch.distributions.transforms.``AffineTransform`( _loc_ , _scale_ ,
_event_dim=0_ , _cache_size=0_
)[[source]](_modules/torch/distributions/transforms.html#AffineTransform)

    

通过逐点仿射映射 变换Y  =  LOC  \+  规模 × × Y = \文本{LOC} + \文本{规模} \乘以x  Y  =  LOC  \+
规模 × × 。

Parameters

    

  * **LOC** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 位置的参数。

  * **规模** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 缩放参数。

  * **event_dim** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的 event_shape 可选大小。这应该是零为单变量随机变量，1用于在矢量分布，2超过矩阵等分布

_class_`torch.distributions.transforms.``SoftmaxTransform`( _cache_size=0_
)[[source]](_modules/torch/distributions/transforms.html#SoftmaxTransform)

    

通过从不受约束的空间变换至单面 Y  =  EXP  ⁡ （ × ） Y = \ EXP（X） Y  =  EXP  （ × ） 然后正火。

这不是双射的，不能用于HMC。然而，这种作用主要是坐标明智（除了最后标准化），并且因此适合于坐标明智的优化算法。

_class_`torch.distributions.transforms.``StickBreakingTransform`(
_cache_size=0_
)[[source]](_modules/torch/distributions/transforms.html#StickBreakingTransform)

    

通过棒折断处理从不受约束的空间转换到一个额外维的单纯。

此变换产生作为迭代乙状结肠在狄利克雷分布的棒破结构变换：第一分对数是通过乙状结肠变换到第一概率和其他一切的概率，然后处理递归。

这是双射和适合于HMC使用;但是它混合在一起的坐标，是优化不太合适。

_class_`torch.distributions.transforms.``LowerCholeskyTransform`(
_cache_size=0_
)[[source]](_modules/torch/distributions/transforms.html#LowerCholeskyTransform)

    

从约束矩阵以下三角矩阵非负对角线项变换。

这是在他们的Cholesky分解方面参数化正定矩阵有用。

_class_`torch.distributions.transforms.``CatTransform`( _tseq_ , _dim=0_ ,
_lengths=None_
)[[source]](_modules/torch/distributions/transforms.html#CatTransform)

    

变换应用于变换的序列函子 TSEQ 逐个分量的每个子矩阵在暗淡，长度的长度[暗淡] ，与[HTG6兼容的方式] `torch.cat（） `。

Example::

    

X0 = torch.cat（[torch.range（1，10），torch.range（1，10）]，暗淡= 0）X =
torch.cat（[X0，X0]，暗淡= 0）T0 = CatTransform（
[ExpTransform（），identity_transform]，暗淡= 0，长度= [10,10]）T =
CatTransform（[T0，T0]，暗淡= 0，长度= [20,20]）Y = T（X）

_class_`torch.distributions.transforms.``StackTransform`( _tseq_ , _dim=0_
)[[source]](_modules/torch/distributions/transforms.html#StackTransform)

    

变换应用于变换的序列函子 TSEQ 逐个分量的每个子矩阵在暗淡的方式与[ `torch.stack（兼容） `
](torch.html#torch.stack "torch.stack")。

Example::

    

X = torch.stack（[torch.range（1，10），torch.range（1，10）]，暗淡= 1）T =
StackTransform（[ExpTransform（），identity_transform]，暗淡= 1）Y = T（ X）

## 约束

下面的约束来实现：

  * `constraints.boolean`

  * `constraints.cat`

  * `constraints.dependent`

  * `constraints.greater_than（LOWER_BOUND） `

  * `constraints.integer_interval（LOWER_BOUND， UPPER_BOUND） `

  * `constraints.interval（LOWER_BOUND， UPPER_BOUND） `

  * `constraints.lower_cholesky`

  * `constraints.lower_triangular`

  * `constraints.nonnegative_integer`

  * `constraints.positive`

  * `constraints.positive_definite`

  * `constraints.positive_integer`

  * `constraints.real`

  * `constraints.real_vector`

  * `constraints.simplex`

  * `constraints.stack`

  * `constraints.unit_interval`

_class_`torch.distributions.constraints.``Constraint`[[source]](_modules/torch/distributions/constraints.html#Constraint)

    

抽象基类的约束。

约束对象表示在其上可变是有效的，例如一个区域在其内的变量可以被优化。

`check`( _value_
)[[source]](_modules/torch/distributions/constraints.html#Constraint.check)

    

返回的字节张量sample_shape + batch_shape 指示是否在值满足每一个事件此约束。

`torch.distributions.constraints.``dependent_property`

    

的别名`torch.distributions.constraints._DependentProperty`

`torch.distributions.constraints.``integer_interval`

    

的`别名torch.distributions.constraints._IntegerInterval`

`torch.distributions.constraints.``greater_than`

    

的别名`torch.distributions.constraints._GreaterThan`

`torch.distributions.constraints.``greater_than_eq`

    

的别名`torch.distributions.constraints._GreaterThanEq`

`torch.distributions.constraints.``less_than`

    

的别名`torch.distributions.constraints._LessThan`

`torch.distributions.constraints.``interval`

    

的`别名torch.distributions.constraints._Interval`

`torch.distributions.constraints.``half_open_interval`

    

的`别名torch.distributions.constraints._HalfOpenInterval`

`torch.distributions.constraints.``cat`

    

的别名`torch.distributions.constraints._Cat`

`torch.distributions.constraints.``stack`

    

的别名`torch.distributions.constraints._Stack`

## 约束注册表

PyTorch提供了两个全球 `ConstraintRegistry`对象链接 `约束 `对象 `变换 `
对象。这些对象都输入约束和返回变换，但它们对双射不同的担保。

  1. `biject_to（约束） `查找一个双射 `变换 `从`constraints.real`为给定的`约束 `。返回变换保证具有`.bijective  =  真 `，并执行`.log_abs_det_jacobian（） `。

  2. `transform_to（约束） `查找未一定双射 `变换 `从`约束。真正的 `为给定的`约束 `。返回的变换不能保证实现`.log_abs_det_jacobian（） `。

的`transform_to（） `注册表是上的概率分布的约束条件下参数，这是由每个分布的`.arg_constraints指示执行无约束优化有用
`字典。这些变换通常，为了避免旋转overparameterize的空间;因此，它们更适合于坐标明智优化算法像亚当：

    
    
    loc = torch.zeros(100, requires_grad=True)
    unconstrained = torch.zeros(100, requires_grad=True)
    scale = transform_to(Normal.arg_constraints['scale'])(unconstrained)
    loss = -Normal(loc, scale).log_prob(data).sum()
    

的`biject_to（） `注册表是哈密顿蒙特卡洛，有用的，其中从具有约束`。支持 `在一个传播的概率分布的样本不受约束的空间，和算法通常是旋转不变：

    
    
    dist = Exponential(rate)
    unconstrained = torch.zeros(100, requires_grad=True)
    sample = biject_to(dist.support)(unconstrained)
    potential_energy = -dist.log_prob(sample).sum()
    

Note

一个例子，其中`transform_to`和`biject_to`不同的是`constraints.simplex`：`
transform_to（constraints.simplex） `返回 `SoftmaxTransform`
，简单地exponentiates和归一化其输入;这是一个价格便宜，主要是协调明智的操作适合于像SVI算法。相比之下，`
biject_to（constraints.simplex） `返回 `StickBreakingTransform`
为bijects其输入降低到一个-fewer维空间;这样更昂贵更少数值稳定变换，但需要用于像HMC算法。

的`biject_to`和`transform_to`目的可以通过用户定义的约束扩展和变换使用他们`.register（ ）
`方法既可以作为单上的约束的函数：

    
    
    transform_to.register(my_constraint, my_transform)
    

或作为参数约束的装饰：

    
    
    @transform_to.register(MyConstraintClass)
    def my_factory(constraint):
        assert isinstance(constraint, MyConstraintClass)
        return MyTransform(constraint.param1, constraint.param2)
    

您可以通过创建一个新的 `ConstraintRegistry`对象创建自己的注册表。

_class_`torch.distributions.constraint_registry.``ConstraintRegistry`[[source]](_modules/torch/distributions/constraint_registry.html#ConstraintRegistry)

    

注册表来约束链接转换。

`register`( _constraint_ , _factory=None_
)[[source]](_modules/torch/distributions/constraint_registry.html#ConstraintRegistry.register)

    

注册在此注册表一个 `约束 `子类。用法：

    
    
    @my_registry.register(MyConstraintClass)
    def construct_transform(constraint):
        assert isinstance(constraint, MyConstraint)
        return MyTransform(constraint.arg_constraints)
    

Parameters

    

  * **约束** （ `约束 `的子类） - 的 `甲亚类约束 `，或所需的类的单一对象。

  * **工厂** （ _可调用_ ） - ，其输入约束对象，并返回可调用一个 `变换 `对象。

[Next ![](_static/images/chevron-right-orange.svg)](hub.html "torch.hub")
[![](_static/images/chevron-right-orange.svg) Previous](distributed.html
"Distributed communication package - torch.distributed")

* * *

©版权所有2019年，Torch 贡献者。
