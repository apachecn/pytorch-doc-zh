

# 概率分布 - torch.distributions

`distributions` 包含可参数化的概率分布和采样函数。这允许构造用于优化的随机计算图和随机梯度估计器。 这个包一般遵循 [TensorFlow Distributions](https://arxiv.org/abs/1711.10604) 包的设计.

通常，不可能直接通过随机样本反向传播。 但是，有两种主要方法可创建可以反向传播的代理函数。 即得分函数估计器/似然比估计器/REINFORCE和pathwise derivative估计器。 REINFORCE通常被视为强化学习中策略梯度方法的基础，并且pathwise derivative估计器常见于变分自动编码器中的重新参数化技巧. 得分函数仅需要样本的值 ![](img/cb804637f7fdaaf91569cfe4f047b418.jpg), pathwise derivative 需要导数 ![](img/385dbaaac9dd8aad33acc31ac64d2f27.jpg). 接下来的部分将在一个强化学习示例中讨论这两个问题.  有关详细信息，请参阅 [Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254) .

## 得分函数

当概率密度函数相对于其参数可微分时，我们只需要`sample()`和`log_prob()`来实现REINFORCE:

![](img/b50e881c13615b1d9aa00ad0c9cdfa99.jpg)

![](img/51b8359f970d2bfe2ad4cdc3ac1aed3c.jpg) 是参数, ![](img/82005cc2e0087e2a52c7e43df4a19a00.jpg) 是学习速率, ![](img/f9f040e861365a0560b2552b4e4e17da.jpg) 是奖励 并且 ![](img/2e84bb32ea0808870a16b888aeaf8d0d.jpg) 是在状态 ![](img/0492c0bfd615cb5e61c847ece512ff51.jpg) 以及给定策略 ![](img/5f3ddae3395c04f9346a3ac1d327ae2a.jpg)执行动作 ![](img/070b1af5eca3a5c5d72884b536090f17.jpg) 的概率.

在实践中，我们将从网络输出中采样一个动作，将这个动作应用于一个环境中，然后使用`log_prob`构造一个等效的损失函数。请注意，我们使用负数是因为优化器使用梯度下降，而上面的规则假设梯度上升。有了确定的策略，REINFORCE的实现代码如下:

```py
probs = policy_network(state)
# Note that this is equivalent to what used to be called multinomial
m = Categorical(probs)
action = m.sample()
next_state, reward = env.step(action)
loss = -m.log_prob(action) * reward
loss.backward()

```

## Pathwise derivative

实现这些随机/策略梯度的另一种方法是使用来自`rsample()`方法的重新参数化技巧，其中参数化随机变量可以通过无参数随机变量的参数确定性函数构造。 因此，重新参数化的样本变得可微分。 实现Pathwise derivative的代码如下:

```py
params = policy_network(state)
m = Normal(*params)
# Any distribution with .has_rsample == True could work based on the application
action = m.rsample()
next_state, reward = env.step(action)  # Assuming that reward is differentiable
loss = -reward
loss.backward()

```

## 分布

```py
class torch.distributions.distribution.Distribution(batch_shape=torch.Size([]), event_shape=torch.Size([]), validate_args=None)
```

基类: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.7)")

Distribution是概率分布的抽象基类.

```py
arg_constraints
```

从参数名称返回字典到 [`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint") 对象（应该满足这个分布的每个参数）.不是张量的arg不需要出现在这个字典中.

```py
batch_shape
```

返回批量参数的形状.

```py
cdf(value)
```

返回`value`处的累积密度/质量函数估计.

| 参数: | **value** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – |


```py
entropy()
```

返回分布的熵, 批量的形状为 batch_shape.

| 返回值: | Tensor 形状为 batch_shape. |


```py
enumerate_support(expand=True)
```

返回包含离散分布支持的所有值的张量. 结果将在维度0上枚举, 所以结果的形状将是 `(cardinality,) + batch_shape + event_shape` (对于单变量分布 `event_shape = ()`).

注意，这在lock-step中枚举了所有批处理张量`[[0, 0], [1, 1], …]`. 当 `expand=False`, 枚举沿着维度 0进行, 但是剩下的批处理维度是单维度, `[[0], [1], ..`.

遍历整个笛卡尔积的使用 `itertools.product(m.enumerate_support())`.

| 参数: | **expand** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 是否扩展对批处理dim的支持以匹配分布的 `batch_shape`. |

| 返回值: | 张量在维上0迭代. |


```py
event_shape
```

返回单个样本的形状 (非批量).

```py
expand(batch_shape, _instance=None)
```

返回一个新的分布实例(或填充派生类提供的现有实例)，其批处理维度扩展为 `batch_shape`.  这个方法调用 [`expand`](tensors.html#torch.Tensor.expand "torch.Tensor.expand") 在分布的参数上. 因此，这不会为扩展的分布实例分配新的内存.  此外，第一次创建实例时，这不会在中重复任何参数检查或参数广播在 `__init__.py`.

参数: 

*   **batch_shape** (_torch.Size_) – 所需的扩展尺寸.
*   **_instance** – 由需要重写`.expand`的子类提供的新实例.


| 返回值: | 批处理维度扩展为`batch_size`的新分布实例. |


```py
icdf(value)
```

 返回按`value`计算的反向累积密度/质量函数.

| 参数: | **value** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – |


```py
log_prob(value)
```

返回按`value`计算的概率密度/质量函数的对数.

| 参数: | **value** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – |


```py
mean
```

返回分布的平均值.

```py
perplexity()
```

返回分布的困惑度, 批量的关于 batch_shape.

| 返回值: | 形状为 batch_shape 的张量. |


```py
rsample(sample_shape=torch.Size([]))
```

如果分布的参数是批量的，则生成sample_shape形状的重新参数化样本或sample_shape形状的批量重新参数化样本.

```py
sample(sample_shape=torch.Size([]))
```

如果分布的参数是批量的，则生成sample_shape形状的样本或sample_shape形状的批量样本.

```py
sample_n(n)
```

如果分布参数是分批的，则生成n个样本或n批样本.

```py
stddev
```

返回分布的标准差.

```py
support
```

返回[`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint") 对象表示该分布的支持.

```py
variance
```

返回分布的方差.

## ExponentialFamily

```py
class torch.distributions.exp_family.ExponentialFamily(batch_shape=torch.Size([]), event_shape=torch.Size([]), validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

指数族是指数族概率分布的抽象基类，其概率质量/密度函数的形式定义如下

![](img/0c8313886f5c82dfae90e21b65152815.jpg)

![](img/51b8359f970d2bfe2ad4cdc3ac1aed3c.jpg) 表示自然参数, ![](img/e705d3772de12f4df3b0cd75af5110a1.jpg) 表示充分统计量, ![](img/f876c4d8353c747436006e70fb6c4f5d.jpg) 是给定族的对数归一化函数  ![](img/d3b6af2f20ffbc8480c6ee97c42958b2.jpg) 是carrier measure.

注意

该类是`Distribution`类与指数族分布之间的中介，主要用于检验`.entropy()`和解析KL散度方法的正确性。我们使用这个类来计算熵和KL散度使用AD框架和Bregman散度 (出自: Frank Nielsen and Richard Nock, Entropies and Cross-entropies of Exponential Families).

```py
entropy()
```

利用对数归一化器的Bregman散度计算熵的方法.

## Bernoulli

```py
class torch.distributions.bernoulli.Bernoulli(probs=None, logits=None, validate_args=None)
```

基类: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

创建参数化的伯努利分布，根据 [`probs`](#torch.distributions.bernoulli.Bernoulli.probs "torch.distributions.bernoulli.Bernoulli.probs") 或者 [`logits`](#torch.distributions.bernoulli.Bernoulli.logits "torch.distributions.bernoulli.Bernoulli.logits") (但不是同时都有).

样本是二值的 (0 或者 1). 取值 `1` 伴随概率 `p` ，或者 `0` 伴随概率 `1 - p`.

例子:

```py
>>> m = Bernoulli(torch.tensor([0.3]))
>>> m.sample()  # 30% chance 1; 70% chance 0
tensor([ 0.])

```

参数: 

*   **probs** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the probabilty of sampling `1`
*   **logits** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log-odds of sampling `1`



```py
arg_constraints = {'logits': Real(), 'probs': Interval(lower_bound=0.0, upper_bound=1.0)}
```

```py
entropy()
```

```py
enumerate_support(expand=True)
```

```py
expand(batch_shape, _instance=None)
```

```py
has_enumerate_support = True
```

```py
log_prob(value)
```

```py
logits
```

```py
mean
```

```py
param_shape
```

```py
probs
```

```py
sample(sample_shape=torch.Size([]))
```

```py
support = Boolean()
```

```py
variance
```

## Beta

```py
class torch.distributions.beta.Beta(concentration1, concentration0, validate_args=None)
```

基类: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Beta 分布，参数为 [`concentration1`](#torch.distributions.beta.Beta.concentration1 "torch.distributions.beta.Beta.concentration1") 和 [`concentration0`](#torch.distributions.beta.Beta.concentration0 "torch.distributions.beta.Beta.concentration0").

例子:

```py
>>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
>>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
tensor([ 0.1046])

```

参数: 

*   **concentration1** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 分布的第一个浓度参数（通常称为alpha）
*   **concentration0** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 分布的第二个浓度参数(通常称为beta)



```py
arg_constraints = {'concentration0': GreaterThan(lower_bound=0.0), 'concentration1': GreaterThan(lower_bound=0.0)}
```

```py
concentration0
```

```py
concentration1
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=())
```

```py
support = Interval(lower_bound=0.0, upper_bound=1.0)
```

```py
variance
```

## Binomial

```py
class torch.distributions.binomial.Binomial(total_count=1, probs=None, logits=None, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

创建一个Binomial 分布，参数为 `total_count` 和 [`probs`](#torch.distributions.binomial.Binomial.probs "torch.distributions.binomial.Binomial.probs") 或者 [`logits`](#torch.distributions.binomial.Binomial.logits "torch.distributions.binomial.Binomial.logits") (但不是同时都有使用). `total_count` 必须和 [`probs`] 之间可广播(#torch.distributions.binomial.Binomial.probs "torch.distributions.binomial.Binomial.probs")/[`logits`](#torch.distributions.binomial.Binomial.logits "torch.distributions.binomial.Binomial.logits").

例子:

```py
>>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
>>> x = m.sample()
tensor([   0.,   22.,   71.,  100.])

>>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
>>> x = m.sample()
tensor([[ 4.,  5.],
 [ 7.,  6.]])

```

参数: 

*   **total_count** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 伯努利试验次数
*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 事件概率
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 事件 log-odds



```py
arg_constraints = {'logits': Real(), 'probs': Interval(lower_bound=0.0, upper_bound=1.0), 'total_count': IntegerGreaterThan(lower_bound=0)}
```

```py
enumerate_support(expand=True)
```

```py
expand(batch_shape, _instance=None)
```

```py
has_enumerate_support = True
```

```py
log_prob(value)
```

```py
logits
```

```py
mean
```

```py
param_shape
```

```py
probs
```

```py
sample(sample_shape=torch.Size([]))
```

```py
support
```

```py
variance
```

## Categorical

```py
class torch.distributions.categorical.Categorical(probs=None, logits=None, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

创建一个 categorical 分布，参数为 [`probs`](#torch.distributions.categorical.Categorical.probs "torch.distributions.categorical.Categorical.probs") 或者 [`logits`](#torch.distributions.categorical.Categorical.logits "torch.distributions.categorical.Categorical.logits") (但不是同时都有).

注意

它等价于从 [`torch.multinomial()`](torch.html#torch.multinomial "torch.multinomial") 的采样.

样本是整数来自![](img/7c6904e60a8ff7044a079e10eaee1f57.jpg) `K` 是 `probs.size(-1)`.

如果 [`probs`](#torch.distributions.categorical.Categorical.probs "torch.distributions.categorical.Categorical.probs") 是 1D 的，长度为`K`, 每个元素是在该索引处对类进行抽样的相对概率.

如果 [`probs`](#torch.distributions.categorical.Categorical.probs "torch.distributions.categorical.Categorical.probs") 是 2D 的, 它被视为一组相对概率向量.

注意

[`probs`](#torch.distributions.categorical.Categorical.probs "torch.distributions.categorical.Categorical.probs")  必须是非负的、有限的并且具有非零和，并且它将被归一化为和为1.

请参阅: [`torch.multinomial()`](torch.html#torch.multinomial "torch.multinomial")

例子:

```py
>>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
>>> m.sample()  # equal probability of 0, 1, 2, 3
tensor(3)

```

参数: 

*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event log probabilities


```py
arg_constraints = {'logits': Real(), 'probs': Simplex()}
```

```py
entropy()
```

```py
enumerate_support(expand=True)
```

```py
expand(batch_shape, _instance=None)
```

```py
has_enumerate_support = True
```

```py
log_prob(value)
```

```py
logits
```

```py
mean
```

```py
param_shape
```

```py
probs
```

```py
sample(sample_shape=torch.Size([]))
```

```py
support
```

```py
variance
```

## Cauchy

```py
class torch.distributions.cauchy.Cauchy(loc, scale, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

样本来自柯西(洛伦兹)分布。均值为0的独立正态分布随机变量之比服从柯西分布。

例子:

```py
>>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
>>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
tensor([ 2.3214])

```

参数: 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 分布的模态或中值.
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – half width at half maximum.



```py
arg_constraints = {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}
```

```py
cdf(value)
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
icdf(value)
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
support = Real()
```

```py
variance
```

## Chi2

```py
class torch.distributions.chi2.Chi2(df, validate_args=None)
```

基类: [`torch.distributions.gamma.Gamma`](#torch.distributions.gamma.Gamma "torch.distributions.gamma.Gamma")

 创建由形状参数[`df`](#torch.distributions.chi2.Chi2.df "torch.distributions.chi2.Chi2.df")参数化的Chi2分布.  这完全等同于 `Gamma(alpha=0.5*df, beta=0.5)`

例子:

```py
>>> m = Chi2(torch.tensor([1.0]))
>>> m.sample()  # Chi2 distributed with shape df=1
tensor([ 0.1046])

```

| 参数: | **df** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 分布的形状参数 |


```py
arg_constraints = {'df': GreaterThan(lower_bound=0.0)}
```

```py
df
```

```py
expand(batch_shape, _instance=None)
```

## Dirichlet

```py
class torch.distributions.dirichlet.Dirichlet(concentration, validate_args=None)
```

基类: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

创建一个 Dirichlet 分布，参数为`concentration`.

例子:

```py
>>> m = Dirichlet(torch.tensor([0.5, 0.5]))
>>> m.sample()  # Dirichlet distributed with concentrarion concentration
tensor([ 0.1046,  0.8954])

```

| 参数: | **concentration** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) –  分布的浓度参数（通常称为alpha） |


```py
arg_constraints = {'concentration': GreaterThan(lower_bound=0.0)}
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=())
```

```py
support = Simplex()
```

```py
variance
```

## Exponential

```py
class torch.distributions.exponential.Exponential(rate, validate_args=None)
```

基类: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a Exponential distribution parameterized by `rate`.

例子:

```py
>>> m = Exponential(torch.tensor([1.0]))
>>> m.sample()  # Exponential distributed with rate=1
tensor([ 0.1046])

```

| 参数: | **rate** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – rate = 1 / scale of the distribution |


```py
arg_constraints = {'rate': GreaterThan(lower_bound=0.0)}
```

```py
cdf(value)
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
icdf(value)
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
stddev
```

```py
support = GreaterThan(lower_bound=0.0)
```

```py
variance
```

## FisherSnedecor

```py
class torch.distributions.fishersnedecor.FisherSnedecor(df1, df2, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Fisher-Snedecor distribution parameterized by `df1` and `df2`.

例子:

```py
>>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))
>>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
tensor([ 0.2453])

```

参数: 

*   **df1** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – degrees of freedom parameter 1
*   **df2** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – degrees of freedom parameter 2



```py
arg_constraints = {'df1': GreaterThan(lower_bound=0.0), 'df2': GreaterThan(lower_bound=0.0)}
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
support = GreaterThan(lower_bound=0.0)
```

```py
variance
```

## Gamma

```py
class torch.distributions.gamma.Gamma(concentration, rate, validate_args=None)
```

基类: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a Gamma distribution parameterized by shape `concentration` and `rate`.

例子:

```py
>>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
>>> m.sample()  # Gamma distributed with concentration=1 and rate=1
tensor([ 0.1046])

```

参数: 

*   **concentration** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – shape parameter of the distribution (often referred to as alpha)
*   **rate** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – rate = 1 / scale of the distribution (often referred to as beta)



```py
arg_constraints = {'concentration': GreaterThan(lower_bound=0.0), 'rate': GreaterThan(lower_bound=0.0)}
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
support = GreaterThan(lower_bound=0.0)
```

```py
variance
```

## Geometric

```py
class torch.distributions.geometric.Geometric(probs=None, logits=None, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Geometric distribution parameterized by [`probs`](#torch.distributions.geometric.Geometric.probs "torch.distributions.geometric.Geometric.probs"), where [`probs`](#torch.distributions.geometric.Geometric.probs "torch.distributions.geometric.Geometric.probs") is the probability of success of Bernoulli trials. It represents the probability that in ![](img/10396db36bab7b7242cfe94f04374444.jpg) Bernoulli trials, the first ![](img/a1c2f8d5b1226e67bdb44b12a6ddf18b.jpg) trials failed, before seeing a success.

Samples are non-negative integers [0, ![](img/06485c2c6e992cf346fdfe033a86a10d.jpg)).

例子:

```py
>>> m = Geometric(torch.tensor([0.3]))
>>> m.sample()  # underlying Bernoulli has 30% chance 1; 70% chance 0
tensor([ 2.])

```

参数: 

*   **probs** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the probabilty of sampling `1`. Must be in range (0, 1]
*   **logits** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log-odds of sampling `1`.



```py
arg_constraints = {'logits': Real(), 'probs': Interval(lower_bound=0.0, upper_bound=1.0)}
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
log_prob(value)
```

```py
logits
```

```py
mean
```

```py
probs
```

```py
sample(sample_shape=torch.Size([]))
```

```py
support = IntegerGreaterThan(lower_bound=0)
```

```py
variance
```

## Gumbel

```py
class torch.distributions.gumbel.Gumbel(loc, scale, validate_args=None)
```

基类: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Samples from a Gumbel Distribution.

Examples:

```py
>>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
>>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
tensor([ 1.0124])

```

参数: 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Location parameter of the distribution
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Scale parameter of the distribution



```py
arg_constraints = {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
mean
```

```py
stddev
```

```py
support = Real()
```

```py
variance
```

## HalfCauchy

```py
class torch.distributions.half_cauchy.HalfCauchy(scale, validate_args=None)
```

基类: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a half-normal distribution parameterized by `scale` where:

```py
X ~ Cauchy(0, scale)
Y = |X| ~ HalfCauchy(scale)

```

例子:

```py
>>> m = HalfCauchy(torch.tensor([1.0]))
>>> m.sample()  # half-cauchy distributed with scale=1
tensor([ 2.3214])

```

| 参数: | **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – scale of the full Cauchy distribution |


```py
arg_constraints = {'scale': GreaterThan(lower_bound=0.0)}
```

```py
cdf(value)
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
icdf(prob)
```

```py
log_prob(value)
```

```py
mean
```

```py
scale
```

```py
support = GreaterThan(lower_bound=0.0)
```

```py
variance
```

## HalfNormal

```py
class torch.distributions.half_normal.HalfNormal(scale, validate_args=None)
```

基类: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a half-normal distribution parameterized by `scale` where:

```py
X ~ Normal(0, scale)
Y = |X| ~ HalfNormal(scale)

```

例子:

```py
>>> m = HalfNormal(torch.tensor([1.0]))
>>> m.sample()  # half-normal distributed with scale=1
tensor([ 0.1046])

```

| 参数: | **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – scale of the full Normal distribution |


```py
arg_constraints = {'scale': GreaterThan(lower_bound=0.0)}
```

```py
cdf(value)
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
icdf(prob)
```

```py
log_prob(value)
```

```py
mean
```

```py
scale
```

```py
support = GreaterThan(lower_bound=0.0)
```

```py
variance
```

## Independent

```py
class torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Reinterprets some of the batch dims of a distribution as event dims.

This is mainly useful for changing the shape of the result of [`log_prob()`](#torch.distributions.independent.Independent.log_prob "torch.distributions.independent.Independent.log_prob"). For example to create a diagonal Normal distribution with the same shape as a Multivariate Normal distribution (so they are interchangeable), you can:

```py
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

```

参数: 

*   **base_distribution** ([_torch.distributions.distribution.Distribution_](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")) – a base distribution
*   **reinterpreted_batch_ndims** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the number of batch dims to reinterpret as event dims



```py
arg_constraints = {}
```

```py
entropy()
```

```py
enumerate_support(expand=True)
```

```py
expand(batch_shape, _instance=None)
```

```py
has_enumerate_support
```

```py
has_rsample
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
sample(sample_shape=torch.Size([]))
```

```py
support
```

```py
variance
```

## Laplace

```py
class torch.distributions.laplace.Laplace(loc, scale, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Laplace distribution parameterized by `loc` and :attr:’scale’.

例子:

```py
>>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
>>> m.sample()  # Laplace distributed with loc=0, scale=1
tensor([ 0.1046])

```

参数: 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – scale of the distribution



```py
arg_constraints = {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}
```

```py
cdf(value)
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
icdf(value)
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
stddev
```

```py
support = Real()
```

```py
variance
```

## LogNormal

```py
class torch.distributions.log_normal.LogNormal(loc, scale, validate_args=None)
```

基类: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a log-normal distribution parameterized by [`loc`](#torch.distributions.log_normal.LogNormal.loc "torch.distributions.log_normal.LogNormal.loc") and [`scale`](#torch.distributions.log_normal.LogNormal.scale "torch.distributions.log_normal.LogNormal.scale") where:

```py
X ~ Normal(loc, scale)
Y = exp(X) ~ LogNormal(loc, scale)

```

例子:

```py
>>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
>>> m.sample()  # log-normal distributed with mean=0 and stddev=1
tensor([ 0.1046])

```

参数: 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of log of distribution
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – standard deviation of log of the distribution



```py
arg_constraints = {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
loc
```

```py
mean
```

```py
scale
```

```py
support = GreaterThan(lower_bound=0.0)
```

```py
variance
```

## LowRankMultivariateNormal

```py
class torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(loc, cov_factor, cov_diag, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a multivariate normal distribution with covariance matrix having a low-rank form parameterized by `cov_factor` and `cov_diag`:

```py
covariance_matrix = cov_factor @ cov_factor.T + cov_diag

```

Example

```py
>>> m = LowRankMultivariateNormal(torch.zeros(2), torch.tensor([1, 0]), torch.tensor([1, 1]))
>>> m.sample()  # normally distributed with mean=`[0,0]`, cov_factor=`[1,0]`, cov_diag=`[1,1]`
tensor([-0.2102, -0.5429])

```

参数: 

*   **loc** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution with shape `batch_shape + event_shape`
*   **cov_factor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – factor part of low-rank form of covariance matrix with shape `batch_shape + event_shape + (rank,)`
*   **cov_diag** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – diagonal part of low-rank form of covariance matrix with shape `batch_shape + event_shape`



Note

The computation for determinant and inverse of covariance matrix is avoided when `cov_factor.shape[1] &lt;&lt; cov_factor.shape[0]` thanks to [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) and [matrix determinant lemma](https://en.wikipedia.org/wiki/Matrix_determinant_lemma). Thanks to these formulas, we just need to compute the determinant and inverse of the small size “capacitance” matrix:

```py
capacitance = I + cov_factor.T @ inv(cov_diag) @ cov_factor

```

```py
arg_constraints = {'cov_diag': GreaterThan(lower_bound=0.0), 'cov_factor': Real(), 'loc': Real()}
```

```py
covariance_matrix
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
log_prob(value)
```

```py
mean
```

```py
precision_matrix
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
scale_tril
```

```py
support = Real()
```

```py
variance
```

## Multinomial

```py
class torch.distributions.multinomial.Multinomial(total_count=1, probs=None, logits=None, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Multinomial distribution parameterized by `total_count` and either [`probs`](#torch.distributions.multinomial.Multinomial.probs "torch.distributions.multinomial.Multinomial.probs") or [`logits`](#torch.distributions.multinomial.Multinomial.logits "torch.distributions.multinomial.Multinomial.logits") (但不是同时都有). The innermost dimension of [`probs`](#torch.distributions.multinomial.Multinomial.probs "torch.distributions.multinomial.Multinomial.probs") indexes over categories. All other dimensions index over batches.

Note that `total_count` need not be specified if only [`log_prob()`](#torch.distributions.multinomial.Multinomial.log_prob "torch.distributions.multinomial.Multinomial.log_prob") is called (see example below)

Note

[`probs`](#torch.distributions.multinomial.Multinomial.probs "torch.distributions.multinomial.Multinomial.probs") must be non-negative, finite and have a non-zero sum, and it will be normalized to sum to 1.

*   [`sample()`](#torch.distributions.multinomial.Multinomial.sample "torch.distributions.multinomial.Multinomial.sample") requires a single shared `total_count` for all parameters and samples.
*   [`log_prob()`](#torch.distributions.multinomial.Multinomial.log_prob "torch.distributions.multinomial.Multinomial.log_prob") allows different `total_count` for each parameter and sample.

例子:

```py
>>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
>>> x = m.sample()  # equal probability of 0, 1, 2, 3
tensor([ 21.,  24.,  30.,  25.])

>>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)
tensor([-4.1338])

```

参数: 

*   **total_count** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – number of trials
*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event log probabilities



```py
arg_constraints = {'logits': Real(), 'probs': Simplex()}
```

```py
expand(batch_shape, _instance=None)
```

```py
log_prob(value)
```

```py
logits
```

```py
mean
```

```py
param_shape
```

```py
probs
```

```py
sample(sample_shape=torch.Size([]))
```

```py
support
```

```py
variance
```

## MultivariateNormal

```py
class torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a multivariate normal (also called Gaussian) distribution parameterized by a mean vector and a covariance matrix.

The multivariate normal distribution can be parameterized either in terms of a positive definite covariance matrix ![](img/ea86c11eaef9af2b4d699b88c2474ffd.jpg) or a positive definite precision matrix ![](img/1949bfcc1decf198a2ff50b6e25f4cf6.jpg) or a lower-triangular matrix ![](img/f4996f1b5056dd364eab16f975b808ff.jpg) with positive-valued diagonal entries, such that ![](img/6749b6afc75abfc8e0652ac8e5c0b8d8.jpg). This triangular matrix can be obtained via e.g. Cholesky decomposition of the covariance.

Example

```py
>>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
>>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
tensor([-0.2102, -0.5429])

```

参数: 

*   **loc** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution
*   **covariance_matrix** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – positive-definite covariance matrix
*   **precision_matrix** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – positive-definite precision matrix
*   **scale_tril** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – lower-triangular factor of covariance, with positive-valued diagonal



Note

Only one of [`covariance_matrix`](#torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix "torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix") or [`precision_matrix`](#torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix "torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix") or [`scale_tril`](#torch.distributions.multivariate_normal.MultivariateNormal.scale_tril "torch.distributions.multivariate_normal.MultivariateNormal.scale_tril") can be specified.

Using [`scale_tril`](#torch.distributions.multivariate_normal.MultivariateNormal.scale_tril "torch.distributions.multivariate_normal.MultivariateNormal.scale_tril") will be more efficient: all computations internally are based on [`scale_tril`](#torch.distributions.multivariate_normal.MultivariateNormal.scale_tril "torch.distributions.multivariate_normal.MultivariateNormal.scale_tril"). If [`covariance_matrix`](#torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix "torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix") or [`precision_matrix`](#torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix "torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix") is passed instead, it is only used to compute the corresponding lower triangular matrices using a Cholesky decomposition.

```py
arg_constraints = {'covariance_matrix': PositiveDefinite(), 'loc': RealVector(), 'precision_matrix': PositiveDefinite(), 'scale_tril': LowerCholesky()}
```

```py
covariance_matrix
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
log_prob(value)
```

```py
mean
```

```py
precision_matrix
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
scale_tril
```

```py
support = Real()
```

```py
variance
```

## NegativeBinomial

```py
class torch.distributions.negative_binomial.NegativeBinomial(total_count, probs=None, logits=None, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Negative Binomial distribution, i.e. distribution of the number of independent identical Bernoulli trials needed before `total_count` failures are achieved. The probability of success of each Bernoulli trial is [`probs`](#torch.distributions.negative_binomial.NegativeBinomial.probs "torch.distributions.negative_binomial.NegativeBinomial.probs").

参数: 

*   **total_count** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – non-negative number of negative Bernoulli trials to stop, although the distribution is still valid for real valued count
*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Event probabilities of success in the half open interval [0, 1)
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Event log-odds for probabilities of success



```py
arg_constraints = {'logits': Real(), 'probs': HalfOpenInterval(lower_bound=0.0, upper_bound=1.0), 'total_count': GreaterThanEq(lower_bound=0)}
```

```py
expand(batch_shape, _instance=None)
```

```py
log_prob(value)
```

```py
logits
```

```py
mean
```

```py
param_shape
```

```py
probs
```

```py
sample(sample_shape=torch.Size([]))
```

```py
support = IntegerGreaterThan(lower_bound=0)
```

```py
variance
```

## Normal

```py
class torch.distributions.normal.Normal(loc, scale, validate_args=None)
```

基类: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a normal (also called Gaussian) distribution parameterized by `loc` and `scale`.

例子:

```py
>>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
>>> m.sample()  # normally distributed with loc=0 and scale=1
tensor([ 0.1046])

```

参数: 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution (often referred to as mu)
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – standard deviation of the distribution (often referred to as sigma)



```py
arg_constraints = {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}
```

```py
cdf(value)
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
icdf(value)
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
sample(sample_shape=torch.Size([]))
```

```py
stddev
```

```py
support = Real()
```

```py
variance
```

## OneHotCategorical

```py
class torch.distributions.one_hot_categorical.OneHotCategorical(probs=None, logits=None, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a one-hot categorical distribution parameterized by [`probs`](#torch.distributions.one_hot_categorical.OneHotCategorical.probs "torch.distributions.one_hot_categorical.OneHotCategorical.probs") or [`logits`](#torch.distributions.one_hot_categorical.OneHotCategorical.logits "torch.distributions.one_hot_categorical.OneHotCategorical.logits").

Samples are one-hot coded vectors of size `probs.size(-1)`.

Note

[`probs`](#torch.distributions.one_hot_categorical.OneHotCategorical.probs "torch.distributions.one_hot_categorical.OneHotCategorical.probs") must be non-negative, finite and have a non-zero sum, and it will be normalized to sum to 1.

See also: `torch.distributions.Categorical()` for specifications of [`probs`](#torch.distributions.one_hot_categorical.OneHotCategorical.probs "torch.distributions.one_hot_categorical.OneHotCategorical.probs") and [`logits`](#torch.distributions.one_hot_categorical.OneHotCategorical.logits "torch.distributions.one_hot_categorical.OneHotCategorical.logits").

例子:

```py
>>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
>>> m.sample()  # equal probability of 0, 1, 2, 3
tensor([ 0.,  0.,  0.,  1.])

```

参数: 

*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event log probabilities



```py
arg_constraints = {'logits': Real(), 'probs': Simplex()}
```

```py
entropy()
```

```py
enumerate_support(expand=True)
```

```py
expand(batch_shape, _instance=None)
```

```py
has_enumerate_support = True
```

```py
log_prob(value)
```

```py
logits
```

```py
mean
```

```py
param_shape
```

```py
probs
```

```py
sample(sample_shape=torch.Size([]))
```

```py
support = Simplex()
```

```py
variance
```

## Pareto

```py
class torch.distributions.pareto.Pareto(scale, alpha, validate_args=None)
```

基类: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Samples from a Pareto Type 1 distribution.

例子:

```py
>>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
>>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
tensor([ 1.5623])

```

参数: 

*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Scale parameter of the distribution
*   **alpha** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Shape parameter of the distribution



```py
arg_constraints = {'alpha': GreaterThan(lower_bound=0.0), 'scale': GreaterThan(lower_bound=0.0)}
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
mean
```

```py
support
```

```py
variance
```

## Poisson

```py
class torch.distributions.poisson.Poisson(rate, validate_args=None)
```

基类: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a Poisson distribution parameterized by `rate`, the rate parameter.

Samples are nonnegative integers, with a pmf given by

![](img/32c47de57300c954795486fea3201bdc.jpg)

例子:

```py
>>> m = Poisson(torch.tensor([4]))
>>> m.sample()
tensor([ 3.])

```

| 参数: | **rate** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the rate parameter |


```py
arg_constraints = {'rate': GreaterThan(lower_bound=0.0)}
```

```py
expand(batch_shape, _instance=None)
```

```py
log_prob(value)
```

```py
mean
```

```py
sample(sample_shape=torch.Size([]))
```

```py
support = IntegerGreaterThan(lower_bound=0)
```

```py
variance
```

## RelaxedBernoulli

```py
class torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature, probs=None, logits=None, validate_args=None)
```

基类: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a RelaxedBernoulli distribution, parametrized by [`temperature`](#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature "torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature"), and either [`probs`](#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs "torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs") or [`logits`](#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits "torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits") (但不是同时都有). This is a relaxed version of the `Bernoulli` distribution, so the values are in (0, 1), and has reparametrizable samples.

例子:

```py
>>> m = RelaxedBernoulli(torch.tensor([2.2]),
 torch.tensor([0.1, 0.2, 0.3, 0.99]))
>>> m.sample()
tensor([ 0.2951,  0.3442,  0.8918,  0.9021])

```

参数: 

*   **temperature** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – relaxation temperature
*   **probs** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the probabilty of sampling `1`
*   **logits** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log-odds of sampling `1`



```py
arg_constraints = {'logits': Real(), 'probs': Interval(lower_bound=0.0, upper_bound=1.0)}
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
logits
```

```py
probs
```

```py
support = Interval(lower_bound=0.0, upper_bound=1.0)
```

```py
temperature
```

## RelaxedOneHotCategorical

```py
class torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(temperature, probs=None, logits=None, validate_args=None)
```

基类: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a RelaxedOneHotCategorical distribution parametrized by [`temperature`](#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature "torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature"), and either [`probs`](#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs "torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs") or [`logits`](#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits "torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits"). This is a relaxed version of the `OneHotCategorical` distribution, so its samples are on simplex, and are reparametrizable.

例子:

```py
>>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
 torch.tensor([0.1, 0.2, 0.3, 0.4]))
>>> m.sample()
tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

```

参数: 

*   **temperature** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – relaxation temperature
*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log probability of each event.



```py
arg_constraints = {'logits': Real(), 'probs': Simplex()}
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
logits
```

```py
probs
```

```py
support = Simplex()
```

```py
temperature
```

## StudentT

```py
class torch.distributions.studentT.StudentT(df, loc=0.0, scale=1.0, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Student’s t-distribution parameterized by degree of freedom `df`, mean `loc` and scale `scale`.

例子:

```py
>>> m = StudentT(torch.tensor([2.0]))
>>> m.sample()  # Student's t-distributed with degrees of freedom=2
tensor([ 0.1046])

```

参数: 

*   **df** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – degrees of freedom
*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – scale of the distribution



```py
arg_constraints = {'df': GreaterThan(lower_bound=0.0), 'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
support = Real()
```

```py
variance
```

## TransformedDistribution

```py
class torch.distributions.transformed_distribution.TransformedDistribution(base_distribution, transforms, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Extension of the Distribution class, which applies a sequence of Transforms to a base distribution. Let f be the composition of transforms applied:

```py
X ~ BaseDistribution
Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
log p(Y) = log p(X) + log |det (dX/dY)|

```

Note that the `.event_shape` of a [`TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution") is the maximum shape of its base distribution and its transforms, since transforms can introduce correlations among events.

An example for the usage of [`TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution") would be:

```py
# Building a Logistic Distribution
# X ~ Uniform(0, 1)
# f = a + b * logit(X)
# Y ~ f(X) ~ Logistic(a, b)
base_distribution = Uniform(0, 1)
transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
logistic = TransformedDistribution(base_distribution, transforms)

```

For more examples, please look at the implementations of [`Gumbel`](#torch.distributions.gumbel.Gumbel "torch.distributions.gumbel.Gumbel"), [`HalfCauchy`](#torch.distributions.half_cauchy.HalfCauchy "torch.distributions.half_cauchy.HalfCauchy"), [`HalfNormal`](#torch.distributions.half_normal.HalfNormal "torch.distributions.half_normal.HalfNormal"), [`LogNormal`](#torch.distributions.log_normal.LogNormal "torch.distributions.log_normal.LogNormal"), [`Pareto`](#torch.distributions.pareto.Pareto "torch.distributions.pareto.Pareto"), [`Weibull`](#torch.distributions.weibull.Weibull "torch.distributions.weibull.Weibull"), [`RelaxedBernoulli`](#torch.distributions.relaxed_bernoulli.RelaxedBernoulli "torch.distributions.relaxed_bernoulli.RelaxedBernoulli") and [`RelaxedOneHotCategorical`](#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical "torch.distributions.relaxed_categorical.RelaxedOneHotCategorical")

```py
arg_constraints = {}
```

```py
cdf(value)
```

Computes the cumulative distribution function by inverting the transform(s) and computing the score of the base distribution.

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample
```

```py
icdf(value)
```

Computes the inverse cumulative distribution function using transform(s) and computing the score of the base distribution.

```py
log_prob(value)
```

Scores the sample by inverting the transform(s) and computing the score using the score of the base distribution and the log abs det jacobian.

```py
rsample(sample_shape=torch.Size([]))
```

Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples if the distribution parameters are batched. Samples first from base distribution and applies `transform()` for every transform in the list.

```py
sample(sample_shape=torch.Size([]))
```

Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched. Samples first from base distribution and applies `transform()` for every transform in the list.

```py
support
```

## Uniform

```py
class torch.distributions.uniform.Uniform(low, high, validate_args=None)
```

基类: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Generates uniformly distributed random samples from the half-open interval `[low, high)`.

例子:

```py
>>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
>>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
tensor([ 2.3418])

```

参数: 

*   **low** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – lower range (inclusive).
*   **high** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – upper range (exclusive).



```py
arg_constraints = {'high': Dependent(), 'low': Dependent()}
```

```py
cdf(value)
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
has_rsample = True
```

```py
icdf(value)
```

```py
log_prob(value)
```

```py
mean
```

```py
rsample(sample_shape=torch.Size([]))
```

```py
stddev
```

```py
support
```

```py
variance
```

## Weibull

```py
class torch.distributions.weibull.Weibull(scale, concentration, validate_args=None)
```

基类: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Samples from a two-parameter Weibull distribution.

Example

```py
>>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
>>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
tensor([ 0.4784])

```

参数: 

*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Scale parameter of distribution (lambda).
*   **concentration** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Concentration parameter of distribution (k/shape).



```py
arg_constraints = {'concentration': GreaterThan(lower_bound=0.0), 'scale': GreaterThan(lower_bound=0.0)}
```

```py
entropy()
```

```py
expand(batch_shape, _instance=None)
```

```py
mean
```

```py
support = GreaterThan(lower_bound=0.0)
```

```py
variance
```

## `KL Divergence`

```py
torch.distributions.kl.kl_divergence(p, q)
```

Compute Kullback-Leibler divergence ![](img/739a8e4cd0597805c3e4daf35c0fc7c6.jpg) between two distributions.

![](img/ff8dcec3abe559720f8b0b464d2471b2.jpg)

参数: 

*   **p** ([_Distribution_](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")) – A `Distribution` object.
*   **q** ([_Distribution_](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")) – A `Distribution` object.


| 返回值: | A batch of KL divergences of shape `batch_shape`. |

| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |

| Raises: | [`NotImplementedError`](https://docs.python.org/3/library/exceptions.html#NotImplementedError "(in Python v3.7)") – If the distribution types have not been registered via [`register_kl()`](#torch.distributions.kl.register_kl "torch.distributions.kl.register_kl"). |


```py
torch.distributions.kl.register_kl(type_p, type_q)
```

Decorator to register a pairwise function with [`kl_divergence()`](#torch.distributions.kl.kl_divergence "torch.distributions.kl.kl_divergence"). Usage:

```py
@register_kl(Normal, Normal)
def kl_normal_normal(p, q):
    # insert implementation here

```

Lookup returns the most specific (type,type) match ordered by subclass. If the match is ambiguous, a `RuntimeWarning` is raised. For example to resolve the ambiguous situation:

```py
@register_kl(BaseP, DerivedQ)
def kl_version1(p, q): ...
@register_kl(DerivedP, BaseQ)
def kl_version2(p, q): ...

```

you should register a third most-specific implementation, e.g.:

```py
register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

```

参数: 

*   **type_p** ([_type_](https://docs.python.org/3/library/functions.html#type "(in Python v3.7)")) – A subclass of `Distribution`.
*   **type_q** ([_type_](https://docs.python.org/3/library/functions.html#type "(in Python v3.7)")) – A subclass of `Distribution`.



## `Transforms`

```py
class torch.distributions.transforms.Transform(cache_size=0)
```

Abstract class for invertable transformations with computable log det jacobians. They are primarily used in `torch.distributions.TransformedDistribution`.

Caching is useful for tranforms whose inverses are either expensive or numerically unstable. Note that care must be taken with memoized values since the autograd graph may be reversed. For example while the following works with or without caching:

```py
y = t(x)
t.log_abs_det_jacobian(x, y).backward()  # x will receive gradients.

```

However the following will error when caching due to dependency reversal:

```py
y = t(x)
z = t.inv(y)
grad(z.sum(), [y])  # error because z is x

```

Derived classes should implement one or both of `_call()` or `_inverse()`. Derived classes that set `bijective=True` should also implement [`log_abs_det_jacobian()`](#torch.distributions.transforms.Transform.log_abs_det_jacobian "torch.distributions.transforms.Transform.log_abs_det_jacobian").

| 参数: | **cache_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Size of cache. If zero, no caching is done. If one, the latest single value is cached. Only 0 and 1 are supported. |

| Variables: | 

*   **domain** ([`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint")) – The constraint representing valid inputs to this transform.
*   **codomain** ([`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint")) – The constraint representing valid outputs to this transform which are inputs to the inverse transform.
*   **bijective** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – Whether this transform is bijective. A transform `t` is bijective iff `t.inv(t(x)) == x` and `t(t.inv(y)) == y` for every `x` in the domain and `y` in the codomain. Transforms that are not bijective should at least maintain the weaker pseudoinverse properties `t(t.inv(t(x)) == t(x)` and `t.inv(t(t.inv(y))) == t.inv(y)`.
*   **sign** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – For bijective univariate transforms, this should be +1 or -1 depending on whether transform is monotone increasing or decreasing.
*   **event_dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Number of dimensions that are correlated together in the transform `event_shape`. This should be 0 for pointwise transforms, 1 for transforms that act jointly on vectors, 2 for transforms that act jointly on matrices, etc.



```py
inv
```

Returns the inverse [`Transform`](#torch.distributions.transforms.Transform "torch.distributions.transforms.Transform") of this transform. This should satisfy `t.inv.inv is t`.

```py
sign
```

Returns the sign of the determinant of the Jacobian, if applicable. In general this only makes sense for bijective transforms.

```py
log_abs_det_jacobian(x, y)
```

Computes the log det jacobian `log |dy/dx|` given input and output.

```py
class torch.distributions.transforms.ComposeTransform(parts)
```

Composes multiple transforms in a chain. The transforms being composed are responsible for caching.

| 参数: | **parts** (list of [`Transform`](#torch.distributions.transforms.Transform "torch.distributions.transforms.Transform")) – A list of transforms to compose. |


```py
class torch.distributions.transforms.ExpTransform(cache_size=0)
```

Transform via the mapping ![](img/ec8d939394f24908d017d86153e312ea.jpg).

```py
class torch.distributions.transforms.PowerTransform(exponent, cache_size=0)
```

Transform via the mapping ![](img/2062af7179e0c19c3599816de6768cee.jpg).

```py
class torch.distributions.transforms.SigmoidTransform(cache_size=0)
```

Transform via the mapping ![](img/749abef3418941161a1c6ff80d9eae76.jpg) and ![](img/6feb73eb74f2267e5caa87d9693362cb.jpg).

```py
class torch.distributions.transforms.AbsTransform(cache_size=0)
```

Transform via the mapping ![](img/dca0dc2e17c81b7ec261e70549de5507.jpg).

```py
class torch.distributions.transforms.AffineTransform(loc, scale, event_dim=0, cache_size=0)
```

Transform via the pointwise affine mapping ![](img/e1df459e7ff26d682fc956b62868f7c4.jpg).

参数: 

*   **loc** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Location parameter.
*   **scale** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Scale parameter.
*   **event_dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Optional size of `event_shape`. This should be zero for univariate random variables, 1 for distributions over vectors, 2 for distributions over matrices, etc.



```py
class torch.distributions.transforms.SoftmaxTransform(cache_size=0)
```

Transform from unconstrained space to the simplex via ![](img/ec8d939394f24908d017d86153e312ea.jpg) then normalizing.

This is not bijective and cannot be used for HMC. However this acts mostly coordinate-wise (except for the final normalization), and thus is appropriate for coordinate-wise optimization algorithms.

```py
class torch.distributions.transforms.StickBreakingTransform(cache_size=0)
```

Transform from unconstrained space to the simplex of one additional dimension via a stick-breaking process.

This transform arises as an iterated sigmoid transform in a stick-breaking construction of the `Dirichlet` distribution: the first logit is transformed via sigmoid to the first probability and the probability of everything else, and then the process recurses.

This is bijective and appropriate for use in HMC; however it mixes coordinates together and is less appropriate for optimization.

```py
class torch.distributions.transforms.LowerCholeskyTransform(cache_size=0)
```

Transform from unconstrained matrices to lower-triangular matrices with nonnegative diagonal entries.

This is useful for parameterizing positive definite matrices in terms of their Cholesky factorization.

## `Constraints`

The following constraints are implemented:

*   `constraints.boolean`
*   `constraints.dependent`
*   `constraints.greater_than(lower_bound)`
*   `constraints.integer_interval(lower_bound, upper_bound)`
*   `constraints.interval(lower_bound, upper_bound)`
*   `constraints.lower_cholesky`
*   `constraints.lower_triangular`
*   `constraints.nonnegative_integer`
*   `constraints.positive`
*   `constraints.positive_definite`
*   `constraints.positive_integer`
*   `constraints.real`
*   `constraints.real_vector`
*   `constraints.simplex`
*   `constraints.unit_interval`

```py
class torch.distributions.constraints.Constraint
```

Abstract base class for constraints.

A constraint object represents a region over which a variable is valid, e.g. within which a variable can be optimized.

```py
check(value)
```

Returns a byte tensor of `sample_shape + batch_shape` indicating whether each event in value satisfies this constraint.

```py
torch.distributions.constraints.dependent_property
```

alias of `torch.distributions.constraints._DependentProperty`

```py
torch.distributions.constraints.integer_interval
```

alias of `torch.distributions.constraints._IntegerInterval`

```py
torch.distributions.constraints.greater_than
```

alias of `torch.distributions.constraints._GreaterThan`

```py
torch.distributions.constraints.greater_than_eq
```

alias of `torch.distributions.constraints._GreaterThanEq`

```py
torch.distributions.constraints.less_than
```

alias of `torch.distributions.constraints._LessThan`

```py
torch.distributions.constraints.interval
```

alias of `torch.distributions.constraints._Interval`

```py
torch.distributions.constraints.half_open_interval
```

alias of `torch.distributions.constraints._HalfOpenInterval`

## `Constraint Registry`

PyTorch provides two global [`ConstraintRegistry`](#torch.distributions.constraint_registry.ConstraintRegistry "torch.distributions.constraint_registry.ConstraintRegistry") objects that link [`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint") objects to [`Transform`](#torch.distributions.transforms.Transform "torch.distributions.transforms.Transform") objects. These objects both input constraints and return transforms, but they have different guarantees on bijectivity.

1.  `biject_to(constraint)` looks up a bijective [`Transform`](#torch.distributions.transforms.Transform "torch.distributions.transforms.Transform") from `constraints.real` to the given `constraint`. The returned transform is guaranteed to have `.bijective = True` and should implement `.log_abs_det_jacobian()`.
2.  `transform_to(constraint)` looks up a not-necessarily bijective [`Transform`](#torch.distributions.transforms.Transform "torch.distributions.transforms.Transform") from `constraints.real` to the given `constraint`. The returned transform is not guaranteed to implement `.log_abs_det_jacobian()`.

The `transform_to()` registry is useful for performing unconstrained optimization on constrained parameters of probability distributions, which are indicated by each distribution’s `.arg_constraints` dict. These transforms often overparameterize a space in order to avoid rotation; they are thus more suitable for coordinate-wise optimization algorithms like Adam:

```py
loc = torch.zeros(100, requires_grad=True)
unconstrained = torch.zeros(100, requires_grad=True)
scale = transform_to(Normal.arg_constraints['scale'])(unconstrained)
loss = -Normal(loc, scale).log_prob(data).sum()

```

The `biject_to()` registry is useful for Hamiltonian Monte Carlo, where samples from a probability distribution with constrained `.support` are propagated in an unconstrained space, and algorithms are typically rotation invariant.:

```py
dist = Exponential(rate)
unconstrained = torch.zeros(100, requires_grad=True)
sample = biject_to(dist.support)(unconstrained)
potential_energy = -dist.log_prob(sample).sum()

```

Note

An example where `transform_to` and `biject_to` differ is `constraints.simplex`: `transform_to(constraints.simplex)` returns a [`SoftmaxTransform`](#torch.distributions.transforms.SoftmaxTransform "torch.distributions.transforms.SoftmaxTransform") that simply exponentiates and normalizes its inputs; this is a cheap and mostly coordinate-wise operation appropriate for algorithms like SVI. In contrast, `biject_to(constraints.simplex)` returns a [`StickBreakingTransform`](#torch.distributions.transforms.StickBreakingTransform "torch.distributions.transforms.StickBreakingTransform") that bijects its input down to a one-fewer-dimensional space; this a more expensive less numerically stable transform but is needed for algorithms like HMC.

The `biject_to` and `transform_to` objects can be extended by user-defined constraints and transforms using their `.register()` method either as a function on singleton constraints:

```py
transform_to.register(my_constraint, my_transform)

```

or as a decorator on parameterized constraints:

```py
@transform_to.register(MyConstraintClass)
def my_factory(constraint):
    assert isinstance(constraint, MyConstraintClass)
    return MyTransform(constraint.param1, constraint.param2)

```

You can create your own registry by creating a new [`ConstraintRegistry`](#torch.distributions.constraint_registry.ConstraintRegistry "torch.distributions.constraint_registry.ConstraintRegistry") object.

```py
class torch.distributions.constraint_registry.ConstraintRegistry
```

Registry to link constraints to transforms.

```py
register(constraint, factory=None)
```

Registers a [`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint") subclass in this registry. Usage:

```py
@my_registry.register(MyConstraintClass)
def construct_transform(constraint):
    assert isinstance(constraint, MyConstraint)
    return MyTransform(constraint.arg_constraints)

```

参数: 

*   **constraint** (subclass of [`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint")) – A subclass of [`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint"), or a singleton object of the desired class.
*   **factory** (_callable_) – A callable that inputs a constraint object and returns a [`Transform`](#torch.distributions.transforms.Transform "torch.distributions.transforms.Transform") object.



