

# Probability distributions - torch.distributions

The `distributions` package contains parameterizable probability distributions and sampling functions. This allows the construction of stochastic computation graphs and stochastic gradient estimators for optimization. This package generally follows the design of the [TensorFlow Distributions](https://arxiv.org/abs/1711.10604) package.

It is not possible to directly backpropagate through random samples. However, there are two main methods for creating surrogate functions that can be backpropagated through. These are the score function estimator/likelihood ratio estimator/REINFORCE and the pathwise derivative estimator. REINFORCE is commonly seen as the basis for policy gradient methods in reinforcement learning, and the pathwise derivative estimator is commonly seen in the reparameterization trick in variational autoencoders. Whilst the score function only requires the value of samples ![](img/cb804637f7fdaaf91569cfe4f047b418.jpg), the pathwise derivative requires the derivative ![](img/385dbaaac9dd8aad33acc31ac64d2f27.jpg). The next sections discuss these two in a reinforcement learning example. For more details see [Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254) .

## Score function

When the probability density function is differentiable with respect to its parameters, we only need `sample()` and `log_prob()` to implement REINFORCE:

![](img/b50e881c13615b1d9aa00ad0c9cdfa99.jpg)

where ![](img/51b8359f970d2bfe2ad4cdc3ac1aed3c.jpg) are the parameters, ![](img/82005cc2e0087e2a52c7e43df4a19a00.jpg) is the learning rate, ![](img/f9f040e861365a0560b2552b4e4e17da.jpg) is the reward and ![](img/2e84bb32ea0808870a16b888aeaf8d0d.jpg) is the probability of taking action ![](img/070b1af5eca3a5c5d72884b536090f17.jpg) in state ![](img/0492c0bfd615cb5e61c847ece512ff51.jpg) given policy ![](img/5f3ddae3395c04f9346a3ac1d327ae2a.jpg).

In practice we would sample an action from the output of a network, apply this action in an environment, and then use `log_prob` to construct an equivalent loss function. Note that we use a negative because optimizers use gradient descent, whilst the rule above assumes gradient ascent. With a categorical policy, the code for implementing REINFORCE would be as follows:

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

The other way to implement these stochastic/policy gradients would be to use the reparameterization trick from the `rsample()` method, where the parameterized random variable can be constructed via a parameterized deterministic function of a parameter-free random variable. The reparameterized sample therefore becomes differentiable. The code for implementing the pathwise derivative would be as follows:

```py
params = policy_network(state)
m = Normal(*params)
# Any distribution with .has_rsample == True could work based on the application
action = m.rsample()
next_state, reward = env.step(action)  # Assuming that reward is differentiable
loss = -reward
loss.backward()

```

## Distribution

```py
class torch.distributions.distribution.Distribution(batch_shape=torch.Size([]), event_shape=torch.Size([]), validate_args=None)
```

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.7)")

Distribution is the abstract base class for probability distributions.

```py
arg_constraints
```

Returns a dictionary from argument names to [`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint") objects that should be satisfied by each argument of this distribution. Args that are not tensors need not appear in this dict.

```py
batch_shape
```

Returns the shape over which parameters are batched.

```py
cdf(value)
```

Returns the cumulative density/mass function evaluated at &lt;cite&gt;value&lt;/cite&gt;.

| Parameters: | **value** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – |
| --- | --- |

```py
entropy()
```

Returns entropy of distribution, batched over batch_shape.

| Returns: | Tensor of shape batch_shape. |
| --- | --- |

```py
enumerate_support(expand=True)
```

Returns tensor containing all values supported by a discrete distribution. The result will enumerate over dimension 0, so the shape of the result will be &lt;cite&gt;(cardinality,) + batch_shape + event_shape&lt;/cite&gt; (where &lt;cite&gt;event_shape = ()&lt;/cite&gt; for univariate distributions).

Note that this enumerates over all batched tensors in lock-step &lt;cite&gt;[[0, 0], [1, 1], …]&lt;/cite&gt;. With &lt;cite&gt;expand=False&lt;/cite&gt;, enumeration happens along dim 0, but with the remaining batch dimensions being singleton dimensions, &lt;cite&gt;[[0], [1], ..&lt;/cite&gt;.

To iterate over the full Cartesian product use &lt;cite&gt;itertools.product(m.enumerate_support())&lt;/cite&gt;.

| Parameters: | **expand** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether to expand the support over the batch dims to match the distribution’s &lt;cite&gt;batch_shape&lt;/cite&gt;. |
| --- | --- |
| Returns: | Tensor iterating over dimension 0. |
| --- | --- |

```py
event_shape
```

Returns the shape of a single sample (without batching).

```py
expand(batch_shape, _instance=None)
```

Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to &lt;cite&gt;batch_shape&lt;/cite&gt;. This method calls [`expand`](tensors.html#torch.Tensor.expand "torch.Tensor.expand") on the distribution’s parameters. As such, this does not allocate new memory for the expanded distribution instance. Additionally, this does not repeat any args checking or parameter broadcasting in &lt;cite&gt;__init__.py&lt;/cite&gt;, when an instance is first created.

| Parameters: | 

*   **batch_shape** (_torch.Size_) – the desired expanded size.
*   **_instance** – new instance provided by subclasses that need to override &lt;cite&gt;.expand&lt;/cite&gt;.

 |
| --- | --- |
| Returns: | New distribution instance with batch dimensions expanded to &lt;cite&gt;batch_size&lt;/cite&gt;. |
| --- | --- |

```py
icdf(value)
```

Returns the inverse cumulative density/mass function evaluated at &lt;cite&gt;value&lt;/cite&gt;.

| Parameters: | **value** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – |
| --- | --- |

```py
log_prob(value)
```

Returns the log of the probability density/mass function evaluated at &lt;cite&gt;value&lt;/cite&gt;.

| Parameters: | **value** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – |
| --- | --- |

```py
mean
```

Returns the mean of the distribution.

```py
perplexity()
```

Returns perplexity of distribution, batched over batch_shape.

| Returns: | Tensor of shape batch_shape. |
| --- | --- |

```py
rsample(sample_shape=torch.Size([]))
```

Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples if the distribution parameters are batched.

```py
sample(sample_shape=torch.Size([]))
```

Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.

```py
sample_n(n)
```

Generates n samples or n batches of samples if the distribution parameters are batched.

```py
stddev
```

Returns the standard deviation of the distribution.

```py
support
```

Returns a [`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint") object representing this distribution’s support.

```py
variance
```

Returns the variance of the distribution.

## ExponentialFamily

```py
class torch.distributions.exp_family.ExponentialFamily(batch_shape=torch.Size([]), event_shape=torch.Size([]), validate_args=None)
```

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

ExponentialFamily is the abstract base class for probability distributions belonging to an exponential family, whose probability mass/density function has the form is defined below

![](img/0c8313886f5c82dfae90e21b65152815.jpg)

where ![](img/51b8359f970d2bfe2ad4cdc3ac1aed3c.jpg) denotes the natural parameters, ![](img/e705d3772de12f4df3b0cd75af5110a1.jpg) denotes the sufficient statistic, ![](img/f876c4d8353c747436006e70fb6c4f5d.jpg) is the log normalizer function for a given family and ![](img/d3b6af2f20ffbc8480c6ee97c42958b2.jpg) is the carrier measure.

Note

This class is an intermediary between the &lt;cite&gt;Distribution&lt;/cite&gt; class and distributions which belong to an exponential family mainly to check the correctness of the &lt;cite&gt;.entropy()&lt;/cite&gt; and analytic KL divergence methods. We use this class to compute the entropy and KL divergence using the AD frame- work and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and Cross-entropies of Exponential Families).

```py
entropy()
```

Method to compute the entropy using Bregman divergence of the log normalizer.

## Bernoulli

```py
class torch.distributions.bernoulli.Bernoulli(probs=None, logits=None, validate_args=None)
```

Bases: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a Bernoulli distribution parameterized by [`probs`](#torch.distributions.bernoulli.Bernoulli.probs "torch.distributions.bernoulli.Bernoulli.probs") or [`logits`](#torch.distributions.bernoulli.Bernoulli.logits "torch.distributions.bernoulli.Bernoulli.logits") (but not both).

Samples are binary (0 or 1). They take the value &lt;cite&gt;1&lt;/cite&gt; with probability &lt;cite&gt;p&lt;/cite&gt; and &lt;cite&gt;0&lt;/cite&gt; with probability &lt;cite&gt;1 - p&lt;/cite&gt;.

Example:

```py
>>> m = Bernoulli(torch.tensor([0.3]))
>>> m.sample()  # 30% chance 1; 70% chance 0
tensor([ 0.])

```

| Parameters: | 

*   **probs** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the probabilty of sampling &lt;cite&gt;1&lt;/cite&gt;
*   **logits** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log-odds of sampling &lt;cite&gt;1&lt;/cite&gt;

 |
| --- | --- |

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

Bases: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Beta distribution parameterized by [`concentration1`](#torch.distributions.beta.Beta.concentration1 "torch.distributions.beta.Beta.concentration1") and [`concentration0`](#torch.distributions.beta.Beta.concentration0 "torch.distributions.beta.Beta.concentration0").

Example:

```py
>>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
>>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
tensor([ 0.1046])

```

| Parameters: | 

*   **concentration1** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 1st concentration parameter of the distribution (often referred to as alpha)
*   **concentration0** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 2nd concentration parameter of the distribution (often referred to as beta)

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Binomial distribution parameterized by `total_count` and either [`probs`](#torch.distributions.binomial.Binomial.probs "torch.distributions.binomial.Binomial.probs") or [`logits`](#torch.distributions.binomial.Binomial.logits "torch.distributions.binomial.Binomial.logits") (but not both). `total_count` must be broadcastable with [`probs`](#torch.distributions.binomial.Binomial.probs "torch.distributions.binomial.Binomial.probs")/[`logits`](#torch.distributions.binomial.Binomial.logits "torch.distributions.binomial.Binomial.logits").

Example:

```py
>>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
>>> x = m.sample()
tensor([   0.,   22.,   71.,  100.])

>>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
>>> x = m.sample()
tensor([[ 4.,  5.],
 [ 7.,  6.]])

```

| Parameters: | 

*   **total_count** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – number of Bernoulli trials
*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Event probabilities
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Event log-odds

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a categorical distribution parameterized by either [`probs`](#torch.distributions.categorical.Categorical.probs "torch.distributions.categorical.Categorical.probs") or [`logits`](#torch.distributions.categorical.Categorical.logits "torch.distributions.categorical.Categorical.logits") (but not both).

Note

It is equivalent to the distribution that [`torch.multinomial()`](torch.html#torch.multinomial "torch.multinomial") samples from.

Samples are integers from ![](img/7c6904e60a8ff7044a079e10eaee1f57.jpg) where &lt;cite&gt;K&lt;/cite&gt; is `probs.size(-1)`.

If [`probs`](#torch.distributions.categorical.Categorical.probs "torch.distributions.categorical.Categorical.probs") is 1D with length-&lt;cite&gt;K&lt;/cite&gt;, each element is the relative probability of sampling the class at that index.

If [`probs`](#torch.distributions.categorical.Categorical.probs "torch.distributions.categorical.Categorical.probs") is 2D, it is treated as a batch of relative probability vectors.

Note

[`probs`](#torch.distributions.categorical.Categorical.probs "torch.distributions.categorical.Categorical.probs") must be non-negative, finite and have a non-zero sum, and it will be normalized to sum to 1.

See also: [`torch.multinomial()`](torch.html#torch.multinomial "torch.multinomial")

Example:

```py
>>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
>>> m.sample()  # equal probability of 0, 1, 2, 3
tensor(3)

```

| Parameters: | 

*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event log probabilities

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of independent normally distributed random variables with means &lt;cite&gt;0&lt;/cite&gt; follows a Cauchy distribution.

Example:

```py
>>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
>>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
tensor([ 2.3214])

```

| Parameters: | 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mode or median of the distribution.
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – half width at half maximum.

 |
| --- | --- |

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

Bases: [`torch.distributions.gamma.Gamma`](#torch.distributions.gamma.Gamma "torch.distributions.gamma.Gamma")

Creates a Chi2 distribution parameterized by shape parameter [`df`](#torch.distributions.chi2.Chi2.df "torch.distributions.chi2.Chi2.df"). This is exactly equivalent to `Gamma(alpha=0.5*df, beta=0.5)`

Example:

```py
>>> m = Chi2(torch.tensor([1.0]))
>>> m.sample()  # Chi2 distributed with shape df=1
tensor([ 0.1046])

```

| Parameters: | **df** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – shape parameter of the distribution |
| --- | --- |

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

Bases: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a Dirichlet distribution parameterized by concentration `concentration`.

Example:

```py
>>> m = Dirichlet(torch.tensor([0.5, 0.5]))
>>> m.sample()  # Dirichlet distributed with concentrarion concentration
tensor([ 0.1046,  0.8954])

```

| Parameters: | **concentration** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – concentration parameter of the distribution (often referred to as alpha) |
| --- | --- |

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

Bases: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a Exponential distribution parameterized by `rate`.

Example:

```py
>>> m = Exponential(torch.tensor([1.0]))
>>> m.sample()  # Exponential distributed with rate=1
tensor([ 0.1046])

```

| Parameters: | **rate** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – rate = 1 / scale of the distribution |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Fisher-Snedecor distribution parameterized by `df1` and `df2`.

Example:

```py
>>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))
>>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
tensor([ 0.2453])

```

| Parameters: | 

*   **df1** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – degrees of freedom parameter 1
*   **df2** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – degrees of freedom parameter 2

 |
| --- | --- |

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

Bases: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a Gamma distribution parameterized by shape `concentration` and `rate`.

Example:

```py
>>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
>>> m.sample()  # Gamma distributed with concentration=1 and rate=1
tensor([ 0.1046])

```

| Parameters: | 

*   **concentration** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – shape parameter of the distribution (often referred to as alpha)
*   **rate** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – rate = 1 / scale of the distribution (often referred to as beta)

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Geometric distribution parameterized by [`probs`](#torch.distributions.geometric.Geometric.probs "torch.distributions.geometric.Geometric.probs"), where [`probs`](#torch.distributions.geometric.Geometric.probs "torch.distributions.geometric.Geometric.probs") is the probability of success of Bernoulli trials. It represents the probability that in ![](img/10396db36bab7b7242cfe94f04374444.jpg) Bernoulli trials, the first ![](img/a1c2f8d5b1226e67bdb44b12a6ddf18b.jpg) trials failed, before seeing a success.

Samples are non-negative integers [0, ![](img/06485c2c6e992cf346fdfe033a86a10d.jpg)).

Example:

```py
>>> m = Geometric(torch.tensor([0.3]))
>>> m.sample()  # underlying Bernoulli has 30% chance 1; 70% chance 0
tensor([ 2.])

```

| Parameters: | 

*   **probs** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the probabilty of sampling &lt;cite&gt;1&lt;/cite&gt;. Must be in range (0, 1]
*   **logits** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log-odds of sampling &lt;cite&gt;1&lt;/cite&gt;.

 |
| --- | --- |

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

Bases: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Samples from a Gumbel Distribution.

Examples:

```py
>>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
>>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
tensor([ 1.0124])

```

| Parameters: | 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Location parameter of the distribution
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Scale parameter of the distribution

 |
| --- | --- |

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

Bases: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a half-normal distribution parameterized by &lt;cite&gt;scale&lt;/cite&gt; where:

```py
X ~ Cauchy(0, scale)
Y = |X| ~ HalfCauchy(scale)

```

Example:

```py
>>> m = HalfCauchy(torch.tensor([1.0]))
>>> m.sample()  # half-cauchy distributed with scale=1
tensor([ 2.3214])

```

| Parameters: | **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – scale of the full Cauchy distribution |
| --- | --- |

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

Bases: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a half-normal distribution parameterized by &lt;cite&gt;scale&lt;/cite&gt; where:

```py
X ~ Normal(0, scale)
Y = |X| ~ HalfNormal(scale)

```

Example:

```py
>>> m = HalfNormal(torch.tensor([1.0]))
>>> m.sample()  # half-normal distributed with scale=1
tensor([ 0.1046])

```

| Parameters: | **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – scale of the full Normal distribution |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

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

| Parameters: | 

*   **base_distribution** ([_torch.distributions.distribution.Distribution_](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")) – a base distribution
*   **reinterpreted_batch_ndims** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the number of batch dims to reinterpret as event dims

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Laplace distribution parameterized by `loc` and :attr:’scale’.

Example:

```py
>>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
>>> m.sample()  # Laplace distributed with loc=0, scale=1
tensor([ 0.1046])

```

| Parameters: | 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – scale of the distribution

 |
| --- | --- |

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

Bases: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a log-normal distribution parameterized by [`loc`](#torch.distributions.log_normal.LogNormal.loc "torch.distributions.log_normal.LogNormal.loc") and [`scale`](#torch.distributions.log_normal.LogNormal.scale "torch.distributions.log_normal.LogNormal.scale") where:

```py
X ~ Normal(loc, scale)
Y = exp(X) ~ LogNormal(loc, scale)

```

Example:

```py
>>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
>>> m.sample()  # log-normal distributed with mean=0 and stddev=1
tensor([ 0.1046])

```

| Parameters: | 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of log of distribution
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – standard deviation of log of the distribution

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

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

| Parameters: | 

*   **loc** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution with shape &lt;cite&gt;batch_shape + event_shape&lt;/cite&gt;
*   **cov_factor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – factor part of low-rank form of covariance matrix with shape &lt;cite&gt;batch_shape + event_shape + (rank,)&lt;/cite&gt;
*   **cov_diag** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – diagonal part of low-rank form of covariance matrix with shape &lt;cite&gt;batch_shape + event_shape&lt;/cite&gt;

 |
| --- | --- |

Note

The computation for determinant and inverse of covariance matrix is avoided when &lt;cite&gt;cov_factor.shape[1] &lt;&lt; cov_factor.shape[0]&lt;/cite&gt; thanks to [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) and [matrix determinant lemma](https://en.wikipedia.org/wiki/Matrix_determinant_lemma). Thanks to these formulas, we just need to compute the determinant and inverse of the small size “capacitance” matrix:

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Multinomial distribution parameterized by `total_count` and either [`probs`](#torch.distributions.multinomial.Multinomial.probs "torch.distributions.multinomial.Multinomial.probs") or [`logits`](#torch.distributions.multinomial.Multinomial.logits "torch.distributions.multinomial.Multinomial.logits") (but not both). The innermost dimension of [`probs`](#torch.distributions.multinomial.Multinomial.probs "torch.distributions.multinomial.Multinomial.probs") indexes over categories. All other dimensions index over batches.

Note that `total_count` need not be specified if only [`log_prob()`](#torch.distributions.multinomial.Multinomial.log_prob "torch.distributions.multinomial.Multinomial.log_prob") is called (see example below)

Note

[`probs`](#torch.distributions.multinomial.Multinomial.probs "torch.distributions.multinomial.Multinomial.probs") must be non-negative, finite and have a non-zero sum, and it will be normalized to sum to 1.

*   [`sample()`](#torch.distributions.multinomial.Multinomial.sample "torch.distributions.multinomial.Multinomial.sample") requires a single shared &lt;cite&gt;total_count&lt;/cite&gt; for all parameters and samples.
*   [`log_prob()`](#torch.distributions.multinomial.Multinomial.log_prob "torch.distributions.multinomial.Multinomial.log_prob") allows different &lt;cite&gt;total_count&lt;/cite&gt; for each parameter and sample.

Example:

```py
>>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
>>> x = m.sample()  # equal probability of 0, 1, 2, 3
tensor([ 21.,  24.,  30.,  25.])

>>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)
tensor([-4.1338])

```

| Parameters: | 

*   **total_count** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – number of trials
*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event log probabilities

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a multivariate normal (also called Gaussian) distribution parameterized by a mean vector and a covariance matrix.

The multivariate normal distribution can be parameterized either in terms of a positive definite covariance matrix ![](img/ea86c11eaef9af2b4d699b88c2474ffd.jpg) or a positive definite precision matrix ![](img/1949bfcc1decf198a2ff50b6e25f4cf6.jpg) or a lower-triangular matrix ![](img/f4996f1b5056dd364eab16f975b808ff.jpg) with positive-valued diagonal entries, such that ![](img/6749b6afc75abfc8e0652ac8e5c0b8d8.jpg). This triangular matrix can be obtained via e.g. Cholesky decomposition of the covariance.

Example

```py
>>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
>>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
tensor([-0.2102, -0.5429])

```

| Parameters: | 

*   **loc** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution
*   **covariance_matrix** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – positive-definite covariance matrix
*   **precision_matrix** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – positive-definite precision matrix
*   **scale_tril** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – lower-triangular factor of covariance, with positive-valued diagonal

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Negative Binomial distribution, i.e. distribution of the number of independent identical Bernoulli trials needed before `total_count` failures are achieved. The probability of success of each Bernoulli trial is [`probs`](#torch.distributions.negative_binomial.NegativeBinomial.probs "torch.distributions.negative_binomial.NegativeBinomial.probs").

| Parameters: | 

*   **total_count** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – non-negative number of negative Bernoulli trials to stop, although the distribution is still valid for real valued count
*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Event probabilities of success in the half open interval [0, 1)
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Event log-odds for probabilities of success

 |
| --- | --- |

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

Bases: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a normal (also called Gaussian) distribution parameterized by `loc` and `scale`.

Example:

```py
>>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
>>> m.sample()  # normally distributed with loc=0 and scale=1
tensor([ 0.1046])

```

| Parameters: | 

*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution (often referred to as mu)
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – standard deviation of the distribution (often referred to as sigma)

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a one-hot categorical distribution parameterized by [`probs`](#torch.distributions.one_hot_categorical.OneHotCategorical.probs "torch.distributions.one_hot_categorical.OneHotCategorical.probs") or [`logits`](#torch.distributions.one_hot_categorical.OneHotCategorical.logits "torch.distributions.one_hot_categorical.OneHotCategorical.logits").

Samples are one-hot coded vectors of size `probs.size(-1)`.

Note

[`probs`](#torch.distributions.one_hot_categorical.OneHotCategorical.probs "torch.distributions.one_hot_categorical.OneHotCategorical.probs") must be non-negative, finite and have a non-zero sum, and it will be normalized to sum to 1.

See also: `torch.distributions.Categorical()` for specifications of [`probs`](#torch.distributions.one_hot_categorical.OneHotCategorical.probs "torch.distributions.one_hot_categorical.OneHotCategorical.probs") and [`logits`](#torch.distributions.one_hot_categorical.OneHotCategorical.logits "torch.distributions.one_hot_categorical.OneHotCategorical.logits").

Example:

```py
>>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
>>> m.sample()  # equal probability of 0, 1, 2, 3
tensor([ 0.,  0.,  0.,  1.])

```

| Parameters: | 

*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event log probabilities

 |
| --- | --- |

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

Bases: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Samples from a Pareto Type 1 distribution.

Example:

```py
>>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
>>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
tensor([ 1.5623])

```

| Parameters: | 

*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Scale parameter of the distribution
*   **alpha** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Shape parameter of the distribution

 |
| --- | --- |

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

Bases: [`torch.distributions.exp_family.ExponentialFamily`](#torch.distributions.exp_family.ExponentialFamily "torch.distributions.exp_family.ExponentialFamily")

Creates a Poisson distribution parameterized by `rate`, the rate parameter.

Samples are nonnegative integers, with a pmf given by

![](img/32c47de57300c954795486fea3201bdc.jpg)

Example:

```py
>>> m = Poisson(torch.tensor([4]))
>>> m.sample()
tensor([ 3.])

```

| Parameters: | **rate** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the rate parameter |
| --- | --- |

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

Bases: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a RelaxedBernoulli distribution, parametrized by [`temperature`](#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature "torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature"), and either [`probs`](#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs "torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs") or [`logits`](#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits "torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits") (but not both). This is a relaxed version of the &lt;cite&gt;Bernoulli&lt;/cite&gt; distribution, so the values are in (0, 1), and has reparametrizable samples.

Example:

```py
>>> m = RelaxedBernoulli(torch.tensor([2.2]),
 torch.tensor([0.1, 0.2, 0.3, 0.99]))
>>> m.sample()
tensor([ 0.2951,  0.3442,  0.8918,  0.9021])

```

| Parameters: | 

*   **temperature** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – relaxation temperature
*   **probs** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the probabilty of sampling &lt;cite&gt;1&lt;/cite&gt;
*   **logits** (_Number__,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log-odds of sampling &lt;cite&gt;1&lt;/cite&gt;

 |
| --- | --- |

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

Bases: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Creates a RelaxedOneHotCategorical distribution parametrized by [`temperature`](#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature "torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature"), and either [`probs`](#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs "torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs") or [`logits`](#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits "torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits"). This is a relaxed version of the `OneHotCategorical` distribution, so its samples are on simplex, and are reparametrizable.

Example:

```py
>>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
 torch.tensor([0.1, 0.2, 0.3, 0.4]))
>>> m.sample()
tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

```

| Parameters: | 

*   **temperature** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – relaxation temperature
*   **probs** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – event probabilities
*   **logits** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the log probability of each event.

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Creates a Student’s t-distribution parameterized by degree of freedom `df`, mean `loc` and scale `scale`.

Example:

```py
>>> m = StudentT(torch.tensor([2.0]))
>>> m.sample()  # Student's t-distributed with degrees of freedom=2
tensor([ 0.1046])

```

| Parameters: | 

*   **df** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – degrees of freedom
*   **loc** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – mean of the distribution
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – scale of the distribution

 |
| --- | --- |

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

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

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

Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples if the distribution parameters are batched. Samples first from base distribution and applies &lt;cite&gt;transform()&lt;/cite&gt; for every transform in the list.

```py
sample(sample_shape=torch.Size([]))
```

Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched. Samples first from base distribution and applies &lt;cite&gt;transform()&lt;/cite&gt; for every transform in the list.

```py
support
```

## Uniform

```py
class torch.distributions.uniform.Uniform(low, high, validate_args=None)
```

Bases: [`torch.distributions.distribution.Distribution`](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")

Generates uniformly distributed random samples from the half-open interval `[low, high)`.

Example:

```py
>>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
>>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
tensor([ 2.3418])

```

| Parameters: | 

*   **low** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – lower range (inclusive).
*   **high** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – upper range (exclusive).

 |
| --- | --- |

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

Bases: [`torch.distributions.transformed_distribution.TransformedDistribution`](#torch.distributions.transformed_distribution.TransformedDistribution "torch.distributions.transformed_distribution.TransformedDistribution")

Samples from a two-parameter Weibull distribution.

Example

```py
>>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
>>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
tensor([ 0.4784])

```

| Parameters: | 

*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Scale parameter of distribution (lambda).
*   **concentration** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Concentration parameter of distribution (k/shape).

 |
| --- | --- |

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

## &lt;cite&gt;KL Divergence&lt;/cite&gt;

```py
torch.distributions.kl.kl_divergence(p, q)
```

Compute Kullback-Leibler divergence ![](img/739a8e4cd0597805c3e4daf35c0fc7c6.jpg) between two distributions.

![](img/ff8dcec3abe559720f8b0b464d2471b2.jpg)

| Parameters: | 

*   **p** ([_Distribution_](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")) – A `Distribution` object.
*   **q** ([_Distribution_](#torch.distributions.distribution.Distribution "torch.distributions.distribution.Distribution")) – A `Distribution` object.

 |
| --- | --- |
| Returns: | A batch of KL divergences of shape &lt;cite&gt;batch_shape&lt;/cite&gt;. |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |
| Raises: | [`NotImplementedError`](https://docs.python.org/3/library/exceptions.html#NotImplementedError "(in Python v3.7)") – If the distribution types have not been registered via [`register_kl()`](#torch.distributions.kl.register_kl "torch.distributions.kl.register_kl"). |
| --- | --- |

```py
torch.distributions.kl.register_kl(type_p, type_q)
```

Decorator to register a pairwise function with [`kl_divergence()`](#torch.distributions.kl.kl_divergence "torch.distributions.kl.kl_divergence"). Usage:

```py
@register_kl(Normal, Normal)
def kl_normal_normal(p, q):
    # insert implementation here

```

Lookup returns the most specific (type,type) match ordered by subclass. If the match is ambiguous, a &lt;cite&gt;RuntimeWarning&lt;/cite&gt; is raised. For example to resolve the ambiguous situation:

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

| Parameters: | 

*   **type_p** ([_type_](https://docs.python.org/3/library/functions.html#type "(in Python v3.7)")) – A subclass of `Distribution`.
*   **type_q** ([_type_](https://docs.python.org/3/library/functions.html#type "(in Python v3.7)")) – A subclass of `Distribution`.

 |
| --- | --- |

## &lt;cite&gt;Transforms&lt;/cite&gt;

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

Derived classes should implement one or both of `_call()` or `_inverse()`. Derived classes that set &lt;cite&gt;bijective=True&lt;/cite&gt; should also implement [`log_abs_det_jacobian()`](#torch.distributions.transforms.Transform.log_abs_det_jacobian "torch.distributions.transforms.Transform.log_abs_det_jacobian").

| Parameters: | **cache_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Size of cache. If zero, no caching is done. If one, the latest single value is cached. Only 0 and 1 are supported. |
| --- | --- |
| Variables: | 

*   **domain** ([`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint")) – The constraint representing valid inputs to this transform.
*   **codomain** ([`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint")) – The constraint representing valid outputs to this transform which are inputs to the inverse transform.
*   **bijective** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – Whether this transform is bijective. A transform `t` is bijective iff `t.inv(t(x)) == x` and `t(t.inv(y)) == y` for every `x` in the domain and `y` in the codomain. Transforms that are not bijective should at least maintain the weaker pseudoinverse properties `t(t.inv(t(x)) == t(x)` and `t.inv(t(t.inv(y))) == t.inv(y)`.
*   **sign** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – For bijective univariate transforms, this should be +1 or -1 depending on whether transform is monotone increasing or decreasing.
*   **event_dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Number of dimensions that are correlated together in the transform `event_shape`. This should be 0 for pointwise transforms, 1 for transforms that act jointly on vectors, 2 for transforms that act jointly on matrices, etc.

 |
| --- | --- |

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

Computes the log det jacobian &lt;cite&gt;log |dy/dx|&lt;/cite&gt; given input and output.

```py
class torch.distributions.transforms.ComposeTransform(parts)
```

Composes multiple transforms in a chain. The transforms being composed are responsible for caching.

| Parameters: | **parts** (list of [`Transform`](#torch.distributions.transforms.Transform "torch.distributions.transforms.Transform")) – A list of transforms to compose. |
| --- | --- |

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

| Parameters: | 

*   **loc** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Location parameter.
*   **scale** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Scale parameter.
*   **event_dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Optional size of &lt;cite&gt;event_shape&lt;/cite&gt;. This should be zero for univariate random variables, 1 for distributions over vectors, 2 for distributions over matrices, etc.

 |
| --- | --- |

```py
class torch.distributions.transforms.SoftmaxTransform(cache_size=0)
```

Transform from unconstrained space to the simplex via ![](img/ec8d939394f24908d017d86153e312ea.jpg) then normalizing.

This is not bijective and cannot be used for HMC. However this acts mostly coordinate-wise (except for the final normalization), and thus is appropriate for coordinate-wise optimization algorithms.

```py
class torch.distributions.transforms.StickBreakingTransform(cache_size=0)
```

Transform from unconstrained space to the simplex of one additional dimension via a stick-breaking process.

This transform arises as an iterated sigmoid transform in a stick-breaking construction of the &lt;cite&gt;Dirichlet&lt;/cite&gt; distribution: the first logit is transformed via sigmoid to the first probability and the probability of everything else, and then the process recurses.

This is bijective and appropriate for use in HMC; however it mixes coordinates together and is less appropriate for optimization.

```py
class torch.distributions.transforms.LowerCholeskyTransform(cache_size=0)
```

Transform from unconstrained matrices to lower-triangular matrices with nonnegative diagonal entries.

This is useful for parameterizing positive definite matrices in terms of their Cholesky factorization.

## &lt;cite&gt;Constraints&lt;/cite&gt;

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

Returns a byte tensor of &lt;cite&gt;sample_shape + batch_shape&lt;/cite&gt; indicating whether each event in value satisfies this constraint.

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

## &lt;cite&gt;Constraint Registry&lt;/cite&gt;

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

| Parameters: | 

*   **constraint** (subclass of [`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint")) – A subclass of [`Constraint`](#torch.distributions.constraints.Constraint "torch.distributions.constraints.Constraint"), or a singleton object of the desired class.
*   **factory** (_callable_) – A callable that inputs a constraint object and returns a [`Transform`](#torch.distributions.transforms.Transform "torch.distributions.transforms.Transform") object.

 |
| --- | --- |

