## Random sampling

```py
torch.manual_seed(seed)
```

Sets the seed for generating random numbers. Returns a `torch._C.Generator` object.

| Parameters: | **seed** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The desired seed. |
| --- | --- |

```py
torch.initial_seed()
```

Returns the initial seed for generating random numbers as a Python `long`.

```py
torch.get_rng_state()
```

Returns the random number generator state as a `torch.ByteTensor`.

```py
torch.set_rng_state(new_state)
```

Sets the random number generator state.

| Parameters: | **new_state** ([_torch.ByteTensor_](tensors.html#torch.ByteTensor "torch.ByteTensor")) – The desired state |
| --- | --- |

```py
torch.default_generator = <torch._C.Generator object>
```

```py
torch.bernoulli(input, *, generator=None, out=None) → Tensor
```

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

The `input` tensor should be a tensor containing probabilities to be used for drawing the binary random number. Hence, all values in `input` have to be in the range: ![](img/1ac351f229015d1047c7d979a9233916.jpg).

The ![](img/511f5a204e4e69e0f1c374e9a5738214.jpg) element of the output tensor will draw a value ![](img/a3ea24a1f2a3549d3e5b0cacf3ecb7c7.jpg) according to the ![](img/511f5a204e4e69e0f1c374e9a5738214.jpg) probability value given in `input`.

![](img/b3f6cd5a237f587278432aa96dd0fd96.jpg)

The returned `out` tensor only has values 0 or 1 and is of the same shape as `input`.

`out` can have integral `dtype`, but :attr`input` must have floating point `dtype`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor of probability values for the Bernoulli distribution
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
>>> a
tensor([[ 0.1737,  0.0950,  0.3609],
 [ 0.7148,  0.0289,  0.2676],
 [ 0.9456,  0.8937,  0.7202]])
>>> torch.bernoulli(a)
tensor([[ 1.,  0.,  0.],
 [ 0.,  0.,  0.],
 [ 1.,  1.,  1.]])

>>> a = torch.ones(3, 3) # probability of drawing "1" is 1
>>> torch.bernoulli(a)
tensor([[ 1.,  1.,  1.],
 [ 1.,  1.,  1.],
 [ 1.,  1.,  1.]])
>>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
>>> torch.bernoulli(a)
tensor([[ 0.,  0.,  0.],
 [ 0.,  0.,  0.],
 [ 0.,  0.,  0.]])

```

```py
torch.multinomial(input, num_samples, replacement=False, out=None) → LongTensor
```

Returns a tensor where each row contains `num_samples` indices sampled from the multinomial probability distribution located in the corresponding row of tensor `input`.

Note

The rows of `input` do not need to sum to one (in which case we use the values as weights), but must be non-negative, finite and have a non-zero sum.

Indices are ordered from left to right according to when each was sampled (first samples are placed in first column).

If `input` is a vector, `out` is a vector of size `num_samples`.

If `input` is a matrix with `m` rows, `out` is an matrix of shape ![](img/c52c825df9f5a9934f74be6777337b15.jpg).

If replacement is `True`, samples are drawn with replacement.

If not, they are drawn without replacement, which means that when a sample index is drawn for a row, it cannot be drawn again for that row.

This implies the constraint that `num_samples` must be lower than `input` length (or number of columns of `input` if it is a matrix).

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor containing probabilities
*   **num_samples** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – number of samples to draw
*   **replacement** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether to draw with replacement or not
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> weights = torch.tensor([0, 10, 3, 0], dtype=torch.float) # create a tensor of weights
>>> torch.multinomial(weights, 4)
tensor([ 1,  2,  0,  0])
>>> torch.multinomial(weights, 4, replacement=True)
tensor([ 2,  1,  1,  1])

```

```py
torch.normal()
```

```py
torch.normal(mean, std, out=None) → Tensor
```

Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.

The [`mean`](#torch.mean "torch.mean") is a tensor with the mean of each output element’s normal distribution

The [`std`](#torch.std "torch.std") is a tensor with the standard deviation of each output element’s normal distribution

The shapes of [`mean`](#torch.mean "torch.mean") and [`std`](#torch.std "torch.std") don’t need to match, but the total number of elements in each tensor need to be the same.

Note

When the shapes do not match, the shape of [`mean`](#torch.mean "torch.mean") is used as the shape for the returned output tensor

Parameters: 

*   **mean** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor of per-element means
*   **std** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor of per-element standard deviations
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
 8.0505,   8.1408,   9.0563,  10.0566])

```

```py
torch.normal(mean=0.0, std, out=None) → Tensor
```

Similar to the function above, but the means are shared among all drawn elements.

Parameters: 

*   **mean** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – the mean for all distributions
*   **std** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor of per-element standard deviations
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> torch.normal(mean=0.5, std=torch.arange(1., 6.))
tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])

```

```py
torch.normal(mean, std=1.0, out=None) → Tensor
```

Similar to the function above, but the standard-deviations are shared among all drawn elements.

Parameters: 

*   **mean** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor of per-element means
*   **std** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – the standard deviation for all distributions
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> torch.normal(mean=torch.arange(1., 6.))
tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])

```

```py
torch.rand(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

Returns a tensor filled with random numbers from a uniform distribution on the interval ![](img/a686b817a52173e9e124e756a19344be.jpg)

The shape of the tensor is defined by the variable argument `sizes`.

Parameters: 

*   **sizes** (_int..._) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")).
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.



Example:

```py
>>> torch.rand(4)
tensor([ 0.5204,  0.2503,  0.3525,  0.5673])
>>> torch.rand(2, 3)
tensor([[ 0.8237,  0.5781,  0.6879],
 [ 0.3816,  0.7249,  0.0998]])

```

```py
torch.rand_like(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor
```

Returns a tensor with the same size as `input` that is filled with random numbers from a uniform distribution on the interval ![](img/a686b817a52173e9e124e756a19344be.jpg). `torch.rand_like(input)` is equivalent to `torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the size of `input` will determine size of the output tensor
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned Tensor. Default: if `None`, defaults to the dtype of `input`.
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned tensor. Default: if `None`, defaults to the layout of `input`.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, defaults to the device of `input`.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.



```py
torch.randint(low=0, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

Returns a tensor filled with random integers generated uniformly between `low` (inclusive) and `high` (exclusive).

The shape of the tensor is defined by the variable argument `size`.

Parameters: 

*   **low** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Lowest integer to be drawn from the distribution. Default: 0.
*   **high** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – One above the highest integer to be drawn from the distribution.
*   **size** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – a tuple defining the shape of the output tensor.
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")).
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.



Example:

```py
>>> torch.randint(3, 5, (3,))
tensor([4, 3, 4])

>>> torch.randint(10, (2, 2))
tensor([[0, 2],
 [5, 5]])

>>> torch.randint(3, 10, (2, 2))
tensor([[4, 5],
 [6, 7]])

```

```py
torch.randint_like(input, low=0, high, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

Returns a tensor with the same shape as Tensor `input` filled with random integers generated uniformly between `low` (inclusive) and `high` (exclusive).

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the size of `input` will determine size of the output tensor
*   **low** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Lowest integer to be drawn from the distribution. Default: 0.
*   **high** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – One above the highest integer to be drawn from the distribution.
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned Tensor. Default: if `None`, defaults to the dtype of `input`.
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned tensor. Default: if `None`, defaults to the layout of `input`.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, defaults to the device of `input`.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.



```py
torch.randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

Returns a tensor filled with random numbers from a normal distribution with mean `0` and variance `1` (also called the standard normal distribution).

![](img/71f756d003530899b04dfd92986cea2f.jpg)

The shape of the tensor is defined by the variable argument `sizes`.

Parameters: 

*   **sizes** (_int..._) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")).
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.



Example:

```py
>>> torch.randn(4)
tensor([-2.1436,  0.9966,  2.3426, -0.6366])
>>> torch.randn(2, 3)
tensor([[ 1.5954,  2.8929, -1.0923],
 [ 1.1719, -0.4709, -0.1996]])

```

```py
torch.randn_like(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor
```

Returns a tensor with the same size as `input` that is filled with random numbers from a normal distribution with mean 0 and variance 1. `torch.randn_like(input)` is equivalent to `torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the size of `input` will determine size of the output tensor
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned Tensor. Default: if `None`, defaults to the dtype of `input`.
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned tensor. Default: if `None`, defaults to the layout of `input`.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, defaults to the device of `input`.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.



```py
torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) → LongTensor
```

Returns a random permutation of integers from `0` to `n - 1`.

Parameters: 

*   **n** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the upper bound (exclusive)
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: `torch.int64`.
*   **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.
*   **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type")). `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
*   **requires_grad** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If autograd should record operations on the returned tensor. Default: `False`.



Example:

```py
>>> torch.randperm(4)
tensor([2, 1, 0, 3])

```

### In-place random sampling

There are a few more in-place random sampling functions defined on Tensors as well. Click through to refer to their documentation:

*   [`torch.Tensor.bernoulli_()`](tensors.html#torch.Tensor.bernoulli_ "torch.Tensor.bernoulli_") - in-place version of [`torch.bernoulli()`](#torch.bernoulli "torch.bernoulli")
*   [`torch.Tensor.cauchy_()`](tensors.html#torch.Tensor.cauchy_ "torch.Tensor.cauchy_") - numbers drawn from the Cauchy distribution
*   [`torch.Tensor.exponential_()`](tensors.html#torch.Tensor.exponential_ "torch.Tensor.exponential_") - numbers drawn from the exponential distribution
*   [`torch.Tensor.geometric_()`](tensors.html#torch.Tensor.geometric_ "torch.Tensor.geometric_") - elements drawn from the geometric distribution
*   [`torch.Tensor.log_normal_()`](tensors.html#torch.Tensor.log_normal_ "torch.Tensor.log_normal_") - samples from the log-normal distribution
*   [`torch.Tensor.normal_()`](tensors.html#torch.Tensor.normal_ "torch.Tensor.normal_") - in-place version of [`torch.normal()`](#torch.normal "torch.normal")
*   [`torch.Tensor.random_()`](tensors.html#torch.Tensor.random_ "torch.Tensor.random_") - numbers sampled from the discrete uniform distribution
*   [`torch.Tensor.uniform_()`](tensors.html#torch.Tensor.uniform_ "torch.Tensor.uniform_") - numbers sampled from the continuous uniform distribution
