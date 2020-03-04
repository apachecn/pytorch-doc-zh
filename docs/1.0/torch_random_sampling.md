

## 随机抽样

> 译者：[ApacheCN](https://github.com/apachecn)

```py
torch.manual_seed(seed)
```

设置用于生成随机数的种子。返回`torch._C.Generator`对象。

| 参数： | **种子** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 所需种子。 |
| --- | --- |

```py
torch.initial_seed()
```

返回用于生成随机数的初始种子，如Python `long`。

```py
torch.get_rng_state()
```

将随机数生成器状态返回为`torch.ByteTensor`。

```py
torch.set_rng_state(new_state)
```

设置随机数生成器状态。

| Parameters: | **new_state**  ([_torch.ByteTensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.ByteTensor "torch.ByteTensor")） - 理想状态 |
| --- | --- |

```py
torch.default_generator = <torch._C.Generator object>
```

```py
torch.bernoulli(input, *, generator=None, out=None) → Tensor
```

从伯努利分布中绘制二进制随机数(0或1）。

`input`张量应该是包含用于绘制二进制随机数的概率的张量。因此，`input`中的所有值必须在以下范围内： [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1ac351f229015d1047c7d979a9233916.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1ac351f229015d1047c7d979a9233916.jpg) 。

输出张量的 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/511f5a204e4e69e0f1c374e9a5738214.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/511f5a204e4e69e0f1c374e9a5738214.jpg) 元素将根据`input`中给出的 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/511f5a204e4e69e0f1c374e9a5738214.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/511f5a204e4e69e0f1c374e9a5738214.jpg) 概率值绘制值 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a3ea24a1f2a3549d3e5b0cacf3ecb7c7.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a3ea24a1f2a3549d3e5b0cacf3ecb7c7.jpg) 。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b3f6cd5a237f587278432aa96dd0fd96.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b3f6cd5a237f587278432aa96dd0fd96.jpg)

返回的`out`张量仅具有值0或1，并且与`input`具有相同的形状。

`out`可以有整数`dtype`，但是：attr `input`必须有浮点`dtype`。

参数：

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 伯努利分布的概率值的输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

例：

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

返回张量，其中每行包含从位于张量`input`的相应行中的多项概率分布中采样的`num_samples`索引。

注意

`input`的行不需要求和为一(在这种情况下我们使用值作为权重），但必须是非负的，有限的并且具有非零和。

根据每个样本的时间(第一个样本放在第一列中）从左到右排序指数。

如果`input`是矢量，`out`是大小为`num_samples`的矢量。

如果`input`是具有`m`行的矩阵，则`out`是形状矩阵 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/c52c825df9f5a9934f74be6777337b15.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/c52c825df9f5a9934f74be6777337b15.jpg) 。

如果更换为`True`，则更换样品。

如果没有，则绘制它们而不替换它们，这意味着当为一行绘制样本索引时，不能再为该行绘制它。

这意味着`num_samples`必须低于`input`长度(或者`input`的列数，如果它是矩阵）的约束。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 包含概率的输入张量
*   **num_samples**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 要抽取的样本数量
*   **替代** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 是否与替换有关
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

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

返回从单独的正态分布中提取的随机数的张量，其中给出了均值和标准差。

[`mean`](#torch.mean "torch.mean") 是一个张量，具有每个输出元素正态分布的均值

[`std`](#torch.std "torch.std") 是一个张量，每个输出元素的正态分布的标准差

[`mean`](#torch.mean "torch.mean") 和 [`std`](#torch.std "torch.std") 的形状不需要匹配，但每个张量中的元素总数需要相同。

Note

当形状不匹配时， [`mean`](#torch.mean "torch.mean") 的形状用作返回输出张量的形状

Parameters:

*   **意味着** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 每个元素的张量意味着
*   **std**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 每元素标准差的张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
 8.0505,   8.1408,   9.0563,  10.0566])

```

```py
torch.normal(mean=0.0, std, out=None) → Tensor
```

与上面的函数类似，但是所有绘制元素之间共享均值。

Parameters:

*   **意味着** ([_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _，_ _任选_） - 所有分布的均值
*   **std**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 每元素标准差的张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.normal(mean=0.5, std=torch.arange(1., 6.))
tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])

```

```py
torch.normal(mean, std=1.0, out=None) → Tensor
```

与上述功能类似，但标准偏差在所有绘制元素之间共享。

Parameters:

*   **意味着** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 每个元素的张量意味着
*   **std**  ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_，_ _可选_） - 所有分布的标准差
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.normal(mean=torch.arange(1., 6.))
tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])

```

```py
torch.rand(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

从区间 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a686b817a52173e9e124e756a19344be.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a686b817a52173e9e124e756a19344be.jpg) 上的均匀分布返回填充随机数的张量

张量的形状由变量参数`sizes`定义。

Parameters:

*   **sizes**  (_int ..._ ) - 定义输出张量形状的整数序列。可以是可变数量的参数，也可以是列表或元组之类的集合。
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。默认值：if `None`，使用全局默认值(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回Tensor的理想布局。默认值：`torch.strided`。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，则使用当前设备作为默认张量类型(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。 `device`将是CPU张量类型的CPU和CUDA张量类型的当前CUDA设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

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

返回与`input`大小相同的张量，该张量用间隔 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a686b817a52173e9e124e756a19344be.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a686b817a52173e9e124e756a19344be.jpg) 上的均匀分布填充随机数。 `torch.rand_like(input)`相当于`torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - `input`的大小将决定输出张量的大小
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回的Tensor的理想数据类型。默认值：if `None`，默认为`input`的dtype。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回张量的理想布局。默认值：if `None`，默认为`input`的布局。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，默认为`input`的设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

```py
torch.randint(low=0, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

返回填充了在`low`(包括）和`high`(不包括）之间统一生成的随机整数的张量。

张量的形状由变量参数`size`定义。

Parameters:

*   **低** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 从分布中得出的最小整数。默认值：0。
*   **高** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 高于从分布中提取的最高整数。
*   **大小** ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) - 定义输出张量形状的元组。
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。默认值：if `None`，使用全局默认值(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回Tensor的理想布局。默认值：`torch.strided`。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，则使用当前设备作为默认张量类型(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。 `device`将是CPU张量类型的CPU和CUDA张量类型的当前CUDA设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

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

返回与Tensor `input`具有相同形状的张量，填充在`low`(包括）和`high`(不包括）之间均匀生成的随机整数。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - `input`的大小将决定输出张量的大小
*   **低** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 从分布中得出的最小整数。默认值：0。
*   **高** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 高于从分布中提取的最高整数。
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回的Tensor的理想数据类型。默认值：if `None`，默认为`input`的dtype。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回张量的理想布局。默认值：if `None`，默认为`input`的布局。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，默认为`input`的设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

```py
torch.randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

返回一个填充了正态分布中随机数的张量，其均值为`0`和方差`1`(也称为标准正态分布）。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/71f756d003530899b04dfd92986cea2f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/71f756d003530899b04dfd92986cea2f.jpg)

The shape of the tensor is defined by the variable argument `sizes`.

Parameters:

*   **sizes**  (_int ..._ ) - 定义输出张量形状的整数序列。可以是可变数量的参数，也可以是列表或元组之类的集合。
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。默认值：if `None`，使用全局默认值(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回Tensor的理想布局。默认值：`torch.strided`。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，则使用当前设备作为默认张量类型(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。 `device`将是CPU张量类型的CPU和CUDA张量类型的当前CUDA设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

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

返回与`input`具有相同大小的张量，该张量用正态分布中的随机数填充，均值为0且方差为1\. `torch.randn_like(input)`等效于`torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - `input`的大小将决定输出张量的大小
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回的Tensor的理想数据类型。默认值：if `None`，默认为`input`的dtype。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回张量的理想布局。默认值：if `None`，默认为`input`的布局。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，默认为`input`的设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

```py
torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) → LongTensor
```

返回从`0`到`n - 1`的整数的随机排列。

Parameters:

*   **n**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 上限(不包括）
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。默认值：`torch.int64`。
*   **布局** ([`torch.layout`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.layout "torch.torch.layout") ，可选） - 返回Tensor的理想布局。默认值：`torch.strided`。
*   **设备** ([`torch.device`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.device "torch.torch.device") ，可选） - 返回张量的所需设备。默认值：如果`None`，则使用当前设备作为默认张量类型(参见 [`torch.set_default_tensor_type()`](#torch.set_default_tensor_type "torch.set_default_tensor_type"))。 `device`将是CPU张量类型的CPU和CUDA张量类型的当前CUDA设备。
*   **requires_grad**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 如果autograd应该记录对返回张量的操作。默认值：`False`。

Example:

```py
>>> torch.randperm(4)
tensor([2, 1, 0, 3])

```

### 就地随机抽样

Tensors还定义了一些更多的就地随机抽样函数。点击查看他们的文档：

*   [`torch.Tensor.bernoulli_()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor.bernoulli_ "torch.Tensor.bernoulli_") - [`torch.bernoulli()`](#torch.bernoulli "torch.bernoulli") 的原位版本
*   [`torch.Tensor.cauchy_()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor.cauchy_ "torch.Tensor.cauchy_") - 从Cauchy分布中提取的数字
*   [`torch.Tensor.exponential_()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor.exponential_ "torch.Tensor.exponential_") - 从指数分布中提取的数字
*   [`torch.Tensor.geometric_()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor.geometric_ "torch.Tensor.geometric_") - 从几何分布中提取的元素
*   [`torch.Tensor.log_normal_()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor.log_normal_ "torch.Tensor.log_normal_") - 来自对数正态分布的样本
*   [`torch.Tensor.normal_()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor.normal_ "torch.Tensor.normal_") - [`torch.normal()`](#torch.normal "torch.normal") 的原位版本
*   [`torch.Tensor.random_()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor.random_ "torch.Tensor.random_") - 从离散均匀分布中采样的数字
*   [`torch.Tensor.uniform_()`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor.uniform_ "torch.Tensor.uniform_") - 从连续均匀分布中采样的数字

