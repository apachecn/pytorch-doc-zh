

### 减少行动

> 译者：[ApacheCN](https://github.com/apachecn)

```py
torch.argmax(input, dim=None, keepdim=False)
```

返回维度上张量的最大值的索引。

这是 [`torch.max()`](#torch.max "torch.max") 返回的第二个值。有关此方法的确切语义，请参阅其文档。

参数：

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **dim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 降低的维数。如果`None`，则返回展平输入的argmax。
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`。如果`dim=None`，则忽略。

例：

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
 [-0.7401, -0.8805, -0.3402, -1.1936],
 [ 0.4907, -1.3948, -1.0691, -0.3132],
 [-1.6092,  0.5419, -0.2993,  0.3195]])

>>> torch.argmax(a, dim=1)
tensor([ 0,  2,  0,  1])

```

```py
torch.argmin(input, dim=None, keepdim=False)
```

返回维度上张量的最小值的索引。

这是 [`torch.min()`](#torch.min "torch.min") 返回的第二个值。有关此方法的确切语义，请参阅其文档。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **dim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 降低的维数。如果`None`，则返回展平输入的argmin。
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`。如果`dim=None`，则忽略。

Example:

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
 [ 1.0100, -1.1975, -0.0102, -0.4732],
 [-0.9240,  0.1207, -0.7506, -1.0213],
 [ 1.7809, -1.2960,  0.9384,  0.1438]])

>>> torch.argmin(a, dim=1)
tensor([ 2,  1,  3,  1])

```

```py
torch.cumprod(input, dim, dtype=None) → Tensor
```

返回维度`dim`中`input`元素的累积乘积。

例如，如果`input`是大小为N的向量，则结果也将是具有元素的大小为N的向量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9e9045e46c1b7fca7acb598cf474f16e.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9e9045e46c1b7fca7acb598cf474f16e.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 执行操作的维度
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。如果指定，则在执行操作之前将输入张量转换为`dtype`。这对于防止数据类型溢出很有用。默认值：无。

Example:

```py
>>> a = torch.randn(10)
>>> a
tensor([ 0.6001,  0.2069, -0.1919,  0.9792,  0.6727,  1.0062,  0.4126,
 -0.2129, -0.4206,  0.1968])
>>> torch.cumprod(a, dim=0)
tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065,
 0.0014, -0.0006, -0.0001])

>>> a[5] = 0.0
>>> torch.cumprod(a, dim=0)
tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000,
 0.0000, -0.0000, -0.0000])

```

```py
torch.cumsum(input, dim, out=None, dtype=None) → Tensor
```

返回维度`dim`中`input`的元素的累积和。

For example, if `input` is a vector of size N, the result will also be a vector of size N, with elements.

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/70f741993caea1156636d61e6f21e463.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/70f741993caea1156636d61e6f21e463.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 执行操作的维度
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。如果指定，则在执行操作之前将输入张量转换为`dtype`。这对于防止数据类型溢出很有用。默认值：无。

Example:

```py
>>> a = torch.randn(10)
>>> a
tensor([-0.8286, -0.4890,  0.5155,  0.8443,  0.1865, -0.1752, -2.0595,
 0.1850, -1.1571, -0.4243])
>>> torch.cumsum(a, dim=0)
tensor([-0.8286, -1.3175, -0.8020,  0.0423,  0.2289,  0.0537, -2.0058,
 -1.8209, -2.9780, -3.4022])

```

```py
torch.dist(input, other, p=2) → Tensor
```

返回(`input` - `other`）的p范数

`input`和`other`的形状必须是[可播放的](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **其他** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 右侧输入张量
*   **p**  ([_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _，_ _任选_） - 要计算的范数

Example:

```py
>>> x = torch.randn(4)
>>> x
tensor([-1.5393, -0.8675,  0.5916,  1.6321])
>>> y = torch.randn(4)
>>> y
tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
>>> torch.dist(x, y, 3.5)
tensor(1.6727)
>>> torch.dist(x, y, 3)
tensor(1.6973)
>>> torch.dist(x, y, 0)
tensor(inf)
>>> torch.dist(x, y, 1)
tensor(2.6537)

```

```py
torch.logsumexp(input, dim, keepdim=False, out=None)
```

返回给定维`dim`中`input`张量的每一行的求和指数的对数。计算在数值上是稳定的。

对于由`dim`和其他指数 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/31df9c730e19ca29b59dce64b99d98c1.jpg) 给出的总和指数 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/d8fdd0e28cfb03738fc5227885ee035a.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/d8fdd0e28cfb03738fc5227885ee035a.jpg) ，结果是

> [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5acb5b22a7a5c1cfbfda7f648a00c656.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5acb5b22a7a5c1cfbfda7f648a00c656.jpg)

如果`keepdim`为`True`，则输出张量与`input`的大小相同，但尺寸为`dim`的大小为1.否则，`dim`被挤压(参见 [`torch.squeeze()`](#torch.squeeze "torch.squeeze"))，导致输出张量比`input`少1个维度。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_或_ _元组python：整数_） - 要减少的维度或维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

```py
Example::
```

```py
>>> a = torch.randn(3, 3)
>>> torch.logsumexp(a, 1)
tensor([ 0.8442,  1.4322,  0.8711])

```

```py
torch.mean()
```

```py
torch.mean(input) → Tensor
```

返回`input`张量中所有元素的平均值。

| 参数： | **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量 |
| --- | --- |

Example:

```py
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.2294, -0.5481,  1.3288]])
>>> torch.mean(a)
tensor(0.3367)

```

```py
torch.mean(input, dim, keepdim=False, out=None) → Tensor
```

返回给定维`dim`中`input`张量的每一行的平均值。如果`dim`是维度列表，请减少所有维度。

如果`keepdim`为`True`，则输出张量与`input`的大小相同，但尺寸为1的尺寸`dim`除外。`dim`被挤压(见[） `torch.squeeze()`](#torch.squeeze "torch.squeeze"))，导致输出张量具有1(或`len(dim)`）更少的维度。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 减少的维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 输出张量是否保留`dim`
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输出张量

Example:

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
 [-0.9644,  1.0131, -0.6549, -1.4279],
 [-0.2951, -1.3350, -0.7694,  0.5600],
 [ 1.0842, -0.9580,  0.3623,  0.2343]])
>>> torch.mean(a, 1)
tensor([-0.0163, -0.5085, -0.4599,  0.1807])
>>> torch.mean(a, 1, True)
tensor([[-0.0163],
 [-0.5085],
 [-0.4599],
 [ 0.1807]])

```

```py
torch.median()
```

```py
torch.median(input) → Tensor
```

返回`input`张量中所有元素的中值。

| Parameters: | **input** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) – the input tensor |
| --- | --- |

Example:

```py
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 1.5219, -1.5212,  0.2202]])
>>> torch.median(a)
tensor(0.2202)

```

```py
torch.median(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)
```

返回给定维`dim`中`input`张量的每一行的中值。还将中值的索引位置返回为`LongTensor`。

默认情况下，`dim`是`input`张量的最后一个维度。

如果`keepdim`为`True`，则输出张量与`input`的尺寸相同，但尺寸为`dim`的尺寸为1.否则，`dim`被挤压(参见 [`torch.squeeze()`](#torch.squeeze "torch.squeeze"))，导致输出张量比`input`少1个维度。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 减少的维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **值** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_） - 输出张量
*   **指数** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _，_ _任选_） - 输出指数张量

Example:

```py
>>> a = torch.randn(4, 5)
>>> a
tensor([[ 0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
 [ 0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
 [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
 [ 1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
>>> torch.median(a, 1)
(tensor([-0.3982,  0.2270,  0.2488,  0.4742]), tensor([ 1,  4,  4,  3]))

```

```py
torch.mode(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)
```

返回给定维`dim`中`input`张量的每一行的模式值。还将模式值的索引位置作为`LongTensor`返回。

By default, `dim` is the last dimension of the `input` tensor.

如果`keepdim`为`True`，则输出张量与`input`的尺寸相同，但尺寸为`dim`的尺寸为1.否则，`dim`被挤压(参见 [`torch.squeeze()`](#torch.squeeze "torch.squeeze"))，导致输出张量的尺寸比`input`少1。

注意

尚未为`torch.cuda.Tensor`定义此功能。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 减少的维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **值** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_） - 输出张量
*   **指数** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _，_ _任选_） - 输出指数张量

Example:

```py
>>> a = torch.randn(4, 5)
>>> a
tensor([[-1.2808, -1.0966, -1.5946, -0.1148,  0.3631],
 [ 1.1395,  1.1452, -0.6383,  0.3667,  0.4545],
 [-0.4061, -0.3074,  0.4579, -1.3514,  1.2729],
 [-1.0130,  0.3546, -1.4689, -0.1254,  0.0473]])
>>> torch.mode(a, 1)
(tensor([-1.5946, -0.6383, -1.3514, -1.4689]), tensor([ 2,  2,  3,  2]))

```

```py
torch.norm(input, p='fro', dim=None, keepdim=False, out=None)
```

返回给定张量的矩阵范数或向量范数。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量

*   **p**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ [_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _，_ _] inf_ _，_ _-inf_ _，_ _'来'__，_ _'nuc'__，_ _任选_） -

    规范的顺序。默认值：`'fro'`可以计算以下规范：

    | ord |矩阵规范|矢量规范| | --- | --- | --- | |没有| Frobenius规范| 2范数| | '来'| Frobenius规范| - | | 'nuc'|核规范| - | |其他|当昏暗是无|时，作为vec规范sum(abs(x） **ord）**(1./ord）|

*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _2元组python：ints_ _，_ _2-list of python：ints_ _，_ _可选_） - 如果是int，将计算向量范数，如果是2元组的int，将计算矩阵范数。如果值为None，则当输入张量仅具有两个维度时将计算矩阵范数，当输入张量仅具有一个维度时将计算向量范数。如果输入张量具有两个以上的维度，则向量范数将应用于最后一个维度。

*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 输出张量是否保留`dim`。如果`dim` = `None`和`out` = `None`，则忽略。默认值：`False`

*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _可选_） - 输出张量。如果`dim` = `None`和`out` = `None`，则忽略。

Example:

```py
>>> import torch
>>> a = torch.arange(9, dtype= torch.float) - 4
>>> b = a.reshape((3, 3))
>>> torch.norm(a)
tensor(7.7460)
>>> torch.norm(b)
tensor(7.7460)
>>> torch.norm(a, float('inf'))
tensor(4.)
>>> torch.norm(b, float('inf'))
tensor([4., 3., 4.])
>>> c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
>>> torch.norm(c, dim=0)
tensor([1.4142, 2.2361, 5.0000])
>>> torch.norm(c, dim=1)
tensor([3.7417, 4.2426])
>>> torch.norm(c, p=1, dim=1)
tensor([6., 6.])
>>> d = torch.arange(8, dtype= torch.float).reshape(2,2,2)
>>> torch.norm(d, dim=(1,2))
tensor([ 3.7417, 11.2250])
>>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
(tensor(3.7417), tensor(11.2250))

```

```py
torch.prod()
```

```py
torch.prod(input, dtype=None) → Tensor
```

返回`input`张量中所有元素的乘积。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。如果指定，则在执行操作之前将输入张量转换为`dtype`。这对于防止数据类型溢出很有用。默认值：无。

Example:

```py
>>> a = torch.randn(1, 3)
>>> a
tensor([[-0.8020,  0.5428, -1.5854]])
>>> torch.prod(a)
tensor(0.6902)

```

```py
torch.prod(input, dim, keepdim=False, dtype=None) → Tensor
```

返回给定维`dim`中`input`张量的每一行的乘积。

If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension `dim` where it is of size 1\. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensor having 1 fewer dimension than `input`.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 减少的维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。如果指定，则在执行操作之前将输入张量转换为`dtype`。这对于防止数据类型溢出很有用。默认值：无。

Example:

```py
>>> a = torch.randn(4, 2)
>>> a
tensor([[ 0.5261, -0.3837],
 [ 1.1857, -0.2498],
 [-1.1646,  0.0705],
 [ 1.1131, -1.0629]])
>>> torch.prod(a, 1)
tensor([-0.2018, -0.2962, -0.0821, -1.1831])

```

```py
torch.std()
```

```py
torch.std(input, unbiased=True) → Tensor
```

返回`input`张量中所有元素的标准偏差。

如果`unbiased`为`False`，则将通过偏差估算器计算标准偏差。否则，将使用贝塞尔的修正。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **无偏** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 是否使用无偏估计

Example:

```py
>>> a = torch.randn(1, 3)
>>> a
tensor([[-0.8166, -1.3802, -0.3560]])
>>> torch.std(a)
tensor(0.5130)

```

```py
torch.std(input, dim, keepdim=False, unbiased=True, out=None) → Tensor
```

返回给定维`dim`中`input`张量的每一行的标准偏差。

If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension `dim` where it is of size 1\. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensor having 1 fewer dimension than `input`.

If `unbiased` is `False`, then the standard-deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 减少的维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **无偏** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 是否使用无偏估计
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.2035,  1.2959,  1.8101, -0.4644],
 [ 1.5027, -0.3270,  0.5905,  0.6538],
 [-1.5745,  1.3330, -0.5596, -0.6548],
 [ 0.1264, -0.5080,  1.6420,  0.1992]])
>>> torch.std(a, dim=1)
tensor([ 1.0311,  0.7477,  1.2204,  0.9087])

```

```py
torch.sum()
```

```py
torch.sum(input, dtype=None) → Tensor
```

返回`input`张量中所有元素的总和。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。如果指定，则在执行操作之前将输入张量转换为`dtype`。这对于防止数据类型溢出很有用。默认值：无。

Example:

```py
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.1133, -0.9567,  0.2958]])
>>> torch.sum(a)
tensor(-0.5475)

```

```py
torch.sum(input, dim, keepdim=False, dtype=None) → Tensor
```

返回给定维`dim`中`input`张量的每一行的总和。如果`dim`是维度列表，请减少所有维度。

If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1\. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s).

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_或_ _元组python：整数_） - 要减少的维度或维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **dtype**  ([`torch.dtype`](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可选） - 返回张量的所需数据类型。如果指定，则在执行操作之前将输入张量转换为`dtype`。这对于防止数据类型溢出很有用。默认值：无。

Example:

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
 [-0.2993,  0.9138,  0.9337, -1.6864],
 [ 0.1132,  0.7892, -0.1003,  0.5688],
 [ 0.3637, -0.9906, -0.4752, -1.5197]])
>>> torch.sum(a, 1)
tensor([-0.4598, -0.1381,  1.3708, -2.6217])
>>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
>>> torch.sum(b, (2, 1))
tensor([  435.,  1335.,  2235.,  3135.])

```

```py
torch.unique(input, sorted=False, return_inverse=False, dim=None)
```

返回输入张量的唯一标量元素作为1-D张量。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **排序** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 是否在返回作为输出之前按升序对唯一元素进行排序。
*   **return_inverse**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 是否还返回原始输入中元素在返回的唯一列表中结束的索引。
*   **dim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 应用唯一的维度。如果是`None`，则返回展平输入的唯一值。默认值：`None`

|返回：|包含张量的张量或元组

＆GT; * **输出** (_Tensor_ )：唯一标量元素的输出列表。 ＆GT; * **inverse_indices**  (_Tensor_ ):(可选）如果`return_inverse`为True，将会有第二个返回的张量(与输入相同的形状），表示原始元素的索引输入映射到输出中;否则，此函数只返回单个张量。

| 返回类型： |  ([Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") ， [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") (可选）） |
| --- | --- |

Example:

```py
>>> output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long))
>>> output
tensor([ 2,  3,  1])

>>> output, inverse_indices = torch.unique(
 torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True)
>>> output
tensor([ 1,  2,  3])
>>> inverse_indices
tensor([ 0,  2,  1,  2])

>>> output, inverse_indices = torch.unique(
 torch.tensor([[1, 3], [2, 3]], dtype=torch.long), sorted=True, return_inverse=True)
>>> output
tensor([ 1,  2,  3])
>>> inverse_indices
tensor([[ 0,  2],
 [ 1,  2]])

```

```py
torch.var()
```

```py
torch.var(input, unbiased=True) → Tensor
```

返回`input`张量中所有元素的方差。

如果`unbiased`是`False`，则通过偏差估计器计算方差。否则，将使用贝塞尔的修正。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **无偏** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 是否使用无偏估计

Example:

```py
>>> a = torch.randn(1, 3)
>>> a
tensor([[-0.3425, -1.2636, -0.4864]])
>>> torch.var(a)
tensor(0.2455)

```

```py
torch.var(input, dim, keepdim=False, unbiased=True, out=None) → Tensor
```

返回给定维`dim`中`input`张量的每一行的方差。

If `keepdim` is `True`, the output tensors are of the same size as `input` except in the dimension `dim` where they are of size 1\. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the outputs tensor having 1 fewer dimension than `input`.

If `unbiased` is `False`, then the variance will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 减少的维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **无偏** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 是否使用无偏估计
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3567,  1.7385, -1.3042,  0.7423],
 [ 1.3436, -0.1015, -0.9834, -0.8438],
 [ 0.6056,  0.1089, -0.3112, -1.4085],
 [-0.7700,  0.6074, -0.1469,  0.7777]])
>>> torch.var(a, 1)
tensor([ 1.7444,  1.1363,  0.7356,  0.5112])

```

