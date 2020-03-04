

### 比较行动

> 译者：[ApacheCN](https://github.com/apachecn)

```py
torch.allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False) → bool
```

此函数检查所有`self`和`other`是否满足条件：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5dd778c75ab74a08d08025aa32f15e20.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5dd778c75ab74a08d08025aa32f15e20.jpg)

元素，对于`self`和`other`的所有元素。此函数的行为类似于 [numpy.allclose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html)

参数：

*   **自** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 首先进行张量比较
*   **其他** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二张量来比较
*   **atol**  ([_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _，_ _任选_） - 绝对耐受。默认值：1e-08
*   **rtol**  ([_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _，_ _任选_） - 相对耐受。默认值：1e-05
*   **equal_nan**  ([_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _，_ _任选_） - 如果`True`，那么两个`NaN` s将是比较平等。默认值：`False`

例：

```py
>>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
False
>>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
True
>>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
False
>>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
True

```

```py
torch.argsort(input, dim=None, descending=False)
```

返回按值按升序对给定维度的张量进行排序的索引。

这是 [`torch.sort()`](#torch.sort "torch.sort") 返回的第二个值。有关此方法的确切语义，请参阅其文档。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _可选_） - 排序的维度
*   **降序** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 控制排序顺序(升序或降序）

Example:

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
 [ 0.1598,  0.0788, -0.0745, -1.2700],
 [ 1.2208,  1.0722, -0.7064,  1.2564],
 [ 0.0669, -0.2318, -0.8229, -0.9280]])

>>> torch.argsort(a, dim=1)
tensor([[2, 0, 3, 1],
 [3, 2, 1, 0],
 [2, 1, 0, 3],
 [3, 2, 1, 0]])

```

```py
torch.eq(input, other, out=None) → Tensor
```

计算元素明确的平等

第二个参数可以是数字或张量，其形状为[可广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)的第一个参数。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要比较的张量
*   **其他** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _或_ [_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) - 张量或值比较
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _可选_） - 输出张量。必须是`ByteTensor`

| 返回： | 在比较为真的每个位置包含1的`torch.ByteTensor` |
| --- | --- |
| 返回类型： | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 1,  0],
 [ 0,  1]], dtype=torch.uint8)

```

```py
torch.equal(tensor1, tensor2) → bool
```

`True`如果两个张量具有相同的尺寸和元素，则`False`。

Example:

```py
>>> torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
True

```

```py
torch.ge(input, other, out=None) → Tensor
```

按元素计算 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/7cedf1d52401b16530bccf12f96edd5a.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/7cedf1d52401b16530bccf12f96edd5a.jpg) 。

The second argument can be a number or a tensor whose shape is [broadcastable](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要比较的张量
*   **其他** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _或_ [_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) - 张量或值比较
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量必须是`ByteTensor`

| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 1,  1],
 [ 0,  1]], dtype=torch.uint8)

```

```py
torch.gt(input, other, out=None) → Tensor
```

按元素计算 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9d325c4b6e2e06380eb65ceae1e84d76.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9d325c4b6e2e06380eb65ceae1e84d76.jpg) 。

The second argument can be a number or a tensor whose shape is [broadcastable](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要比较的张量
*   **其他** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _或_ [_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) - 张量或值比较
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量必须是`ByteTensor`

| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 0,  1],
 [ 0,  0]], dtype=torch.uint8)

```

```py
torch.isfinite(tensor)
```

返回一个新的张量，其布尔元素表示每个元素是否为`Finite`。

| 参数： | **张量** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 张量来检查 |
| --- | --- |
| 返回： | `torch.ByteTensor`在有限元的每个位置包含1，否则为0 |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
tensor([ 1,  0,  1,  0,  0], dtype=torch.uint8)

```

```py
torch.isinf(tensor)
```

返回一个新的张量，其布尔元素表示每个元素是否为`+/-INF`。

| Parameters: | **tensor** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) – A tensor to check |
| --- | --- |
| Returns: | `torch.ByteTensor`在`+/-INF`元素的每个位置包含1，否则为0 |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
tensor([ 0,  1,  0,  1,  0], dtype=torch.uint8)

```

```py
torch.isnan(tensor)
```

返回一个新的张量，其布尔元素表示每个元素是否为`NaN`。

| Parameters: | **tensor** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) – A tensor to check |
| --- | --- |
| Returns: | `torch.ByteTensor`在`NaN`元素的每个位置包含1。 |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.isnan(torch.tensor([1, float('nan'), 2]))
tensor([ 0,  1,  0], dtype=torch.uint8)

```

```py
torch.kthvalue(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor)
```

返回给定维度上给定`input`张量的`k`个最小元素。

如果未给出`dim`，则选择`input`的最后一个尺寸。

返回`(values, indices)`的元组，其中`indices`是维度`dim`中原始`input`张量中第k个最小元素的索引。

如果`keepdim`为`True`，`values`和`indices`张量都与`input`的尺寸相同，但尺寸为`dim`的尺寸除外。否则，`dim`被挤压(见 [`torch.squeeze()`](#torch.squeeze "torch.squeeze"))，导致`values`和`indices`张量的尺寸比`input`张量小1。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **k**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - k为第k个最小元素
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _可选_） - 找到kth值的维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _任选_） - (Tensor，LongTensor）的输出元组可以任意给出用作输出缓冲区

Example:

```py
>>> x = torch.arange(1., 6.)
>>> x
tensor([ 1.,  2.,  3.,  4.,  5.])
>>> torch.kthvalue(x, 4)
(tensor(4.), tensor(3))

>>> x=torch.arange(1.,7.).resize_(2,3)
>>> x
tensor([[ 1.,  2.,  3.],
 [ 4.,  5.,  6.]])
>>> torch.kthvalue(x,2,0,True)
(tensor([[ 4.,  5.,  6.]]), tensor([[ 1,  1,  1]]))

```

```py
torch.le(input, other, out=None) → Tensor
```

按元素计算 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/7cce834f96c0e53d76c3ec5ed63cf099.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/7cce834f96c0e53d76c3ec5ed63cf099.jpg) 。

The second argument can be a number or a tensor whose shape is [broadcastable](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要比较的张量
*   **其他** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _或_ [_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) - 张量或值比较
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量必须是`ByteTensor`

| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 1,  0],
 [ 1,  1]], dtype=torch.uint8)

```

```py
torch.lt(input, other, out=None) → Tensor
```

按元素计算 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/26b8f23e09743e63a71bcf53c650f1a8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/26b8f23e09743e63a71bcf53c650f1a8.jpg) 。

The second argument can be a number or a tensor whose shape is [broadcastable](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要比较的张量
*   **其他** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _或_ [_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) - 张量或值比较
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量必须是`ByteTensor`

| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 0,  0],
 [ 1,  0]], dtype=torch.uint8)

```

```py
torch.max()
```

```py
torch.max(input) → Tensor
```

返回`input`张量中所有元素的最大值。

| Parameters: | **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量 |
| --- | --- |

Example:

```py
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.6763,  0.7445, -2.2369]])
>>> torch.max(a)
tensor(0.7445)

```

```py
torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
```

返回给定维`dim`中`input`张量的每一行的最大值。第二个返回值是找到的每个最大值的索引位置(argmax）。

如果`keepdim`为`True`，则输出张量与`input`的尺寸相同，但尺寸为`dim`的尺寸为1.否则，`dim`被挤压(参见 [`torch.squeeze()`](#torch.squeeze "torch.squeeze"))，导致输出张量的尺寸比`input`少1。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 减少的维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _可选_） - 两个输出张量的结果元组(max，max_indices）

Example:

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
 [ 1.1949, -1.1127, -2.2379, -0.6702],
 [ 1.5717, -0.9207,  0.1297, -1.8768],
 [-0.6172,  1.0036, -0.6060, -0.2432]])
>>> torch.max(a, 1)
(tensor([ 0.8475,  1.1949,  1.5717,  1.0036]), tensor([ 3,  0,  0,  1]))

```

```py
torch.max(input, other, out=None) → Tensor
```

张量`input`的每个元素与张量`other`的对应元素进行比较，并采用逐元素最大值。

`input`和`other`的形状不需要匹配，但它们必须是[可广播的](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/adfdd713f1ca11c4b41b733a5f452f47.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/adfdd713f1ca11c4b41b733a5f452f47.jpg)

注意

当形状不匹配时，返回的输出张量的形状遵循[广播规则](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **其他** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二个输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.2942, -0.7416,  0.2653, -0.1584])
>>> b = torch.randn(4)
>>> b
tensor([ 0.8722, -1.7421, -0.4141, -0.5055])
>>> torch.max(a, b)
tensor([ 0.8722, -0.7416,  0.2653, -0.1584])

```

```py
torch.min()
```

```py
torch.min(input) → Tensor
```

返回`input`张量中所有元素的最小值。

| Parameters: | **input** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) – the input tensor |
| --- | --- |

Example:

```py
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.6750,  1.0857,  1.7197]])
>>> torch.min(a)
tensor(0.6750)

```

```py
torch.min(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
```

返回给定维`dim`中`input`张量的每一行的最小值。第二个返回值是找到的每个最小值的索引位置(argmin）。

If `keepdim` is `True`, the output tensors are of the same size as `input` except in the dimension `dim` where they are of size 1\. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensors having 1 fewer dimension than `input`.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 减少的维度
*   **keepdim**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 输出张量是否保留`dim`
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _任选_） - 两个输出张量的元组(min，min_indices）

Example:

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
 [-1.4644, -0.2635, -0.3651,  0.6134],
 [ 0.2457,  0.0384,  1.0128,  0.7015],
 [-0.1153,  2.9849,  2.1458,  0.5788]])
>>> torch.min(a, 1)
(tensor([-1.1899, -1.4644,  0.0384, -0.1153]), tensor([ 2,  0,  1,  0]))

```

```py
torch.min(input, other, out=None) → Tensor
```

将张量`input`的每个元素与张量`other`的对应元素进行比较，并采用逐元素最小值。返回结果张量。

The shapes of `input` and `other` don’t need to match, but they must be [broadcastable](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics).

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/173d7b4bb5cacbc18349fb16cd6130ab.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/173d7b4bb5cacbc18349fb16cd6130ab.jpg)

Note

When the shapes do not match, the shape of the returned output tensor follows the [broadcasting rules](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics).

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **其他** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二个输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.8137, -1.1740, -0.6460,  0.6308])
>>> b = torch.randn(4)
>>> b
tensor([-0.1369,  0.1555,  0.4019, -0.1929])
>>> torch.min(a, b)
tensor([-0.1369, -1.1740, -0.6460, -0.1929])

```

```py
torch.ne(input, other, out=None) → Tensor
```

按元素计算 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9b7e900c499f533c33aac63d7b487c24.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9b7e900c499f533c33aac63d7b487c24.jpg) 。

The second argument can be a number or a tensor whose shape is [broadcastable](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要比较的张量
*   **其他** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _或_ [_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) - 张量或值比较
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量必须是`ByteTensor`

| Returns: | 在比较为真的每个位置包含1的`torch.ByteTensor`。 |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 0,  1],
 [ 1,  0]], dtype=torch.uint8)

```

```py
torch.sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)
```

按值按升序对给定维度的`input`张量元素进行排序。

If `dim` is not given, the last dimension of the `input` is chosen.

如果`descending`是`True`，则元素按值按降序排序。

返回元组(sorted_tensor，sorted_indices），其中sorted_indices是原始`input`张量中元素的索引。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _可选_） - 排序的维度
*   **降序** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 控制排序顺序(升序或降序）
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _任选_） - (`Tensor`，`LongTensor`）的输出元组可以选择将其用作输出缓冲区

Example:

```py
>>> x = torch.randn(3, 4)
>>> sorted, indices = torch.sort(x)
>>> sorted
tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
 [-0.5793,  0.0061,  0.6058,  0.9497],
 [-0.5071,  0.3343,  0.9553,  1.0960]])
>>> indices
tensor([[ 1,  0,  2,  3],
 [ 3,  1,  0,  2],
 [ 0,  3,  1,  2]])

>>> sorted, indices = torch.sort(x, 0)
>>> sorted
tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
 [ 0.0608,  0.0061,  0.9497,  0.3343],
 [ 0.6058,  0.9553,  1.0960,  2.3332]])
>>> indices
tensor([[ 2,  0,  0,  1],
 [ 0,  1,  1,  2],
 [ 1,  2,  2,  0]])

```

```py
torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
```

返回给定维度上给定`input`张量的`k`最大元素。

If `dim` is not given, the last dimension of the `input` is chosen.

如果`largest`为`False`，则返回`k`最小元素。

返回`(values, indices)`元组，其中`indices`是原始`input`张量中元素的索引。

布尔选项`sorted`如果`True`，将确保返回的`k`元素本身已排序

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **k**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - “top-k”中的k
*   **昏暗** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _可选_） - 排序的维度
*   **最大** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 控制是否返回最大或最小元素
*   **排序** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 控制是否按排序顺序返回元素
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _任选_） - (Tensor，LongTensor）的输出元组，可以选择性给予用作输出缓冲区

Example:

```py
>>> x = torch.arange(1., 6.)
>>> x
tensor([ 1.,  2.,  3.,  4.,  5.])
>>> torch.topk(x, 3)
(tensor([ 5.,  4.,  3.]), tensor([ 4,  3,  2]))

```

