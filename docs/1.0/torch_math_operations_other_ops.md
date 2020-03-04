

### 其他行动

> 译者：[ApacheCN](https://github.com/apachecn)

```py
torch.bincount(self, weights=None, minlength=0) → Tensor
```

计算非负的int数组中每个值的频率。

除非`input`为空，否则箱数(大小为1）比`input`中的最大值大1，在这种情况下，结果是大小为0.如果指定`minlength`，则箱数为至少`minlength`并且如果`input`为空，则结果是填充零的大小`minlength`的张量。如果`n`是位置`i`的值，`out[n] += weights[i]`如果指定了`weights`，则`out[n] += 1`。

注意

使用CUDA后端时，此操作可能会导致不容易关闭的不确定行为。有关背景，请参阅[再现性](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/randomness.html)的注释。

参数：

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 1-d int张量
*   **权重** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 可选，输入张量中每个值的权重。应与输入张量大小相同。
*   **minlength**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 可选的最小二进制数。应该是非负面的。

| 返回： | 如果`input`非空，则为形状张量`Size([max(input) + 1])`，否则为`Size(0)` |
| --- | --- |
| 返回类型： | 输出 ([Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) |

例：

```py
>>> input = torch.randint(0, 8, (5,), dtype=torch.int64)
>>> weights = torch.linspace(0, 1, steps=5)
>>> input, weights
(tensor([4, 3, 6, 3, 4]),
 tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])

>>> torch.bincount(input)
tensor([0, 0, 0, 2, 2, 0, 1])

>>> input.bincount(weights)
tensor([0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.5000])

```

```py
torch.broadcast_tensors(*tensors) → List of Tensors
```

根据_broadcasting-semantics广播给定的张量。

| 参数： | * **张量** - 任何数量的相同类型的张量 |
| --- | --- |

Example:

```py
>>> x = torch.arange(3).view(1, 3)
>>> y = torch.arange(2).view(2, 1)
>>> a, b = torch.broadcast_tensors(x, y)
>>> a.size()
torch.Size([2, 3])
>>> a
tensor([[0, 1, 2],
 [0, 1, 2]])

```

```py
torch.cross(input, other, dim=-1, out=None) → Tensor
```

返回`input`和`other`的维度`dim`中矢量的叉积。

`input`和`other`必须具有相同的尺寸，并且`dim`尺寸的大小应为3。

如果未给出`dim`，则默认为找到大小为3的第一个维度。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **其他** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二个输入张量
*   **dim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 采取交叉积的维度。
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4, 3)
>>> a
tensor([[-0.3956,  1.1455,  1.6895],
 [-0.5849,  1.3672,  0.3599],
 [-1.1626,  0.7180, -0.0521],
 [-0.1339,  0.9902, -2.0225]])
>>> b = torch.randn(4, 3)
>>> b
tensor([[-0.0257, -1.4725, -1.2251],
 [-1.1479, -0.7005, -1.9757],
 [-1.3904,  0.3726, -1.1836],
 [-0.9688, -0.7153,  0.2159]])
>>> torch.cross(a, b, dim=1)
tensor([[ 1.0844, -0.5281,  0.6120],
 [-2.4490, -1.5687,  1.9792],
 [-0.8304, -1.3037,  0.5650],
 [-1.2329,  1.9883,  1.0551]])
>>> torch.cross(a, b)
tensor([[ 1.0844, -0.5281,  0.6120],
 [-2.4490, -1.5687,  1.9792],
 [-0.8304, -1.3037,  0.5650],
 [-1.2329,  1.9883,  1.0551]])

```

```py
torch.diag(input, diagonal=0, out=None) → Tensor
```

*   如果`input`是矢量(1-D张量），则返回2-D平方张量，其中`input`的元素作为对角线。
*   如果`input`是矩阵(2-D张量），则返回具有`input`的对角元素的1-D张量。

参数 [`diagonal`](#torch.diagonal "torch.diagonal") 控制要考虑的对角线：

*   如果 [`diagonal`](#torch.diagonal "torch.diagonal") = 0，则它是主对角线。
*   如果 [`diagonal`](#torch.diagonal "torch.diagonal") &gt; 0，它在主对角线上方。
*   如果 [`diagonal`](#torch.diagonal "torch.diagonal") ＆lt; 0，它在主对角线下面。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **对角线** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _可选_） - 要考虑的对角线
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

也可以看看

[`torch.diagonal()`](#torch.diagonal "torch.diagonal") 始终返回其输入的对角线。

[`torch.diagflat()`](#torch.diagflat "torch.diagflat") 总是构造一个由输入指定的对角元素的张量。

例子：

获取输入向量为对角线的方阵：

```py
>>> a = torch.randn(3)
>>> a
tensor([ 0.5950,-0.0872, 2.3298])
>>> torch.diag(a)
tensor([[ 0.5950, 0.0000, 0.0000],
 [ 0.0000,-0.0872, 0.0000],
 [ 0.0000, 0.0000, 2.3298]])
>>> torch.diag(a, 1)
tensor([[ 0.0000, 0.5950, 0.0000, 0.0000],
 [ 0.0000, 0.0000,-0.0872, 0.0000],
 [ 0.0000, 0.0000, 0.0000, 2.3298],
 [ 0.0000, 0.0000, 0.0000, 0.0000]])

```

获取给定矩阵的第k个对角线：

```py
>>> a = torch.randn(3, 3)
>>> a
tensor([[-0.4264, 0.0255,-0.1064],
 [ 0.8795,-0.2429, 0.1374],
 [ 0.1029,-0.6482,-1.6300]])
>>> torch.diag(a, 0)
tensor([-0.4264,-0.2429,-1.6300])
>>> torch.diag(a, 1)
tensor([ 0.0255, 0.1374])

```

```py
torch.diag_embed(input, offset=0, dim1=-2, dim2=-1) → Tensor
```

创建一个张量，其某些2D平面的对角线(由`dim1`和`dim2`指定）由`input`填充。为了便于创建批量对角矩阵，默认选择由返回张量的最后两个维度形成的2D平面。

参数`offset`控制要考虑的对角线：

*   如果`offset` = 0，则它是主对角线。
*   如果`offset`&gt; 0，它在主对角线上方。
*   如果`offset`＆lt; 0，它在主对角线下面。

将计算新矩阵的大小以使得指定的对角线具有最后输入维度的大小。注意，对于 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/28256dd5af833c877d63bfabfaa7b301.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/28256dd5af833c877d63bfabfaa7b301.jpg) 以外的`offset`，`dim1`和`dim2`的顺序很重要。交换它们相当于改变`offset`的符号。

将 [`torch.diagonal()`](#torch.diagonal "torch.diagonal") 应用于具有相同参数的此函数的输出，将产生与输入相同的矩阵。但是， [`torch.diagonal()`](#torch.diagonal "torch.diagonal") 具有不同的默认尺寸，因此需要明确指定。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量。必须至少是一维的。
*   **偏移** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 对角线考虑。默认值：0(主对角线）。
*   **dim1**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 相对于其采取对角线的第一维度。默认值：-2。
*   **dim2**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 相对于其采取对角线的第二维度。默认值：-1。

Example:

```py
>>> a = torch.randn(2, 3)
>>> torch.diag_embed(a)
tensor([[[ 1.5410,  0.0000,  0.0000],
 [ 0.0000, -0.2934,  0.0000],
 [ 0.0000,  0.0000, -2.1788]],

 [[ 0.5684,  0.0000,  0.0000],
 [ 0.0000, -1.0845,  0.0000],
 [ 0.0000,  0.0000, -1.3986]]])

>>> torch.diag_embed(a, offset=1, dim1=0, dim2=2)
tensor([[[ 0.0000,  1.5410,  0.0000,  0.0000],
 [ 0.0000,  0.5684,  0.0000,  0.0000]],

 [[ 0.0000,  0.0000, -0.2934,  0.0000],
 [ 0.0000,  0.0000, -1.0845,  0.0000]],

 [[ 0.0000,  0.0000,  0.0000, -2.1788],
 [ 0.0000,  0.0000,  0.0000, -1.3986]],

 [[ 0.0000,  0.0000,  0.0000,  0.0000],
 [ 0.0000,  0.0000,  0.0000,  0.0000]]])

```

```py
torch.diagflat(input, diagonal=0) → Tensor
```

*   如果`input`是矢量(1-D张量），则返回2-D平方张量，其中`input`的元素作为对角线。
*   如果`input`是一个具有多个维度的张量，则返回一个二维张量，其对角线元素等于一个展平的`input`。

The argument `offset` controls which diagonal to consider:

*   如果`offset` = 0，则它是主对角线。
*   如果`offset`&gt; 0，它在主对角线上方。
*   如果`offset`＆lt; 0，它在主对角线下面。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **偏移** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 对角线考虑。默认值：0(主对角线）。

Examples:

```py
>>> a = torch.randn(3)
>>> a
tensor([-0.2956, -0.9068,  0.1695])
>>> torch.diagflat(a)
tensor([[-0.2956,  0.0000,  0.0000],
 [ 0.0000, -0.9068,  0.0000],
 [ 0.0000,  0.0000,  0.1695]])
>>> torch.diagflat(a, 1)
tensor([[ 0.0000, -0.2956,  0.0000,  0.0000],
 [ 0.0000,  0.0000, -0.9068,  0.0000],
 [ 0.0000,  0.0000,  0.0000,  0.1695],
 [ 0.0000,  0.0000,  0.0000,  0.0000]])

>>> a = torch.randn(2, 2)
>>> a
tensor([[ 0.2094, -0.3018],
 [-0.1516,  1.9342]])
>>> torch.diagflat(a)
tensor([[ 0.2094,  0.0000,  0.0000,  0.0000],
 [ 0.0000, -0.3018,  0.0000,  0.0000],
 [ 0.0000,  0.0000, -0.1516,  0.0000],
 [ 0.0000,  0.0000,  0.0000,  1.9342]])

```

```py
torch.diagonal(input, offset=0, dim1=0, dim2=1) → Tensor
```

返回`input`的局部视图，其对角线元素相对于`dim1`和`dim2`作为形状末尾的尺寸附加。

The argument `offset` controls which diagonal to consider:

*   如果`offset` = 0，则它是主对角线。
*   如果`offset`&gt; 0，它在主对角线上方。
*   如果`offset`＆lt; 0，它在主对角线下面。

将 [`torch.diag_embed()`](#torch.diag_embed "torch.diag_embed") 应用于具有相同参数的此函数的输出，将生成带有输入对角线条目的对角矩阵。但是， [`torch.diag_embed()`](#torch.diag_embed "torch.diag_embed") 具有不同的默认尺寸，因此需要明确指定。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量。必须至少是二维的。
*   **偏移** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 对角线考虑。默认值：0(主对角线）。
*   **dim1**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 相对于其采取对角线的第一维度。默认值：0。
*   **dim2**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _任选_） - 相对于其采取对角线的第二维度。默认值：1。

Note

要采用批对角线，传入dim1 = -2，dim2 = -1。

Examples:

```py
>>> a = torch.randn(3, 3)
>>> a
tensor([[-1.0854,  1.1431, -0.1752],
 [ 0.8536, -0.0905,  0.0360],
 [ 0.6927, -0.3735, -0.4945]])

>>> torch.diagonal(a, 0)
tensor([-1.0854, -0.0905, -0.4945])

>>> torch.diagonal(a, 1)
tensor([ 1.1431,  0.0360])

>>> x = torch.randn(2, 5, 4, 2)
>>> torch.diagonal(x, offset=-1, dim1=1, dim2=2)
tensor([[[-1.2631,  0.3755, -1.5977, -1.8172],
 [-1.1065,  1.0401, -0.2235, -0.7938]],

 [[-1.7325, -0.3081,  0.6166,  0.2335],
 [ 1.0500,  0.7336, -0.3836, -1.1015]]])

```

```py
torch.einsum(equation, *operands) → Tensor
```

该函数提供了一种使用爱因斯坦求和约定来计算多线性表达式(即乘积和）的方法。

Parameters:

*   **等式** (_string_ ) - 该等式根据与操作数和结果的每个维度相关联的小写字母(索引）给出。左侧列出了操作数尺寸，以逗号分隔。每个张量维度应该有一个索引字母。右侧跟在`-&gt;`之后，并给出输出的索引。如果省略`-&gt;`和右侧，则它隐式地定义为在左侧恰好出现一次的所有索引的按字母顺序排序的列表。在操作数输入之后，将输出中未显示的索引求和。如果索引对同一操作数多次出现，则采用对角线。省略号`…`表示固定数量的维度。如果推断出右侧，则省略号维度位于输出的开头。
*   **操作数**(_张量列表_） - 计算爱因斯坦和的操作数。请注意，操作数作为列表传递，而不是作为单个参数传递。

Examples:

```py
>>> x = torch.randn(5)
>>> y = torch.randn(4)
>>> torch.einsum('i,j->ij', x, y)  # outer product
tensor([[-0.0570, -0.0286, -0.0231,  0.0197],
 [ 1.2616,  0.6335,  0.5113, -0.4351],
 [ 1.4452,  0.7257,  0.5857, -0.4984],
 [-0.4647, -0.2333, -0.1883,  0.1603],
 [-1.1130, -0.5588, -0.4510,  0.3838]])

>>> A = torch.randn(3,5,4)
>>> l = torch.randn(2,5)
>>> r = torch.randn(2,4)
>>> torch.einsum('bn,anm,bm->ba', l, A, r) # compare torch.nn.functional.bilinear
tensor([[-0.3430, -5.2405,  0.4494],
 [ 0.3311,  5.5201, -3.0356]])

>>> As = torch.randn(3,2,5)
>>> Bs = torch.randn(3,5,4)
>>> torch.einsum('bij,bjk->bik', As, Bs) # batch matrix multiplication
tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
 [-1.6706, -0.8097, -0.8025, -2.1183]],

 [[ 4.2239,  0.3107, -0.5756, -0.2354],
 [-1.4558, -0.3460,  1.5087, -0.8530]],

 [[ 2.8153,  1.8787, -4.3839, -1.2112],
 [ 0.3728, -2.1131,  0.0921,  0.8305]]])

>>> A = torch.randn(3, 3)
>>> torch.einsum('ii->i', A) # diagonal
tensor([-0.7825,  0.8291, -0.1936])

>>> A = torch.randn(4, 3, 3)
>>> torch.einsum('...ii->...i', A) # batch diagonal
tensor([[-1.0864,  0.7292,  0.0569],
 [-0.9725, -1.0270,  0.6493],
 [ 0.5832, -1.1716, -1.5084],
 [ 0.4041, -1.1690,  0.8570]])

>>> A = torch.randn(2, 3, 4, 5)
>>> torch.einsum('...ij->...ji', A).shape # batch permute
torch.Size([2, 3, 5, 4])

```

```py
torch.flatten(input, start_dim=0, end_dim=-1) → Tensor
```

在张量中展平连续的一系列变暗。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **start_dim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 第一个暗淡变平
*   **end_dim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 最后的暗淡变平

Example:

```py
>>> t = torch.tensor([[[1, 2],
 [3, 4]],
 [[5, 6],
 [7, 8]]])
>>> torch.flatten(t)
tensor([1, 2, 3, 4, 5, 6, 7, 8])
>>> torch.flatten(t, start_dim=1)
tensor([[1, 2, 3, 4],
 [5, 6, 7, 8]])

```

```py
torch.flip(input, dims) → Tensor
```

在dims中沿给定轴反转n-D张量的顺序。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **暗淡**(_一个列表_ _或_ [_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") - 轴要翻转

Example:

```py
>>> x = torch.arange(8).view(2, 2, 2)
>>> x
tensor([[[ 0,  1],
 [ 2,  3]],

 [[ 4,  5],
 [ 6,  7]]])
>>> torch.flip(x, [0, 1])
tensor([[[ 6,  7],
 [ 4,  5]],

 [[ 2,  3],
 [ 0,  1]]])

```

```py
torch.histc(input, bins=100, min=0, max=0, out=None) → Tensor
```

计算张量的直方图。

元素在 [`min`](#torch.min "torch.min") 和 [`max`](#torch.max "torch.max") 之间分成相等的宽度区间。如果 [`min`](#torch.min "torch.min") 和 [`max`](#torch.max "torch.max") 都为零，则使用数据的最小值和最大值。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **箱** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 直方图箱数
*   **min**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 范围的下限(含）
*   **max**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 范围的上限(含）
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

| Returns: | 直方图表示为张量 |
| --- | --- |
| Return type: | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

Example:

```py
>>> torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)
tensor([ 0.,  2.,  1.,  0.])

```

```py
torch.meshgrid(*tensors, **kwargs)
```

取 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) 张量，每个张量可以是标量或1维向量，并创建 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) N维网格，其中：math：[`](#id2)i`通过扩展：math：[`](#id4)i` th输入定义由其他输入定义的维度来定义网格。

> ```py
> Args:
> ```
> 
> 张量(Tensor列表）：标量列表或1维张量。标量将被自动视为大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/e800eead21f1007b4005a268169586f7.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/e800eead21f1007b4005a268169586f7.jpg) 的张量
> 
> ```py
> Returns:
> ```
> 
> seq(张量序列）：如果输入的 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a1c2f8d5b1226e67bdb44b12a6ddf18b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a1c2f8d5b1226e67bdb44b12a6ddf18b.jpg) 张量大小为 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5871a78f7096a5c43c0b08b090b8c4f1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5871a78f7096a5c43c0b08b090b8c4f1.jpg) ，那么输出也会有 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a1c2f8d5b1226e67bdb44b12a6ddf18b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a1c2f8d5b1226e67bdb44b12a6ddf18b.jpg) 张量，其中所有张量均为 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/771af95b2d780e68358b78d5124091fa.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/771af95b2d780e68358b78d5124091fa.jpg) 。
> 
> Example:
> 
> ```py
> &gt;&gt;&gt; x = torch.tensor([1, 2, 3])
> &gt;&gt;&gt; y = torch.tensor([4, 5, 6])
> &gt;&gt;&gt; grid_x, grid_y = torch.meshgrid(x, y)
> &gt;&gt;&gt; grid_x
> tensor([[1, 1, 1],
>  [2, 2, 2],
>  [3, 3, 3]])
> &gt;&gt;&gt; grid_y
> tensor([[4, 5, 6],
>  [4, 5, 6],
>  [4, 5, 6]])
> 
> ```

```py
torch.renorm(input, p, dim, maxnorm, out=None) → Tensor
```

返回张量，其中沿着维度`dim`的`input`的每个子张量被归一化，使得子张量的`p` - 范数低于值`maxnorm`

Note

如果行的范数低于`maxnorm`，则该行不变

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **p**  ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")） - 规范计算的动力
*   **dim**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 切片以获得子张量的维数
*   **maxnorm**  ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")） - 保持每个子张量的最大范数
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> x = torch.ones(3, 3)
>>> x[1].fill_(2)
tensor([ 2.,  2.,  2.])
>>> x[2].fill_(3)
tensor([ 3.,  3.,  3.])
>>> x
tensor([[ 1.,  1.,  1.],
 [ 2.,  2.,  2.],
 [ 3.,  3.,  3.]])
>>> torch.renorm(x, 1, 0, 5)
tensor([[ 1.0000,  1.0000,  1.0000],
 [ 1.6667,  1.6667,  1.6667],
 [ 1.6667,  1.6667,  1.6667]])

```

```py
torch.tensordot(a, b, dims=2)
```

返回多维度上a和b的收缩。

[`tensordot`](#torch.tensordot "torch.tensordot") 实现了矩阵乘积的推广。

Parameters:

*   **a**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 左张量收缩
*   **b**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 右张量收缩
*   **暗淡** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_或_ _元组的两个python列表：整数_） - 要收缩的维数或`a`和`b`的明确维度列表

当用整数参数`dims` = [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg) 调用时，`a`和`b`的维数是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg) 和 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg) ，它分别计算

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/99b9f5b0cd445ecb19920e143987f33e.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/99b9f5b0cd445ecb19920e143987f33e.jpg)

当使用列表形式的`dims`调用时，将收缩给定的维度来代替`a`的最后 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg) 和[的第一个](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6872867a863714d15d9a0d64c20734ce.jpg) [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9566974d45a96737f7e0ecf302d877b8.jpg) ] ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6872867a863714d15d9a0d64c20734ce.jpg) 。这些尺寸的尺寸必须匹配，但 [`tensordot`](#torch.tensordot "torch.tensordot") 将处理广播尺寸。

Examples:

```py
>>> a = torch.arange(60.).reshape(3, 4, 5)
>>> b = torch.arange(24.).reshape(4, 3, 2)
>>> torch.tensordot(a, b, dims=([1, 0], [0, 1]))
tensor([[4400., 4730.],
 [4532., 4874.],
 [4664., 5018.],
 [4796., 5162.],
 [4928., 5306.]])

>>> a = torch.randn(3, 4, 5, device='cuda')
>>> b = torch.randn(4, 5, 6, device='cuda')
>>> c = torch.tensordot(a, b, dims=2).cpu()
tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
 [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
 [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])

```

```py
torch.trace(input) → Tensor
```

返回输入2-D矩阵的对角线元素的总和。

Example:

```py
>>> x = torch.arange(1., 10.).view(3, 3)
>>> x
tensor([[ 1.,  2.,  3.],
 [ 4.,  5.,  6.],
 [ 7.,  8.,  9.]])
>>> torch.trace(x)
tensor(15.)

```

```py
torch.tril(input, diagonal=0, out=None) → Tensor
```

返回矩阵的下三角部分(2-D张量）`input`，结果张量`out`的其他元素设置为0。

矩阵的下三角形部分被定义为对角线上和下方的元素。

参数 [`diagonal`](#torch.diagonal "torch.diagonal") 控制要考虑的对角线。如果 [`diagonal`](#torch.diagonal "torch.diagonal") = 0，则保留主对角线上和下方的所有元素。正值包括主对角线上方的对角线数量，同样负值也不包括主对角线下方的对角线数量。主对角线是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1bfb3770a124b38b3aba63186b7c8f46.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1bfb3770a124b38b3aba63186b7c8f46.jpg) 的指数 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6291ea635817db74920cd048cc3cb8d4.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6291ea635817db74920cd048cc3cb8d4.jpg) 的集合，其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5ccb16cb4e75340b0c2b2d022fd778a7.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5ccb16cb4e75340b0c2b2d022fd778a7.jpg) 是基质的维度。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **对角线** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _可选_） - 要考虑的对角线
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(3, 3)
>>> a
tensor([[-1.0813, -0.8619,  0.7105],
 [ 0.0935,  0.1380,  2.2112],
 [-0.3409, -0.9828,  0.0289]])
>>> torch.tril(a)
tensor([[-1.0813,  0.0000,  0.0000],
 [ 0.0935,  0.1380,  0.0000],
 [-0.3409, -0.9828,  0.0289]])

>>> b = torch.randn(4, 6)
>>> b
tensor([[ 1.2219,  0.5653, -0.2521, -0.2345,  1.2544,  0.3461],
 [ 0.4785, -0.4477,  0.6049,  0.6368,  0.8775,  0.7145],
 [ 1.1502,  3.2716, -1.1243, -0.5413,  0.3615,  0.6864],
 [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0978]])
>>> torch.tril(b, diagonal=1)
tensor([[ 1.2219,  0.5653,  0.0000,  0.0000,  0.0000,  0.0000],
 [ 0.4785, -0.4477,  0.6049,  0.0000,  0.0000,  0.0000],
 [ 1.1502,  3.2716, -1.1243, -0.5413,  0.0000,  0.0000],
 [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0000]])
>>> torch.tril(b, diagonal=-1)
tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
 [ 0.4785,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
 [ 1.1502,  3.2716,  0.0000,  0.0000,  0.0000,  0.0000],
 [-0.0614, -0.7344, -1.3164,  0.0000,  0.0000,  0.0000]])

```

```py
torch.triu(input, diagonal=0, out=None) → Tensor
```

返回矩阵的上三角部分(2-D张量）`input`，结果张量`out`的其他元素设置为0。

矩阵的上三角形部分被定义为对角线上方和上方的元素。

参数 [`diagonal`](#torch.diagonal "torch.diagonal") 控制要考虑的对角线。如果 [`diagonal`](#torch.diagonal "torch.diagonal") = 0，则保留主对角线上和下方的所有元素。正值排除了主对角线上方的对角线数量，同样负值也包括主对角线下方的对角线数量。主对角线是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1bfb3770a124b38b3aba63186b7c8f46.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1bfb3770a124b38b3aba63186b7c8f46.jpg) 的指数 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6291ea635817db74920cd048cc3cb8d4.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6291ea635817db74920cd048cc3cb8d4.jpg) 的集合，其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5ccb16cb4e75340b0c2b2d022fd778a7.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5ccb16cb4e75340b0c2b2d022fd778a7.jpg) 是基质的维度。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **对角线** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_，_ _可选_） - 要考虑的对角线
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(3, 3)
>>> a
tensor([[ 0.2309,  0.5207,  2.0049],
 [ 0.2072, -1.0680,  0.6602],
 [ 0.3480, -0.5211, -0.4573]])
>>> torch.triu(a)
tensor([[ 0.2309,  0.5207,  2.0049],
 [ 0.0000, -1.0680,  0.6602],
 [ 0.0000,  0.0000, -0.4573]])
>>> torch.triu(a, diagonal=1)
tensor([[ 0.0000,  0.5207,  2.0049],
 [ 0.0000,  0.0000,  0.6602],
 [ 0.0000,  0.0000,  0.0000]])
>>> torch.triu(a, diagonal=-1)
tensor([[ 0.2309,  0.5207,  2.0049],
 [ 0.2072, -1.0680,  0.6602],
 [ 0.0000, -0.5211, -0.4573]])

>>> b = torch.randn(4, 6)
>>> b
tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
 [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
 [ 0.4333,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
 [-0.9888,  1.0679, -1.3337, -1.6556,  0.4798,  0.2830]])
>>> torch.tril(b, diagonal=1)
tensor([[ 0.5876, -0.0794,  0.0000,  0.0000,  0.0000,  0.0000],
 [-0.2447,  0.9556, -1.2919,  0.0000,  0.0000,  0.0000],
 [ 0.4333,  0.3146,  0.6576, -1.0432,  0.0000,  0.0000],
 [-0.9888,  1.0679, -1.3337, -1.6556,  0.4798,  0.0000]])
>>> torch.tril(b, diagonal=-1)
tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
 [-0.2447,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
 [ 0.4333,  0.3146,  0.0000,  0.0000,  0.0000,  0.0000],
 [-0.9888,  1.0679, -1.3337,  0.0000,  0.0000,  0.0000]])

```

