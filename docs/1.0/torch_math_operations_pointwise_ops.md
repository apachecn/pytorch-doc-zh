

### 逐点行动

> 译者：[ApacheCN](https://github.com/apachecn)

```py
torch.abs(input, out=None) → Tensor
```

计算给定`input`张量的逐元素绝对值。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1a4bcc75ec995f7b04a37cccd88b214b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1a4bcc75ec995f7b04a37cccd88b214b.jpg)

参数：

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

例：

```py
>>> torch.abs(torch.tensor([-1, -2, 3]))
tensor([ 1,  2,  3])

```

```py
torch.acos(input, out=None) → Tensor
```

返回带有`input`元素的反余弦的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/3533abc4adcb633e8fb0bfc683c437bb.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/3533abc4adcb633e8fb0bfc683c437bb.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.3348, -0.5889,  0.2005, -0.1584])
>>> torch.acos(a)
tensor([ 1.2294,  2.2004,  1.3690,  1.7298])

```

```py
torch.add()
```

```py
torch.add(input, value, out=None)
```

将标量`value`添加到输入`input`的每个元素并返回新的结果张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/76a7103f78e0443c4ad36bbf203db638.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/76a7103f78e0443c4ad36bbf203db638.jpg)

如果`input`的类型为FloatTensor或DoubleTensor，则`value`必须是实数，否则应为整数。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **值**(_号码_） - 要添加到`input`的每个元素的数字

| 关键字参数： |
| --- |
| ？ |

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
>>> torch.add(a, 20)
tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

```

```py
torch.add(input, value=1, other, out=None)
```

张量`other`的每个元素乘以标量`value`并添加到张量`input`的每个元素。返回结果张量。

`input`和`other`的形状必须是[可播放的](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/bc929133d25d93686f6106f171de0de3.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/bc929133d25d93686f6106f171de0de3.jpg)

如果`other`的类型为FloatTensor或DoubleTensor，则`value`必须是实数，否则应为整数。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第一个输入张量
*   **值**(_数字_） - `other`的标量乘数
*   **其他** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二个输入张量

| Keyword Arguments: |
| --- |
| ? |

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-0.9732, -0.3497,  0.6245,  0.4022])
>>> b = torch.randn(4, 1)
>>> b
tensor([[ 0.3743],
 [-1.7724],
 [-0.5811],
 [-0.8017]])
>>> torch.add(a, 10, b)
tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
 [-18.6971, -18.0736, -17.0994, -17.3216],
 [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
 [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])

```

```py
torch.addcdiv(tensor, value=1, tensor1, tensor2, out=None) → Tensor
```

通过`tensor2`执行`tensor1`的逐元素划分，将结果乘以标量`value`并将其添加到 [`tensor`](#torch.tensor "torch.tensor") 。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/e6bfafde43b0e449b24255b208acc8e0.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/e6bfafde43b0e449b24255b208acc8e0.jpg)

[`tensor`](#torch.tensor "torch.tensor") ，`tensor1`和`tensor2`的形状必须是[可播放的](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

对于`FloatTensor`或`DoubleTensor`类型的输入，`value`必须是实数，否则是整数。

Parameters:

*   **张量** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要加的张量
*   **值**(_数_ _，_ _可选_） - [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1b168c1663790fbd38202af8bfea37bc.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1b168c1663790fbd38202af8bfea37bc.jpg) 的乘数
*   **tensor1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 分子张量
*   **张量2**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 分母张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> t = torch.randn(1, 3)
>>> t1 = torch.randn(3, 1)
>>> t2 = torch.randn(1, 3)
>>> torch.addcdiv(t, 0.1, t1, t2)
tensor([[-0.2312, -3.6496,  0.1312],
 [-1.0428,  3.4292, -0.1030],
 [-0.5369, -0.9829,  0.0430]])

```

```py
torch.addcmul(tensor, value=1, tensor1, tensor2, out=None) → Tensor
```

通过`tensor2`执行`tensor1`的逐元素乘法，将结果乘以标量`value`并将其添加到 [`tensor`](#torch.tensor "torch.tensor") 。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ab575ca50c2fce8e335280dff71f26b0.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ab575ca50c2fce8e335280dff71f26b0.jpg)

The shapes of [`tensor`](#torch.tensor "torch.tensor"), `tensor1`, and `tensor2` must be [broadcastable](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics).

For inputs of type `FloatTensor` or `DoubleTensor`, `value` must be a real number, otherwise an integer.

Parameters:

*   **张量** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要加的张量
*   **值**(_数_ _，_ _可选_） - [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b345d1a46cebf2308e450926de3195ef.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b345d1a46cebf2308e450926de3195ef.jpg) 的乘数
*   **tensor1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要倍增的张量
*   **tensor2**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要倍增的张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> t = torch.randn(1, 3)
>>> t1 = torch.randn(3, 1)
>>> t2 = torch.randn(1, 3)
>>> torch.addcmul(t, 0.1, t1, t2)
tensor([[-0.8635, -0.6391,  1.6174],
 [-0.7617, -0.5879,  1.7388],
 [-0.8353, -0.6249,  1.6511]])

```

```py
torch.asin(input, out=None) → Tensor
```

返回具有`input`元素的反正弦的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/eb9c70310a0ed5b865beb34bc1e28a99.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/eb9c70310a0ed5b865beb34bc1e28a99.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-0.5962,  1.4985, -0.4396,  1.4525])
>>> torch.asin(a)
tensor([-0.6387,     nan, -0.4552,     nan])

```

```py
torch.atan(input, out=None) → Tensor
```

返回带有`input`元素反正切的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/cd3367165f341b3ab7dd3ee6dcfbb92c.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/cd3367165f341b3ab7dd3ee6dcfbb92c.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
>>> torch.atan(a)
tensor([ 0.2299,  0.2487, -0.5591, -0.5727])

```

```py
torch.atan2(input1, input2, out=None) → Tensor
```

返回带有`input1`和`input2`元素的反正切的新张量。

`input1`和`input2`的形状必须是[可播放的](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

Parameters:

*   **input1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第一个输入张量
*   **input2**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二个输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
>>> torch.atan2(a, torch.randn(4))
tensor([ 0.9833,  0.0811, -1.9743, -1.4151])

```

```py
torch.ceil(input, out=None) → Tensor
```

返回具有`input`元素的ceil的新张量，该元素是大于或等于每个元素的最小整数。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efa1e00e060e0787c8b7ea48fe74745d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efa1e00e060e0787c8b7ea48fe74745d.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-0.6341, -1.4208, -1.0900,  0.5826])
>>> torch.ceil(a)
tensor([-0., -1., -1.,  1.])

```

```py
torch.clamp(input, min, max, out=None) → Tensor
```

将`input`中的所有元素钳位到`[` [`min`](#torch.min "torch.min") ， [`max`](#torch.max "torch.max") `]`范围内并返回结果张量：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a116639ce05e419a65971fdffeaa2d81.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a116639ce05e419a65971fdffeaa2d81.jpg)

如果`input`的类型为`FloatTensor`或`DoubleTensor`，则 [`min`](#torch.min "torch.min") 和 [`max`](#torch.max "torch.max") 必须为实数，否则它们应为整数。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **min**  (_Number_ ) - 要被钳位的范围的下限
*   **max**  (_Number_ ) - 要钳位的范围的上限
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-1.7120,  0.1734, -0.0478, -0.0922])
>>> torch.clamp(a, min=-0.5, max=0.5)
tensor([-0.5000,  0.1734, -0.0478, -0.0922])

```

```py
torch.clamp(input, *, min, out=None) → Tensor
```

将`input`中的所有元素钳位为大于或等于 [`min`](#torch.min "torch.min") 。

如果`input`的类型为`FloatTensor`或`DoubleTensor`，则`value`应为实数，否则应为整数。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **值**(_数字_） - 输出中每个元素的最小值
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-0.0299, -2.3184,  2.1593, -0.8883])
>>> torch.clamp(a, min=0.5)
tensor([ 0.5000,  0.5000,  2.1593,  0.5000])

```

```py
torch.clamp(input, *, max, out=None) → Tensor
```

将`input`中的所有元素钳位为小于或等于 [`max`](#torch.max "torch.max") 。

If `input` is of type `FloatTensor` or `DoubleTensor`, `value` should be a real number, otherwise it should be an integer.

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **值**(_数字_） - 输出中每个元素的最大值
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.7753, -0.4702, -0.4599,  1.1899])
>>> torch.clamp(a, max=0.5)
tensor([ 0.5000, -0.4702, -0.4599,  0.5000])

```

```py
torch.cos(input, out=None) → Tensor
```

返回具有`input`元素的余弦的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ba6a1422eca60e84b7e3e9c551761d18.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ba6a1422eca60e84b7e3e9c551761d18.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
>>> torch.cos(a)
tensor([ 0.1395,  0.2957,  0.6553,  0.5574])

```

```py
torch.cosh(input, out=None) → Tensor
```

返回具有`input`元素的双曲余弦值的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/676476d8f75c5c5d3a52347cb5576435.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/676476d8f75c5c5d3a52347cb5576435.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.1632,  1.1835, -0.6979, -0.7325])
>>> torch.cosh(a)
tensor([ 1.0133,  1.7860,  1.2536,  1.2805])

```

```py
torch.div()
```

```py
torch.div(input, value, out=None) → Tensor
```

将输入`input`的每个元素与标量`value`分开，并返回一个新的结果张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/92ada503afc46afd1ea338c293ed0b48.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/92ada503afc46afd1ea338c293ed0b48.jpg)

如果`input`的类型为`FloatTensor`或`DoubleTensor`，`value`应为实数，否则应为整数

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **值**(_号码_） - 要分配给`input`的每个元素的数字
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(5)
>>> a
tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
>>> torch.div(a, 0.5)
tensor([ 0.7620,  2.5548, -0.5944, -0.7439,  0.9275])

```

```py
torch.div(input, other, out=None) → Tensor
```

张量`input`的每个元素除以张量`other`的每个元素。返回结果张量。 `input`和`other`的形状必须是[可播放的](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/19b90d4ca4770702635c981d243185b9.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/19b90d4ca4770702635c981d243185b9.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 分子张量
*   **其他** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 分母张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
 [ 0.1815, -1.0111,  0.9805, -1.5923],
 [ 0.1062,  1.4581,  0.7759, -1.2344],
 [-0.1830, -0.0313,  1.1908, -1.4757]])
>>> b = torch.randn(4)
>>> b
tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
>>> torch.div(a, b)
tensor([[-0.4620, -6.6051,  0.5676,  1.2637],
 [ 0.2260, -3.4507, -1.2086,  6.8988],
 [ 0.1322,  4.9764, -0.9564,  5.3480],
 [-0.2278, -0.1068, -1.4678,  6.3936]])

```

```py
torch.digamma(input, out=None) → Tensor
```

计算`input`上伽玛函数的对数导数。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/369a7b2257c669fcc4fcd12afa5cfde7.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/369a7b2257c669fcc4fcd12afa5cfde7.jpg)

| 参数： | **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 计算digamma函数的张量 |
| --- | --- |

Example:

```py
>>> a = torch.tensor([1, 0.5])
>>> torch.digamma(a)
tensor([-0.5772, -1.9635])

```

```py
torch.erf(tensor, out=None) → Tensor
```

计算每个元素的错误函数。错误函数定义如下：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/486fba0c8b6d762f89942dff1e3067f8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/486fba0c8b6d762f89942dff1e3067f8.jpg)

Parameters:

*   **张量** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.erf(torch.tensor([0, -1., 10.]))
tensor([ 0.0000, -0.8427,  1.0000])

```

```py
torch.erfc(input, out=None) → Tensor
```

计算`input`的每个元素的互补误差函数。互补误差函数定义如下：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6100237da75310da52ebc2247d9918f1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6100237da75310da52ebc2247d9918f1.jpg)

Parameters:

*   **张量** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.erfc(torch.tensor([0, -1., 10.]))
tensor([ 1.0000, 1.8427,  0.0000])

```

```py
torch.erfinv(input, out=None) → Tensor
```

计算`input`的每个元素的反向误差函数。反向误差函数在 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/2454c5f08b77e60915c698acbc0eec91.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/2454c5f08b77e60915c698acbc0eec91.jpg) 范围内定义为：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ff1b16ddc6ea5e8c13cd48cf7e4e26c4.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ff1b16ddc6ea5e8c13cd48cf7e4e26c4.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.erfinv(torch.tensor([0, 0.5, -1.]))
tensor([ 0.0000,  0.4769,    -inf])

```

```py
torch.exp(input, out=None) → Tensor
```

返回具有输入张量`input`元素的指数的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/c1c7df2e920de2c586fe0c1040d8e7cd.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/c1c7df2e920de2c586fe0c1040d8e7cd.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.exp(torch.tensor([0, math.log(2.)]))
tensor([ 1.,  2.])

```

```py
torch.expm1(input, out=None) → Tensor
```

返回一个新的张量，其元素的指数减去`input`的1。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/3f5d893e1a9355354b0f64666f45b4ff.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/3f5d893e1a9355354b0f64666f45b4ff.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.expm1(torch.tensor([0, math.log(2.)]))
tensor([ 0.,  1.])

```

```py
torch.floor(input, out=None) → Tensor
```

返回一个新的张量，其中包含`input`元素的最低值，这是每个元素小于或等于的最大整数。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/17860fe2f89c3d742fd5a35e3616d8b4.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/17860fe2f89c3d742fd5a35e3616d8b4.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-0.8166,  1.5308, -0.2530, -0.2091])
>>> torch.floor(a)
tensor([-1.,  1., -1., -1.])

```

```py
torch.fmod(input, divisor, out=None) → Tensor
```

计算除法的元素余数。

被除数和除数可以包含整数和浮点数。余数与被除数`input`具有相同的符号。

当`divisor`是张量时，`input`和`divisor`的形状必须是[可广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 股息
*   **除数** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _或_ [_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) - 除数，可能是与被除数相同形状的数字或张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
tensor([-1., -0., -1.,  1.,  0.,  1.])
>>> torch.fmod(torch.tensor([1., 2, 3, 4, 5]), 1.5)
tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])

```

```py
torch.frac(input, out=None) → Tensor
```

计算`input`中每个元素的小数部分。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/0e649c6142e2ff9cde94388354dc3638.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/0e649c6142e2ff9cde94388354dc3638.jpg)

Example:

```py
>>> torch.frac(torch.tensor([1, 2.5, -3.2]))
tensor([ 0.0000,  0.5000, -0.2000])

```

```py
torch.lerp(start, end, weight, out=None)
```

是否基于标量`weight`对两个张量`start`和`end`进行线性插值，并返回得到的`out`张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/52c0d270ea337a2b6d51bf86fb6f2d45.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/52c0d270ea337a2b6d51bf86fb6f2d45.jpg)

`start`和`end`的形状必须是[可播放的](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

Parameters:

*   **启动** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 张量与起点
*   **结束** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 带有终点的张量
*   **体重** ([_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) - 插值公式的权重
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> start = torch.arange(1., 5.)
>>> end = torch.empty(4).fill_(10)
>>> start
tensor([ 1.,  2.,  3.,  4.])
>>> end
tensor([ 10.,  10.,  10.,  10.])
>>> torch.lerp(start, end, 0.5)
tensor([ 5.5000,  6.0000,  6.5000,  7.0000])

```

```py
torch.log(input, out=None) → Tensor
```

返回具有`input`元素的自然对数的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/63dc128af37016ef7e59d39837eccc3d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/63dc128af37016ef7e59d39837eccc3d.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(5)
>>> a
tensor([-0.7168, -0.5471, -0.8933, -1.4428, -0.1190])
>>> torch.log(a)
tensor([ nan,  nan,  nan,  nan,  nan])

```

```py
torch.log10(input, out=None) → Tensor
```

返回一个新的张量，其对数为`input`元素的基数10。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8020f6f65d1d242403c13ad15b32ad43.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8020f6f65d1d242403c13ad15b32ad43.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.rand(5)
>>> a
tensor([ 0.5224,  0.9354,  0.7257,  0.1301,  0.2251])

>>> torch.log10(a)
tensor([-0.2820, -0.0290, -0.1392, -0.8857, -0.6476])

```

```py
torch.log1p(input, out=None) → Tensor
```

返回一个自然对数为(1 + `input`）的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/553908dc43850f56bb79cbef6e776136.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/553908dc43850f56bb79cbef6e776136.jpg)

注意

对于`input`的小值，此函数比 [`torch.log()`](#torch.log "torch.log") 更准确

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(5)
>>> a
tensor([-1.0090, -0.9923,  1.0249, -0.5372,  0.2492])
>>> torch.log1p(a)
tensor([    nan, -4.8653,  0.7055, -0.7705,  0.2225])

```

```py
torch.log2(input, out=None) → Tensor
```

返回一个新的张量，其对数为`input`元素的基数2。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a6be5b946e4f6d5d157679d60642a747.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a6be5b946e4f6d5d157679d60642a747.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.rand(5)
>>> a
tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])

>>> torch.log2(a)
tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])

```

```py
torch.mul()
```

```py
torch.mul(input, value, out=None)
```

将输入`input`的每个元素与标量`value`相乘，并返回一个新的结果张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/7dd1caf9162104803cc11bfb0de7a8fa.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/7dd1caf9162104803cc11bfb0de7a8fa.jpg)

If `input` is of type `FloatTensor` or `DoubleTensor`, `value` should be a real number, otherwise it should be an integer

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **值**(_号码_） - 要与`input`的每个元素相乘的数字
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(3)
>>> a
tensor([ 0.2015, -0.4255,  2.6087])
>>> torch.mul(a, 100)
tensor([  20.1494,  -42.5491,  260.8663])

```

```py
torch.mul(input, other, out=None)
```

张量`input`的每个元素乘以张量`other`的每个元素。返回结果张量。

The shapes of `input` and `other` must be [broadcastable](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics).

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/13296a8d428f985a8702d83e100d4153.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/13296a8d428f985a8702d83e100d4153.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第一个被乘数张量
*   **其他** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二个被乘数张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4, 1)
>>> a
tensor([[ 1.1207],
 [-0.3137],
 [ 0.0700],
 [ 0.8378]])
>>> b = torch.randn(1, 4)
>>> b
tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
>>> torch.mul(a, b)
tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
 [-0.1614, -0.0382,  0.1645, -0.7021],
 [ 0.0360,  0.0085, -0.0367,  0.1567],
 [ 0.4312,  0.1019, -0.4394,  1.8753]])

```

```py
torch.mvlgamma(input, p) → Tensor
```

用维度 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/648811253cdbfe19389964c25be56518.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/648811253cdbfe19389964c25be56518.jpg) 元素计算多变量log-gamma函数，由下式给出：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/d1cecf35ffe071cbcf420549cd030664.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/d1cecf35ffe071cbcf420549cd030664.jpg)

其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8890ddee156302958d8906a2799dc16b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8890ddee156302958d8906a2799dc16b.jpg) 和 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/94c1ba406fbf0f76780513dfd005e6f5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/94c1ba406fbf0f76780513dfd005e6f5.jpg) 是Gamma函数。

如果任何元素小于或等于 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/80e27a556a24dac8f2985689098c1a82.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/80e27a556a24dac8f2985689098c1a82.jpg) ，则抛出错误。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 计算多变量log-gamma函数的张量
*   **p**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 维数

Example:

```py
>>> a = torch.empty(2, 3).uniform_(1, 2)
>>> a
tensor([[1.6835, 1.8474, 1.1929],
 [1.0475, 1.7162, 1.4180]])
>>> torch.mvlgamma(a, 2)
tensor([[0.3928, 0.4007, 0.7586],
 [1.0311, 0.3901, 0.5049]])

```

```py
torch.neg(input, out=None) → Tensor
```

返回一个新的张量，其元素为`input`。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5ea71e988dcc7de6c27b28ca79f4e893.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5ea71e988dcc7de6c27b28ca79f4e893.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(5)
>>> a
tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
>>> torch.neg(a)
tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])

```

```py
torch.pow()
```

```py
torch.pow(input, exponent, out=None) → Tensor
```

使用`exponent`获取`input`中每个元素的功效，并返回带有结果的张量。

`exponent`可以是单个`float`编号，也可以是`Tensor`，其元素数与`input`相同。

当`exponent`是标量值时，应用的操作是：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a04a68f6eedb206fc30bd425f792afdb.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a04a68f6eedb206fc30bd425f792afdb.jpg)

当`exponent`是张量时，应用的操作是：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5499cae563d1d075246e3a18b191e870.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5499cae563d1d075246e3a18b191e870.jpg)

当`exponent`是张量时，`input`和`exponent`的形状必须是[可广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **指数** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_或_ _tensor_） - 指数值
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
>>> torch.pow(a, 2)
tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
>>> exp = torch.arange(1., 5.)

>>> a = torch.arange(1., 5.)
>>> a
tensor([ 1.,  2.,  3.,  4.])
>>> exp
tensor([ 1.,  2.,  3.,  4.])
>>> torch.pow(a, exp)
tensor([   1.,    4.,   27.,  256.])

```

```py
torch.pow(base, input, out=None) → Tensor
```

`base`是标量`float`值，`input`是张量。返回的张量`out`与`input`具有相同的形状

适用的操作是：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/0c98533a385eed5ae6f333583d9d239e.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/0c98533a385eed5ae6f333583d9d239e.jpg)

Parameters:

*   **base**  ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")） - 电源操作的标量基值
*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 指数张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> exp = torch.arange(1., 5.)
>>> base = 2
>>> torch.pow(base, exp)
tensor([  2.,   4.,   8.,  16.])

```

```py
torch.reciprocal(input, out=None) → Tensor
```

返回具有`input`元素倒数的新张量

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a4cb9bbdd43eddd6583c288380fe9704.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a4cb9bbdd43eddd6583c288380fe9704.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-0.4595, -2.1219, -1.4314,  0.7298])
>>> torch.reciprocal(a)
tensor([-2.1763, -0.4713, -0.6986,  1.3702])

```

```py
torch.remainder(input, divisor, out=None) → Tensor
```

Computes the element-wise remainder of division.

除数和被除数可以包含整数和浮点数。其余部分与除数具有相同的符号。

When `divisor` is a tensor, the shapes of `input` and `divisor` must be [broadcastable](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics).

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 股息
*   **除数** ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _或_ [_漂浮_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) - 可能是一个除数数字或与被除数相同形状的张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
>>> torch.remainder(torch.tensor([1., 2, 3, 4, 5]), 1.5)
tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])

```

也可以看看

[`torch.fmod()`](#torch.fmod "torch.fmod") ，它计算与C库函数`fmod()`等效的除法的元素余数。

```py
torch.round(input, out=None) → Tensor
```

返回一个新的张量，`input`的每个元素四舍五入到最接近的整数。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.9920,  0.6077,  0.9734, -1.0362])
>>> torch.round(a)
tensor([ 1.,  1.,  1., -1.])

```

```py
torch.rsqrt(input, out=None) → Tensor
```

返回一个新的张量，其具有`input`的每个元素的平方根的倒数。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ba016159f6eee3d6e907b3f1f4690148.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ba016159f6eee3d6e907b3f1f4690148.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-0.0370,  0.2970,  1.5420, -0.9105])
>>> torch.rsqrt(a)
tensor([    nan,  1.8351,  0.8053,     nan])

```

```py
torch.sigmoid(input, out=None) → Tensor
```

返回带有`input`元素的sigmoid的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/35490743ae06a50e628101c524fa3557.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/35490743ae06a50e628101c524fa3557.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
>>> torch.sigmoid(a)
tensor([ 0.7153,  0.7481,  0.2920,  0.1458])

```

```py
torch.sign(input, out=None) → Tensor
```

返回带有`input`元素符号的新张量。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.tensor([0.7, -1.2, 0., 2.3])
>>> a
tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
>>> torch.sign(a)
tensor([ 1., -1.,  0.,  1.])

```

```py
torch.sin(input, out=None) → Tensor
```

返回带有`input`元素正弦的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/4bb3f3689a942de005b6ed433517a99a.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/4bb3f3689a942de005b6ed433517a99a.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-0.5461,  0.1347, -2.7266, -0.2746])
>>> torch.sin(a)
tensor([-0.5194,  0.1343, -0.4032, -0.2711])

```

```py
torch.sinh(input, out=None) → Tensor
```

返回具有`input`元素的双曲正弦的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ef01436f55bc1cd4c0407857bb6b41d0.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ef01436f55bc1cd4c0407857bb6b41d0.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.5380, -0.8632, -0.1265,  0.9399])
>>> torch.sinh(a)
tensor([ 0.5644, -0.9744, -0.1268,  1.0845])

```

```py
torch.sqrt(input, out=None) → Tensor
```

返回具有`input`元素的平方根的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5407c7228f589f6c48f1bdd755f1e4c8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5407c7228f589f6c48f1bdd755f1e4c8.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-2.0755,  1.0226,  0.0831,  0.4806])
>>> torch.sqrt(a)
tensor([    nan,  1.0112,  0.2883,  0.6933])

```

```py
torch.tan(input, out=None) → Tensor
```

返回具有`input`元素正切的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/0cab489cdb9e93ea59ac064d58876397.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/0cab489cdb9e93ea59ac064d58876397.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([-1.2027, -1.7687,  0.4412, -1.3856])
>>> torch.tan(a)
tensor([-2.5930,  4.9859,  0.4722, -5.3366])

```

```py
torch.tanh(input, out=None) → Tensor
```

返回具有`input`元素的双曲正切的新张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/29f29380f07881b913efa1bcc641e2ae.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/29f29380f07881b913efa1bcc641e2ae.jpg)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
>>> torch.tanh(a)
tensor([ 0.7156, -0.6218,  0.8257,  0.2553])

```

```py
torch.trunc(input, out=None) → Tensor
```

返回具有`input`元素的截断整数值的新张量。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

Example:

```py
>>> a = torch.randn(4)
>>> a
tensor([ 3.4742,  0.5466, -0.8008, -0.9079])
>>> torch.trunc(a)
tensor([ 3.,  0., -0., -0.])

```

