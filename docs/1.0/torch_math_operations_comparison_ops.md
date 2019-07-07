### Comparison Ops

```py
torch.allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False) → bool
```

This function checks if all `self` and `other` satisfy the condition:

![](img/5dd778c75ab74a08d08025aa32f15e20.jpg)

elementwise, for all elements of `self` and `other`. The behaviour of this function is analogous to [numpy.allclose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html)

Parameters: 

*   **self** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – first tensor to compare
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – second tensor to compare
*   **atol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – absolute tolerance. Default: 1e-08
*   **rtol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – relative tolerance. Default: 1e-05
*   **equal_nan** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – if `True`, then two `NaN` s will be compared as equal. Default: `False`



Example:

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

Returns the indices that sort a tensor along a given dimension in ascending order by value.

This is the second value returned by [`torch.sort()`](#torch.sort "torch.sort"). See its documentation for the exact semantics of this method.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the dimension to sort along
*   **descending** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls the sorting order (ascending or descending)



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

Computes element-wise equality

The second argument can be a number or a tensor whose shape is [broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – the tensor or value to compare
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor. Must be a `ByteTensor`


| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 1,  0],
 [ 0,  1]], dtype=torch.uint8)

```

```py
torch.equal(tensor1, tensor2) → bool
```

`True` if two tensors have the same size and elements, `False` otherwise.

Example:

```py
>>> torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
True

```

```py
torch.ge(input, other, out=None) → Tensor
```

Computes ![](img/7cedf1d52401b16530bccf12f96edd5a.jpg) element-wise.

The second argument can be a number or a tensor whose shape is [broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – the tensor or value to compare
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor that must be a `ByteTensor`


| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 1,  1],
 [ 0,  1]], dtype=torch.uint8)

```

```py
torch.gt(input, other, out=None) → Tensor
```

Computes ![](img/9d325c4b6e2e06380eb65ceae1e84d76.jpg) element-wise.

The second argument can be a number or a tensor whose shape is [broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – the tensor or value to compare
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor that must be a `ByteTensor`


| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 0,  1],
 [ 0,  0]], dtype=torch.uint8)

```

```py
torch.isfinite(tensor)
```

Returns a new tensor with boolean elements representing if each element is `Finite` or not.

| Parameters: | **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – A tensor to check |
| --- | --- |
| Returns: | A `torch.ByteTensor` containing a 1 at each location of finite elements and 0 otherwise |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
tensor([ 1,  0,  1,  0,  0], dtype=torch.uint8)

```

```py
torch.isinf(tensor)
```

Returns a new tensor with boolean elements representing if each element is `+/-INF` or not.

| Parameters: | **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – A tensor to check |
| --- | --- |
| Returns: | A `torch.ByteTensor` containing a 1 at each location of `+/-INF` elements and 0 otherwise |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
tensor([ 0,  1,  0,  1,  0], dtype=torch.uint8)

```

```py
torch.isnan(tensor)
```

Returns a new tensor with boolean elements representing if each element is `NaN` or not.

| Parameters: | **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – A tensor to check |
| --- | --- |
| Returns: | A `torch.ByteTensor` containing a 1 at each location of `NaN` elements. |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> torch.isnan(torch.tensor([1, float('nan'), 2]))
tensor([ 0,  1,  0], dtype=torch.uint8)

```

```py
torch.kthvalue(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor)
```

Returns the `k` th smallest element of the given `input` tensor along a given dimension.

If `dim` is not given, the last dimension of the `input` is chosen.

A tuple of `(values, indices)` is returned, where the `indices` is the indices of the kth-smallest element in the original `input` tensor in dimension `dim`.

If `keepdim` is `True`, both the `values` and `indices` tensors are the same size as `input`, except in the dimension `dim` where they are of size 1\. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in both the `values` and `indices` tensors having 1 fewer dimension than the `input` tensor.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **k** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – k for the k-th smallest element
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the dimension to find the kth value along
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensors have `dim` retained or not
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the output tuple of (Tensor, LongTensor) can be optionally given to be used as output buffers



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

Computes ![](img/7cce834f96c0e53d76c3ec5ed63cf099.jpg) element-wise.

The second argument can be a number or a tensor whose shape is [broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – the tensor or value to compare
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor that must be a `ByteTensor`


| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 1,  0],
 [ 1,  1]], dtype=torch.uint8)

```

```py
torch.lt(input, other, out=None) → Tensor
```

Computes ![](img/26b8f23e09743e63a71bcf53c650f1a8.jpg) element-wise.

The second argument can be a number or a tensor whose shape is [broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – the tensor or value to compare
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor that must be a `ByteTensor`


| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

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

Returns the maximum value of all elements in the `input` tensor.

| Parameters: | **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor |
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

Returns the maximum value of each row of the `input` tensor in the given dimension `dim`. The second return value is the index location of each maximum value found (argmax).

If `keepdim` is `True`, the output tensors are of the same size as `input` except in the dimension `dim` where they are of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensors having 1 fewer dimension than `input`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensors have `dim` retained or not
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the result tuple of two output tensors (max, max_indices)



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

Each element of the tensor `input` is compared with the corresponding element of the tensor `other` and an element-wise maximum is taken.

The shapes of `input` and `other` don’t need to match, but they must be [broadcastable](notes/broadcasting.html#broadcasting-semantics).

![](img/adfdd713f1ca11c4b41b733a5f452f47.jpg)

Note

When the shapes do not match, the shape of the returned output tensor follows the [broadcasting rules](notes/broadcasting.html#broadcasting-semantics).

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second input tensor
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



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

Returns the minimum value of all elements in the `input` tensor.

| Parameters: | **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor |
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

Returns the minimum value of each row of the `input` tensor in the given dimension `dim`. The second return value is the index location of each minimum value found (argmin).

If `keepdim` is `True`, the output tensors are of the same size as `input` except in the dimension `dim` where they are of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensors having 1 fewer dimension than `input`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensors have `dim` retained or not
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the tuple of two output tensors (min, min_indices)



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

Each element of the tensor `input` is compared with the corresponding element of the tensor `other` and an element-wise minimum is taken. The resulting tensor is returned.

The shapes of `input` and `other` don’t need to match, but they must be [broadcastable](notes/broadcasting.html#broadcasting-semantics).

![](img/173d7b4bb5cacbc18349fb16cd6130ab.jpg)

Note

When the shapes do not match, the shape of the returned output tensor follows the [broadcasting rules](notes/broadcasting.html#broadcasting-semantics).

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second input tensor
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



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

Computes ![](img/9b7e900c499f533c33aac63d7b487c24.jpg) element-wise.

The second argument can be a number or a tensor whose shape is [broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first argument.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – the tensor or value to compare
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor that must be a `ByteTensor`


| Returns: | A `torch.ByteTensor` containing a 1 at each location where comparison is true. |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ 0,  1],
 [ 1,  0]], dtype=torch.uint8)

```

```py
torch.sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)
```

Sorts the elements of the `input` tensor along a given dimension in ascending order by value.

If `dim` is not given, the last dimension of the `input` is chosen.

If `descending` is `True` then the elements are sorted in descending order by value.

A tuple of (sorted_tensor, sorted_indices) is returned, where the sorted_indices are the indices of the elements in the original `input` tensor.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the dimension to sort along
*   **descending** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls the sorting order (ascending or descending)
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the output tuple of (`Tensor`, `LongTensor`) that can be optionally given to be used as output buffers



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

Returns the `k` largest elements of the given `input` tensor along a given dimension.

If `dim` is not given, the last dimension of the `input` is chosen.

If `largest` is `False` then the `k` smallest elements are returned.

A tuple of `(values, indices)` is returned, where the `indices` are the indices of the elements in the original `input` tensor.

The boolean option `sorted` if `True`, will make sure that the returned `k` elements are themselves sorted

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **k** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the k in “top-k”
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the dimension to sort along
*   **largest** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether to return largest or smallest elements
*   **sorted** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether to return the elements in sorted order
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the output tuple of (Tensor, LongTensor) that can be optionally given to be used as output buffers



Example:

```py
>>> x = torch.arange(1., 6.)
>>> x
tensor([ 1.,  2.,  3.,  4.,  5.])
>>> torch.topk(x, 3)
(tensor([ 5.,  4.,  3.]), tensor([ 4,  3,  2]))

```
