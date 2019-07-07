### Reduction Ops

```py
torch.argmax(input, dim=None, keepdim=False)
```

Returns the indices of the maximum values of a tensor across a dimension.

This is the second value returned by [`torch.max()`](#torch.max "torch.max"). See its documentation for the exact semantics of this method.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce. If `None`, the argmax of the flattened input is returned.
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensors have `dim` retained or not. Ignored if `dim=None`.



Example:

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

Returns the indices of the minimum values of a tensor across a dimension.

This is the second value returned by [`torch.min()`](#torch.min "torch.min"). See its documentation for the exact semantics of this method.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce. If `None`, the argmin of the flattened input is returned.
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensors have `dim` retained or not. Ignored if `dim=None`.



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

Returns the cumulative product of elements of `input` in the dimension `dim`.

For example, if `input` is a vector of size N, the result will also be a vector of size N, with elements.

![](img/9e9045e46c1b7fca7acb598cf474f16e.jpg)

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to do the operation over
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. This is useful for preventing data type overflows. Default: None.



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

Returns the cumulative sum of elements of `input` in the dimension `dim`.

For example, if `input` is a vector of size N, the result will also be a vector of size N, with elements.

![](img/70f741993caea1156636d61e6f21e463.jpg)

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to do the operation over
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. This is useful for preventing data type overflows. Default: None.



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

Returns the p-norm of (`input` - `other`)

The shapes of `input` and `other` must be [broadcastable](notes/broadcasting.html#broadcasting-semantics).

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the Right-hand-side input tensor
*   **p** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – the norm to be computed



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

Returns the log of summed exponentials of each row of the `input` tensor in the given dimension `dim`. The computation is numerically stabilized.

For summation index ![](img/d8fdd0e28cfb03738fc5227885ee035a.jpg) given by `dim` and other indices ![](img/31df9c730e19ca29b59dce64b99d98c1.jpg), the result is

> ![](img/5acb5b22a7a5c1cfbfda7f648a00c656.jpg)

If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension `dim` where it is of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensor having 1 fewer dimension than `input`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _tuple of python:ints_) – the dimension or dimensions to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensor has `dim` retained or not
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



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

Returns the mean value of all elements in the `input` tensor.

| Parameters: | **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor |
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

Returns the mean value of each row of the `input` tensor in the given dimension `dim`. If `dim` is a list of dimensions, reduce over all of them.

If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s).

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether the output tensor has `dim` retained or not
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the output tensor



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

Returns the median value of all elements in the `input` tensor.

| Parameters: | **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor |
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

Returns the median value of each row of the `input` tensor in the given dimension `dim`. Also returns the index location of the median value as a `LongTensor`.

By default, `dim` is the last dimension of the `input` tensor.

If `keepdim` is `True`, the output tensors are of the same size as `input` except in the dimension `dim` where they are of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the outputs tensor having 1 fewer dimension than `input`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensors have `dim` retained or not
*   **values** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor
*   **indices** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output index tensor



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

Returns the mode value of each row of the `input` tensor in the given dimension `dim`. Also returns the index location of the mode value as a `LongTensor`.

By default, `dim` is the last dimension of the `input` tensor.

If `keepdim` is `True`, the output tensors are of the same size as `input` except in the dimension `dim` where they are of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensors having 1 fewer dimension than `input`.

Note

This function is not defined for `torch.cuda.Tensor` yet.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensors have `dim` retained or not
*   **values** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor
*   **indices** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output index tensor



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

Returns the matrix norm or vector norm of a given tensor.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **p** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _inf__,_ _-inf__,_ _'fro'__,_ _'nuc'__,_ _optional_) –

    the order of norm. Default: `'fro'` The following norms can be calculated:

    &#124; ord &#124; matrix norm &#124; vector norm &#124;
    &#124; --- &#124; --- &#124; --- &#124;
    &#124; None &#124; Frobenius norm &#124; 2-norm &#124;
    &#124; ’fro’ &#124; Frobenius norm &#124; – &#124;
    &#124; ‘nuc’ &#124; nuclear norm &#124; – &#124;
    &#124; Other &#124; as vec norm when dim is None &#124; sum(abs(x)**ord)**(1./ord) &#124;

*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _2-tuple of python:ints__,_ _2-list of python:ints__,_ _optional_) – If it is an int, vector norm will be calculated, if it is 2-tuple of ints, matrix norm will be calculated. If the value is None, matrix norm will be calculated when the input tensor only has two dimensions, vector norm will be calculated when the input tensor only has one dimension. If the input tensor has more than two dimensions, the vector norm will be applied to last dimension.
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether the output tensors have `dim` retained or not. Ignored if `dim` = `None` and `out` = `None`. Default: `False`
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor. Ignored if `dim` = `None` and `out` = `None`.



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

Returns the product of all elements in the `input` tensor.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. This is useful for preventing data type overflows. Default: None.



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

Returns the product of each row of the `input` tensor in the given dimension `dim`.

If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension `dim` where it is of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensor having 1 fewer dimension than `input`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensor has `dim` retained or not
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. This is useful for preventing data type overflows. Default: None.



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

Returns the standard-deviation of all elements in the `input` tensor.

If `unbiased` is `False`, then the standard-deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **unbiased** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether to use the unbiased estimation or not



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

Returns the standard-deviation of each row of the `input` tensor in the given dimension `dim`.

If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension `dim` where it is of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensor having 1 fewer dimension than `input`.

If `unbiased` is `False`, then the standard-deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensor has `dim` retained or not
*   **unbiased** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether to use the unbiased estimation or not
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



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

Returns the sum of all elements in the `input` tensor.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. This is useful for preventing data type overflows. Default: None.



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

Returns the sum of each row of the `input` tensor in the given dimension `dim`. If `dim` is a list of dimensions, reduce over all of them.

If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s).

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _tuple of python:ints_) – the dimension or dimensions to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensor has `dim` retained or not
*   **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. This is useful for preventing data type overflows. Default: None.



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

Returns the unique scalar elements of the input tensor as a 1-D tensor.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **sorted** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – Whether to sort the unique elements in ascending order before returning as output.
*   **return_inverse** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – Whether to also return the indices for where elements in the original input ended up in the returned unique list.
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to apply unique. If `None`, the unique of the flattened input is returned. default: `None`


| Returns: | A tensor or a tuple of tensors containing

&gt; *   **output** (_Tensor_): the output list of unique scalar elements.
&gt; *   **inverse_indices** (_Tensor_): (optional) if `return_inverse` is True, there will be a 2nd returned tensor (same shape as input) representing the indices for where elements in the original input map to in the output; otherwise, this function will only return a single tensor.


| Return type: | ([Tensor](tensors.html#torch.Tensor "torch.Tensor"), [Tensor](tensors.html#torch.Tensor "torch.Tensor") (optional)) |
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

Returns the variance of all elements in the `input` tensor.

If `unbiased` is `False`, then the variance will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **unbiased** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether to use the unbiased estimation or not



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

Returns the variance of each row of the `input` tensor in the given dimension `dim`.

If `keepdim` is `True`, the output tensors are of the same size as `input` except in the dimension `dim` where they are of size 1. Otherwise, `dim` is squeezed (see [`torch.squeeze()`](#torch.squeeze "torch.squeeze")), resulting in the outputs tensor having 1 fewer dimension than `input`.

If `unbiased` is `False`, then the variance will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to reduce
*   **keepdim** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether the output tensor has `dim` retained or not
*   **unbiased** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether to use the unbiased estimation or not
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



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
