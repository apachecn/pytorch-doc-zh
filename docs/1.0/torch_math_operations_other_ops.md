### Other Operations

```py
torch.bincount(self, weights=None, minlength=0) → Tensor
```

Count the frequency of each value in an array of non-negative ints.

The number of bins (size 1) is one larger than the largest value in `input` unless `input` is empty, in which case the result is a tensor of size 0\. If `minlength` is specified, the number of bins is at least `minlength` and if `input` is empty, then the result is tensor of size `minlength` filled with zeros. If `n` is the value at position `i`, `out[n] += weights[i]` if `weights` is specified else `out[n] += 1`.

Note

When using the CUDA backend, this operation may induce nondeterministic behaviour that is not easily switched off. Please see the notes on [Reproducibility](notes/randomness.html) for background.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 1-d int tensor
*   **weights** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – optional, weight for each value in the input tensor. Should be of same size as input tensor.
*   **minlength** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – optional, minimum number of bins. Should be non-negative.


| Returns: | a tensor of shape `Size([max(input) + 1])` if `input` is non-empty, else `Size(0)` |
| --- | --- |
| Return type: | output ([Tensor](tensors.html#torch.Tensor "torch.Tensor")) |
| --- | --- |

Example:

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

Broadcasts the given tensors according to _broadcasting-semantics.

| Parameters: | ***tensors** – any number of tensors of the same type |
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

Returns the cross product of vectors in dimension `dim` of `input` and `other`.

`input` and `other` must have the same size, and the size of their `dim` dimension should be 3.

If `dim` is not given, it defaults to the first dimension found with the size 3.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **other** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the dimension to take the cross-product in.
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



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

*   If `input` is a vector (1-D tensor), then returns a 2-D square tensor with the elements of `input` as the diagonal.
*   If `input` is a matrix (2-D tensor), then returns a 1-D tensor with the diagonal elements of `input`.

The argument [`diagonal`](#torch.diagonal "torch.diagonal") controls which diagonal to consider:

*   If [`diagonal`](#torch.diagonal "torch.diagonal") = 0, it is the main diagonal.
*   If [`diagonal`](#torch.diagonal "torch.diagonal") &gt; 0, it is above the main diagonal.
*   If [`diagonal`](#torch.diagonal "torch.diagonal") &lt; 0, it is below the main diagonal.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **diagonal** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the diagonal to consider
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



See also

[`torch.diagonal()`](#torch.diagonal "torch.diagonal") always returns the diagonal of its input.

[`torch.diagflat()`](#torch.diagflat "torch.diagflat") always constructs a tensor with diagonal elements specified by the input.

Examples:

Get the square matrix where the input vector is the diagonal:

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

Get the k-th diagonal of a given matrix:

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

Creates a tensor whose diagonals of certain 2D planes (specified by `dim1` and `dim2`) are filled by `input`. To facilitate creating batched diagonal matrices, the 2D planes formed by the last two dimensions of the returned tensor are chosen by default.

The argument `offset` controls which diagonal to consider:

*   If `offset` = 0, it is the main diagonal.
*   If `offset` &gt; 0, it is above the main diagonal.
*   If `offset` &lt; 0, it is below the main diagonal.

The size of the new matrix will be calculated to make the specified diagonal of the size of the last input dimension. Note that for `offset` other than ![](img/28256dd5af833c877d63bfabfaa7b301.jpg), the order of `dim1` and `dim2` matters. Exchanging them is equivalent to changing the sign of `offset`.

Applying [`torch.diagonal()`](#torch.diagonal "torch.diagonal") to the output of this function with the same arguments yields a matrix identical to input. However, [`torch.diagonal()`](#torch.diagonal "torch.diagonal") has different default dimensions, so those need to be explicitly specified.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor. Must be at least 1-dimensional.
*   **offset** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – which diagonal to consider. Default: 0 (main diagonal).
*   **dim1** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – first dimension with respect to which to take diagonal. Default: -2.
*   **dim2** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – second dimension with respect to which to take diagonal. Default: -1.



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

*   If `input` is a vector (1-D tensor), then returns a 2-D square tensor with the elements of `input` as the diagonal.
*   If `input` is a tensor with more than one dimension, then returns a 2-D tensor with diagonal elements equal to a flattened `input`.

The argument `offset` controls which diagonal to consider:

*   If `offset` = 0, it is the main diagonal.
*   If `offset` &gt; 0, it is above the main diagonal.
*   If `offset` &lt; 0, it is below the main diagonal.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **offset** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the diagonal to consider. Default: 0 (main diagonal).



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

Returns a partial view of `input` with the its diagonal elements with respect to `dim1` and `dim2` appended as a dimension at the end of the shape.

The argument `offset` controls which diagonal to consider:

*   If `offset` = 0, it is the main diagonal.
*   If `offset` &gt; 0, it is above the main diagonal.
*   If `offset` &lt; 0, it is below the main diagonal.

Applying [`torch.diag_embed()`](#torch.diag_embed "torch.diag_embed") to the output of this function with the same arguments yields a diagonal matrix with the diagonal entries of the input. However, [`torch.diag_embed()`](#torch.diag_embed "torch.diag_embed") has different default dimensions, so those need to be explicitly specified.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor. Must be at least 2-dimensional.
*   **offset** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – which diagonal to consider. Default: 0 (main diagonal).
*   **dim1** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – first dimension with respect to which to take diagonal. Default: 0.
*   **dim2** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – second dimension with respect to which to take diagonal. Default: 1.



Note

To take a batch diagonal, pass in dim1=-2, dim2=-1.

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

This function provides a way of computing multilinear expressions (i.e. sums of products) using the Einstein summation convention.

Parameters: 

*   **equation** (_string_) – The equation is given in terms of lower case letters (indices) to be associated with each dimension of the operands and result. The left hand side lists the operands dimensions, separated by commas. There should be one index letter per tensor dimension. The right hand side follows after `-&gt;` and gives the indices for the output. If the `-&gt;` and right hand side are omitted, it implicitly defined as the alphabetically sorted list of all indices appearing exactly once in the left hand side. The indices not apprearing in the output are summed over after multiplying the operands entries. If an index appears several times for the same operand, a diagonal is taken. Ellipses `…` represent a fixed number of dimensions. If the right hand side is inferred, the ellipsis dimensions are at the beginning of the output.
*   **operands** (_list of Tensors_) – The operands to compute the Einstein sum of. Note that the operands are passed as a list, not as individual arguments.



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

Flattens a contiguous range of dims in a tensor.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **start_dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the first dim to flatten
*   **end_dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the last dim to flatten



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

Reverse the order of a n-D tensor along given axis in dims.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **dims** (_a list_ _or_ [_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – axis to flip on



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

Computes the histogram of a tensor.

The elements are sorted into equal width bins between [`min`](#torch.min "torch.min") and [`max`](#torch.max "torch.max"). If [`min`](#torch.min "torch.min") and [`max`](#torch.max "torch.max") are both zero, the minimum and maximum values of the data are used.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **bins** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – number of histogram bins
*   **min** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – lower end of the range (inclusive)
*   **max** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – upper end of the range (inclusive)
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor


| Returns: | Histogram represented as a tensor |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)
tensor([ 0.,  2.,  1.,  0.])

```

```py
torch.meshgrid(*tensors, **kwargs)
```

Take ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) tensors, each of which can be either scalar or 1-dimensional vector, and create ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) N-dimensional grids, where the :math:[`](#id2)i`th grid is defined by expanding the :math:[`](#id4)i`th input over dimensions defined by other inputs.

> ```py
> Args:
> ```
> 
> tensors (list of Tensor): list of scalars or 1 dimensional tensors. Scalars will be treated as tensors of size ![](img/e800eead21f1007b4005a268169586f7.jpg) automatically
> 
> ```py
> Returns:
> ```
> 
> seq (sequence of Tensors): If the input has ![](img/a1c2f8d5b1226e67bdb44b12a6ddf18b.jpg) tensors of size ![](img/5871a78f7096a5c43c0b08b090b8c4f1.jpg), then the output would also has ![](img/a1c2f8d5b1226e67bdb44b12a6ddf18b.jpg) tensors, where all tensors are of size ![](img/771af95b2d780e68358b78d5124091fa.jpg).
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

Returns a tensor where each sub-tensor of `input` along dimension `dim` is normalized such that the `p`-norm of the sub-tensor is lower than the value `maxnorm`

Note

If the norm of a row is lower than `maxnorm`, the row is unchanged

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **p** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – the power for the norm computation
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the dimension to slice over to get the sub-tensors
*   **maxnorm** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – the maximum norm to keep each sub-tensor under
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



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

Returns a contraction of a and b over multiple dimensions.

[`tensordot`](#torch.tensordot "torch.tensordot") implements a generalizes the matrix product.

Parameters: 

*   **a** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Left tensor to contract
*   **b** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Right tensor to contract
*   **dims** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _tuple of two lists of python:integers_) – number of dimensions to contract or explicit lists of dimensions for `a` and `b` respectively



When called with an integer argument `dims` = ![](img/9566974d45a96737f7e0ecf302d877b8.jpg), and the number of dimensions of `a` and `b` is ![](img/20ddd8181c2e0d0fb893637e8572d475.jpg) and ![](img/493731e423d5db62086d0b8705dda0c8.jpg), respectively, it computes

![](img/99b9f5b0cd445ecb19920e143987f33e.jpg)

When called with `dims` of the list form, the given dimensions will be contracted in place of the last ![](img/9566974d45a96737f7e0ecf302d877b8.jpg) of `a` and the first ![](img/9566974d45a96737f7e0ecf302d877b8.jpg) of ![](img/6872867a863714d15d9a0d64c20734ce.jpg). The sizes in these dimensions must match, but [`tensordot`](#torch.tensordot "torch.tensordot") will deal with broadcasted dimensions.

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

Returns the sum of the elements of the diagonal of the input 2-D matrix.

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

Returns the lower triangular part of the matrix (2-D tensor) `input`, the other elements of the result tensor `out` are set to 0.

The lower triangular part of the matrix is defined as the elements on and below the diagonal.

The argument [`diagonal`](#torch.diagonal "torch.diagonal") controls which diagonal to consider. If [`diagonal`](#torch.diagonal "torch.diagonal") = 0, all elements on and below the main diagonal are retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative value excludes just as many diagonals below the main diagonal. The main diagonal are the set of indices ![](img/6291ea635817db74920cd048cc3cb8d4.jpg) for ![](img/1bfb3770a124b38b3aba63186b7c8f46.jpg) where ![](img/5ccb16cb4e75340b0c2b2d022fd778a7.jpg) are the dimensions of the matrix.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **diagonal** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the diagonal to consider
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



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

Returns the upper triangular part of the matrix (2-D tensor) `input`, the other elements of the result tensor `out` are set to 0.

The upper triangular part of the matrix is defined as the elements on and above the diagonal.

The argument [`diagonal`](#torch.diagonal "torch.diagonal") controls which diagonal to consider. If [`diagonal`](#torch.diagonal "torch.diagonal") = 0, all elements on and below the main diagonal are retained. A positive value excludes just as many diagonals above the main diagonal, and similarly a negative value includes just as many diagonals below the main diagonal. The main diagonal are the set of indices ![](img/6291ea635817db74920cd048cc3cb8d4.jpg) for ![](img/1bfb3770a124b38b3aba63186b7c8f46.jpg) where ![](img/5ccb16cb4e75340b0c2b2d022fd778a7.jpg) are the dimensions of the matrix.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **diagonal** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – the diagonal to consider
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



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
