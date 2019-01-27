

# torch.sparse

Warning

This API is currently experimental and may change in the near future.

Torch supports sparse tensors in COO(rdinate) format, which can efficiently store and process tensors for which the majority of elements are zeros.

A sparse tensor is represented as a pair of dense tensors: a tensor of values and a 2D tensor of indices. A sparse tensor can be constructed by providing these two tensors, as well as the size of the sparse tensor (which cannot be inferred from these tensors!) Suppose we want to define a sparse tensor with the entry 3 at location (0, 2), entry 4 at location (1, 0), and entry 5 at location (1, 2). We would then write:

```py
>>> i = torch.LongTensor([[0, 1, 1],
 [2, 0, 2]])
>>> v = torch.FloatTensor([3, 4, 5])
>>> torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense()
 0  0  3
 4  0  5
[torch.FloatTensor of size 2x3]

```

Note that the input to LongTensor is NOT a list of index tuples. If you want to write your indices this way, you should transpose before passing them to the sparse constructor:

```py
>>> i = torch.LongTensor([[0, 2], [1, 0], [1, 2]])
>>> v = torch.FloatTensor([3,      4,      5    ])
>>> torch.sparse.FloatTensor(i.t(), v, torch.Size([2,3])).to_dense()
 0  0  3
 4  0  5
[torch.FloatTensor of size 2x3]

```

You can also construct hybrid sparse tensors, where only the first n dimensions are sparse, and the rest of the dimensions are dense.

```py
>>> i = torch.LongTensor([[2, 4]])
>>> v = torch.FloatTensor([[1, 3], [5, 7]])
>>> torch.sparse.FloatTensor(i, v).to_dense()
 0  0
 0  0
 1  3
 0  0
 5  7
[torch.FloatTensor of size 5x2]

```

An empty sparse tensor can be constructed by specifying its size:

```py
>>> torch.sparse.FloatTensor(2, 3)
SparseFloatTensor of size 2x3 with indices:
[torch.LongTensor with no dimension]
and values:
[torch.FloatTensor with no dimension]

```

```py
SparseTensor has the following invariants:
```

1.  sparse_dim + dense_dim = len(SparseTensor.shape)
2.  SparseTensor._indices().shape = (sparse_dim, nnz)
3.  SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])

Since SparseTensor._indices() is always a 2D tensor, the smallest sparse_dim = 1. Therefore, representation of a SparseTensor of sparse_dim = 0 is simply a dense tensor.

Note

Our sparse tensor format permits _uncoalesced_ sparse tensors, where there may be duplicate coordinates in the indices; in this case, the interpretation is that the value at that index is the sum of all duplicate value entries. Uncoalesced tensors permit us to implement certain operators more efficiently.

For the most part, you shouldn’t have to care whether or not a sparse tensor is coalesced or not, as most operations will work identically given a coalesced or uncoalesced sparse tensor. However, there are two cases in which you may need to care.

First, if you repeatedly perform an operation that can produce duplicate entries (e.g., [`torch.sparse.FloatTensor.add()`](#torch.sparse.FloatTensor.add "torch.sparse.FloatTensor.add")), you should occasionally coalesce your sparse tensors to prevent them from growing too large.

Second, some operators will produce different values depending on whether or not they are coalesced or not (e.g., [`torch.sparse.FloatTensor._values()`](#torch.sparse.FloatTensor._values "torch.sparse.FloatTensor._values") and [`torch.sparse.FloatTensor._indices()`](#torch.sparse.FloatTensor._indices "torch.sparse.FloatTensor._indices"), as well as [`torch.Tensor.sparse_mask()`](tensors.html#torch.Tensor.sparse_mask "torch.Tensor.sparse_mask")). These operators are prefixed by an underscore to indicate that they reveal internal implementation details and should be used with care, since code that works with coalesced sparse tensors may not work with uncoalesced sparse tensors; generally speaking, it is safest to explicitly coalesce before working with these operators.

For example, suppose that we wanted to implement an operator by operating directly on [`torch.sparse.FloatTensor._values()`](#torch.sparse.FloatTensor._values "torch.sparse.FloatTensor._values"). Multiplication by a scalar can be implemented in the obvious way, as multiplication distributes over addition; however, square root cannot be implemented directly, since `sqrt(a + b) != sqrt(a) + sqrt(b)` (which is what would be computed if you were given an uncoalesced tensor.)

```py
class torch.sparse.FloatTensor
```

```py
add()
```

```py
add_()
```

```py
clone()
```

```py
dim()
```

```py
div()
```

```py
div_()
```

```py
get_device()
```

```py
hspmm()
```

```py
mm()
```

```py
mul()
```

```py
mul_()
```

```py
narrow_copy()
```

```py
resizeAs_()
```

```py
size()
```

```py
spadd()
```

```py
spmm()
```

```py
sspaddmm()
```

```py
sspmm()
```

```py
sub()
```

```py
sub_()
```

```py
t_()
```

```py
toDense()
```

```py
transpose()
```

```py
transpose_()
```

```py
zero_()
```

```py
coalesce()
```

```py
is_coalesced()
```

```py
_indices()
```

```py
_values()
```

```py
_nnz()
```

## Functions

```py
torch.sparse.addmm(mat, mat1, mat2, beta=1, alpha=1)
```

This function does exact same thing as [`torch.addmm()`](torch.html#torch.addmm "torch.addmm") in the forward, except that it supports backward for sparse matrix `mat1`. `mat1` need to have &lt;cite&gt;sparse_dim = 2&lt;/cite&gt;. Note that the gradients of `mat1` is a coalesced sparse tensor.

| Parameters: | 

*   **mat** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – a dense matrix to be added
*   **mat1** (_SparseTensor_) – a sparse matrix to be multiplied
*   **mat2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – a dense matrix be multiplied
*   **beta** (_Number__,_ _optional_) – multiplier for `mat` (![](img/50705df736e9a7919e768cf8c4e4f794.jpg))
*   **alpha** (_Number__,_ _optional_) – multiplier for ![](img/c4fda0ec33ee23096c7bac6105f7a619.jpg) (![](img/82005cc2e0087e2a52c7e43df4a19a00.jpg))

 |
| --- | --- |

```py
torch.sparse.mm(mat1, mat2)
```

Performs a matrix multiplication of the sparse matrix `mat1` and dense matrix `mat2`. Similar to [`torch.mm()`](torch.html#torch.mm "torch.mm"), If `mat1` is a ![](img/b2d82f601df5521e215e30962b942ad1.jpg) tensor, `mat2` is a ![](img/ec84c2d649caa2a7d4dc59b6b23b0278.jpg) tensor, out will be a ![](img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) dense tensor. `mat1` need to have &lt;cite&gt;sparse_dim = 2&lt;/cite&gt;. This function also supports backward for both matrices. Note that the gradients of `mat1` is a coalesced sparse tensor.

| Parameters: | 

*   **mat1** (_SparseTensor_) – the first sparse matrix to be multiplied
*   **mat2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second dense matrix to be multiplied

 |
| --- | --- |

Example:

```py
>>> a = torch.randn(2, 3).to_sparse().requires_grad_(True)
>>> a
tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
 [0, 1, 2, 0, 1, 2]]),
 values=tensor([ 1.5901,  0.0183, -0.6146,  1.8061, -0.0112,  0.6302]),
 size=(2, 3), nnz=6, layout=torch.sparse_coo, requires_grad=True)

>>> b = torch.randn(3, 2, requires_grad=True)
>>> b
tensor([[-0.6479,  0.7874],
 [-1.2056,  0.5641],
 [-1.1716, -0.9923]], requires_grad=True)

>>> y = torch.sparse.mm(a, b)
>>> y
tensor([[-0.3323,  1.8723],
 [-1.8951,  0.7904]], grad_fn=<SparseAddmmBackward>)
>>> y.sum().backward()
>>> a.grad
tensor(indices=tensor([[0, 0, 0, 1, 1, 1],
 [0, 1, 2, 0, 1, 2]]),
 values=tensor([ 0.1394, -0.6415, -2.1639,  0.1394, -0.6415, -2.1639]),
 size=(2, 3), nnz=6, layout=torch.sparse_coo)

```

```py
torch.sparse.sum(input, dim=None, dtype=None)
```

Returns the sum of each row of SparseTensor `input` in the given dimensions `dim`. If :attr::&lt;cite&gt;dim&lt;/cite&gt; is a list of dimensions, reduce over all of them. When sum over all `sparse_dim`, this method returns a Tensor instead of SparseTensor.

All summed `dim` are squeezed (see [`torch.squeeze()`](torch.html#torch.squeeze "torch.squeeze")), resulting an output tensor having :attr::&lt;cite&gt;dim&lt;/cite&gt; fewer dimensions than `input`.

During backward, only gradients at `nnz` locations of `input` will propagate back. Note that the gradients of `input` is coalesced.

| Parameters: | 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input SparseTensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _tuple of python:ints_) – a dimension or a list of dimensions to reduce. Default: reduce over all dims.
*   **dtype** (`torch.dtype`, optional) – the desired data type of returned Tensor. Default: dtype of `input`.

 |
| --- | --- |

Example:

```py
>>> nnz = 3
>>> dims = [5, 5, 2, 3]
>>> I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
 torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
>>> V = torch.randn(nnz, dims[2], dims[3])
>>> size = torch.Size(dims)
>>> S = torch.sparse_coo_tensor(I, V, size)
>>> S
tensor(indices=tensor([[2, 0, 3],
 [2, 4, 1]]),
 values=tensor([[[-0.6438, -1.6467,  1.4004],
 [ 0.3411,  0.0918, -0.2312]],

 [[ 0.5348,  0.0634, -2.0494],
 [-0.7125, -1.0646,  2.1844]],

 [[ 0.1276,  0.1874, -0.6334],
 [-1.9682, -0.5340,  0.7483]]]),
 size=(5, 5, 2, 3), nnz=3, layout=torch.sparse_coo)

# when sum over only part of sparse_dims, return a SparseTensor
>>> torch.sparse.sum(S, [1, 3])
tensor(indices=tensor([[0, 2, 3]]),
 values=tensor([[-1.4512,  0.4073],
 [-0.8901,  0.2017],
 [-0.3183, -1.7539]]),
 size=(5, 2), nnz=3, layout=torch.sparse_coo)

# when sum over all sparse dim, return a dense Tensor
# with summed dims squeezed
>>> torch.sparse.sum(S, [0, 1, 3])
tensor([-2.6596, -1.1450])

```

