### BLAS and LAPACK Operations

```py
torch.addbmm(beta=1, mat, alpha=1, batch1, batch2, out=None) → Tensor
```

Performs a batch matrix-matrix product of matrices stored in `batch1` and `batch2`, with a reduced add step (all matrix multiplications get accumulated along the first dimension). `mat` is added to the final result.

`batch1` and `batch2` must be 3-D tensors each containing the same number of matrices.

If `batch1` is a ![](img/eccd104fbbbbb116c7e98ca54b2214a0.jpg) tensor, `batch2` is a ![](img/f8b603730e091b70ad24e5a089cdd30f.jpg) tensor, `mat` must be [broadcastable](notes/broadcasting.html#broadcasting-semantics) with a ![](img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) tensor and `out` will be a ![](img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) tensor.

![](img/e9a3c5b413385d813461f90cc06b1454.jpg)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be real numbers, otherwise they should be integers.

Parameters: 

*   **beta** (_Number__,_ _optional_) – multiplier for `mat` (![](img/50705df736e9a7919e768cf8c4e4f794.jpg))
*   **mat** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – matrix to be added
*   **alpha** (_Number__,_ _optional_) – multiplier for `batch1 @ batch2` (![](img/82005cc2e0087e2a52c7e43df4a19a00.jpg))
*   **batch1** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the first batch of matrices to be multiplied
*   **batch2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second batch of matrices to be multiplied
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> M = torch.randn(3, 5)
>>> batch1 = torch.randn(10, 3, 4)
>>> batch2 = torch.randn(10, 4, 5)
>>> torch.addbmm(M, batch1, batch2)
tensor([[  6.6311,   0.0503,   6.9768, -12.0362,  -2.1653],
 [ -4.8185,  -1.4255,  -6.6760,   8.9453,   2.5743],
 [ -3.8202,   4.3691,   1.0943,  -1.1109,   5.4730]])

```

```py
torch.addmm(beta=1, mat, alpha=1, mat1, mat2, out=None) → Tensor
```

Performs a matrix multiplication of the matrices `mat1` and `mat2`. The matrix `mat` is added to the final result.

If `mat1` is a ![](img/b2d82f601df5521e215e30962b942ad1.jpg) tensor, `mat2` is a ![](img/ec84c2d649caa2a7d4dc59b6b23b0278.jpg) tensor, then `mat` must be [broadcastable](notes/broadcasting.html#broadcasting-semantics) with a ![](img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) tensor and `out` will be a ![](img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) tensor.

`alpha` and `beta` are scaling factors on matrix-vector product between `mat1` and :attr`mat2` and the added matrix `mat` respectively.

![](img/8d4b0912f137549bc9b2dc4ee38a0a40.jpg)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be real numbers, otherwise they should be integers.

Parameters: 

*   **beta** (_Number__,_ _optional_) – multiplier for `mat` (![](img/50705df736e9a7919e768cf8c4e4f794.jpg))
*   **mat** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – matrix to be added
*   **alpha** (_Number__,_ _optional_) – multiplier for ![](img/c4fda0ec33ee23096c7bac6105f7a619.jpg) (![](img/82005cc2e0087e2a52c7e43df4a19a00.jpg))
*   **mat1** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the first matrix to be multiplied
*   **mat2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second matrix to be multiplied
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> M = torch.randn(2, 3)
>>> mat1 = torch.randn(2, 3)
>>> mat2 = torch.randn(3, 3)
>>> torch.addmm(M, mat1, mat2)
tensor([[-4.8716,  1.4671, -1.3746],
 [ 0.7573, -3.9555, -2.8681]])

```

```py
torch.addmv(beta=1, tensor, alpha=1, mat, vec, out=None) → Tensor
```

Performs a matrix-vector product of the matrix `mat` and the vector `vec`. The vector [`tensor`](#torch.tensor "torch.tensor") is added to the final result.

If `mat` is a ![](img/b2d82f601df5521e215e30962b942ad1.jpg) tensor, `vec` is a 1-D tensor of size `m`, then [`tensor`](#torch.tensor "torch.tensor") must be [broadcastable](notes/broadcasting.html#broadcasting-semantics) with a 1-D tensor of size `n` and `out` will be 1-D tensor of size `n`.

`alpha` and `beta` are scaling factors on matrix-vector product between `mat` and `vec` and the added tensor [`tensor`](#torch.tensor "torch.tensor") respectively.

![](img/4188eb7768951ccca87969272bcfa3a7.jpg)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be real numbers, otherwise they should be integers

Parameters: 

*   **beta** (_Number__,_ _optional_) – multiplier for [`tensor`](#torch.tensor "torch.tensor") (![](img/50705df736e9a7919e768cf8c4e4f794.jpg))
*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – vector to be added
*   **alpha** (_Number__,_ _optional_) – multiplier for ![](img/a901c2282b0dbdcf23379ddd5a3c274b.jpg) (![](img/82005cc2e0087e2a52c7e43df4a19a00.jpg))
*   **mat** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – matrix to be multiplied
*   **vec** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – vector to be multiplied
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> M = torch.randn(2)
>>> mat = torch.randn(2, 3)
>>> vec = torch.randn(3)
>>> torch.addmv(M, mat, vec)
tensor([-0.3768, -5.5565])

```

```py
torch.addr(beta=1, mat, alpha=1, vec1, vec2, out=None) → Tensor
```

Performs the outer-product of vectors `vec1` and `vec2` and adds it to the matrix `mat`.

Optional values `beta` and `alpha` are scaling factors on the outer product between `vec1` and `vec2` and the added matrix `mat` respectively.

![](img/171f2173f3a92cea6433a9dd012888ad.jpg)

If `vec1` is a vector of size `n` and `vec2` is a vector of size `m`, then `mat` must be [broadcastable](notes/broadcasting.html#broadcasting-semantics) with a matrix of size ![](img/b2d82f601df5521e215e30962b942ad1.jpg) and `out` will be a matrix of size ![](img/b2d82f601df5521e215e30962b942ad1.jpg).

For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be real numbers, otherwise they should be integers

Parameters: 

*   **beta** (_Number__,_ _optional_) – multiplier for `mat` (![](img/50705df736e9a7919e768cf8c4e4f794.jpg))
*   **mat** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – matrix to be added
*   **alpha** (_Number__,_ _optional_) – multiplier for ![](img/3f2eb83c372296996af0ac869a078ebd.jpg) (![](img/82005cc2e0087e2a52c7e43df4a19a00.jpg))
*   **vec1** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the first vector of the outer product
*   **vec2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second vector of the outer product
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> vec1 = torch.arange(1., 4.)
>>> vec2 = torch.arange(1., 3.)
>>> M = torch.zeros(3, 2)
>>> torch.addr(M, vec1, vec2)
tensor([[ 1.,  2.],
 [ 2.,  4.],
 [ 3.,  6.]])

```

```py
torch.baddbmm(beta=1, mat, alpha=1, batch1, batch2, out=None) → Tensor
```

Performs a batch matrix-matrix product of matrices in `batch1` and `batch2`. `mat` is added to the final result.

`batch1` and `batch2` must be 3-D tensors each containing the same number of matrices.

If `batch1` is a ![](img/eccd104fbbbbb116c7e98ca54b2214a0.jpg) tensor, `batch2` is a ![](img/f8b603730e091b70ad24e5a089cdd30f.jpg) tensor, then `mat` must be [broadcastable](notes/broadcasting.html#broadcasting-semantics) with a ![](img/29f0e4a370460668f7e257b22d08622d.jpg) tensor and `out` will be a ![](img/29f0e4a370460668f7e257b22d08622d.jpg) tensor. Both `alpha` and `beta` mean the same as the scaling factors used in [`torch.addbmm()`](#torch.addbmm "torch.addbmm").

![](img/069d82fba319e5aec62a5ad55fd0d01c.jpg)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be real numbers, otherwise they should be integers.

Parameters: 

*   **beta** (_Number__,_ _optional_) – multiplier for `mat` (![](img/50705df736e9a7919e768cf8c4e4f794.jpg))
*   **mat** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to be added
*   **alpha** (_Number__,_ _optional_) – multiplier for ![](img/c9ac2542d6edbedec1234ae90d5bf79f.jpg) (![](img/82005cc2e0087e2a52c7e43df4a19a00.jpg))
*   **batch1** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the first batch of matrices to be multiplied
*   **batch2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second batch of matrices to be multiplied
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> M = torch.randn(10, 3, 5)
>>> batch1 = torch.randn(10, 3, 4)
>>> batch2 = torch.randn(10, 4, 5)
>>> torch.baddbmm(M, batch1, batch2).size()
torch.Size([10, 3, 5])

```

```py
torch.bmm(batch1, batch2, out=None) → Tensor
```

Performs a batch matrix-matrix product of matrices stored in `batch1` and `batch2`.

`batch1` and `batch2` must be 3-D tensors each containing the same number of matrices.

If `batch1` is a ![](img/eccd104fbbbbb116c7e98ca54b2214a0.jpg) tensor, `batch2` is a ![](img/f8b603730e091b70ad24e5a089cdd30f.jpg) tensor, `out` will be a ![](img/29f0e4a370460668f7e257b22d08622d.jpg) tensor.

![](img/699b5d44b53e8c67d763dc6fb072e488.jpg)

Note

This function does not [broadcast](notes/broadcasting.html#broadcasting-semantics). For broadcasting matrix products, see [`torch.matmul()`](#torch.matmul "torch.matmul").

Parameters: 

*   **batch1** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the first batch of matrices to be multiplied
*   **batch2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second batch of matrices to be multiplied
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> batch1 = torch.randn(10, 3, 4)
>>> batch2 = torch.randn(10, 4, 5)
>>> res = torch.bmm(batch1, batch2)
>>> res.size()
torch.Size([10, 3, 5])

```

```py
torch.btrifact(A, info=None, pivot=True)
```

Batch LU factorization.

Returns a tuple containing the LU factorization and pivots. Pivoting is done if `pivot` is set.

The optional argument `info` stores information if the factorization succeeded for each minibatch example. The `info` is provided as an `IntTensor`, its values will be filled from dgetrf and a non-zero value indicates an error occurred. Specifically, the values are from cublas if cuda is being used, otherwise LAPACK.

Warning

The `info` argument is deprecated in favor of [`torch.btrifact_with_info()`](#torch.btrifact_with_info "torch.btrifact_with_info").

Parameters: 

*   **A** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to factor
*   **info** (_IntTensor__,_ _optional_) – (deprecated) an `IntTensor` to store values indicating whether factorization succeeds
*   **pivot** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether pivoting is done


| Returns: | A tuple containing factorization and pivots. |
| --- | --- |

Example:

```py
>>> A = torch.randn(2, 3, 3)
>>> A_LU, pivots = torch.btrifact(A)
>>> A_LU
tensor([[[ 1.3506,  2.5558, -0.0816],
 [ 0.1684,  1.1551,  0.1940],
 [ 0.1193,  0.6189, -0.5497]],

 [[ 0.4526,  1.2526, -0.3285],
 [-0.7988,  0.7175, -0.9701],
 [ 0.2634, -0.9255, -0.3459]]])

>>> pivots
tensor([[ 3,  3,  3],
 [ 3,  3,  3]], dtype=torch.int32)

```

```py
torch.btrifact_with_info(A, pivot=True) -> (Tensor, IntTensor, IntTensor)
```

Batch LU factorization with additional error information.

This is a version of [`torch.btrifact()`](#torch.btrifact "torch.btrifact") that always creates an info `IntTensor`, and returns it as the third return value.

Parameters: 

*   **A** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to factor
*   **pivot** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls whether pivoting is done


| Returns: | A tuple containing factorization, pivots, and an `IntTensor` where non-zero values indicate whether factorization for each minibatch sample succeeds. |
| --- | --- |

Example:

```py
>>> A = torch.randn(2, 3, 3)
>>> A_LU, pivots, info = A.btrifact_with_info()
>>> if info.nonzero().size(0) == 0:
>>>   print('LU factorization succeeded for all samples!')
LU factorization succeeded for all samples!

```

```py
torch.btrisolve(b, LU_data, LU_pivots) → Tensor
```

Batch LU solve.

Returns the LU solve of the linear system ![](img/79f5b7df86014d0a54c744c91d8b351d.jpg).

Parameters: 

*   **b** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the RHS tensor
*   **LU_data** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the pivoted LU factorization of A from [`btrifact()`](#torch.btrifact "torch.btrifact").
*   **LU_pivots** (_IntTensor_) – the pivots of the LU factorization



Example:

```py
>>> A = torch.randn(2, 3, 3)
>>> b = torch.randn(2, 3)
>>> A_LU = torch.btrifact(A)
>>> x = torch.btrisolve(b, *A_LU)
>>> torch.norm(torch.bmm(A, x.unsqueeze(2)) - b.unsqueeze(2))
tensor(1.00000e-07 *
 2.8312)

```

```py
torch.btriunpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True)
```

Unpacks the data and pivots from a batched LU factorization (btrifact) of a tensor.

Returns a tuple of tensors as `(the pivots, the L tensor, the U tensor)`.

Parameters: 

*   **LU_data** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the packed LU factorization data
*   **LU_pivots** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the packed LU factorization pivots
*   **unpack_data** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – flag indicating if the data should be unpacked
*   **unpack_pivots** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – flag indicating if the pivots should be unpacked



Example:

```py
>>> A = torch.randn(2, 3, 3)
>>> A_LU, pivots = A.btrifact()
>>> P, A_L, A_U = torch.btriunpack(A_LU, pivots)
>>>
>>> # can recover A from factorization
>>> A_ = torch.bmm(P, torch.bmm(A_L, A_U))

```

```py
torch.chain_matmul(*matrices)
```

Returns the matrix product of the ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) 2-D tensors. This product is efficiently computed using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms of arithmetic operations ([[CLRS]](https://mitpress.mit.edu/books/introduction-algorithms-third-edition)). Note that since this is a function to compute the product, ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) needs to be greater than or equal to 2; if equal to 2 then a trivial matrix-matrix product is returned. If ![](img/9341d9048ac485106d2b2ee8de14876f.jpg) is 1, then this is a no-op - the original matrix is returned as is.

| Parameters: | **matrices** (_Tensors..._) – a sequence of 2 or more 2-D tensors whose product is to be determined. |
| --- | --- |
| Returns: | if the ![](img/5c5e7583f110d90e938149340dd42e92.jpg) tensor was of dimensions ![](img/fd285b0b789ab6ce131b7a0208da2fe0.jpg), then the product would be of dimensions ![](img/61c4b45c29064296a380ab945a449672.jpg). |
| --- | --- |
| Return type: | [Tensor](tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

Example:

```py
>>> a = torch.randn(3, 4)
>>> b = torch.randn(4, 5)
>>> c = torch.randn(5, 6)
>>> d = torch.randn(6, 7)
>>> torch.chain_matmul(a, b, c, d)
tensor([[ -2.3375,  -3.9790,  -4.1119,  -6.6577,   9.5609, -11.5095,  -3.2614],
 [ 21.4038,   3.3378,  -8.4982,  -5.2457, -10.2561,  -2.4684,   2.7163],
 [ -0.9647,  -5.8917,  -2.3213,  -5.2284,  12.8615, -12.2816,  -2.5095]])

```

```py
torch.cholesky(A, upper=False, out=None) → Tensor
```

Computes the Cholesky decomposition of a symmetric positive-definite matrix ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg) or for batches of symmetric positive-definite matrices.

If `upper` is `True`, the returned matrix `U` is upper-triangular, and the decomposition has the form:

![](img/7100a5dce6b64985eeb45416a640b7e6.jpg)

If `upper` is `False`, the returned matrix `L` is lower-triangular, and the decomposition has the form:

![](img/3ec14c9e61e88b877808fab3bbdd17ca.jpg)

If `upper` is `True`, and `A` is a batch of symmetric positive-definite matrices, then the returned tensor will be composed of upper-triangular Cholesky factors of each of the individual matrices. Similarly, when `upper` is `False`, the returned tensor will be composed of lower-triangular Cholesky factors of each of the individual matrices.

Parameters: 

*   **a** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor of size ([*](#id6), n, n) where `*` is zero or more batch dimensions consisting of symmetric positive-definite matrices.
*   **upper** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – flag that indicates whether to return a upper or lower triangular matrix. Default: `False`
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output matrix



Example:

```py
>>> a = torch.randn(3, 3)
>>> a = torch.mm(a, a.t()) # make symmetric positive-definite
>>> l = torch.cholesky(a)
>>> a
tensor([[ 2.4112, -0.7486,  1.4551],
 [-0.7486,  1.3544,  0.1294],
 [ 1.4551,  0.1294,  1.6724]])
>>> l
tensor([[ 1.5528,  0.0000,  0.0000],
 [-0.4821,  1.0592,  0.0000],
 [ 0.9371,  0.5487,  0.7023]])
>>> torch.mm(l, l.t())
tensor([[ 2.4112, -0.7486,  1.4551],
 [-0.7486,  1.3544,  0.1294],
 [ 1.4551,  0.1294,  1.6724]])
>>> a = torch.randn(3, 2, 2)
>>> a = torch.matmul(a, a.transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
>>> l = torch.cholesky(a)
>>> z = torch.matmul(l, l.transpose(-1, -2))
>>> torch.max(torch.abs(z - a)) # Max non-zero
tensor(2.3842e-07)

```

```py
torch.dot(tensor1, tensor2) → Tensor
```

Computes the dot product (inner product) of two tensors.

Note

This function does not [broadcast](notes/broadcasting.html#broadcasting-semantics).

Example:

```py
>>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
tensor(7)

```

```py
torch.eig(a, eigenvectors=False, out=None) -> (Tensor, Tensor)
```

Computes the eigenvalues and eigenvectors of a real square matrix.

Parameters: 

*   **a** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the square matrix of shape ![](img/7819768bc0adceb9951cf2ce9a0525f2.jpg) for which the eigenvalues and eigenvectors will be computed
*   **eigenvectors** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – `True` to compute both eigenvalues and eigenvectors; otherwise, only eigenvalues will be computed
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the output tensors


| Returns: | A tuple containing

&gt; *   **e** (_Tensor_): Shape ![](img/6bb1e4cc787b2a2a3e362c6385033b7d.jpg). Each row is an eigenvalue of `a`, where the first element is the real part and the second element is the imaginary part. The eigenvalues are not necessarily ordered.
&gt; *   **v** (_Tensor_): If `eigenvectors=False`, it’s an empty tensor. Otherwise, this tensor of shape ![](img/7819768bc0adceb9951cf2ce9a0525f2.jpg) can be used to compute normalized (unit length) eigenvectors of corresponding eigenvalues `e` as follows. If the corresponding e[j] is a real number, column v[:, j] is the eigenvector corresponding to eigenvalue e[j]. If the corresponding e[j] and e[j + 1] eigenvalues form a complex conjugate pair, then the true eigenvectors can be computed as ![](img/ec9513691a2c7521c03807425da807ed.jpg), ![](img/a532b8aa12f3a5051c8105bf8e226b64.jpg).


| Return type: | ([Tensor](tensors.html#torch.Tensor "torch.Tensor"), [Tensor](tensors.html#torch.Tensor "torch.Tensor")) |
| --- | --- |

```py
torch.gels(B, A, out=None) → Tensor
```

Computes the solution to the least squares and least norm problems for a full rank matrix ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg) of size ![](img/cc3ea6b8d05f85433fd7aa6a20c33408.jpg) and a matrix ![](img/813135a6280e2672503128d3d2080d4a.jpg) of size ![](img/602cd3c92249bd53b21908f902ff6089.jpg).

If ![](img/199260d72e51fe506909a150c6f77020.jpg), [`gels()`](#torch.gels "torch.gels") solves the least-squares problem:

![](img/6d9885b2d1646d00061288d0c063790b.jpg)

If ![](img/bc0a7901ab359873a58a64e43f9fc85a.jpg), [`gels()`](#torch.gels "torch.gels") solves the least-norm problem:

![](img/921685e425167b15563b245ca59e3ac3.jpg)

Returned tensor ![](img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg) has shape ![](img/b10f10ee19a21653d24c909eb3e0877a.jpg). The first ![](img/493731e423d5db62086d0b8705dda0c8.jpg) rows of ![](img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg) contains the solution. If ![](img/199260d72e51fe506909a150c6f77020.jpg), the residual sum of squares for the solution in each column is given by the sum of squares of elements in the remaining ![](img/51e590a6a852e7b2d4cd0a3476859fc5.jpg) rows of that column.

Parameters: 

*   **B** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the matrix ![](img/813135a6280e2672503128d3d2080d4a.jpg)
*   **A** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the ![](img/20ddd8181c2e0d0fb893637e8572d475.jpg) by ![](img/493731e423d5db62086d0b8705dda0c8.jpg) matrix ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg)
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the optional destination tensor


| Returns: | A tuple containing:

&gt; *   **X** (_Tensor_): the least squares solution
&gt; *   **qr** (_Tensor_): the details of the QR factorization


| Return type: | ([Tensor](tensors.html#torch.Tensor "torch.Tensor"), [Tensor](tensors.html#torch.Tensor "torch.Tensor")) |
| --- | --- |

Note

The returned matrices will always be transposed, irrespective of the strides of the input matrices. That is, they will have stride `(1, m)` instead of `(m, 1)`.

Example:

```py
>>> A = torch.tensor([[1., 1, 1],
 [2, 3, 4],
 [3, 5, 2],
 [4, 2, 5],
 [5, 4, 3]])
>>> B = torch.tensor([[-10., -3],
 [ 12, 14],
 [ 14, 12],
 [ 16, 16],
 [ 18, 16]])
>>> X, _ = torch.gels(B, A)
>>> X
tensor([[  2.0000,   1.0000],
 [  1.0000,   1.0000],
 [  1.0000,   2.0000],
 [ 10.9635,   4.8501],
 [  8.9332,   5.2418]])

```

```py
torch.geqrf(input, out=None) -> (Tensor, Tensor)
```

This is a low-level function for calling LAPACK directly.

You’ll generally want to use [`torch.qr()`](#torch.qr "torch.qr") instead.

Computes a QR decomposition of `input`, but without constructing ![](img/1d680db5f32fd278f8d48e5407691154.jpg) and ![](img/502cdd9c79852b33d2a6d18ba5ec3102.jpg) as explicit separate matrices.

Rather, this directly calls the underlying LAPACK function `?geqrf` which produces a sequence of ‘elementary reflectors’.

See [LAPACK documentation for geqrf](https://software.intel.com/en-us/node/521004) for further details.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input matrix
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the output tuple of (Tensor, Tensor)



```py
torch.ger(vec1, vec2, out=None) → Tensor
```

Outer product of `vec1` and `vec2`. If `vec1` is a vector of size ![](img/493731e423d5db62086d0b8705dda0c8.jpg) and `vec2` is a vector of size ![](img/20ddd8181c2e0d0fb893637e8572d475.jpg), then `out` must be a matrix of size ![](img/b2d82f601df5521e215e30962b942ad1.jpg).

Note

This function does not [broadcast](notes/broadcasting.html#broadcasting-semantics).

Parameters: 

*   **vec1** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 1-D input vector
*   **vec2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 1-D input vector
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – optional output matrix



Example:

```py
>>> v1 = torch.arange(1., 5.)
>>> v2 = torch.arange(1., 4.)
>>> torch.ger(v1, v2)
tensor([[  1.,   2.,   3.],
 [  2.,   4.,   6.],
 [  3.,   6.,   9.],
 [  4.,   8.,  12.]])

```

```py
torch.gesv(B, A) -> (Tensor, Tensor)
```

This function returns the solution to the system of linear equations represented by ![](img/9c11b6313ae06c752584c5c1b2c03964.jpg) and the LU factorization of A, in order as a tuple `X, LU`.

`LU` contains `L` and `U` factors for LU factorization of `A`.

`torch.gesv(B, A)` can take in 2D inputs `B, A` or inputs that are batches of 2D matrices. If the inputs are batches, then returns batched outputs `X, LU`.

Note

The `out` keyword only supports 2D matrix inputs, that is, `B, A` must be 2D matrices.

Note

Irrespective of the original strides, the returned matrices `X` and `LU` will be transposed, i.e. with strides like `B.contiguous().transpose(-1, -2).strides()` and `A.contiguous().transpose(-1, -2).strides()` respectively.

Parameters: 

*   **B** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input matrix of size ![](img/d9795910f977049c4df2084f47c592ed.jpg) , where ![](img/28ec51e742166ea3400be6e7343bbfa5.jpg) is zero or more batch dimensions.
*   **A** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input square matrix of size ![](img/494aaae2a24df44c813ce87b9f21d745.jpg), where ![](img/28ec51e742166ea3400be6e7343bbfa5.jpg) is zero or more batch dimensions.
*   **out** (_(_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_)__,_ _optional_) – optional output tuple.



Example:

```py
>>> A = torch.tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
 [-6.05, -3.30,  5.36, -4.44,  1.08],
 [-0.45,  2.58, -2.70,  0.27,  9.04],
 [8.32,  2.71,  4.35,  -7.17,  2.14],
 [-9.67, -5.14, -7.26,  6.08, -6.87]]).t()
>>> B = torch.tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
 [-1.56,  4.00, -8.67,  1.75,  2.86],
 [9.81, -4.09, -4.57, -8.61,  8.99]]).t()
>>> X, LU = torch.gesv(B, A)
>>> torch.dist(B, torch.mm(A, X))
tensor(1.00000e-06 *
 7.0977)

>>> # Batched solver example
>>> A = torch.randn(2, 3, 1, 4, 4)
>>> B = torch.randn(2, 3, 1, 4, 6)
>>> X, LU = torch.gesv(B, A)
>>> torch.dist(B, A.matmul(X))
tensor(1.00000e-06 *
 3.6386)

```

```py
torch.inverse(input, out=None) → Tensor
```

Takes the inverse of the square matrix `input`. `input` can be batches of 2D square tensors, in which case this function would return a tensor composed of individual inverses.

Note

Irrespective of the original strides, the returned tensors will be transposed, i.e. with strides like `input.contiguous().transpose(-2, -1).strides()`

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor of size ([*](#id8), n, n) where `*` is zero or more batch dimensions
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the optional output tensor



Example:

```py
>>> x = torch.rand(4, 4)
>>> y = torch.inverse(x)
>>> z = torch.mm(x, y)
>>> z
tensor([[ 1.0000, -0.0000, -0.0000,  0.0000],
 [ 0.0000,  1.0000,  0.0000,  0.0000],
 [ 0.0000,  0.0000,  1.0000,  0.0000],
 [ 0.0000, -0.0000, -0.0000,  1.0000]])
>>> torch.max(torch.abs(z - torch.eye(4))) # Max non-zero
tensor(1.1921e-07)
>>> # Batched inverse example
>>> x = torch.randn(2, 3, 4, 4)
>>> y = torch.inverse(x)
>>> z = torch.matmul(x, y)
>>> torch.max(torch.abs(z - torch.eye(4).expand_as(x))) # Max non-zero
tensor(1.9073e-06)

```

```py
torch.det(A) → Tensor
```

Calculates determinant of a 2D square tensor.

Note

Backward through [`det()`](#torch.det "torch.det") internally uses SVD results when `A` is not invertible. In this case, double backward through [`det()`](#torch.det "torch.det") will be unstable in when `A` doesn’t have distinct singular values. See [`svd()`](#torch.svd "torch.svd") for details.

| Parameters: | **A** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – The input 2D square tensor |
| --- | --- |

Example:

```py
>>> A = torch.randn(3, 3)
>>> torch.det(A)
tensor(3.7641)

```

```py
torch.logdet(A) → Tensor
```

Calculates log determinant of a 2D square tensor.

Note

Result is `-inf` if `A` has zero log determinant, and is `nan` if `A` has negative determinant.

Note

Backward through [`logdet()`](#torch.logdet "torch.logdet") internally uses SVD results when `A` is not invertible. In this case, double backward through [`logdet()`](#torch.logdet "torch.logdet") will be unstable in when `A` doesn’t have distinct singular values. See [`svd()`](#torch.svd "torch.svd") for details.

| Parameters: | **A** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – The input 2D square tensor |
| --- | --- |

Example:

```py
>>> A = torch.randn(3, 3)
>>> torch.det(A)
tensor(0.2611)
>>> torch.logdet(A)
tensor(-1.3430)

```

```py
torch.slogdet(A) -> (Tensor, Tensor)
```

Calculates the sign and log value of a 2D square tensor’s determinant.

Note

If `A` has zero determinant, this returns `(0, -inf)`.

Note

Backward through [`slogdet()`](#torch.slogdet "torch.slogdet") internally uses SVD results when `A` is not invertible. In this case, double backward through [`slogdet()`](#torch.slogdet "torch.slogdet") will be unstable in when `A` doesn’t have distinct singular values. See [`svd()`](#torch.svd "torch.svd") for details.

| Parameters: | **A** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – The input 2D square tensor |
| --- | --- |
| Returns: | A tuple containing the sign of the determinant, and the log value of the absolute determinant. |
| --- | --- |

Example:

```py
>>> A = torch.randn(3, 3)
>>> torch.det(A)
tensor(-4.8215)
>>> torch.logdet(A)
tensor(nan)
>>> torch.slogdet(A)
(tensor(-1.), tensor(1.5731))

```

```py
torch.matmul(tensor1, tensor2, out=None) → Tensor
```

Matrix product of two tensors.

The behavior depends on the dimensionality of the tensors as follows:

*   If both tensors are 1-dimensional, the dot product (scalar) is returned.
*   If both arguments are 2-dimensional, the matrix-matrix product is returned.
*   If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
*   If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
*   If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N &gt; 2), then a batched matrix multiply is returned. If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after. If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after. The non-matrix (i.e. batch) dimensions are [broadcasted](notes/broadcasting.html#broadcasting-semantics) (and thus must be broadcastable). For example, if `tensor1` is a ![](img/a4697ce48760baf0633769e49f46b335.jpg) tensor and `tensor2` is a ![](img/ad9fbe324dcc50cc2232a9c1a2675daf.jpg) tensor, `out` will be an ![](img/6082fde6b0f498f9ed21c0ac7a9709d3.jpg) tensor.

Note

The 1-dimensional dot product version of this function does not support an `out` parameter.

Parameters: 

*   **tensor1** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the first tensor to be multiplied
*   **tensor2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second tensor to be multiplied
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> # vector x vector
>>> tensor1 = torch.randn(3)
>>> tensor2 = torch.randn(3)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([])
>>> # matrix x vector
>>> tensor1 = torch.randn(3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([3])
>>> # batched matrix x broadcasted vector
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3])
>>> # batched matrix x batched matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(10, 4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
>>> # batched matrix x broadcasted matrix
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])

```

```py
torch.matrix_power(input, n) → Tensor
```

Returns the matrix raised to the power `n` for square matrices. For batch of matrices, each individual matrix is raised to the power `n`.

If `n` is negative, then the inverse of the matrix (if invertible) is raised to the power `n`. For a batch of matrices, the batched inverse (if invertible) is raised to the power `n`. If `n` is 0, then an identity matrix is returned.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **n** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – the power to raise the matrix to



Example:

```py
>>> a = torch.randn(2, 2, 2)
>>> a
tensor([[[-1.9975, -1.9610],
 [ 0.9592, -2.3364]],

 [[-1.2534, -1.3429],
 [ 0.4153, -1.4664]]])
>>> torch.matrix_power(a, 3)
tensor([[[  3.9392, -23.9916],
 [ 11.7357,  -0.2070]],

 [[  0.2468,  -6.7168],
 [  2.0774,  -0.8187]]])

```

```py
torch.matrix_rank(input, tol=None, bool symmetric=False) → Tensor
```

Returns the numerical rank of a 2-D tensor. The method to compute the matrix rank is done using SVD by default. If `symmetric` is `True`, then `input` is assumed to be symmetric, and the computation of the rank is done by obtaining the eigenvalues.

`tol` is the threshold below which the singular values (or the eigenvalues when `symmetric` is `True`) are considered to be 0\. If `tol` is not specified, `tol` is set to `S.max() * max(S.size()) * eps` where `S` is the singular values (or the eigenvalues when `symmetric` is `True`), and `eps` is the epsilon value for the datatype of `input`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input 2-D tensor
*   **tol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – the tolerance value. Default: `None`
*   **symmetric** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – indicates whether `input` is symmetric. Default: `False`



Example:

```py
>>> a = torch.eye(10)
>>> torch.matrix_rank(a)
tensor(10)
>>> b = torch.eye(10)
>>> b[0, 0] = 0
>>> torch.matrix_rank(b)
tensor(9)

```

```py
torch.mm(mat1, mat2, out=None) → Tensor
```

Performs a matrix multiplication of the matrices `mat1` and `mat2`.

If `mat1` is a ![](img/b2d82f601df5521e215e30962b942ad1.jpg) tensor, `mat2` is a ![](img/ec84c2d649caa2a7d4dc59b6b23b0278.jpg) tensor, `out` will be a ![](img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) tensor.

Note

This function does not [broadcast](notes/broadcasting.html#broadcasting-semantics). For broadcasting matrix products, see [`torch.matmul()`](#torch.matmul "torch.matmul").

Parameters: 

*   **mat1** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the first matrix to be multiplied
*   **mat2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second matrix to be multiplied
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> mat1 = torch.randn(2, 3)
>>> mat2 = torch.randn(3, 3)
>>> torch.mm(mat1, mat2)
tensor([[ 0.4851,  0.5037, -0.3633],
 [-0.0760, -3.6705,  2.4784]])

```

```py
torch.mv(mat, vec, out=None) → Tensor
```

Performs a matrix-vector product of the matrix `mat` and the vector `vec`.

If `mat` is a ![](img/b2d82f601df5521e215e30962b942ad1.jpg) tensor, `vec` is a 1-D tensor of size ![](img/20ddd8181c2e0d0fb893637e8572d475.jpg), `out` will be 1-D of size ![](img/493731e423d5db62086d0b8705dda0c8.jpg).

Note

This function does not [broadcast](notes/broadcasting.html#broadcasting-semantics).

Parameters: 

*   **mat** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – matrix to be multiplied
*   **vec** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – vector to be multiplied
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor



Example:

```py
>>> mat = torch.randn(2, 3)
>>> vec = torch.randn(3)
>>> torch.mv(mat, vec)
tensor([ 1.0404, -0.6361])

```

```py
torch.orgqr(a, tau) → Tensor
```

Computes the orthogonal matrix `Q` of a QR factorization, from the `(a, tau)` tuple returned by [`torch.geqrf()`](#torch.geqrf "torch.geqrf").

This directly calls the underlying LAPACK function `?orgqr`. See [LAPACK documentation for orgqr](https://software.intel.com/en-us/mkl-developer-reference-c-orgqr) for further details.

Parameters: 

*   **a** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the `a` from [`torch.geqrf()`](#torch.geqrf "torch.geqrf").
*   **tau** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the `tau` from [`torch.geqrf()`](#torch.geqrf "torch.geqrf").



```py
torch.ormqr(a, tau, mat, left=True, transpose=False) -> (Tensor, Tensor)
```

Multiplies `mat` by the orthogonal `Q` matrix of the QR factorization formed by [`torch.geqrf()`](#torch.geqrf "torch.geqrf") that is represented by `(a, tau)`.

This directly calls the underlying LAPACK function `?ormqr`. See [LAPACK documentation for ormqr](https://software.intel.com/en-us/mkl-developer-reference-c-ormqr) for further details.

Parameters: 

*   **a** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the `a` from [`torch.geqrf()`](#torch.geqrf "torch.geqrf").
*   **tau** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the `tau` from [`torch.geqrf()`](#torch.geqrf "torch.geqrf").
*   **mat** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the matrix to be multiplied.



```py
torch.pinverse(input, rcond=1e-15) → Tensor
```

Calculates the pseudo-inverse (also known as the Moore-Penrose inverse) of a 2D tensor. Please look at [Moore-Penrose inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) for more details

Note

This method is implemented using the Singular Value Decomposition.

Note

The pseudo-inverse is not necessarily a continuous function in the elements of the matrix [[1]](https://epubs.siam.org/doi/10.1137/0117004). Therefore, derivatives are not always existent, and exist for a constant rank only [[2]](https://www.jstor.org/stable/2156365). However, this method is backprop-able due to the implementation by using SVD results, and could be unstable. Double-backward will also be unstable due to the usage of SVD internally. See [`svd()`](#torch.svd "torch.svd") for more details.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – The input 2D tensor of dimensions ![](img/ee12b6c487a34051534acf84ddb3f98f.jpg)
*   **rcond** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – A floating point value to determine the cutoff for small singular values. Default: 1e-15


| Returns: | The pseudo-inverse of `input` of dimensions ![](img/3380c6697127aa874110f3e6faef8bdf.jpg) |
| --- | --- |

Example:

```py
>>> input = torch.randn(3, 5)
>>> input
tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
 [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
 [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
>>> torch.pinverse(input)
tensor([[ 0.0600, -0.1933, -0.2090],
 [-0.0903, -0.0817, -0.4752],
 [-0.7124, -0.1631, -0.2272],
 [ 0.1356,  0.3933, -0.5023],
 [-0.0308, -0.1725, -0.5216]])

```

```py
torch.potrf(a, upper=True, out=None)
```

Computes the Cholesky decomposition of a symmetric positive-definite matrix ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg).

For more information, regarding [`torch.potrf()`](#torch.potrf "torch.potrf"), please check [`torch.cholesky()`](#torch.cholesky "torch.cholesky").

Warning

torch.potrf is deprecated in favour of torch.cholesky and will be removed in the next release. Please use torch.cholesky instead and note that the `upper` argument in torch.cholesky defaults to `False`.

```py
torch.potri(u, upper=True, out=None) → Tensor
```

Computes the inverse of a positive semidefinite matrix given its Cholesky factor `u`: returns matrix `inv`

If `upper` is `True` or not provided, `u` is upper triangular such that the returned tensor is

![](img/8e80431286b91606da8941100c871bc6.jpg)

If `upper` is `False`, `u` is lower triangular such that the returned tensor is

![](img/41fe857d2b3b29df4a984ddab7b21847.jpg)

Parameters: 

*   **u** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input 2-D tensor, a upper or lower triangular Cholesky factor
*   **upper** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether to return a upper (default) or lower triangular matrix
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor for `inv`



Example:

```py
>>> a = torch.randn(3, 3)
>>> a = torch.mm(a, a.t()) # make symmetric positive definite
>>> u = torch.cholesky(a)
>>> a
tensor([[  0.9935,  -0.6353,   1.5806],
 [ -0.6353,   0.8769,  -1.7183],
 [  1.5806,  -1.7183,  10.6618]])
>>> torch.potri(u)
tensor([[ 1.9314,  1.2251, -0.0889],
 [ 1.2251,  2.4439,  0.2122],
 [-0.0889,  0.2122,  0.1412]])
>>> a.inverse()
tensor([[ 1.9314,  1.2251, -0.0889],
 [ 1.2251,  2.4439,  0.2122],
 [-0.0889,  0.2122,  0.1412]])

```

```py
torch.potrs(b, u, upper=True, out=None) → Tensor
```

Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix `u`.

If `upper` is `True` or not provided, `u` is upper triangular and `c` is returned such that:

![](img/886788a09088ffab8386053266129b3c.jpg)

If `upper` is `False`, `u` is and lower triangular and `c` is returned such that:

![](img/17f8916876d295a6aef6f97efbae20d5.jpg)

`torch.potrs(b, u)` can take in 2D inputs `b, u` or inputs that are batches of 2D matrices. If the inputs are batches, then returns batched outputs `c`

Note

The `out` keyword only supports 2D matrix inputs, that is, `b, u` must be 2D matrices.

Parameters: 

*   **b** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input matrix of size ![](img/d9795910f977049c4df2084f47c592ed.jpg), where ![](img/28ec51e742166ea3400be6e7343bbfa5.jpg) is zero or more batch dimensions
*   **u** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input matrix of size ![](img/494aaae2a24df44c813ce87b9f21d745.jpg), where ![](img/28ec51e742166ea3400be6e7343bbfa5.jpg) is zero of more batch dimensions composed of upper or lower triangular Cholesky factor
*   **upper** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether to return a upper (default) or lower triangular matrix
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _optional_) – the output tensor for `c`



Example:

```py
>>> a = torch.randn(3, 3)
>>> a = torch.mm(a, a.t()) # make symmetric positive definite
>>> u = torch.cholesky(a)
>>> a
tensor([[ 0.7747, -1.9549,  1.3086],
 [-1.9549,  6.7546, -5.4114],
 [ 1.3086, -5.4114,  4.8733]])
>>> b = torch.randn(3, 2)
>>> b
tensor([[-0.6355,  0.9891],
 [ 0.1974,  1.4706],
 [-0.4115, -0.6225]])
>>> torch.potrs(b,u)
tensor([[ -8.1625,  19.6097],
 [ -5.8398,  14.2387],
 [ -4.3771,  10.4173]])
>>> torch.mm(a.inverse(),b)
tensor([[ -8.1626,  19.6097],
 [ -5.8398,  14.2387],
 [ -4.3771,  10.4173]])

```

```py
torch.pstrf(a, upper=True, out=None) -> (Tensor, Tensor)
```

Computes the pivoted Cholesky decomposition of a positive semidefinite matrix `a`. returns matrices `u` and `piv`.

If `upper` is `True` or not provided, `u` is upper triangular such that ![](img/1fb46274ab877a719bfb6aad1055a2ac.jpg), with `p` the permutation given by `piv`.

If `upper` is `False`, `u` is lower triangular such that ![](img/294e0994f5012c83c1e0c122c5a406a2.jpg).

Parameters: 

*   **a** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input 2-D tensor
*   **upper** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether to return a upper (default) or lower triangular matrix
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – tuple of `u` and `piv` tensors



Example:

```py
>>> a = torch.randn(3, 3)
>>> a = torch.mm(a, a.t()) # make symmetric positive definite
>>> a
tensor([[ 3.5405, -0.4577,  0.8342],
 [-0.4577,  1.8244, -0.1996],
 [ 0.8342, -0.1996,  3.7493]])
>>> u,piv = torch.pstrf(a)
>>> u
tensor([[ 1.9363,  0.4308, -0.1031],
 [ 0.0000,  1.8316, -0.2256],
 [ 0.0000,  0.0000,  1.3277]])
>>> piv
tensor([ 2,  0,  1], dtype=torch.int32)
>>> p = torch.eye(3).index_select(0,piv.long()).index_select(0,piv.long()).t() # make pivot permutation
>>> torch.mm(torch.mm(p.t(),torch.mm(u.t(),u)),p) # reconstruct
tensor([[ 3.5405, -0.4577,  0.8342],
 [-0.4577,  1.8244, -0.1996],
 [ 0.8342, -0.1996,  3.7493]])

```

```py
torch.qr(input, out=None) -> (Tensor, Tensor)
```

Computes the QR decomposition of a matrix `input`, and returns matrices `Q` and `R` such that ![](img/5dcc06c3a05a06beb80d3f1ef2e078f2.jpg), with ![](img/1d680db5f32fd278f8d48e5407691154.jpg) being an orthogonal matrix and ![](img/502cdd9c79852b33d2a6d18ba5ec3102.jpg) being an upper triangular matrix.

This returns the thin (reduced) QR factorization.

Note

precision may be lost if the magnitudes of the elements of `input` are large

Note

While it should always give you a valid decomposition, it may not give you the same one across platforms - it will depend on your LAPACK implementation.

Note

Irrespective of the original strides, the returned matrix ![](img/1d680db5f32fd278f8d48e5407691154.jpg) will be transposed, i.e. with strides `(1, m)` instead of `(m, 1)`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input 2-D tensor
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – tuple of `Q` and `R` tensors



Example:

```py
>>> a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
>>> q, r = torch.qr(a)
>>> q
tensor([[-0.8571,  0.3943,  0.3314],
 [-0.4286, -0.9029, -0.0343],
 [ 0.2857, -0.1714,  0.9429]])
>>> r
tensor([[ -14.0000,  -21.0000,   14.0000],
 [   0.0000, -175.0000,   70.0000],
 [   0.0000,    0.0000,  -35.0000]])
>>> torch.mm(q, r).round()
tensor([[  12.,  -51.,    4.],
 [   6.,  167.,  -68.],
 [  -4.,   24.,  -41.]])
>>> torch.mm(q.t(), q).round()
tensor([[ 1.,  0.,  0.],
 [ 0.,  1., -0.],
 [ 0., -0.,  1.]])

```

```py
torch.svd(input, some=True, compute_uv=True, out=None) -> (Tensor, Tensor, Tensor)
```

`U, S, V = torch.svd(A)` returns the singular value decomposition of a real matrix `A` of size `(n x m)` such that ![](img/af257a01939a1fbe6211d4a4c168f25b.jpg).

`U` is of shape ![](img/7819768bc0adceb9951cf2ce9a0525f2.jpg).

`S` is a diagonal matrix of shape ![](img/b2d82f601df5521e215e30962b942ad1.jpg), represented as a vector of size ![](img/f72656b2358852a4b20972b707fd8222.jpg) containing the non-negative diagonal entries.

`V` is of shape ![](img/be5a855e888d33755dcdfa9d94e598d4.jpg).

If `some` is `True` (default), the returned `U` and `V` matrices will contain only ![](img/3c84c3f757bca109ec4ed7fc0cada53f.jpg) orthonormal columns.

If `compute_uv` is `False`, the returned `U` and `V` matrices will be zero matrices of shape ![](img/7819768bc0adceb9951cf2ce9a0525f2.jpg) and ![](img/be5a855e888d33755dcdfa9d94e598d4.jpg) respectively. `some` will be ignored here.

Note

The implementation of SVD on CPU uses the LAPACK routine `?gesdd` (a divide-and-conquer algorithm) instead of `?gesvd` for speed. Analogously, the SVD on GPU uses the MAGMA routine `gesdd` as well.

Note

Irrespective of the original strides, the returned matrix `U` will be transposed, i.e. with strides `(1, n)` instead of `(n, 1)`.

Note

Extra care needs to be taken when backward through `U` and `V` outputs. Such operation is really only stable when `input` is full rank with all distinct singular values. Otherwise, `NaN` can appear as the gradients are not properly defined. Also, notice that double backward will usually do an additional backward through `U` and `V` even if the original backward is only on `S`.

Note

When `some` = `False`, the gradients on `U[:, min(n, m):]` and `V[:, min(n, m):]` will be ignored in backward as those vectors can be arbitrary bases of the subspaces.

Note

When `compute_uv` = `False`, backward cannot be performed since `U` and `V` from the forward pass is required for the backward operation.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input 2-D tensor
*   **some** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – controls the shape of returned `U` and `V`
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the output tuple of tensors



Example:

```py
>>> a = torch.tensor([[8.79,  6.11, -9.15,  9.57, -3.49,  9.84],
 [9.93,  6.91, -7.93,  1.64,  4.02,  0.15],
 [9.83,  5.04,  4.86,  8.83,  9.80, -8.99],
 [5.45, -0.27,  4.85,  0.74, 10.00, -6.02],
 [3.16,  7.98,  3.01,  5.80,  4.27, -5.31]]).t()

>>> u, s, v = torch.svd(a)
>>> u
tensor([[-0.5911,  0.2632,  0.3554,  0.3143,  0.2299],
 [-0.3976,  0.2438, -0.2224, -0.7535, -0.3636],
 [-0.0335, -0.6003, -0.4508,  0.2334, -0.3055],
 [-0.4297,  0.2362, -0.6859,  0.3319,  0.1649],
 [-0.4697, -0.3509,  0.3874,  0.1587, -0.5183],
 [ 0.2934,  0.5763, -0.0209,  0.3791, -0.6526]])
>>> s
tensor([ 27.4687,  22.6432,   8.5584,   5.9857,   2.0149])
>>> v
tensor([[-0.2514,  0.8148, -0.2606,  0.3967, -0.2180],
 [-0.3968,  0.3587,  0.7008, -0.4507,  0.1402],
 [-0.6922, -0.2489, -0.2208,  0.2513,  0.5891],
 [-0.3662, -0.3686,  0.3859,  0.4342, -0.6265],
 [-0.4076, -0.0980, -0.4933, -0.6227, -0.4396]])
>>> torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))
tensor(1.00000e-06 *
 9.3738)

```

```py
torch.symeig(input, eigenvectors=False, upper=True, out=None) -> (Tensor, Tensor)
```

This function returns eigenvalues and eigenvectors of a real symmetric matrix `input`, represented by a tuple ![](img/07aa3eb39d8bb2f8d4d17cd4925159b4.jpg).

`input` and ![](img/21ec2ab32d1af3e766487093bb20cf22.jpg) are ![](img/be5a855e888d33755dcdfa9d94e598d4.jpg) matrices and ![](img/be8982d125e27260b5c793cf0d39d70a.jpg) is a ![](img/20ddd8181c2e0d0fb893637e8572d475.jpg) dimensional vector.

This function calculates all eigenvalues (and vectors) of `input` such that ![](img/db9e47c935013aa5b30057ba51ac84b9.jpg).

The boolean argument `eigenvectors` defines computation of eigenvectors or eigenvalues only.

If it is `False`, only eigenvalues are computed. If it is `True`, both eigenvalues and eigenvectors are computed.

Since the input matrix `input` is supposed to be symmetric, only the upper triangular portion is used by default.

If `upper` is `False`, then lower triangular portion is used.

Note: Irrespective of the original strides, the returned matrix `V` will be transposed, i.e. with strides `(1, m)` instead of `(m, 1)`.

Parameters: 

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input symmetric matrix
*   **eigenvectors** (_boolean__,_ _optional_) – controls whether eigenvectors have to be computed
*   **upper** (_boolean__,_ _optional_) – controls whether to consider upper-triangular or lower-triangular region
*   **out** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – the output tuple of (Tensor, Tensor)


| Returns: | A tuple containing

&gt; *   **e** (_Tensor_): Shape ![](img/2f4b02bcd5b11d436474c4c4cdb91683.jpg). Each element is an eigenvalue of `input`, The eigenvalues are in ascending order.
&gt; *   **V** (_Tensor_): Shape ![](img/be5a855e888d33755dcdfa9d94e598d4.jpg). If `eigenvectors=False`, it’s a tensor filled with zeros. Otherwise, this tensor contains the orthonormal eigenvectors of the `input`.


| Return type: | ([Tensor](tensors.html#torch.Tensor "torch.Tensor"), [Tensor](tensors.html#torch.Tensor "torch.Tensor")) |
| --- | --- |

Examples:

```py
>>> a = torch.tensor([[ 1.96,  0.00,  0.00,  0.00,  0.00],
 [-6.49,  3.80,  0.00,  0.00,  0.00],
 [-0.47, -6.39,  4.17,  0.00,  0.00],
 [-7.20,  1.50, -1.51,  5.70,  0.00],
 [-0.65, -6.34,  2.67,  1.80, -7.10]]).t()
>>> e, v = torch.symeig(a, eigenvectors=True)
>>> e
tensor([-11.0656,  -6.2287,   0.8640,   8.8655,  16.0948])
>>> v
tensor([[-0.2981, -0.6075,  0.4026, -0.3745,  0.4896],
 [-0.5078, -0.2880, -0.4066, -0.3572, -0.6053],
 [-0.0816, -0.3843, -0.6600,  0.5008,  0.3991],
 [-0.0036, -0.4467,  0.4553,  0.6204, -0.4564],
 [-0.8041,  0.4480,  0.1725,  0.3108,  0.1622]])

```

```py
torch.trtrs(b, A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)
```

Solves a system of equations with a triangular coefficient matrix ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg) and multiple right-hand sides `b`.

In particular, solves ![](img/79ccd3754eebf815ed3195b42f93bacb.jpg) and assumes ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg) is upper-triangular with the default keyword arguments.

Parameters: 

*   **A** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input triangular coefficient matrix
*   **b** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – multiple right-hand sides. Each column of ![](img/6872867a863714d15d9a0d64c20734ce.jpg) is a right-hand side for the system of equations.
*   **upper** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether to solve the upper-triangular system of equations (default) or the lower-triangular system of equations. Default: True.
*   **transpose** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg) should be transposed before being sent into the solver. Default: False.
*   **unitriangular** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg) is unit triangular. If True, the diagonal elements of ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg) are assumed to be 1 and not referenced from ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg). Default: False.


| Returns: | A tuple ![](img/1044f4a1887b042eb41d12f782c0582f.jpg) where ![](img/f8961918eb987d8916766b1d77790ecb.jpg) is a clone of ![](img/efdb05f076173b39fdd26ef663e7b0d8.jpg) and ![](img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg) is the solution to ![](img/79ccd3754eebf815ed3195b42f93bacb.jpg) (or whatever variant of the system of equations, depending on the keyword arguments.) |
| --- | --- |

```py
Shape:
```

*   A: ![](img/ff833c4d1f13ca018e121d87f6ef1607.jpg)
*   b: ![](img/9b9aebaa467ad07dca05b5086bd21ca2.jpg)
*   output[0]: ![](img/9b9aebaa467ad07dca05b5086bd21ca2.jpg)
*   output[1]: ![](img/ff833c4d1f13ca018e121d87f6ef1607.jpg)

Examples:

```py
>>> A = torch.randn(2, 2).triu()
>>> A
tensor([[ 1.1527, -1.0753],
 [ 0.0000,  0.7986]])
>>> b = torch.randn(2, 3)
>>> b
tensor([[-0.0210,  2.3513, -1.5492],
 [ 1.5429,  0.7403, -1.0243]])
>>> torch.trtrs(b, A)
(tensor([[ 1.7840,  2.9045, -2.5405],
 [ 1.9319,  0.9269, -1.2826]]), tensor([[ 1.1527, -1.0753],
 [ 0.0000,  0.7986]]))

```
