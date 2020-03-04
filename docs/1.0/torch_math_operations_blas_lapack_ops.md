

### BLAS和LAPACK操作

> 译者：[ApacheCN](https://github.com/apachecn)

```py
torch.addbmm(beta=1, mat, alpha=1, batch1, batch2, out=None) → Tensor
```

执行存储在`batch1`和`batch2`中的矩阵的批量矩阵 - 矩阵乘积，减少加法步骤(所有矩阵乘法沿第一维积累）。 `mat`被添加到最终结果中。

`batch1`和`batch2`必须是3-D张量，每个张量包含相同数量的矩阵。

如果`batch1`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/eccd104fbbbbb116c7e98ca54b2214a0.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/eccd104fbbbbb116c7e98ca54b2214a0.jpg) 张量，`batch2`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/f8b603730e091b70ad24e5a089cdd30f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/f8b603730e091b70ad24e5a089cdd30f.jpg) 张量，`mat`必须是[可广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)和 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) 张量和`out`将是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) 张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/e9a3c5b413385d813461f90cc06b1454.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/e9a3c5b413385d813461f90cc06b1454.jpg)

对于`FloatTensor`或`DoubleTensor`类型的输入，参数`beta`和`alpha`必须是实数，否则它们应该是整数。

参数：

*   **beta** (_编号_ _，_ _任选_） - `mat` ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg))的乘数
*   **mat**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要添加的基质
*   **alpha** (_数_ _，_ _任选_） - `batch1 @ batch2` ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg))的乘数
*   **batch1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第一批要乘的矩阵
*   **batch2**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二批矩阵被乘以
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

例：

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

执行矩阵`mat1`和`mat2`的矩阵乘法。矩阵`mat`被添加到最终结果中。

如果`mat1`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg) 张量，`mat2`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ec84c2d649caa2a7d4dc59b6b23b0278.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ec84c2d649caa2a7d4dc59b6b23b0278.jpg) 张量，那么`mat`必须是[可广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)和 [] ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) 张量和`out`将是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) 张量。

`alpha`和`beta`分别是`mat1`和：attr `mat2`与添加的基质`mat`之间的基质 - 载体产物的比例因子。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8d4b0912f137549bc9b2dc4ee38a0a40.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8d4b0912f137549bc9b2dc4ee38a0a40.jpg)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be real numbers, otherwise they should be integers.

Parameters:

*   **beta** (_编号_ _，_ _任选_） - `mat` ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg))的乘数
*   **mat**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要添加的基质
*   **alpha** (_编号_ _，_ _任选_） - [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/c4fda0ec33ee23096c7bac6105f7a619.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/c4fda0ec33ee23096c7bac6105f7a619.jpg)  ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg))的乘数
*   **mat1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第一个被乘法的矩阵
*   **mat2**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要倍增的第二个矩阵
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

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

执行矩阵`mat`和向量`vec`的矩阵向量乘积。将载体 [`tensor`](#torch.tensor "torch.tensor") 添加到最终结果中。

如果`mat`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg) 张量，`vec`是大小为`m`的1-D张量，则 [`tensor`](#torch.tensor "torch.tensor") 必须是[可广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)具有1-D张量的`n`和`out`将是1-D张量的大小`n`。

`alpha`和`beta`分别是`mat`和`vec`之间的基质 - 载体产物和加入的张量 [`tensor`](#torch.tensor "torch.tensor") 的比例因子。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/4188eb7768951ccca87969272bcfa3a7.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/4188eb7768951ccca87969272bcfa3a7.jpg)

对于`FloatTensor`或`DoubleTensor`类型的输入，参数`beta`和`alpha`必须是实数，否则它们应该是整数

Parameters:

*   **beta** (_，_ _任选_） - [`tensor`](#torch.tensor "torch.tensor")  ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg))的乘数
*   **张量** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要添加的载体
*   **alpha** (_编号_ _，_ _任选_） - [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a901c2282b0dbdcf23379ddd5a3c274b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a901c2282b0dbdcf23379ddd5a3c274b.jpg)  ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg))的乘数
*   **mat**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 矩阵成倍增加
*   **vec**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 载体倍增
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

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

执行向量`vec1`和`vec2`的外积并将其添加到矩阵`mat`。

可选值`beta`和`alpha`分别是`vec1`和`vec2`之间的外积和添加的矩阵`mat`的缩放因子。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/171f2173f3a92cea6433a9dd012888ad.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/171f2173f3a92cea6433a9dd012888ad.jpg)

如果`vec1`是大小为`n`的矢量而`vec2`是大小为`m`的矢量，那么`mat`必须是[可广播的](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)，其大小为矩阵 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg) 和`out`将是大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg) 的基质。

For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be real numbers, otherwise they should be integers

Parameters:

*   **beta** (_编号_ _，_ _任选_） - `mat` ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg))的乘数
*   **mat**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要添加的基质
*   **alpha** (_编号_ _，_ _任选_） - [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/3f2eb83c372296996af0ac869a078ebd.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/3f2eb83c372296996af0ac869a078ebd.jpg)  ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg))的乘数
*   **vec1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 外部产品的第一个载体
*   **vec2**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 外产品的第二个载体
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

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

在`batch1`和`batch2`中执行矩阵的批量矩阵 - 矩阵乘积。 `mat`被添加到最终结果中。

`batch1` and `batch2` must be 3-D tensors each containing the same number of matrices.

如果`batch1`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/eccd104fbbbbb116c7e98ca54b2214a0.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/eccd104fbbbbb116c7e98ca54b2214a0.jpg) 张量，`batch2`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/f8b603730e091b70ad24e5a089cdd30f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/f8b603730e091b70ad24e5a089cdd30f.jpg) 张量，那么`mat`必须是[可广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)和 [] ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/29f0e4a370460668f7e257b22d08622d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/29f0e4a370460668f7e257b22d08622d.jpg) 张量和`out`将是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/29f0e4a370460668f7e257b22d08622d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/29f0e4a370460668f7e257b22d08622d.jpg) 张量。 `alpha`和`beta`均与 [`torch.addbmm()`](#torch.addbmm "torch.addbmm") 中使用的比例因子相同。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/069d82fba319e5aec62a5ad55fd0d01c.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/069d82fba319e5aec62a5ad55fd0d01c.jpg)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be real numbers, otherwise they should be integers.

Parameters:

*   **beta** (_编号_ _，_ _任选_） - `mat` ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/50705df736e9a7919e768cf8c4e4f794.jpg))的乘数
*   **垫** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要加的张量
*   **alpha** (_编号_ _，_ _任选_） - [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/c9ac2542d6edbedec1234ae90d5bf79f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/c9ac2542d6edbedec1234ae90d5bf79f.jpg)  ([![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/82005cc2e0087e2a52c7e43df4a19a00.jpg))的乘数
*   **batch1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第一批要乘的矩阵
*   **batch2**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二批矩阵被乘以
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

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

执行存储在`batch1`和`batch2`中的矩阵的批量矩阵 - 矩阵乘积。

`batch1` and `batch2` must be 3-D tensors each containing the same number of matrices.

如果`batch1`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/eccd104fbbbbb116c7e98ca54b2214a0.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/eccd104fbbbbb116c7e98ca54b2214a0.jpg) 张量，`batch2`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/f8b603730e091b70ad24e5a089cdd30f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/f8b603730e091b70ad24e5a089cdd30f.jpg) 张量，`out`将是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/29f0e4a370460668f7e257b22d08622d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/29f0e4a370460668f7e257b22d08622d.jpg) 张量。

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/699b5d44b53e8c67d763dc6fb072e488.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/699b5d44b53e8c67d763dc6fb072e488.jpg)

注意

此功能不[广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。有关广播矩阵产品，请参阅 [`torch.matmul()`](#torch.matmul "torch.matmul") 。

Parameters:

*   **batch1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第一批要乘的矩阵
*   **batch2**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第二批矩阵被乘以
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

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

批量LU分解。

返回包含LU分解和枢轴的元组。如果设置了`pivot`，则完成旋转。

如果每个minibatch示例的分解成功，则可选参数`info`存储信息。 `info`作为`IntTensor`提供，其值将从dgetrf填充，非零值表示发生错误。具体来说，如果使用cuda，则值来自cublas，否则为LAPACK。

警告

`info`参数不推荐使用 [`torch.btrifact_with_info()`](#torch.btrifact_with_info "torch.btrifact_with_info") 。

Parameters:

*   **A**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 因子的张量
*   **info**  (_IntTensor_ _，_ _可选_） - (弃用）`IntTensor`存储指示分解是否成功的值
*   **pivot**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 控制是否完成旋转

| 返回： | 包含分解和枢轴的元组。 |
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

批量LU分解和其他错误信息。

这是 [`torch.btrifact()`](#torch.btrifact "torch.btrifact") 的一个版本，它始终创建一个info `IntTensor`，并将其作为第三个返回值返回。

Parameters:

*   **A**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 因子的张量
*   **pivot**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 控制是否完成旋转

| Returns: | 包含因式分解，枢轴和`IntTensor`的元组，其中非零值表示每个小批量样本的分解是否成功。 |
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

批量LU解决。

返回线性系统 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/79f5b7df86014d0a54c744c91d8b351d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/79f5b7df86014d0a54c744c91d8b351d.jpg) 的LU求解。

Parameters:

*   **b**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - RHS张量
*   **LU_data**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 来自 [`btrifact()`](#torch.btrifact "torch.btrifact") 的A的旋转LU分解。
*   **LU_pivots**  (_IntTensor_ ) - LU分解的关键点

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

从张量的分段LU分解(btrifact）解包数据和枢轴。

返回张量的元组作为`(the pivots, the L tensor, the U tensor)`。

Parameters:

*   **LU_data**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 打包的LU分解数据
*   **LU_pivots**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 打包的LU分解枢轴
*   **unpack_data**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 指示数据是否应解包的标志
*   **unpack_pivots**  ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - 指示枢轴是否应解包的标志

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

返回 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) 2-D张量的矩阵乘积。使用矩阵链序算法有效地计算该乘积，该算法选择在算术运算方面产生最低成本的顺序 ([[CLRS]](https://mitpress.mit.edu/books/introduction-algorithms-third-edition))。请注意，由于这是计算产品的函数， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) 需要大于或等于2;如果等于2，则返回一个平凡的矩阵 - 矩阵乘积。如果 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9341d9048ac485106d2b2ee8de14876f.jpg) 为1，那么这是一个无操作 - 原始矩阵按原样返回。

| 参数： | **矩阵**(_张量..._ ) - 2个或更多个2-D张量的序列，其产物将被确定。 |
| --- | --- |
| 返回： | 如果 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5c5e7583f110d90e938149340dd42e92.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5c5e7583f110d90e938149340dd42e92.jpg) 张量具有 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/fd285b0b789ab6ce131b7a0208da2fe0.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/fd285b0b789ab6ce131b7a0208da2fe0.jpg) 的维度，则产物的尺寸为 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/61c4b45c29064296a380ab945a449672.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/61c4b45c29064296a380ab945a449672.jpg) 。 |
| 返回类型： | [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") |

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

计算对称正定矩阵 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 的Cholesky分解或对称批正对称正定矩阵。

如果`upper`为`True`，则返回的矩阵`U`为上三角形，分解的形式为：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/7100a5dce6b64985eeb45416a640b7e6.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/7100a5dce6b64985eeb45416a640b7e6.jpg)

如果`upper`为`False`，则返回的矩阵`L`为低三角形，分解的形式为：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/3ec14c9e61e88b877808fab3bbdd17ca.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/3ec14c9e61e88b877808fab3bbdd17ca.jpg)

如果`upper`是`True`，并且`A`是一批对称正定矩阵，则返回的张量将由每个单独矩阵的上三角形Cholesky因子组成。类似地，当`upper`是`False`时，返回的张量将由每个单个矩阵的下三角形Cholesky因子组成。

Parameters:

*   **a**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 输入张量大小 ([*](#id6) ，n，n）其中`*`为零或更多批由对称正定矩阵组成的维数。
*   **上** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 表示是否返回上下三角矩阵的标志。默认值：`False`
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _可选_） - 输出矩阵

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

计算两个张量的点积(内积）。

Note

此功能不[广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)。

Example:

```py
>>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
tensor(7)

```

```py
torch.eig(a, eigenvectors=False, out=None) -> (Tensor, Tensor)
```

计算实方阵的特征值和特征向量。

Parameters:

*   **a**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 形状 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/7819768bc0adceb9951cf2ce9a0525f2.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/7819768bc0adceb9951cf2ce9a0525f2.jpg) 的方阵，其特征值和特征向量将被计算
*   **特征向量** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")） - `True`计算特征值和特征向量;否则，只计算特征值
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _任选_） - 输出张量

|返回：|包含元组的元组

＆GT; * **e** (_tensor_）：形状 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6bb1e4cc787b2a2a3e362c6385033b7d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6bb1e4cc787b2a2a3e362c6385033b7d.jpg) 。每行是`a`的特征值，其中第一个元素是实部，第二个元素是虚部。特征值不一定是有序的。 ＆GT; * **v**  (_Tensor_ )：如果`eigenvectors=False`，它是一个空张量。否则，该张量形状 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/7819768bc0adceb9951cf2ce9a0525f2.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/7819768bc0adceb9951cf2ce9a0525f2.jpg) 可用于计算相应特征值`e`的归一化(单位长度）特征向量，如下所述。如果对应的e [j]是实数，则列v [：，j]是对应于特征值e [j]的特征向量。如果相应的e [j]和e [j + 1]特征值形成复共轭对，那么真实的特征向量可以被计算为 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ec9513691a2c7521c03807425da807ed.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ec9513691a2c7521c03807425da807ed.jpg) ， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a532b8aa12f3a5051c8105bf8e226b64.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a532b8aa12f3a5051c8105bf8e226b64.jpg) 。

| 返回类型： |  ([Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") ， [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) |
| --- | --- |

```py
torch.gels(B, A, out=None) → Tensor
```

计算大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/cc3ea6b8d05f85433fd7aa6a20c33408.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/cc3ea6b8d05f85433fd7aa6a20c33408.jpg) 的全秩矩阵 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 和大小的矩阵 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/813135a6280e2672503128d3d2080d4a.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/813135a6280e2672503128d3d2080d4a.jpg) 的最小二乘和最小范数问题的解决方案 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/602cd3c92249bd53b21908f902ff6089.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/602cd3c92249bd53b21908f902ff6089.jpg) 。

如果 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/199260d72e51fe506909a150c6f77020.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/199260d72e51fe506909a150c6f77020.jpg) ， [`gels()`](#torch.gels "torch.gels") 解决了最小二乘问题：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6d9885b2d1646d00061288d0c063790b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6d9885b2d1646d00061288d0c063790b.jpg)

如果 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/bc0a7901ab359873a58a64e43f9fc85a.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/bc0a7901ab359873a58a64e43f9fc85a.jpg) ， [`gels()`](#torch.gels "torch.gels") 解决了最小范数问题：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/921685e425167b15563b245ca59e3ac3.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/921685e425167b15563b245ca59e3ac3.jpg)

返回张量 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg) 具有 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b10f10ee19a21653d24c909eb3e0877a.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b10f10ee19a21653d24c909eb3e0877a.jpg) 的形状。 [ ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg) ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg) ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg) ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg)的第一](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg) [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg) 行包含该溶液。如果 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/199260d72e51fe506909a150c6f77020.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/199260d72e51fe506909a150c6f77020.jpg) ，则每列中溶液的残余平方和由该列的剩余 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/51e590a6a852e7b2d4cd0a3476859fc5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/51e590a6a852e7b2d4cd0a3476859fc5.jpg) 行中的元素的平方和给出。

Parameters:

*   **B**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 基质 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/813135a6280e2672503128d3d2080d4a.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/813135a6280e2672503128d3d2080d4a.jpg)
*    ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg) ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg) ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg) ](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg) [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg) 
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _可选_） - 可选目的地张量

|返回：|包含以下内容的元组：

＆GT; * **X** (_tensor_）：最小二乘解＆gt; * **qr**  (_Tensor_ )：QR分解的细节

| Return type: | ([Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor"), [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) |
| --- | --- |

Note

无论输入矩阵的步幅如何，返回的矩阵将始终被转置。也就是说，他们将有`(1, m)`而不是`(m, 1)`。

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

这是一个直接调用LAPACK的低级函数。

您通常希望使用 [`torch.qr()`](#torch.qr "torch.qr") 。

计算`input`的QR分解，但不构造 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1d680db5f32fd278f8d48e5407691154.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1d680db5f32fd278f8d48e5407691154.jpg) 和 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/502cdd9c79852b33d2a6d18ba5ec3102.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/502cdd9c79852b33d2a6d18ba5ec3102.jpg) 作为显式单独的矩阵。

相反，这直接调用底层LAPACK函数`?geqrf`，它产生一系列“基本反射器”。

有关详细信息，请参阅geqrf 的 [LAPACK文档。](https://software.intel.com/en-us/node/521004)

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入矩阵
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _可选_） - 输出元组(Tensor，Tensor）

```py
torch.ger(vec1, vec2, out=None) → Tensor
```

`vec1`和`vec2`的外产物。如果`vec1`是大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg) 的载体，`vec2`是大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg) 的载体，那么`out`必须是大小的矩阵 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg) 。

Note

This function does not [broadcast](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics).

Parameters:

*   **vec1**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 1-D输入向量
*   **vec2**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 1-D输入向量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _可选_） - 可选输出矩阵

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

该函数将解决方案返回到由 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9c11b6313ae06c752584c5c1b2c03964.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9c11b6313ae06c752584c5c1b2c03964.jpg) 表示的线性方程组和A的LU分解，按顺序作为元组`X, LU`。

`LU`包含`A`的LU分解的`L`和`U`因子。

`torch.gesv(B, A)`可以接收2D输入`B, A`或两批2D矩阵的输入。如果输入是批次，则返回批量输出`X, LU`。

Note

`out`关键字仅支持2D矩阵输入，即`B, A`必须是2D矩阵。

Note

不管原始步幅如何，返回的矩阵`X`和`LU`将被转置，即分别具有诸如`B.contiguous().transpose(-1, -2).strides()`和`A.contiguous().transpose(-1, -2).strides()`的步幅。

Parameters:

*   **B**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/d9795910f977049c4df2084f47c592ed.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/d9795910f977049c4df2084f47c592ed.jpg) 的输入矩阵，其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg) 为零或批量维度更多。
*   **A**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 输入方形矩阵 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/494aaae2a24df44c813ce87b9f21d745.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/494aaae2a24df44c813ce87b9f21d745.jpg) ，其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg) 为零或更多批量维度。
*   **出**(_(_ [_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _，_ [_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor") _]）__，_ _可选_） - 可选输出元组。

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

采用方阵`input`的倒数。 `input`可以是2D方形张量的批次，在这种情况下，该函数将返回由单个反转组成的张量。

Note

无论原始步幅如何，返回的张量都将被转置，即像`input.contiguous().transpose(-2, -1).strides()`这样的步幅

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量大小 ([*](#id8) ，n，n）其中`*`为零或更多批尺寸
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _可选_） - 可选输出张量

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

计算2D平方张量的行列式。

Note

当`A`不可逆时，向后通过 [`det()`](#torch.det "torch.det") 在内部使用SVD结果。在这种情况下，当`A`没有明显的奇异值时，通过 [`det()`](#torch.det "torch.det") 的双向后将是不稳定的。有关详细信息，请参阅 [`svd()`](#torch.svd "torch.svd") 。

| Parameters: | **A**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入2D平方张量 |
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

计算2D平方张量的对数行列式。

Note

如果`A`具有零对数行列式，则结果为`-inf`，如果`A`具有负的行列式，则结果为`nan`。

Note

当`A`不可逆时，向后通过 [`logdet()`](#torch.logdet "torch.logdet") 在内部使用SVD结果。在这种情况下，当`A`没有明显的奇异值时，通过 [`logdet()`](#torch.logdet "torch.logdet") 的双向后将是不稳定的。有关详细信息，请参阅 [`svd()`](#torch.svd "torch.svd") 。

| Parameters: | **A** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) – The input 2D square tensor |
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

计算2D平方张量的行列式的符号和对数值。

Note

如果`A`的行列式为零，则返回`(0, -inf)`。

Note

当`A`不可逆时，向后通过 [`slogdet()`](#torch.slogdet "torch.slogdet") 在内部使用SVD结果。在这种情况下，当`A`没有明显的奇异值时，通过 [`slogdet()`](#torch.slogdet "torch.slogdet") 的双向后将是不稳定的。有关详细信息，请参阅 [`svd()`](#torch.svd "torch.svd") 。

| Parameters: | **A** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) – The input 2D square tensor |
| --- | --- |
| Returns: | 包含行列式符号的元组，以及绝对行列式的对数值。 |

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

两个张量的矩阵乘积。

行为取决于张量的维度如下：

*   如果两个张量都是1维的，则返回点积(标量）。
*   如果两个参数都是二维的，则返回矩阵 - 矩阵乘积。
*   如果第一个参数是1维且第二个参数是2维，则为了矩阵乘法的目的，在其维度之前加1。在矩阵乘法之后，移除前置维度。
*   如果第一个参数是2维且第二个参数是1维，则返回矩阵向量乘积。
*   如果两个参数都是至少一维的并且至少一个参数是N维的(其中N&gt; 2），则返回批量矩阵乘法。如果第一个参数是1维的，则为了批量矩阵的目的，将1加在其维度之前，然后将其删除。如果第二个参数是1维的，则为了批处理矩阵的多个目的，将1附加到其维度，并在之后删除。非矩阵(即批量）维度是[广播](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics)(因此必须是可广播的）。例如，如果`tensor1`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/a4697ce48760baf0633769e49f46b335.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/a4697ce48760baf0633769e49f46b335.jpg) 张量而`tensor2`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ad9fbe324dcc50cc2232a9c1a2675daf.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ad9fbe324dcc50cc2232a9c1a2675daf.jpg) 张量，`out`将是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6082fde6b0f498f9ed21c0ac7a9709d3.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6082fde6b0f498f9ed21c0ac7a9709d3.jpg) 张量。

Note

此功能的1维点积版本不支持`out`参数。

Parameters:

*   **tensor1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第一个要乘的张量
*   **tensor2**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要增加的第二个张量
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

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

返回为矩形矩阵提升到幂`n`的矩阵。对于一批矩阵，每个单独的矩阵被提升到功率`n`。

如果`n`为负，则矩阵的反转(如果可逆）将升至功率`n`。对于一批矩阵，批量反转(如果可逆）则上升到功率`n`。如果`n`为0，则返回单位矩阵。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入张量
*   **n**  ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")） - 将矩阵提升到

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

返回二维张量的数值等级。默认情况下，使用SVD完成计算矩阵秩的方法。如果`symmetric`是`True`，则假设`input`是对称的，并且通过获得特征值来完成秩的计算。

`tol`是一个阈值，低于该阈值时，奇异值(或`symmetric`为`True`时的特征值）被认为是0.如果未指定`tol`，则`tol`设置为`S.max() * max(S.size()) * eps`，其中`S` ]是奇异值(或`symmetric`为`True`时的特征值），`eps`是`input`数据类型的epsilon值。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入2-D张量
*   **tol**  ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_，_ _任选_） - 耐受值。默认值：`None`
*   **对称** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 表示`input`是否对称。默认值：`False`

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

执行矩阵`mat1`和`mat2`的矩阵乘法。

如果`mat1`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg) 张量，`mat2`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ec84c2d649caa2a7d4dc59b6b23b0278.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ec84c2d649caa2a7d4dc59b6b23b0278.jpg) 张量，`out`将是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/42cdcd96fd628658ac0e3e7070ba08d5.jpg) 张量。

Note

This function does not [broadcast](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics). For broadcasting matrix products, see [`torch.matmul()`](#torch.matmul "torch.matmul").

Parameters:

*   **mat1**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 第一个被乘法的矩阵
*   **mat2**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要倍增的第二个矩阵
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

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

执行矩阵`mat`和向量`vec`的矩阵向量乘积。

如果`mat`是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg) 张量，`vec`是1-D张量大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg) ，`out`将是1-D大小 [] ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/493731e423d5db62086d0b8705dda0c8.jpg) 。

Note

This function does not [broadcast](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/broadcasting.html#broadcasting-semantics).

Parameters:

*   **mat**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 矩阵成倍增加
*   **vec**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 载体倍增
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - 输出张量

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

从 [`torch.geqrf()`](#torch.geqrf "torch.geqrf") 返回的`(a, tau)`元组计算QR分解的正交矩阵`Q`。

这直接调用底层LAPACK函数`?orgqr`。有关详细信息，请参阅orgqr 的 [LAPACK文档。](https://software.intel.com/en-us/mkl-developer-reference-c-orgqr)

Parameters:

*   **a**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 来自 [`torch.geqrf()`](#torch.geqrf "torch.geqrf") 的`a`。
*   **tau**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 来自 [`torch.geqrf()`](#torch.geqrf "torch.geqrf") 的`tau`。

```py
torch.ormqr(a, tau, mat, left=True, transpose=False) -> (Tensor, Tensor)
```

将`mat`乘以由`(a, tau)`表示的 [`torch.geqrf()`](#torch.geqrf "torch.geqrf") 形成的QR分解的正交`Q`矩阵。

这直接调用底层LAPACK函数`?ormqr`。有关详细信息，请参阅ormqr 的 [LAPACK文档。](https://software.intel.com/en-us/mkl-developer-reference-c-ormqr)

Parameters:

*   **a**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 来自 [`torch.geqrf()`](#torch.geqrf "torch.geqrf") 的`a`。
*   **tau**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 来自 [`torch.geqrf()`](#torch.geqrf "torch.geqrf") 的`tau`。
*   **mat**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 要倍增的矩阵。

```py
torch.pinverse(input, rcond=1e-15) → Tensor
```

计算2D张量的伪逆(也称为Moore-Penrose逆）。有关详细信息，请查看 [Moore-Penrose逆](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)

Note

该方法使用奇异值分解来实现。

Note

伪逆不一定是矩阵 [[1]](https://epubs.siam.org/doi/10.1137/0117004) 的元素中的连续函数。因此，衍生物并不总是存在，只存在于恒定等级 [[2]](https://www.jstor.org/stable/2156365) 。但是，由于使用SVD结果实现，此方法可以反向使用，并且可能不稳定。由于在内部使用SVD，双向后也将不稳定。有关详细信息，请参阅 [`svd()`](#torch.svd "torch.svd") 。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 维度 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ee12b6c487a34051534acf84ddb3f98f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ee12b6c487a34051534acf84ddb3f98f.jpg) 的输入2D张量
*   **rcond**  ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")） - 一个浮点值，用于确定小奇异值的截止值。默认值：1e-15

| Returns: | 维度 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/3380c6697127aa874110f3e6faef8bdf.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/3380c6697127aa874110f3e6faef8bdf.jpg) 的`input`的伪逆 |
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

计算对称正定矩阵 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 的Cholesky分解。

有关 [`torch.potrf()`](#torch.potrf "torch.potrf") 的更多信息，请查看 [`torch.cholesky()`](#torch.cholesky "torch.cholesky") 。

Warning

torch.potrf不赞成使用torch.cholesky，将在下一个版本中删除。请改用torch.cholesky并注意torch.cholesky中的`upper`参数默认为`False`。

```py
torch.potri(u, upper=True, out=None) → Tensor
```

计算正半定矩阵的倒数，给出其Cholesky因子`u`：返回矩阵`inv`

如果`upper`为`True`或未提供，则`u`为上三角形，使得返回的张量为

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/8e80431286b91606da8941100c871bc6.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/8e80431286b91606da8941100c871bc6.jpg)

如果`upper`为`False`，则`u`为下三角形，使得返回的张量为

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/41fe857d2b3b29df4a984ddab7b21847.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/41fe857d2b3b29df4a984ddab7b21847.jpg)

Parameters:

*   **u**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入2-D张量，上下三角Cholesky因子
*   **上** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 是否返回上限(默认）或下三角矩阵
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - `inv`的输出张量

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

求解具有正半定矩阵的线性方程组，给定其Cholesky因子矩阵`u`。

如果`upper`为`True`或未提供，则`u`为上三角形并返回`c`，以便：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/886788a09088ffab8386053266129b3c.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/886788a09088ffab8386053266129b3c.jpg)

如果`upper`为`False`，则`u`为下三角形并返回`c`，以便：

[![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/17f8916876d295a6aef6f97efbae20d5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/17f8916876d295a6aef6f97efbae20d5.jpg)

`torch.potrs(b, u)`可以接收2D输入`b, u`或两批2D矩阵的输入。如果输入是批次，则返回批量输出`c`

Note

`out`关键字仅支持2D矩阵输入，即`b, u`必须是2D矩阵。

Parameters:

*   **b**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/d9795910f977049c4df2084f47c592ed.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/d9795910f977049c4df2084f47c592ed.jpg) 的输入矩阵，其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg) 为零或批量维度更多
*   **u**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 大小为 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/494aaae2a24df44c813ce87b9f21d745.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/494aaae2a24df44c813ce87b9f21d745.jpg) 的输入矩阵，其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/28ec51e742166ea3400be6e7343bbfa5.jpg) 为零更多批量尺寸由上部或下部三角形Cholesky因子组成
*   **上** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 是否返回上限(默认）或下三角矩阵
*   **out**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")_，_ _任选_） - `c`的输出张量

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

计算正半定矩阵`a`的旋转Cholesky分解。返回矩阵`u`和`piv`。

如果`upper`为`True`或未提供，则`u`为上三角形，使得 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1fb46274ab877a719bfb6aad1055a2ac.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1fb46274ab877a719bfb6aad1055a2ac.jpg) ，`p`为`piv`给出的置换。

如果`upper`为`False`，则`u`为三角形，使得 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/294e0994f5012c83c1e0c122c5a406a2.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/294e0994f5012c83c1e0c122c5a406a2.jpg) 。

Parameters:

*   **a**  ([_tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) - 输入二维张量
*   **上** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 是否返回上限(默认）或下三角矩阵
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _任选_） - `u`和`piv`张量的元组

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

计算矩阵`input`的QR分解，并返回矩阵`Q`和`R`，使 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/5dcc06c3a05a06beb80d3f1ef2e078f2.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/5dcc06c3a05a06beb80d3f1ef2e078f2.jpg) ， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1d680db5f32fd278f8d48e5407691154.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1d680db5f32fd278f8d48e5407691154.jpg) 为正交矩阵和 [] ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/502cdd9c79852b33d2a6d18ba5ec3102.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/502cdd9c79852b33d2a6d18ba5ec3102.jpg) 是一个上三角矩阵。

这将返回瘦(减少）QR分解。

Note

如果`input`的元素的大小很大，则精度可能会丢失

Note

虽然它应该总是给你一个有效的分解，它可能不会跨平台给你相同的 - 它将取决于你的LAPACK实现。

Note

不管原始步幅如何，返回的基质 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1d680db5f32fd278f8d48e5407691154.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1d680db5f32fd278f8d48e5407691154.jpg) 将被转置，即步幅为`(1, m)`而不是`(m, 1)`。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入2-D张量
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _任选_） - `Q`和`R`张量的元组

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

`U, S, V = torch.svd(A)`返回大小为`(n x m)`的实矩阵`A`的奇异值分解，使得 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/af257a01939a1fbe6211d4a4c168f25b.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/af257a01939a1fbe6211d4a4c168f25b.jpg) 。

`U`具有 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/7819768bc0adceb9951cf2ce9a0525f2.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/7819768bc0adceb9951cf2ce9a0525f2.jpg) 的形状。

`S`是形状 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b2d82f601df5521e215e30962b942ad1.jpg) 的对角矩阵，表示为包含非负对角线条目的大小 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/f72656b2358852a4b20972b707fd8222.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/f72656b2358852a4b20972b707fd8222.jpg) 的向量。

`V`具有 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/be5a855e888d33755dcdfa9d94e598d4.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/be5a855e888d33755dcdfa9d94e598d4.jpg) 的形状。

如果`some`为`True`(默认值），则返回的`U`和`V`矩阵将仅包含 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/3c84c3f757bca109ec4ed7fc0cada53f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/3c84c3f757bca109ec4ed7fc0cada53f.jpg) 正交列。

如果`compute_uv`是`False`，则返回的`U`和`V`矩阵将分别为形状 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/7819768bc0adceb9951cf2ce9a0525f2.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/7819768bc0adceb9951cf2ce9a0525f2.jpg) 和 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/be5a855e888d33755dcdfa9d94e598d4.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/be5a855e888d33755dcdfa9d94e598d4.jpg) 的零矩阵。这里将忽略`some`。

Note

在CPU上实现SVD使用LAPACK例程`?gesdd`(分而治之算法）而不是`?gesvd`来提高速度。类似地，GPU上的SVD也使用MAGMA例程`gesdd`。

Note

不管原始步幅如何，返回的矩阵`U`将被转置，即用步幅`(1, n)`代替`(n, 1)`。

Note

向后通过`U`和`V`输出时需要特别小心。当`input`具有所有不同的奇异值的满秩时，这种操作实际上是稳定的。否则，`NaN`可能会出现，因为未正确定义渐变。此外，请注意，即使原始后向仅在`S`上，双向后通常会通过`U`和`V`向后进行。

Note

当`some` = `False`时，`U[:, min(n, m):]`和`V[:, min(n, m):]`上的梯度将被反向忽略，因为这些矢量可以是子空间的任意基数。

Note

当`compute_uv` = `False`时，由于后向操作需要前向通道的`U`和`V`，因此无法执行后退操作。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入2-D张量
*   **一些** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - 控制返回`U`和`V`的形状
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _任选_） - 张量的输出元组

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

该函数返回实对称矩阵`input`的特征值和特征向量，由元组 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/07aa3eb39d8bb2f8d4d17cd4925159b4.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/07aa3eb39d8bb2f8d4d17cd4925159b4.jpg) 表示。

`input`和 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/21ec2ab32d1af3e766487093bb20cf22.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/21ec2ab32d1af3e766487093bb20cf22.jpg) 是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/be5a855e888d33755dcdfa9d94e598d4.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/be5a855e888d33755dcdfa9d94e598d4.jpg) 基质， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/be8982d125e27260b5c793cf0d39d70a.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/be8982d125e27260b5c793cf0d39d70a.jpg) 是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/20ddd8181c2e0d0fb893637e8572d475.jpg) 维向量。

该函数计算`input`的所有特征值(和向量），使得 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/db9e47c935013aa5b30057ba51ac84b9.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/db9e47c935013aa5b30057ba51ac84b9.jpg) 。

布尔参数`eigenvectors`仅定义特征向量或特征值的计算。

如果是`False`，则仅计算特征值。如果是`True`，则计算特征值和特征向量。

由于输入矩阵`input`应该是对称的，因此默认情况下仅使用上三角形部分。

如果`upper`是`False`，则使用下三角形部分。

注意：无论原始步幅如何，返回的矩阵`V`都将被转置，即使用步幅`(1, m)`而不是`(m, 1)`。

Parameters:

*   **输入** ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入对称矩阵
*   **特征向量**(_布尔_ _，_ _可选_） - 控制是否必须计算特征向量
*   **上**(_布尔_ _，_ _可选_） - 控制是否考虑上三角或下三角区域
*   **out**  ([_元组_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") _，_ _可选_） - 输出元组(Tensor，Tensor）

| Returns: | A tuple containing

＆GT; * **e** (_tensor_）：形状 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/2f4b02bcd5b11d436474c4c4cdb91683.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/2f4b02bcd5b11d436474c4c4cdb91683.jpg) 。每个元素是`input`的特征值，特征值按升序排列。 ＆GT; * **V** (_tensor_）：形状 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/be5a855e888d33755dcdfa9d94e598d4.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/be5a855e888d33755dcdfa9d94e598d4.jpg) 。如果`eigenvectors=False`，它是一个充满零的张量。否则，该张量包含`input`的标准正交特征向量。

| Return type: | ([Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor"), [Tensor](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")) |
| --- | --- |

例子：

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

求解具有三角系数矩阵 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 和多个右侧`b`的方程组。

特别是，解决 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/79ccd3754eebf815ed3195b42f93bacb.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/79ccd3754eebf815ed3195b42f93bacb.jpg) 并假设 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 是默认关键字参数的上三角形。

Parameters:

*   **A**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 输入三角系数矩阵
*   **b**  ([_Tensor_](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/tensors.html#torch.Tensor "torch.Tensor")） - 多个右侧。 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/6872867a863714d15d9a0d64c20734ce.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/6872867a863714d15d9a0d64c20734ce.jpg) 的每一列是方程组的右侧。
*   **上** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _可选_） - 是否解决上三角方程组(默认）或者下三角方程组。默认值：True。
*   **转座** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 是否应转置在被送到解算器之前。默认值：False。
*   **三角** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_，_ _任选_） - [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 是单位三角形。如果为True，则假定 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 的对角元素为1，并且未参考 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 。默认值：False。

| Returns: | 元组 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1044f4a1887b042eb41d12f782c0582f.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1044f4a1887b042eb41d12f782c0582f.jpg) 其中 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/f8961918eb987d8916766b1d77790ecb.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/f8961918eb987d8916766b1d77790ecb.jpg) 是 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/efdb05f076173b39fdd26ef663e7b0d8.jpg) 和 [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/1284cf6bcb6c2ffc47e2dd24cd1c51b8.jpg) 的克隆是[的解决方案] ![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/79ccd3754eebf815ed3195b42f93bacb.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/79ccd3754eebf815ed3195b42f93bacb.jpg) (或等式系统的任何变体，取决于关键字参数。） |
| --- | --- |

```py
Shape:
```

*   答： [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ff833c4d1f13ca018e121d87f6ef1607.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ff833c4d1f13ca018e121d87f6ef1607.jpg)
*   b： [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9b9aebaa467ad07dca05b5086bd21ca2.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9b9aebaa467ad07dca05b5086bd21ca2.jpg)
*   输出[0]： [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/9b9aebaa467ad07dca05b5086bd21ca2.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/9b9aebaa467ad07dca05b5086bd21ca2.jpg)
*   输出[1]： [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/ff833c4d1f13ca018e121d87f6ef1607.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/ff833c4d1f13ca018e121d87f6ef1607.jpg)

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

