# torch.sparse

警告

这个API目前处于试验阶段，并在不久的将来可能会改变。

Torch 支持COO（rdinate）格式稀疏张量，其可有效地存储和处理张量的量，大部分元素都为零。

稀疏张量被表示为一对密张量：值的张量和指数的2D张量。稀疏张量可以通过提供这两个张量，以及所述稀疏张量的大小来构造（其不能从这些张量推断！）假设我们要定义与在位置的条目3的稀疏张量（0，2）
，条目4在位置（1,0），和条目5在位置（1,2）。然后，我们可以这样写：

    
    
    >>> i = torch.LongTensor([[0, 1, 1],
                              [2, 0, 2]])
    >>> v = torch.FloatTensor([3, 4, 5])
    >>> torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense()
     0  0  3
     4  0  5
    [torch.FloatTensor of size 2x3]
    

请注意，输入到LongTensor不是索引元组的列表。如果你想写的指标这样，你应该将它们传递到稀疏的构造函数之前转：

    
    
    >>> i = torch.LongTensor([[0, 2], [1, 0], [1, 2]])
    >>> v = torch.FloatTensor([3,      4,      5    ])
    >>> torch.sparse.FloatTensor(i.t(), v, torch.Size([2,3])).to_dense()
     0  0  3
     4  0  5
    [torch.FloatTensor of size 2x3]
    

您还可以构建混合稀疏张量，其中只有第N维稀疏，并且尺寸的其余密集。

    
    
    >>> i = torch.LongTensor([[2, 4]])
    >>> v = torch.FloatTensor([[1, 3], [5, 7]])
    >>> torch.sparse.FloatTensor(i, v).to_dense()
     0  0
     0  0
     1  3
     0  0
     5  7
    [torch.FloatTensor of size 5x2]
    

空稀疏张量可以通过指定其大小来构造：

    
    
    >>> torch.sparse.FloatTensor(2, 3)
    SparseFloatTensor of size 2x3 with indices:
    [torch.LongTensor with no dimension]
    and values:
    [torch.FloatTensor with no dimension]
    

SparseTensor has the following invariants:

    

  1. sparse_dim + dense_dim = LEN（SparseTensor.shape）

  2. SparseTensor._indices（）。形状=（sparse_dim，NNZ）

  3. 。SparseTensor._values（）形状=（NNZ，SparseTensor.shape [sparse_dim：]）

由于SparseTensor._indices（）始终是一个2D张量，最小sparse_dim = 1。因此，sparse_dim = 0
SparseTensor的表示仅仅是一个致密的张量。

注意

我们稀疏张量格式许可 _未聚_
稀疏张量，哪里有可能在指数复制坐标;在这种情况下，解释是该索引的值是所有重复的值项的总和。未聚张量使我们能够更有效地实现某些运营商。

在大多数情况下，你不应该去关心稀疏张量是否被合并与否，大多数操作将工作给予相同的聚结或未聚稀疏张量。然而，有两种情况中，你可能需要关心。

首先，如果你重复执行一个操作，即可以产生重复的条目（例如， `torch.sparse.FloatTensor.add（） `
），你应该偶尔凝聚你的稀疏张量，以防止它们变得太大。

第二，一些运营商将根据它们是否被合并或不（例如， `torch.sparse.FloatTensor._values（） `并产生不同的值 `
torch.sparse.FloatTensor._indices（） `，以及[ `torch.Tensor.sparse_mask（） `
](tensors.html#torch.Tensor.sparse_mask
"torch.Tensor.sparse_mask")）。这些运营商正在通过一个下划线前缀，以表明他们露出内部的实现细节，应小心使用，因为这与凝聚的稀疏张量运行的代码可能无法与非联合稀疏张量工作;一般来说，它是最安全的前明确合并与这些运营商合作。

例如，假设我们希望通过直接在 `操作以实现操作员torch.sparse.FloatTensor._values（） `
。乘以一个标量可以在明显的方式来实现，如乘法分布在另外;然而，平方根不能直接实现的，因为`SQRT（一个 +  [ HTG11 b） ！=
SQRT（一） +  SQRT（b）中 `（这是将如果给你一个未聚张量来计算。）

_class_`torch.sparse.``FloatTensor`

    

`add`()

    

`add_`()

    

`clone`()

    

`dim`()

    

`div`()

    

`div_`()

    

`get_device`()

    

`hspmm`()

    

`mm`()

    

`mul`()

    

`mul_`()

    

`narrow_copy`()

    

`resizeAs_`()

    

`size`()

    

`spadd`()

    

`spmm`()

    

`sspaddmm`()

    

`sspmm`()

    

`sub`()

    

`sub_`()

    

`t_`()

    

`toDense`()

    

`transpose`()

    

`transpose_`()

    

`zero_`()

    

`coalesce`()

    

`is_coalesced`()

    

`_indices`()

    

`_values`()

    

`_nnz`()

    

## 功能

`torch.sparse.``addmm`( _mat_ , _mat1_ , _mat2_ , _beta=1_ , _alpha=1_
)[[source]](_modules/torch/sparse.html#addmm)

    

这个函数完全相同的东西作为[ `torch.addmm（） `](torch.html#torch.addmm
"torch.addmm")在向前，不同之处在于它支持向后对稀疏矩阵`MAT1`。 `MAT1`需要有 sparse_dim = 2
。注意，MAT1的`梯度 `是一个聚结的稀疏张量。

Parameters

    

  * **垫** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要添加致密基质

  * **MAT1** （ _SparseTensor_ ） - 要乘以一个稀疏矩阵

  * **MAT2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 致密的矩阵相乘

  * **的β** （ _号码_ _，_ _可选_ ） - 乘数`垫 `（ β \的β β ）

  * **阿尔法** （ _号码_ _，_ _可选_ ） - 乘数 M  一 T  1  @  M  一 T  2  MAT1 @ MAT2  M  一 T  1  @  M  一 吨 2  （ α \阿尔法 α ）

`torch.sparse.``mm`( _mat1_ , _mat2_
)[[source]](_modules/torch/sparse.html#mm)

    

执行稀疏矩阵的矩阵乘法`MAT1`和稠密矩阵`MAT2`。类似于[ `torch.mm（） `](torch.html#torch.mm
"torch.mm")，如果`MAT1`是 （ n的 × M  ） （N \乘以m） （ n的 × M  ） 张量，`MAT2`是 （ M  ×
p  ） （M \倍p） （ M  × P  ） 张量，进行将是 （ n的 × p  ） （ ñ\倍p） （ n的 × p  ） 密集的张量。 `MAT1
`需要有 sparse_dim = 2 。此功能还支持向后两个矩阵。注意，MAT1的`梯度 `是一个聚结的稀疏张量。

Parameters

    

  * **MAT1** （ _SparseTensor_ ） - 第一稀疏矩阵相乘

  * **MAT2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要被相乘的第二密集矩阵

例：

    
    
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
    

`torch.sparse.``sum`( _input_ , _dim=None_ , _dtype=None_
)[[source]](_modules/torch/sparse.html#sum)

    

返回在给定尺寸SparseTensor `输入 `中的每一行的总和`暗淡 `。如果`暗淡 `为维度的列表，减少过度所有的人。当总和在所有`
sparse_dim`，此方法返回一个张量代替SparseTensor。

所有求和`暗淡 `被挤压（见[ `torch.squeeze（） `](torch.html#torch.squeeze
"torch.squeeze")），从而导致具有输出张量`暗淡 `比尺寸较少`输入 `。

期间落后，仅在梯度`NNZ`的`位置的输入 `将传播回来。请注意，输入 的`梯度被聚结。`

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入SparseTensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _蟒的元组：整数_ ） - 一个维度或维度列表，以减少。默认值：减少对所有变暗。

  * **DTYPE** （`torch.dtype`，可选） - 所需的数据返回张量的类型。默认：`输入 `D型。

Example:

    
    
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
    

[Next ![](_static/images/chevron-right-orange.svg)](cuda.html "torch.cuda")
[![](_static/images/chevron-right-orange.svg) Previous](type_info.html "Type
Info")

* * *

©版权所有2019年，Torch 贡献者。