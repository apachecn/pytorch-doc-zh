.. currentmodule:: torch.sparse

torch.sparse
============

.. warning::

    This API is currently experimental and may change in the near future.
    当前,此 API 是实验性质并且即将可能会改变.

Torch supports sparse tensors in COO(rdinate) format, which can
efficiently store and process tensors for which the majority of elements
are zeros.
Torch 支持 COO(rdinate) 格式的稀疏张量,它能有效的存储和驱动即使是多数是0的
元素的张量.

A sparse tensor is represented as a pair of dense tensors: a tensor
of values and a 2D tensor of indices.  A sparse tensor can be constructed
by providing these two tensors, as well as the size of the sparse tensor
(which cannot be inferred from these tensors!)  Suppose we want to define
a sparse tensor with the entry 3 at location (0, 2), entry 4 at
location (1, 0), and entry 5 at location (1, 2).  We would then write:
一个稀疏张量被表示为一对致密张量:一个张量的值和一个2D张量的索引.
可以通过这两个张量来构造稀疏张量,以及稀疏张量的大小(不能从这些张量推断!).
假设我们要在 (0,2) 处定义条目3,位置 (1,0) 的条目4,位置 (1,2) 的5的
稀疏张量,我们可以这样写:

    >>> i = torch.LongTensor([[0, 1, 1],
                              [2, 0, 2]])
    >>> v = torch.FloatTensor([3, 4, 5])
    >>> torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense()
     0  0  3
     4  0  5
    [torch.FloatTensor of size 2x3]

Note that the input to LongTensor is NOT a list of index tuples.  If you want
to write your indices this way, you should transpose before passing them to
the sparse constructor:
请注意, LongTensor 输入的不是索引元组的列表.如果您想用这种方式编写索引,您应该在将它们
传递给稀疏构造函数之前,进行转换:

    >>> i = torch.LongTensor([[0, 2], [1, 0], [1, 2]])
    >>> v = torch.FloatTensor([3,      4,      5    ])
    >>> torch.sparse.FloatTensor(i.t(), v, torch.Size([2,3])).to_dense()
     0  0  3
     4  0  5
    [torch.FloatTensor of size 2x3]

You can also construct hybrid sparse tensors, where only the first n
dimensions are sparse, and the rest of the dimensions are dense.
您还可以构建混合稀疏张量,其中只有第一个n维是稀疏的,其余的维度是密集的.

    >>> i = torch.LongTensor([[2, 4]])
    >>> v = torch.FloatTensor([[1, 3], [5, 7]])
    >>> torch.sparse.FloatTensor(i, v).to_dense()
     0  0
     0  0
     1  3
     0  0
     5  7
    [torch.FloatTensor of size 5x2]

An empty sparse tensor can be constructed by specifying its size:
可以构建一个指定大小的空的稀疏张量:

    >>> torch.sparse.FloatTensor(2, 3)
    SparseFloatTensor of size 2x3 with indices:
    [torch.LongTensor with no dimension]
    and values:
    [torch.FloatTensor with no dimension]

.. note::

    Our sparse tensor format permits *uncoalesced* sparse tensors, where
    there may be duplicate coordinates in the indices; in this case,
    the interpretation is that the value at that index is the sum of all
    duplicate value entries. Uncoalesced tensors permit us to implement
    certain operators more efficiently.
    我们的稀疏张量格式允许 *uncoalesced* 稀疏张量,其中索引中可能有重复的坐标; 在这
    种情况下,该索引处的值代表所有重复条目值的总和. Uncoalesced 张量允许我们更
    有效地实现确定的操作符.
 
    For the most part, you shouldn't have to care whether or not a
    sparse tensor is coalesced or not, as most operations will work
    identically given a coalesced or uncoalesced sparse tensor.
    However, there are two cases in which you may need to care.
    在大多数情况下,您不必关心稀疏张量是否 coalesced ,因为大多数操作 coalesced 和 uncoalesced 
    的张量的工作情况是相同的,但是,您可能需要关心两种情况.

    First, if you repeatedly perform an operation that can produce
    duplicate entries (e.g., :func:`torch.sparse.FloatTensor.add`), you
    should occasionally coalesce your sparse tensors to prevent
    them from growing too large.
    首先,如果您反复执行可以产生重复条目(例如 :func:`torch.sparse.FloatTensor.add` )
    的操作,您需要适当合并您的稀疏张量,以防止它们增长的太大.

    Second, some operators will produce different values depending on
    whether or not they are coalesced or not (e.g.,
    :func:`torch.sparse.FloatTensor._values` and
    :func:`torch.sparse.FloatTensor._indices`, as well as
    :func:`torch.Tensor._sparse_mask`).  These operators are
    prefixed by an underscore to indicate that they reveal internal
    implementation details and should be used with care, since code
    that works with coalesced sparse tensors may not work with
    uncoalesced sparse tensors; generally speaking, it is safest
    to explicitly coalesce before working with these operators.
    其次,一些操作符会产生不同的值这取决于它们是否是被 coalesced 的
    (例如,:func:`torch.sparse.FloatTensor._values` 
    和 :func:`torch.sparse.FloatTensor._indices`,
    还有 :func:`torch.Tensor._sparse_mask`),这些操作符前面加了一个下划线,表明了
    它们的内部实现,并且应当谨慎使用,因为 coalesced 的稀疏张量和 uncoalesced 的
    稀疏张量可能不能一起使用;一般来说,在运用这些操作符之前,最安全的方式是明确的
    coalesced .

    For example, suppose that we wanted to implement an operator
    by operating directly on :func:`torch.sparse.FloatTensor._values`.
    Multiplication by a scalar can be implemented in the obvious way,
    as multiplication distributes over addition; however, square root
    cannot be implemented directly, since ``sqrt(a + b) != sqrt(a) +
    sqrt(b)`` (which is what would be computed if you were given an
    uncoalesced tensor.)
    例如,假设我们想通过直接操作 torch.sparse.FloatTensor._values() 的一个实现.
    随着乘法分布的增加,标量的乘法可以以明显的方式实现; 然而,平方根不能直接实现,
    ``sqrt(a + b) != sqrt(a) +sqrt(b)`` (如果你赋予了一个 uncoalesced 的张量
    那会发生什么.)

.. class:: FloatTensor()

    .. method:: add
    .. method:: add_
    .. method:: clone
    .. method:: dim
    .. method:: div
    .. method:: div_
    .. method:: get_device
    .. method:: hspmm
    .. method:: mm
    .. method:: mul
    .. method:: mul_
    .. method:: resizeAs_
    .. method:: size
    .. method:: spadd
    .. method:: spmm
    .. method:: sspaddmm
    .. method:: sspmm
    .. method:: sub
    .. method:: sub_
    .. method:: t_
    .. method:: toDense
    .. method:: transpose
    .. method:: transpose_
    .. method:: zero_
    .. method:: coalesce
    .. method:: is_coalesced
    .. method:: _indices
    .. method:: _values
    .. method:: _nnz
