.. _broadcasting-semantics:

广播语义
======================

一些 PyTorch 的操作支持基于 :any:`NumPy Broadcasting Semantics <numpy.doc.broadcasting>`.

简而言之, 如果一个 PyTorch 操作支持广播语义, 那么它的张量参数可以自动扩展为相同的大小（不需要复制数据）

一般语义
-----------------
如果两个张量满足如下规则，那么就认为其是 `broadcastable`

- 每个张量至少存在维度
- 在遍历维度大小时, 从尾部维度开始遍历，并且二者维度必须相等, 它们其中一个要么是1要么不存在

示例::

    >>> x=torch.FloatTensor(5,7,3)
    >>> y=torch.FloatTensor(5,7,3)
    # 相同的形状总是满足的（上述规则总是成立的）

    >>> x=torch.FloatTensor()
    >>> y=torch.FloatTensor(2,2)
    # x and y are not broadcastable, because x does not have at least 1 dimension

    # 可以排列尾部维度
    >>> x=torch.FloatTensor(5,3,4,1)
    >>> y=torch.FloatTensor(  3,1,1)
    # x and y are broadcastable.
    # 1st trailing dimension: both have size 1
    # 2nd trailing dimension: y has size 1
    # 3rd trailing dimension: x size == y size
    # 4th trailing dimension: y dimension doesn't exist

    # 但是:
    >>> x=torch.FloatTensor(5,2,4,1)
    >>> y=torch.FloatTensor(  3,1,1)
    # x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3

如果两个张量 :attr:`x`, :attr:`y` 是 `broadcastable`, 则结果张量的大小由如下方式计算:
- 如果维度的数量 :attr:`x` 和 :attr:`y` 不相等, 在维度较少的张量的维度前置 1
- 然后，对于每个维度的大小，生成维度的大小是 attr:`x` 和 :attr:`y` 的最大值

示例 ::

    # 可以排列尾部维度，使阅读更容易
    >>> x=torch.FloatTensor(5,1,4,1)
    >>> y=torch.FloatTensor(  3,1,1)
    >>> (x+y).size()
    torch.Size([5, 3, 4, 1])

    # 但是也可不必排列
    >>> x=torch.FloatTensor(1)
    >>> y=torch.FloatTensor(3,1,7)
    >>> (x+y).size()
    torch.Size([3, 1, 7])

    >>> x=torch.FloatTensor(5,2,4,1)
    >>> y=torch.FloatTensor(3,1,1)
    >>> (x+y).size()
    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1

直接语义
------------------
一个复杂操作就是直接操作不满足情况的两个张量使其改变维度满足条件

示例 ::

    >>> x=torch.FloatTensor(5,3,4,1)
    >>> y=torch.FloatTensor(3,1,1)
    >>> (x.add_(y)).size()
    torch.Size([5, 3, 4, 1])

    # but:
    >>> x=torch.FloatTensor(1,3,1)
    >>> y=torch.FloatTensor(3,1,7)
    >>> (x.add_(y)).size()
    RuntimeError: The expanded size of the tensor (1) must match the existing size (7) at non-singleton dimension 2.

向后兼容
-----------------------
以前的版本的 PyTorch 只要张量中的元素数目是相等的, 便允许某些点状函数在不同的形状的张量上执行, 其中点状操作是通过将每个张量视为 1 维执行
现今 PyTorch 支持广播语义和不推荐使用点状函数操作向量, 并且将在具有相同数量的元素但不支持广播语义的张量操作生成一个 Python 警告

注意, 广播语义的引入可能会导致向后不兼容的情况, 即两个张量形状不同, 但是数量相同且支持广播语义.

示例 ::

    >>> torch.add(torch.ones(4,1), torch.randn(4))

本预生成一个: torch.Size([4,1]) 的张量,但是现在会生成一个: torch.Size([4,4]) 的张量.
为了帮助使用者识别代码中可能存在由引入广播语义的向后不兼容情况, 
你可以将 `torch.utils.backcompat.broadcast_warning.enabled` 设置为 `True`, 在这种情况下是这会生成一个 Python 警告

示例 ::

    >>> torch.utils.backcompat.broadcast_warning.enabled=True
    >>> torch.add(torch.ones(4,1), torch.ones(4))
    __main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, and have the same number of elements.
    Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 1-dimensional.
