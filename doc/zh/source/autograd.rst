.. role:: hidden
    :class: hidden-section

Automatic differentiation package - torch.autograd
==================================================

.. automodule:: torch.autograd
.. currentmodule:: torch.autograd

.. autofunction:: backward

.. autofunction:: grad

Variable（变量）
----------------

API compatibility
^^^^^^^^^^^^^^^^^

Variable API 几乎与常规 Tensor API 相同(除了一些例外)
一对元祖更改的方法, 这将覆盖输入所需的梯度计算.
在大多数情况下, 张量可以安全地替换变量和代码将保持正常工作.
因为这个, 我们没有记录变量上的所有操作, 你应该参阅 :class:`torch.Tensor` 文档以达到我们的目的.

In-place operations on Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 autograd 支持就地操作是一件困难的事情, 我们不鼓励他们在大多数情况下使用.
Autograd 积极的缓冲区释放和重用使得它非常高效, 而且就地操作的场合也很少实际上降低了大量的内存使用量.
除非你正在操作在大量的的内存下, 否则你可能永远不需要使用它们.

In-place correctness checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

所有的 :class:`Variable` 跟踪适用于它们的就地操作, 并且如果实现检测到一个变量被保存在后面的一个这个函数, 但是之后它被修改了, 会在开始求导时会报出异常.
这确保了如果你在就地使用函数并没有看到任何错误, 你可以肯定的是计算变量是正确的.


.. autoclass:: Variable
    :members:

Function（函数）
---------------------------

.. autoclass:: Function
    :members:

Profiler（分析器）
------------------

Autograd 包含一个分析器, 可以让你检查不同的成本在你的模型中的运算符 - 在 CPU 和 GPU 上.
有两种模式目前实现 - 只使用 CPU 的 :class:`~torch.autograd.profiler.profile`.
和基于 nvprof（注册 CPU 和 GPU 活动）的方式使用 :class:`~torch.autograd.profiler.emit_nvtx`.

.. autoclass:: torch.autograd.profiler.profile
    :members:

.. autoclass:: torch.autograd.profiler.emit_nvtx
    :members:

.. autofunction:: torch.autograd.profiler.load_nvprof
