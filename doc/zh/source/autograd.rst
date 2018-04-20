.. role:: hidden
    :class: hidden-section

Automatic differentiation package - torch.autograd
==================================================

.. automodule:: torch.autograd
.. currentmodule:: torch.autograd

.. autofunction:: backward

.. autofunction:: grad

Variable (变量) 
----------------

API compatibility
^^^^^^^^^^^^^^^^^

Variable API 几乎与常规 Tensor API 相同(一些会覆盖梯度计算输入的内置方法除外).
在大多数情况下, 变量量可以安全地替换张量并且代码将保持正常工作.
因为这个, 我们没有记录变量上的所有操作, 你应该参阅 :class:`torch.Tensor` 文档来查看变量上的所有操作.

In-place operations on Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 autograd 支持就地操作是一件困难的事情, 在大多数情况下我们不鼓励使用.
Autograd 积极的缓冲区释放和重用使得它非常高效, 而且很少有就地操作实际上大量地降低了内存使用量的情况.
除非你正在大量的的内存压力下运行, 否则你可能永远不需要使用它们.

In-place correctness checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

所有的 :class:`Variable` 跟踪适用于它们的就地操作, 并且如果实现检测到一个变量是否被其中一个函数后台保存, 但是之后它被就地修改了, 会在开始求导时会报出异常.
这确保了如果你在就地使用函数并没有看到任何错误, 你可以肯定的是计算变量是正确的.


.. autoclass:: Variable
    :members:

:hidden:'Function(函数)'
---------------------------

.. autoclass:: Function
    :members:

Profiler(分析器)
------------------

Autograd 包含一个分析器, 可以让你检查你的模型在CPU 和 GPU 上不同运算的成本.
目前实现有两种模式 - 只使用 CPU 的 :class:`~torch.autograd.profiler.profile`.
和基于 nvprof (注册 CPU 和 GPU 活动) 的方式使用 :class:`~torch.autograd.profiler.emit_nvtx`.

.. autoclass:: torch.autograd.profiler.profile
    :members:

.. autoclass:: torch.autograd.profiler.emit_nvtx
    :members:

.. autofunction:: torch.autograd.profiler.load_nvprof
