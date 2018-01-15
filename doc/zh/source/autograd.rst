.. role:: hidden
    :class: hidden-section

Automatic differentiation package - torch.autograd
==================================================

.. automodule:: torch.autograd
.. currentmodule:: torch.autograd

.. autofunction:: backward

.. autofunction:: grad

Variable
--------

API compatibility
^^^^^^^^^^^^^^^^^

变量API几乎与常规Tensor API相同（除了例外）
一对元祖更改的方法，这将覆盖输入所需的
梯度计算）. 在大多数情况下，张量可以安全地替换
变量和代码将保持正常工作. 因为这个，
我们没有记录变量上的所有操作，你应该这样做
请参阅：class：`torch.Tensor` docs为此目的.

In-place operations on Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在autograd支持就地操作是一件困难的事情，我们不鼓励
他们在大多数情况下使用. Autograd积极的缓冲区释放和重用使得
它非常高效，而且就地操作的场合也很少
实际上降低了大量的内存使用量. 除非你正在操作
在大量的的记忆下，你可能永远不需要使用它们.

In-place correctness checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

所有的：class：`Variable`s跟踪适用于它们的就地操作，并且
如果实现检测到一个变量被保存在后面的一个
这个函数，但是之后它被修改了，会在开始求导时会报出异常. 这确保了如果你在就地使用
函数并没有看到任何错误，你可以肯定的是计算
变量是正确的.


.. autoclass:: Variable
    :members:

:hidden:`Function`
------------------

.. autoclass:: Function
    :members:

Profiler
--------

Autograd包含一个分析器，可以让你检查不同的成本
在你的模型中的运算符 - 在CPU和GPU上. 有两种模式
目前实现 - 只使用CPU：class：`〜torch.autograd.profiler.profile`.
和基于nvprof（注册CPU和GPU活动）使用
产品类别：`〜torch.autograd.profiler.emit_nvtx`.

.. autoclass:: torch.autograd.profiler.profile
    :members:

.. autoclass:: torch.autograd.profiler.emit_nvtx
    :members:

.. autofunction:: torch.autograd.profiler.load_nvprof
