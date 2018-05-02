torch.optim
===================================

.. automodule:: torch.optim

如何使用 optimizer (优化器) 
----------------------------

为了使用 :mod:`torch.optim` 你需要创建一个 optimizer 对象, 这个对象能够保持当前的状态以及依靠梯度计算
来完成参数更新.

构建
^^^^

要构建一个 :class:`Optimizer` 你需要一个可迭代的参数 (全部都应该是 :class:`~torch.autograd.Variable`) 进行优化. 然后,
你能够设置优化器的参数选项, 例如学习率, 权重衰减等.

.. note::

    如果你需要通过 `.cuda()` 将模型移动到 GPU 上, 请在构建优化器之前来移动.
    模型的参数在进行 `.cuda()` 之后将变成不同的对象,该对象与之前调用的参数不同.

    通常来说, 在对优化器进行构建和调用的时候, 你应该要确保优化参数位于相同的
    地点.

例子 ::

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr = 0.0001)

为每个参数单独设置选项
^^^^^^^^^^^^^^^^^^^^^^

:class:`Optimizer` 也支持为每个参数单独设置选项.
若要这么做, 不要直接使用 :class: `~torch.autograd.Variable` 的迭代, 而是使用 :class:`dict` 的迭代.
每一个 dict 都分别定义了一组参数, 并且应该要包含 ``params`` 键,这个键对应列表的参数. 
其他的键应该与 optimizer 所接受的其他参数的关键字相匹配, 并且会被用于对这组参数的优化.

.. note::

    你仍然能够传递选项作为关键字参数.在未重写这些选项的组中, 它们会被用作默认值.
    这非常适用于当你只想改动一个参数组的选项, 但其他参数组的选项不变的情况.


例如, 当我们想指定每一层的学习率时, 这是非常有用的::

    optim.SGD([
                    {'params': model.base.parameters()},
                    {'params': model.classifier.parameters(), 'lr': 1e-3}
                ], lr=1e-2, momentum=0.9)

这意味着 ``model.base`` 的参数将会使用 ``1e-2`` 的学习率, ``model.classifier`` 的参数将会使用 ``1e-3`` 的学习率,
并且 ``0.9`` 的 momentum 将应用于所有参数.

进行单步优化
^^^^^^^^^^^^

所有的优化器都实现了 :func:`~Optimizer.step` 方法, 且更新到所有的参数.
它可以通过以下两种方式来使用:

``optimizer.step()``
~~~~~~~~~~~~~~~~~~~~

这是大多数 optimizer 所支持的简化版本.
一旦使用 :func:`~torch.autograd.Variable.backward` 之类的函数计算出来梯度之后我们就可以调用这个函数了.

例子 ::

    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

``optimizer.step(closure)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

一些优化算法例如 Conjugate Gradient 和 LBFGS 需要重复多次计算函数,
因此你需要传入一个闭包去允许它们重新计算你的模型.
这个闭包应当清空梯度, 计算损失, 然后返回.

例子 ::

    for input, target in dataset:
        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            return loss
        optimizer.step(closure)

算法
----------

.. autoclass:: Optimizer
    :members:
.. autoclass:: Adadelta
    :members:
.. autoclass:: Adagrad
    :members:
.. autoclass:: Adam
    :members:
.. autoclass:: SparseAdam
    :members:
.. autoclass:: Adamax
    :members:
.. autoclass:: ASGD
    :members:
.. autoclass:: LBFGS
    :members:
.. autoclass:: RMSprop
    :members:
.. autoclass:: Rprop
    :members:
.. autoclass:: SGD
    :members:

如何调整学习率
---------------------------

:mod: `torch.optim.lr_scheduler` 基于循环的次数提供了一些方法来调节学习率.
:class: `torch.optim.lr_scheduler.ReduceLROnPlateau` 基于验证测量结果来设置不同的学习率.

.. autoclass:: torch.optim.lr_scheduler.LambdaLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.StepLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.MultiStepLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.ExponentialLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.ReduceLROnPlateau
    :members:
