多进程的最佳实践
==============================

:mod:`torch.multiprocessing` 是 Python 中 :mod:`python:multiprocessing` 模块的替代.
它支持完全相同的操作, 但进一步扩展了它的功能, 使得所有张量可以通过 :class:`python:multiprocessing.Queue` 传输,
将其数据移动到共享内存中, 并且只会向其他进程发送一个句柄. 

.. note::
        
    当一个 :class:`~torch.autograd.Variable` 被发送到另一个进程中, :attr:`Variable.data` 和 :attr:`Variable.grad.data` 都将被共享.

这里允许实现各种训练方法, 例如 Hogwild, A3C, 或者其他需要异步操作的方法.

共享 CUDA 向量
--------------------

只有 Python 3 支持使用 ``spawn`` 或 ``forkserver`` 启动方法在进程中共享 CUDA 向量. :mod:`python:multiprocessing` 在 Python 2 使用 ``fork`` 
只能创建子进程, 但是在 CUDA 运行时不被支持.

.. warning::

    CUDA API 要求被导出到其他进程的分配只要被使用, 就要一直保持有效. 您应该小心, 确保您共享的CUDA张量只要有必要就不要超出范围. 
    这不是共享模型参数的问题, 但传递其他类型的数据应该小心. 注意, 此限制不适用于共享 CPU 内存. 

参考: :ref:`cuda-nn-dataparallel-instead`


最佳实践和提示
-----------------------

避免和抵制死锁
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当新进程被创建时, 可能会发生很多错误, 最常见的原因就是后台线程. 如果有任何线程持有锁或导入模块, 
并且 ``fork`` 已被调用, 则子进程很有可能将会处于毁坏的状态, 并导致死锁或在其他地方失败. 
注意即使你自己没有这样做, Python 内置的库也会这样做 - 不需要比 :mod:`python:multiprocessing` 看得更远.
:class:`python:multiprocessing.Queue` 事实上是一个非常复杂的库, 它可以创建多个线程, 用于序列化, 发送和接收对象, 
但是它们也有可能引起前面提到的问题. 如果你遇到这样的问题, 可以尝试使用 :class:`~python:multiprocessing.queues.SimpleQueue`, 
它不会使用其他额外的线程.

我们正在竭尽全力把它设计得更简单, 并确保这些死锁不会发生, 但有些事情无法控制. 如果有任何问题您一时无法解决, 请尝试在论坛上提出,
我们将看看是否可以解决.


重用经过队列的缓冲区
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

请记住当每次将 :class:`~torch.Tensor` 放入 :class:`python:multiprocessing.Queue`, 
它必须被移至共享内存中. 如果它已经被共享, 它是一个无效操作, 否则会产生一个额外的内存副本,
这会减缓整个进程. 即使你有一个进程池来发送数据到一个进程, 也应该先把它送回缓冲区 —— 这几乎是没有损失的, 
并且允许你在发送下一个 batch 时避免产生副本.

异步多进程训练 (例如 Hogwild) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 :mod:`torch.multiprocessing` 可以异步地训练模型, 其中参数可以一直共享, 或定期同步. 
对于第一种情况, 我们建议传输整个模型对象, 而对于第二种情况, 我们建议只传输 :meth:`~torch.nn.Module.state_dict`.

我们建议使用 :class:`python:multiprocessing.Queue` 来在进程之间传输各种 PyTorch 对象.
例如, 当使用 ``fork`` 启动方法, 有可能会继承共享内存中的张量和存储量. 但这是非常容易出错的, 
应谨慎使用, 最好是成为深度用户以后, 再使用这个方法. 队列虽然有时是一个较不优雅的解决方案, 但基本上能在所有情况下都正常工作.

.. warning::

    当使用全局的声明时, 你应该注意, 因为它们没有被 ``if __name__ == '__main__'`` 限制. 如果使用与 ``fork`` 不同的启动方法, 
    它们将在所有子进程中被执行.

Hogwild
~~~~~~~

一个 Hogwild 的具体实现可以在 `examples repository`__ 中找到. 为了展示代码的整体结构, 下面有一个小例子::

    import torch.multiprocessing as mp
    from model import MyModel

    def train(model):
        # Construct data_loader, optimizer, etc.
        for data, labels in data_loader:
            optimizer.zero_grad()
            loss_fn(model(data), labels).backward()
            optimizer.step()  # This will update the shared parameters

    if __name__ == '__main__':
        num_processes = 4
        model = MyModel()
        # NOTE: this is required for the ``fork`` method to work
        model.share_memory()
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=train, args=(model,))
            p.start()
            processes.append(p)
        for p in processes:
          p.join()

.. __: https://github.com/pytorch/examples/tree/master/mnist_hogwild
