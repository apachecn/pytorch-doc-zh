Multiprocessing package - torch.multiprocessing
===============================================

.. automodule:: torch.multiprocessing
.. currentmodule:: torch.multiprocessing

.. warning::

    如果主进程突然退出 (例如, 由于传入的信号) , Python 的多进程有时无法清理其子进程.
    这是一个已知的警告, 所以如果你在中断解释器之后发现任何资源泄漏, 这可能意味着这只是发生在你身上.

管理策略
-------------------

.. autofunction:: get_all_sharing_strategies
.. autofunction:: get_sharing_strategy
.. autofunction:: set_sharing_strategy

共享 CUDA 张量
--------------------

在进程之间共享 CUDA 张量仅在 Python 3 中支持, 使用 ``spawn`` 或 ``forkserver`` 启动方法.
Python 2 中的 :mod:`python:multiprocessing` 只能使用 ``fork`` 创建子进程, 而中 CUDA 运行时是不支持的.

.. warning::

    CUDA API 要求输出到其他进程的分配保持有效, 只要它们被它们使用.
    您应该注意, 并确保您共享的 CUDA 张量不会超出范围, 在有必要的情况下.
    这不应该是共享模型参数的问题, 而是应该小心地传递其他类型的数据.
    请注意, 此限制不适用于共享 CPU 内存.


共享策略
------------------

本节简要介绍不同分享策略的工作原理.
请注意, 它仅适用于 CPU 张量 - CUDA 张量将始终使用 CUDA API, 因为这是它们可以共享的唯一方式.

File descriptor - ``file_descriptor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. note::

    这是默认的策略 (除了不支持的 macOS 和 OS X之外) 
    This is the default strategy (except for macOS and OS X where it's not
    supported).

这个策略将使用文件描述符作为共享内存句柄.
无论何时将存储移动到共享内存, 从 ``shm_open`` 获取的文件描述符都将与该对象一起缓存,
并且当将要将其发送到其他进程时, 文件描述符将被传送 (例如, 通过 UNIX sockets) 到其中.
接收器还将缓存文件描述符并对其进行 ``mmap``, 以获得存储数据的共享视图.

请注意, 如果共享张量很大, 这个策略会在大部分时间保持大量的文件描述符.
如果您的系统对打开的文件描述符的数量有限制, 并且不能提高它们, 则应该使用 ``file_system`` 策略.

File system - ``file_system``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该策略将使用给 ``shm_open`` 的文件名来标识共享内存区.
这样做的好处是不需要缓存从中获取的文件描述符, 但同时也容易发生共享内存泄漏.
该文件创建后不能被删除, 因为其他进程需要访问它来打开它们各自的视图.
如果该进程崩溃, 并且不调用存储析构函数, 则这些文件将保留在系统中.
这种情况非常严重, 因为它们会一直使用内存, 直到系统重新启动, 或者被手动释放.

为了解决共享内存文件泄漏的问题, :mod:`torch.multiprocessing` 模块会产生一个名为 ``torch_shm_manager`` 的守护进程,
它将把自己从当前进程组中分离出来, 并跟踪所有的共享内存分配.
一旦连接到它的所有进程退出, 它将等待一会儿, 以确保不会有新的连接. 并将迭代组中已分配的所有共享内存文件.
如果发现其中任何一个仍然存在, 它们将被释放.
我们已经测试了这个方法, 并证明它对各种失败都是有效的.
不过, 如果你的系统有足够高的限制, ``file_descriptor`` 是一个所支持的策略, 虽然我们不建议切换到这个策略上.