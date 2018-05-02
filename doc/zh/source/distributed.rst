.. role:: hidden
    :class: hidden-section

Distributed communication package - torch.distributed
=====================================================

.. automodule:: torch.distributed
.. currentmodule:: torch.distributed

目前torch.distributed支持三个后端, 每个都有不同的功能. 下表显示哪些功能可用于 CPU/CUDA 张量.
只有在设备上编译安装PyTorch, 才能在MPI的设备上支持cuda.


+------------+-----------+-----------+-----------+
| Backend    | ``tcp``   | ``gloo``  | ``mpi``   |
+------------+-----+-----+-----+-----+-----+-----+
| Device     | CPU | GPU | CPU | GPU | CPU | GPU |
+============+=====+=====+=====+=====+=====+=====+
| send       | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| recv       | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| broadcast  | ✓   | ✘   | ✓   | ✓   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| all_reduce | ✓   | ✘   | ✓   | ✓   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| reduce     | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| all_gather | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| gather     | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| scatter    | ✓   | ✘   | ✘   | ✘   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+
| barrier    | ✓   | ✘   | ✓   | ✓   | ✓   | ?   |
+------------+-----+-----+-----+-----+-----+-----+

.. _distributed-basics:

Basics
------
`torch.distributed` 为在一台或多台机器上运行的多个计算节点提供多进程并行的通信模块和PyTorch的支持.
类 :func:`torch.nn.parallel.DistributedDataParallel` 建立在这个功能之上, 以提供任何PyTorch模型分布式训练的装饰器.
这个类和 :doc:`multiprocessing` 和 :func:`torch.nn.DataParallel` 并不相同, PyTorch集群分布式计算支持多台机器,
使用时用户必须在主要训练的脚本中, 明确地将每个进程复制到每台机器中.

在单机多节点计算的情况下, 使用 `torch.distributed` 和 :func:`torch.nn.parallel.DistributedDataParallel` 作为
训练的装饰器, 相比于 :func:`torch.nn.DataParallel` 之类的数据并行计算, 任然具有优势:

* 在每次迭代中, 每个进程维护自己的优化器, 执行完整的优化步骤. 虽然这看起来可能是多余的, 但是因为梯度已经被收集在
  一起, 并且计算了梯度的平均值, 因此对于每个进程梯度是相同的, 这可以减少在节点之间传递张量, 再计算参数的时间.

* 每个进程都包含一个独立的Python解释器, 消除了Python解释器的额外开销, 以及由于驱动多线程, 模型副本和GPU造成
   "GIL-thrashing" . 对于需要消耗大量Python解释器运行时间 (包括具有循环图层或许多小组件的模型) 的模型来说是非常重要的.


Initialization
--------------

在调用其他模型之前, 这个包需要使用 :func:`torch.distributed.init_process_group` 函数进行初始化.
在初始化单元中, 所有进程都会参与.


.. autofunction:: init_process_group

.. autofunction:: get_rank

.. autofunction:: get_world_size

--------------------------------------------------------------------------------

目前支持三种初始化的方法:

TCP initialization
^^^^^^^^^^^^^^^^^^

提供两种TCP的初始化的方法, 两种方法都需要各台机器的网络地址和集群机器数目 ``world_size`` .
第一种方法需要指定属于0级进程的地址, 并且初始化时所有进程的等级都由手动指定.

第二种方法是, 地址必须是有效的IP多播地址, 在这种情况下, 可以自动分配等级.
多路通信的初始化也支持 ``group_name`` 参数, 它允许你为多个作业使用相同的地址, 只要它们使用不同的小组名即可.


::

    import torch.distributed as dist

    # Use address of one of the machines
    dist.init_process_group(init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)

    # or a multicast address - rank will be assigned automatically if unspecified
    dist.init_process_group(init_method='tcp://[ff15:1e18:5d4c:4cf0:d02d:b659:53ba:b0a7]:23456',
                            world_size=4)

Shared file-system initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

另一个初始化方法使用一个文件系统, 这个文件系统在一个组中的所有机器上共享和可见, 以及一个所需的 ``world_size`` 参数.
URL应该以 ``file://`` 开头, 并包含一个可以和共享文件系统所有现有目录中的路径相区别的路径, 作为URL. 这个初始化方法也
支持 ``group_name`` 参数, 它允许你为多个作业使用相同的共享文件路径, 只要它们使用不同的小组名.


.. warning::

    这种方法假设文件系统支持使用 ``fcntl`` 进行锁定 -大多数本地系统和NFS都支持它.

::

    import torch.distributed as dist

    # Rank will be assigned automatically if unspecified
    dist.init_process_group(init_method='file:///mnt/nfs/sharedfile', world_size=4,
                            group_name=args.group)

Environment variable initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

此方法将从环境变量中读取配置, 从而可以完全自定义如何获取信息. 要设置的变量是:

* ``MASTER_PORT`` - 需要; 必须是0级机器上的自由端口
* ``MASTER_ADDR`` - 需要 (除了等级0) ; 等级0节点的地址
* ``WORLD_SIZE`` - 需要; 可以在这里设置, 或者在调用init函数
* ``RANK`` - 需要; 可以在这里设置, 或者在调用init函数

等级为0的机器将用于设置所有连接.

这是默认的方法, 这意味着 ``init_method`` 不必被特别指定(或者可以是 ``env://`` )


Groups
------

默认的集群 (collectives) 操作默认的小组 (group), 要求所有的进程进入分布式函数中调用. 一些工作负载可以从可以从更细粒度的通信中受益
这是分布式集群发挥作用的地方. :func:`~torch.distributed.new_group` 函数可以用来创建新的组, 并且包含所有进程的任意子集.
它返回一个不透明的组句柄, 它可以作为集群的 ``group``  参数 (集群 collectives 是一般的编程模式中的交换信息的分布式函数) .


.. autofunction:: new_group

Point-to-point communication
----------------------------

.. autofunction:: send

.. autofunction:: recv

:func:`~torch.distributed.isend` 和 :func:`~torch.distributed.irecv`
使用时返回分布式请求对象. 通常, 这个对象的类型是未指定的, 因为它们不能使用手动创建, 但是它们支持两种方法指定:

* ``is_completed()`` - 如果操作完成返回True
* ``wait()`` - 如果操作完成会阻塞所有的进程.
  ``is_completed()`` 如果结果返回, 保证函数返回True.

当使用MPI作为后端, :func:`~torch.distributed.isend` 和 :func:`~torch.distributed.irecv`
支持 "不超车" 式的工作方式, 这种方式可以保证消息的顺序. 更多的细节可以看
http://mpi-forum.org/docs/mpi-2.2/mpi22-report/node54.htm#Node54

.. autofunction:: isend

.. autofunction:: irecv

Collective functions
--------------------

.. autofunction:: broadcast

.. autofunction:: all_reduce

.. autofunction:: reduce

.. autofunction:: all_gather

.. autofunction:: gather

.. autofunction:: scatter

.. autofunction:: barrier
