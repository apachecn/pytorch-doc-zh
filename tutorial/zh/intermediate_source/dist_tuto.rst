使用 PyTorch 编写分布式程序
=============================================
**作者**: `Séb Arnold <http://seba1511.com>`_

在这个简单的教程中, 我们将介绍 PyTorch 中的 distributed 包.
我们将介绍如何进行分布式设置, 使用不同的通讯策略, 和学习一些包内部的实现。

设置
-----

.. raw:: html

   <!--
   * Processes & machines
   * variables and init_process_group
   -->

PyTorch 中的 distributed 包 (即 ``torch.distributed``) 让研究人员和从业者能够容易的
在进程间和机器集群间并行化他们的计算. 为此, 它利用消息传递语义, 允许每个进程将数据传递给任何其他进程.
与 multiprocessing 包 (``torch.multiprocessing``) 对比, processes 可以使用不同的通讯后端,
并且不限于在同一台机器上执行.

为了可以开始, 我们需要能够同时允许多个进程. 如果你有权访问计算机集群, 
则应该检查本地系统管理员或使用您最喜欢的协助工具. (例如,
`pdsh <https://linux.die.net/man/1/pdsh>`__,
`clustershell <http://cea-hpc.github.io/clustershell/>`__, 或者
`其他 <https://slurm.schedmd.com/>`__) 
为了本教程的目的, 我们将使用单台机器, 然后使用下面的模板 fork 多个进程. 

.. code:: python

    """run.py:"""
    #!/usr/bin/env python
    import os
    import torch
    import torch.distributed as dist
    from torch.multiprocessing import Process

    def run(rank, size):
        """ 之后将实现的分布式函数. """
        pass

    def init_processes(rank, size, fn, backend='tcp'):
        """ 初始化分布式环境. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)


    if __name__ == "__main__":
        size = 2
        processes = []
        for rank in range(size):
            p = Process(target=init_processes, args=(rank, size, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

上面这个脚本生成两个将分别设置分布式环境的进程, 初始化进程组 (``dist.init_process_group``),
最后执行给定的 ``run`` 方法.

让我们看一下 ``init_processes`` 函数. 它使每一个进程可以通过一个主节点互相协调,
使用相同的 IP 和端口. 注意, 我们使用了 TCP 后端, 但我们也能够使用
`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`__ 或者
`Gloo <http://github.com/facebookincubator/gloo>`__ 替代. (参考
`5.1 部分 <#communication-backends>`__) 在教程的最后, 我们将查看在 ``dist.init_process_group``
中的不可思议的事情, 但它本质上是允许进程通过共享位置来相互通信.

点对点通信
----------------------------

.. figure:: /_static/img/distributed/send_recv.png
   :width: 100%
   :align: center
   :alt: 发送和接受

   发送和接受

从一个进程传输数据到另一个进程称之为点对点通信. 这是通过 ``send`` 和 ``recv`` 函数,
或者他们与 ``send`` 和 ``recv`` 地位相当的 *immediate*, ``isend`` 和 ``irecv`` 来实现的.

.. code:: python

    """阻塞的点对点通信."""

    def run(rank, size):
        tensor = torch.zeros(1)
        if rank == 0:
            tensor += 1
            # 发送 tensor 给 process 1
            dist.send(tensor=tensor, dst=1)
        else:
            # 从 process 0 接收 tensor
            dist.recv(tensor=tensor, src=0)
        print('Rank ', rank, ' has data ', tensor[0])

上面的例子, 两个进程开始都有一个值为 0 的 tensor, 然后进程 0 增加 tensor 的值并且发送给进程 1 ,
因此, 两个进程的 tensor 最终都增加了 1.0. 注意, 进程 1 为了保存收到的数据需要分配内存.

另外需要注意, ``send``/``recv`` 是 **阻塞的**:两个程序都会阻塞直到通讯完成.
另一方面 immediates 是 **非租塞的**; 脚本继续执行, 方法最后返回一个 ``DistributedRequest`` 对象.
在这个对象上，我们可以选择 ``wait()``.

.. code:: python

    """非阻塞点对点通信."""

    def run(rank, size):
        tensor = torch.zeros(1)
        req = None
        if rank == 0:
            tensor += 1
            # 发送 tensor 给 process 1
            req = dist.isend(tensor=tensor, dst=1)
            print('Rank 0 started sending')
        else:
            # 从 process 0 接收 tensor
            req = dist.irecv(tensor=tensor, src=0)
            print('Rank 1 started receiving')
        req.wait()
        print('Rank ', rank, ' has data ', tensor[0])

当使用 immediates 时, 我们必须对发送或者接收的 tensor 小心使用.
因为我们不知道数据什么时候会被传达给其他进程, 我们不应该修改发送的 tensor, 也不应该在 ``req.wait()`` 结束之前访问收到的 tensor.
换言之,

-  在执行 ``dist.isend()`` 之后修改发送的 ``tensor`` 会出现未定义行为的结果.
-  在执行 ``dist.irecv()`` 之后读取接受的 ``tensor`` 会出现未定义行为的结果.

但是, 在 ``req.wait()`` 执行之后我们保证信息传递已经发生并且结束, 所以保存在 ``tensor[0]`` 的值是 1.0.

点对点通信, 在我们想要对我们进程间的通信有一个细粒度的控制时有用.
他们可以被用于实现花哨的算法, 例如有一个使用 `百度的
DeepSpeech <https://github.com/baidu-research/baidu-allreduce>`__ 或者
`Facebook 的 大规模实验 <https://research.fb.com/publications/imagenet1kin1h/>`__.
(参考 `4.1 章节 <#our-own-ring-allreduce>`__)

Collective 通信
------------------------

+----------------------------------------------------+-----------------------------------------------------+
| .. figure:: /_static/img/distributed/scatter.png   | .. figure:: /_static/img/distributed/gather.png     |
|   :alt: Scatter                                    |   :alt: Gather                                      |
|   :width: 100%                                     |   :width: 100%                                      |
|   :align: center                                   |   :align: center                                    |
|                                                    |                                                     |
|   Scatter                                          |   Gather                                            |
+----------------------------------------------------+-----------------------------------------------------+
| .. figure:: /_static/img/distributed/reduce.png    | .. figure:: /_static/img/distributed/all_reduce.png |
|   :alt: Reduce                                     |   :alt: All-Reduce                                  |
|   :width: 100%                                     |   :width: 100%                                      |
|   :align: center                                   |   :align: center                                    |
|                                                    |                                                     |
|   Reduce                                           |   All-Reduce                                        |
+----------------------------------------------------+-----------------------------------------------------+
| .. figure:: /_static/img/distributed/broadcast.png | .. figure:: /_static/img/distributed/all_gather.png |
|   :alt: Broadcast                                  |   :alt: All-Gather                                  |
|   :width: 100%                                     |   :width: 100%                                      |
|   :align: center                                   |   :align: center                                    |
|                                                    |                                                     |
|   Broadcast                                        |   All-Gather                                        |
+----------------------------------------------------+-----------------------------------------------------+



与点对点通信对比, collective 允许通信模式跨 **group** 内的所有进程. 一个组是我们所有进程的子集.
我们可以传递一个包含的队列的 list 给 ``dist.new_group(group)`` 来创建一个组.
默认情况, collectives 在所有进程间执行, 又被称为 **world**. 例如, 为了获得所有进程中的 tensor 的和,
我们可以使用 ``dist.all_reduce(tensor, op, group)`` collective.

.. code:: python

    """ All-Reduce 例子."""
    def run(rank, size):
        """ 简单的点对点通信. """
        group = dist.new_group([0, 1]) 
        tensor = torch.ones(1)
        dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
        print('Rank ', rank, ' has data ', tensor[0])

由于我们想得到组内所有 tensor 的和, 我们使用 ``dist.reduce_op.SUM`` 作为 reduce 的 operator.
一般来讲, 任何数学交换运算可以当做 operator. PyTorch 附带了 4 个这样的开箱即用的 operator,
他们都在元素级的工作:


-  ``dist.reduce_op.SUM``,
-  ``dist.reduce_op.PRODUCT``,
-  ``dist.reduce_op.MAX``,
-  ``dist.reduce_op.MIN``.

除 ``dist.all_reduce(tensor, op, group)`` 外, 这里一共有 6 个 collective 在当前的 PyTorch 版本.

-  ``dist.broadcast(tensor, src, group)``: 从 ``src`` 拷贝 ``tensor`` 到所有其他进程.
-  ``dist.reduce(tensor, dst, op, group)``: 对所有 ``tensor`` 执行 ``op`` 然后保存 reduce 结果到 ``dst``.
-  ``dist.all_reduce(tensor, op, group)``: 和 reduce 一样, 不同的是, reduce 结果保存在所有的进程中.
-  ``dist.scatter(tensor, src, scatter_list, group)``: 复制 :math:`i^{\text{th}}` tensor ``scatter_list[i]`` 到
   :math:`i^{\text{th}}` 进程.
-  ``dist.gather(tensor, dst, gather_list, group)``: 在 ``dst`` 中, 从所有进程拷贝 ``tensor``.
-  ``dist.all_gather(tensor_list, tensor, group)``: 在所用进程中，将 ``tensor`` 从所有进程复制到 ``tensor_list``.

分布式训练
--------------------

.. raw:: html

   <!--
   * Gloo Backend
   * Simple all_reduce on the gradients
   * Point to optimized DistributedDataParallel

   TODO: Custom ring-allreduce
   -->

**注意:** 你可以在 ` 这个 GitHub repository <https://github.com/seba-1511/dist_tuto.pth/>`__ 找到这个部分的示例脚本.

既然我们了解分布式模块如何工作, 让我们用它写一些有用的东西. 我们的目标是复制
`DistributedDataParallel <http://pytorch.org/docs/master/nn.html#torch.nn.parallel.DistributedDataParallel>`__
的功能. 当然, 这是个说教的例子, 现实情况你应当使用下面链出的, 官方的, 经过全面测试和优化的版本.

简单来说, 我们要实现一个分布式版本的随机梯度下降. 我们的脚本将让所有的进程计算他们的模型关于他们的批量数据的梯度,
然后计算他们的梯度的平均值. 为了确保在更改进程的数量时有类似的收敛结果, 我们必须首先对我们的数据集进行分区.
( 你也可以使用 `tnt.dataset.SplitDataset <https://github.com/pytorch/tnt/blob/master/torchnet/dataset/splitdataset.py#L4>`__
替换下面的代码片段. )

.. code:: python

    """ 数据集分区工具 """
    class Partition(object):

        def __init__(self, data, index):
            self.data = data
            self.index = index

        def __len__(self):
            return len(self.index)

        def __getitem__(self, index):
            data_idx = self.index[index]
            return self.data[data_idx]


    class DataPartitioner(object):

        def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
            self.data = data
            self.partitions = []
            rng = Random()
            rng.seed(seed)
            data_len = len(data)
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)

            for frac in sizes:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

        def use(self, partition):
            return Partition(self.data, self.partitions[partition])

配合上面的代码, 我们现在可以简单的使用下面的代码分割任何数据集:

.. code:: python

    """ 分割 MNIST """
    def partition_dataset():
        dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
        size = dist.get_world_size()
        bsz = 128 / float(size)
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(dataset, partition_sizes)
        partition = partition.use(dist.get_rank())
        train_set = torch.utils.data.DataLoader(partition,
                                             batch_size=bsz,
                                             shuffle=True)
        return train_set, bsz

假设我们有 2 个复制, 然后每个进程将有一个 60000 / 2 = 30000 样本的 ``train_set``.
为了保存整个批次大小是128, 我们同样用复制的数量除以批次大小 ( 128 ).

我们现在可以编写我们的通常的前向方向优化训练代码, 以及添加一个用于计算我们模型平均梯度的函数.
( 下面的代码主要受到官方的 `PyTorch MNIST 例子 <https://github.com/pytorch/examples/blob/master/mnist/main.py>`__ 的启发.)

.. code:: python

    """ 分布式的同步的随机梯度下降例子 """
    def run(rank, size):
            torch.manual_seed(1234)
            train_set, bsz = partition_dataset()
            model = Net()
            optimizer = optim.SGD(model.parameters(),
                                  lr=0.01, momentum=0.5)

            num_batches = ceil(len(train_set.dataset) / float(bsz)) 
            for epoch in range(10):
                epoch_loss = 0.0
                for data, target in train_set:
                    data, target = Variable(data), Variable(target)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    epoch_loss += loss.data[0]
                    loss.backward()
                    average_gradients(model)
                    optimizer.step()
                print('Rank ', dist.get_rank(), ', epoch ',
                      epoch, ': ', epoch_loss / num_batches) 

这遗留了一个 ``average_gradients(model)`` 函数需要实现, 只需要传入模型然后跨整个 world 计算他的平均梯度.

.. code:: python

    """ 计算平均梯度. """
    def average_gradients(model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size 

*Et voilà*! 我们成功的实现了分布式的异步随机梯度下降, 并且可以在大型计算机集群上训练任何模型.

**注意:** 虽然最后一句在 *技术上* 是正确的, 在实现一个产品级别的异步随机梯度算法时，
还需要 `许多技巧 <http://seba-1511.github.io/dist_blog>`__.
再次声明, 使用 `经过测试和优化的实现 <http://pytorch.org/docs/master/nn.html#torch.nn.parallel.DistributedDataParallel>`__.

我们自己的 Ring-Allreduce
~~~~~~~~~~~~~~~~~~~~~~

作为一个额外的挑战, 想象一下, 我们想实现 DeepSpeech 的高效率 ring allreduce, 
这很容易使用点对点的 collective 来实现.

.. code:: python

    """ 实现一个带有加法操作的 ring-reduce. """
    def allreduce(send, recv):
        rank = dist.get_rank()
        size = dist.get_world_size()
        send_buff = th.zeros(send.size())
        recv_buff = th.zeros(send.size())
        accum = th.zeros(send.size())
        accum[:] = send[:]

        left = ((rank - 1) + size) % size
        right = (rank + 1) % size

        for i in range(size - 1):
            if i % 2 == 0:
                # Send send_buff
                send_req = dist.isend(send_buff, right)
                dist.recv(recv_buff, left)
                accum[:] += recv[:]
            else:
                # Send recv_buff
                send_req = dist.isend(recv_buff, right)
                dist.recv(send_buff, left)
                accum[:] += send[:]
            send_req.wait()
        recv[:] = accum[:]

在上面的脚本中, ``allreduce(send, recv)`` 函数和 PyTorch 中提供的 ``allreduce`` 有一些细微的差别.
这个函数需要一个名为 ``recv`` 的 tensor 参数, 这个参数将保存所有 ``send`` tensor 的和.
作为一个留给读者的联系, 在我们的版本和 DeepSpeech 中的版本之间, 还有一个区别: 他们的实现将梯度张亮分割成块 (*chunks*),
以最优化利用带宽. (提示: `toch.chunk <http://pytorch.org/docs/master/torch.html#torch.chunk>`__)


高级话题
---------------

我们现在准备发现一些 ``torch.distributed`` 的更高级的功能. 由于有很多东西要覆盖, 这章节分成两个子章节:

1. 通讯后端: 在这里, 我们学习如何使用 MPI 和 Gloo 进行 GPU-GPU 通信。
2. 初始化方法: 在这里, 我们了解如何在 ``dist.init_process_group()`` 中最佳的设置初始协调阶段.

通讯后端
~~~~~~~~~~~~~~~~~~~~~~

``torch.distributed`` 其中一个比较优雅的方面是, 他能够在不同的后端上抽象和构建.
如前所述, 目前有三个后端在 PyTorch 中实现: TCP, MPI 和 GLoo。
它们每个都有不同的规格和折衷，取决于所需的使用情况。
可以在 `这里 <http://pytorch.org/docs/master/distributed.html#module-torch.distributed>`__ 找到一个支持函数的比较表.

**TCP 后端**

到目前为止, 我们已经广泛使用 TCP 后端. 他最为一个开发平台非常方便, 以为他保证在大多数的机器和操作系统上工作.
它还支持 CPU 上的所有点对点和 collective 功能. 然而，没有支持 GPU, 它的通信例程不如 MPI 优化。

**Gloo 后端**

`Gloo 后端 <https://github.com/facebookincubator/gloo>`__ 为 CPU 和 GPU 提供 *collective* 通信过程的优化实现.
它特别适合 GPU, 因为它可以执行通信, 而无需使用 `GPUDirect <https://developer.nvidia.com/gpudirect>`__ 将数据传输到 CPU 内存.
它还能够使用 `NCCL <https://github.com/NVIDIA/nccl>`__ 执行快速的节点内通信, 
并实现用于节点间例程的 `自己的算法 <https://github.com/facebookincubator/gloo/blob/master/docs/algorithms.md>`__.

从 0.2.0 版本开始, Gloo 后端自动包含在 PyTorch 的预编译二进制文件中。
正如您已经注意到的那样, 如果您将 ``model`` 放在 GPU 上, 我们的分布式 SGD 示例不起作用.
让我们来解决它, 首先在 ``init_processes（rank，size，fn，backend ='tcp')`` 中替换 ``backend ='gloo``.
此时, 该脚本仍将在CPU上运行, 但在幕后使用Gloo后端.
为了使用多个GPU, 我们还要做以下修改:

0. ``init_processes(rank, size, fn, backend='tcp')`` :math:`\rightarrow`
   ``init_processes(rank, size, fn, backend='gloo')``
1. ``model = Net()`` :math:`\rightarrow` ``model = Net().cuda(rank)``
2. ``data, target = Variable(data), Variable(target)``
   :math:`\rightarrow`
   ``data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))``

通过上面的修改, 我们的模型现在在两个 GPU 上训练, 你可以使用 ``watch nvidia-smi`` 来监视它们的使用.

**MPI 后端**

消息传递接口 (MPI) 是高性能计算领域的标准化工具.
它允许进行点对点和集体通信, 并且是 ``torch.distributed`` 的 API 的主要灵感来源.
存在几种 MPI 的实现 (例如，`Open-MPI <https://www.open-mpi.org/>`__, `MVAPICH2 <http://mvapich.cse.ohio-state.edu/>`__, 
`Intel MPI <https://software.intel.com/en-us/intel-mpi-library>`__) 每个都针对不同的目的而优化.
使用 MPI 后端的优点在于 MPI 在大型计算机集群上的广泛可用性和高层次的优化.
`一些 <https://developer.nvidia.com/mvapich>`__ `最近的  <https://developer.nvidia.com/ibm-spectrum-mpi>`__
`实现 <http://www.open-mpi.org/>`__ 还能够利用 CUDA IPC 的优势和 GPU Direct 技术来避免通过 CPU 内存拷贝.

不幸的是, PyTorch 的二进制文件不能包含 MPI 实现, 我们必须手动编译它.
幸运的是, 这个过程非常简单, 因为在编译时, PyTorch 会 *自行* 寻找可用的MPI实现.
以下步骤通过从`源代码 <https://github.com/pytorch/pytorch#from-source>`__ 安装 PyTorch 来安装 MPI 后端.

1. 创建和激活你的 Anaconda 环境, 跟着 `向导 <https://github.com/pytorch/pytorch#from-source>`__ 安装所有的必要选项. 但 **不要** 运行 ``python setup.py install``.
2. 选择并安装你喜欢的 MPI 实现. 注意启用  CUDA-aware MPI 可能需要一些额外的步骤. 在我们的例子中, 我们将继续使用 *没有* GPU 支持的 Open-MPI: ``conda install -c conda-forge openmpi``
3. 现在, 去你克隆的 PyTorch 库, 并执行 ``python setup.py install``.

为了测试我们新安装的后端, 需要进行一些修改.

1. 用 ``init_processes(0, 0, run, backend='mpi')`` 替换 ``if __name__ == '__main__':`` 下面的内容.
2. 运行 ``mpirun -n 4 python myscript.py``.

做这些修改的原因是 MPI 需要在生成过程之前创建自己的环境. MPI 也会生成自己的进程, 并执行在 `Initialization
Methods <#initialization-methods>`__ 中描述的握手, 使 ``init_process_group`` 的 ``rank``\ 和 ``size`` 参数成为多余.
这实际上非常强大, 因为你可以传递额外的参数给 ``mpirun``, 以便为每个进程调整计算资源. 
(比如每个进程的内核数量, 手动分配机器到特定的序列, 和 `一些其他的 <https://www.open-mpi.org/faq/?category=running#mpirun-hostfile>`__)
这样做, 你应该获得与其他通信后端相同的熟悉的输出。

初始化方法
~~~~~~~~~~~~~~~~~~~~~~

为了完成这个教程, 让我们讨论下我们最先调用的函数: ``dist.init_process_group(backend, init_method)``.
具体来说, 我们将介绍负责每个进程之间初始协调步骤的不同初始化方法.
这些方法可以让你定义这种协调是如何完成的.
根据您的硬件设置, 这些方法中的某个应该比其他方法更合适.
除以下章节外, 你也应该看下 `官方文档 <http://pytorch.org/docs/master/distributed.html#initialization>`__.


在深入研究初始化方法之前, 我们先快速看一下从 C/C++ 的角度来看 ``init_process_group`` 的背后发生了什么.

1. 首先, 参数被解析和验证.
2. 后端通过 ``name2channel.at()`` 函数解析. 返回一个 ``Channel`` 对象, 将被用于执行数据传输.
3. GIL 被抛弃了, 然后调用 ``THDProcessGroupInit()``. 这将实例化信道并添加主节点的地址.
4. 排序为 0 的进程将执行 ``master`` 程序, 其他的排序的进程作为 ``workers``.
5. master

   a. 为所有 worker 创建 socket.
   b. 等待所有 worker 来连接.
   c. 向他们发送有关其他进程位置的信息.

6. 每个 worker

   a. 向 master 创建一个套接字.
   b. 发送他们自己的位置信息.
   c. 接受其他 worker 的信息.
   d. 打开一个 socket 并与其他所有 worker 握手.

7. 初始化完成后, 每个人都连接到每个人.

**环境变量**

在本教程中, 我们已经使用环境变量初始化方法.
通过在所有机器上设置以下四个环境变量, 所有进程将能够正确连接到主机, 获取有关其他进程的信息, 并最终与它们握手.

-  ``MASTER_PORT``: A free port on the machine that will host the
   process with rank 0.
-  ``MASTER_ADDR``: IP address of the machine that will host the process
   with rank 0.
-  ``WORLD_SIZE``: The total number of processes, so that the master
   knows how many workers to wait for.
-  ``RANK``: Rank of each process, so they will know whether it is the
   master of a worker.

**共享文件系统**

共享文件系统要求所有进程可以访问共享文件系统, 并通过一个共享的文件进行协调.
这意味着所有进程将打开这个文件, 写入自己的信息, 并且等待所有进程写完.
在所有要求之后, 信息对所有进程都是容易获得的. 为了避免竟态条件, 
文件系统必须支持通过 `fcntl <http://man7.org/linux/man-pages/man2/fcntl.2.html>`__ 进行锁定.
注意, 你可以自己手动指定序列, 或者让进程自己觉得序列.
为每个作业定义一个唯一的 ``groupname``, 你可以为多个作业使用同一个文件路径, 而且安全的避免冲突。

.. code:: python

    dist.init_process_group(init_method='file:///mnt/nfs/sharedfile', world_size=4,
                            group_name='mygroup')

**TCP Init & 组播**

可以使用两种不同的方法通过 TCP 初始化:

1. 通过提供 rank 0 (排序为0)进程的 IP 地址和 world 的大小.
2. 通过提供 *任意* 有效 IP `组播地址 <https://en.wikipedia.org/wiki/Multicast_address>`__ 和 world 的大小.

在第一种情况下, 所有进程都将能够连接到序号为0的进程, 然后按照上面程序描述的.

.. code:: python

    dist.init_process_group(init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)

第二种情况, 组播地址指定这个组的潜在活动的节点, 协调可以通过允许每个进程在遵循上述过程之前进行初始握手来处理.
此外, TCP 组播初始化还支持  ``group_name``  参数(与共享文件方法一样), 允许在同一个集群上调度多个作业.

.. code:: python

    dist.init_process_group(init_method='tcp://[ff15:1e18:5d4c:4cf0:d02d:b659:53ba:b0a7]:23456',
                            world_size=4)

.. raw:: html

   <!--
   ## Internals
   * The magic behind init_process_group:

   1. validate and parse the arguments
   2. resolve the backend: name2channel.at()
   3. Drop GIL & THDProcessGroupInit: instantiate the channel and add address of master from config
   4. rank 0 inits master, others workers
   5. master: create sockets for all workers -> wait for all workers to connect -> send them each the info about location of other processes
   6. worker: create socket to master, send own info, receive info about each worker, and then handshake with each of them
   7. By this time everyone has handshake with everyone.
   -->

.. raw:: html

   <center>

**感谢**

.. raw:: html

   </center>

我要感谢 PyTorch 开发人员在实现、文档和测试方面做得很好.
当代码不清楚时，我总是可以依靠 `文档 <http://pytorch.org/docs/master/distributed.html>`__ 
或 `测试 <https://github.com/pytorch/pytorch/blob/master/test/test_distributed.py>`__ 来找到答案.
特别是, 我要感谢 Soumith Chintala, Adam Paszke 和 Natalia Gimelshein 在早期草稿中提供深刻见解并回答问题.
