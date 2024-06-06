使用 PyTorch 编写分布式应用程序
作者： Séb Arnold
先决条件：PyTorch 分布式概述
在这个简短的教程中，我们将介绍 PyTorch 的分布式包。我们将了解如何设置分布式设置，使用不同的通信策略，并介绍包的一些内部结构。
设置
PyTorch 中包含的分布式包（即 torch.distributed ）使研究人员和从业者能够轻松地跨进程和机器集群并行化他们的计算。为此，它利用消息传递语义，允许每个进程将数据传达给任何其他进程。与多处理 （ torch.multiprocessing ） 包相反，进程可以使用不同的通信后端，并且不限于在同一台机器上执行。
为了开始，我们需要能够同时运行多个进程。如果您有权访问计算群集，则应咨询本地系统管理员或使用您喜欢的协调工具（例如 pdsh、clustershell 或其他工具）。在本教程中，我们将使用一台机器，并使用以下模板生成多个进程。
"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
上面的脚本生成了两个进程，每个进程将设置分布式环境，初始化进程组 （ dist.init_process_group ），最后执行给定 run 的函数。
让我们看一下 init_process 功能。它确保每个进程都能够使用相同的 IP 地址和端口通过主节点进行协调。请注意，我们使用了后端， gloo 但其他后端可用。（c.f. 第 5.1 节）在本教程的最后，我们将回顾其中 dist.init_process_group 发生的魔术，但它本质上允许进程通过共享其位置来相互通信。
点对点通信
![Alt text](send_recv.png)
发送和接收
从一个进程到另一个进程的数据传输称为点对点通信。这些都是通过 send 和 recv 函数或其直接对应物实现的， isend 并且 irecv .
"""Blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
在上面的例子中，两个进程都以零张量开始，然后进程 0 递增张量并将其发送到进程 1，因此它们都以 1.0 结束。请注意，进程 1 需要分配内存以存储它将接收的数据。
另请注意 send / recv 正在阻塞：两个进程都会停止，直到通信完成。另一方面，即时是非阻塞的;脚本继续执行，方法返回一个 Work 对象，我们可以选择 wait() 。
"""Non-blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
在使用即时张量时，我们必须小心如何使用发送和接收张量。由于我们不知道数据何时会被传达给另一个进程，因此在 req.wait() 完成之前，我们不应该修改发送的张量，也不应该访问接收到的张量。换言之，
写入 tensor after dist.isend() 将导致未定义的行为。
从 tensor 之后 dist.irecv() 读取将导致未定义的行为。
但是，在执行之后 req.wait() ，我们可以保证通信已经发生，并且存储在 1.0 中的 tensor[0] 值。
当我们想要对流程的通信进行更精细的控制时，点对点通信非常有用。它们可以用来实现花哨的算法，例如百度的DeepSpeech或Facebook的大规模实验中使用的算法。（c.f. 第 4.1 节）
集体沟通
![Alt text](scatter.png)
![Alt text](scatter-1.png)
![Alt text](scatter-2.png)
![Alt text](scatter-3.png)
![Alt text](broadcast.png)
![Alt text](broadcast-1.png)
与点对点通信相反，集合体允许跨组中所有流程的通信模式。组是我们所有流程的一个子集。要创建一个组，我们可以将排名列表传递给 dist.new_group(group) 。默认情况下，集合体在所有进程（也称为世界）上执行。例如，为了获得所有进程上所有张量的总和，我们可以使用 dist.all_reduce(tensor, op, group) 集合。
""" All-Reduce example."""
def run(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
由于我们想要组中所有张量的总和，因此我们使用 dist.ReduceOp.SUM reduce 运算符。一般来说，任何交换数学运算都可以用作算子。开箱即用，PyTorch 带有 4 个这样的运算符，它们都在元素级别工作：
dist.ReduceOp.SUM,
dist.ReduceOp.PRODUCT,
dist.ReduceOp.MAX,
dist.ReduceOp.MIN.
除此之外 dist.all_reduce(tensor, op, group) ，目前在 PyTorch 中总共实现了 6 个集合体。
dist.broadcast(tensor, src, group) ：从 src 所有其他进程复制 tensor 。
dist.reduce(tensor, dst, op, group) ：应用于 op ever tensor ，并将结果存储在 dst 中。
dist.all_reduce(tensor, op, group) ：与 reduce 相同，但结果存储在所有进程中。
dist.scatter(tensor, scatter_list, src, group) ：将 ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
dist.gather(tensor, gather_list, dst, group) ：从 中 dst 的所有进程复制 tensor 。
dist.all_gather(tensor_list, tensor, group) ：在所有进程上将所有进程复制 tensor 到 tensor_list 。
dist.barrier(group) ：阻止组中的所有进程，直到每个进程都进入此函数。
分布式训练
注意：您可以在此 GitHub 存储库中找到此部分的示例脚本。
现在我们了解了分布式模块的工作原理，让我们用它写一些有用的东西。我们的目标是复制 DistributedDataParallel 的功能。当然，这将是一个教学示例，在现实世界中，您应该使用上面链接的官方、经过充分测试和优化的版本。
很简单，我们想要实现随机梯度下降的分布式版本。我们的脚本将允许所有进程在其批数据上计算其模型的梯度，然后对梯度进行平均。为了确保在更改进程数量时获得相似的收敛结果，我们首先必须对数据集进行分区。（您也可以使用 tnt.dataset.SplitDataset，而不是下面的代码片段。
""" Dataset partitioning helper """
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
        rng = Random()  # from random import Random
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
通过上面的代码片段，我们现在可以使用以下几行简单地对任何数据集进行分区：
""" Partitioning MNIST """
def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz
假设我们有 2 个副本，那么每个进程将具有 train_set 60000 / 2 = 30000 个样本。我们还将批大小除以副本数，以保持总批大小为 128。
现在，我们可以编写通常的前向-后向优化训练代码，并添加函数调用来平均模型的梯度。（以下内容主要受官方 PyTorch MNIST 示例的启发。
""" Distributed Synchronous SGD Example """
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
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)
它仍然需要实现该 average_gradients(model) 函数，该函数只是接受一个模型并平均其在整个世界中的梯度。
""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
瞧！我们成功地实现了分布式同步 SGD，并且可以在大型计算机集群上训练任何模型。
注意：虽然最后一句话在技术上是正确的，但要实现同步 SGD 的生产级实施，还需要更多技巧。同样，使用经过测试和优化的内容。
我们自己的环形-Allreduce
另一个挑战是，假设我们想要实现 DeepSpeech 的高效环 allreduce。使用点对点集合体可以相当容易地实现这一点。
""" Implementation of a ring-reduce with addition. """
def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv_buff[:]
       else:
           # Send recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:]
在上面的脚本中，该 allreduce(send, recv) 函数的签名与 PyTorch 中的签名略有不同。它需要一个 recv 张量，并将所有 send 张量的总和存储在其中。作为留给读者的练习，我们的版本与 DeepSpeech 中的版本之间仍然存在一个区别：它们的实现将梯度张量划分为块，以便最佳地利用通信带宽。（提示：torch.chunk）
高级主题
我们现在已准备好发现 的一些 torch.distributed 更高级的功能。由于要涵盖的内容很多，本节分为两个小节：
通信后端：我们在这里学习如何使用 MPI 和 Gloo 进行 GPU-GPU 通信。
初始化方法：我们了解如何在 中 dist.init_process_group() 最好地设置初始协调阶段。
通信后端
最 torch.distributed 优雅的方面之一是它能够抽象和构建在不同的后端之上。如前所述，目前在 PyTorch 中实现了三个后端：Gloo、NCCL 和 MPI。它们各自具有不同的规格和权衡，具体取决于所需的用例。可以在此处找到支持的功能的比较表。
后端
消息传递接口（MPI）是高性能计算领域的标准化工具。它允许进行点对点和集体通信，并且是 API 的主要 torch.distributed 灵感来源。MPI 有多种实现（例如 Open-MPI、MVAPICH2、Intel MPI），每种实现都针对不同的目的进行了优化。使用 MPI 后端的优势在于 MPI 在大型计算机群集上的广泛可用性和高级别的优化。一些最近的实现还能够利用 CUDA IPC 和 GPU Direct 技术，以避免通过 CPU 进行内存复制。
不幸的是，PyTorch 的二进制文件不能包含 MPI 实现，我们必须手动重新编译它。幸运的是，这个过程相当简单，因为在编译时，PyTorch 会自行寻找可用的 MPI 实现。以下步骤通过从源安装 PyTorch 来安装 MPI 后端。
1、创建并激活 Anaconda 环境，按照指南安装所有必备组件，但尚未运行 python setup.py install 。
2、选择并安装您喜欢的 MPI 实现。请注意，启用 CUDA 感知 MPI 可能需要一些额外的步骤。在我们的例子中，我们将坚持使用不支持 GPU 的 Open-MPI： conda install -c conda-forge openmpi
3、现在，转到克隆的 PyTorch 存储库并执行 python setup.py install 。
为了测试我们新安装的后端，需要进行一些修改。
1、将 下 if __name__ == '__main__': 的内容替换为 init_process(0, 0, run, backend='mpi') 。
2、这些更改的原因是 MPI 需要在生成进程之前创建自己的环境。MPI 还将生成自己的进程并执行初始化方法中描述的握手，使 rank and size 参数 init_process_group 变得多余。这实际上非常强大，因为您可以传递 mpirun 额外的参数，以便为每个进程定制计算资源。（例如每个进程的内核数、手动将机器分配给特定等级等等）这样做，您应该获得与其他通信后端相同的熟悉输出。
后端
NCCL 后端提供了针对 CUDA 张量的集合运算的优化实现。如果您仅将 CUDA 张量用于集体操作，请考虑使用此后端以获得一流的性能。NCCL 后端包含在支持 CUDA 的预构建二进制文件中。
初始化方法
为了完成本教程，让我们来谈谈我们调用的第一个函数： dist.init_process_group(backend, init_method) .特别是，我们将介绍负责每个进程之间初始协调步骤的不同初始化方法。这些方法允许您定义如何完成此协调。根据您的硬件设置，其中一种方法自然应该比其他方法更合适。除了以下部分之外，您还应该查看官方文档。
 环境变量
在本教程中，我们一直在使用环境变量初始化方法。通过在所有机器上设置以下四个环境变量，所有进程都将能够正确连接到主进程，获取有关其他进程的信息，并最终与它们握手。
MASTER_PORT ：计算机上将托管排名为 0 的进程的自由端口。
MASTER_ADDR ：将托管排名为 0 的进程的计算机的 IP 地址。
WORLD_SIZE ：进程总数，以便主站知道要等待多少个工人。
RANK ：每个工序的等级，这样他们就会知道它是否是工人的主人。
共享文件系统
共享文件系统要求所有进程都有权访问共享文件系统，并将通过共享文件协调它们。这意味着每个进程都将打开文件，写入其信息，并等待每个人都这样做。之后，所有必需的信息都将随时可供所有流程使用。为了避免争用情况，文件系统必须支持通过 fcntl 进行锁定。
dist.init_process_group(
    init_method='file:///mnt/nfs/sharedfile',
    rank=args.rank,
    world_size=4)
TCP
通过TCP进行初始化可以通过提供进程的IP地址来实现，该IP地址的等级为0，并且具有可访问的端口号。在这里，所有工作人员将能够连接到等级为 0 的进程，并交换有关如何相互联系的信息。
dist.init_process_group(
    init_method='tcp://10.1.1.20:23456',
    rank=args.rank,
    world_size=4)
我要感谢 PyTorch 开发人员在他们的实现、文档和测试方面做得如此出色。当代码不清楚时，我总是可以依靠文档或测试来找到答案。我要特别感谢 Soumith Chintala、Adam Paszke 和 Natalia Gimelshein 对早期草稿的深刻评论和回答问题。