# 使用分布式 RPC 框架实现参数服务器

> 原文：<https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html>

**作者**： [Rohan Varma](https://github.com/rohan-varma)

先决条件：

*   [PyTorch 分布式概述](../beginner/dist_overview.html)
*   [RPC API 文档](https://pytorch.org/docs/master/rpc.html)

本教程介绍了一个简单的示例，该示例使用 PyTorch 的[分布式 RPC 框架](https://pytorch.org/docs/stable/rpc.html)实现参数服务器。 参数服务器框架是一种范例，其中一组服务器存储参数（例如大型嵌入表），并且多个训练人员查询参数服务器以检索最新参数。 这些训练器可以在本地运行训练循环，并偶尔与参数服务器同步以获得最新参数。 有关参数服务器方法的更多信息，请查阅[本文](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)。

使用分布式 RPC 框架，我们将构建一个示例，其中多个训练器使用 RPC 与同一个参数服务器进行通信，并使用 [RRef](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef) 访问远程参数服务器实例上的状态。 每位训练器将通过使用分布式 Autograd 跨多个节点拼接 Autograd 图，以分布式方式启动其专用的反向传递。

**注意**：本教程介绍了分布式 RPC 框架的用法，该方法可用于将模型拆分到多台计算机上，或用于实现参数服务器训练策略，在该策略中，网络训练器可以获取托管在另一台计算机上的参数。 相反，如果您要跨多个 GPU 复制模型，请参阅[分布式数据并行教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)。 还有另一个 [RPC 教程](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)，涵盖了强化学习和 RNN 用例。

让我们从熟悉的地方开始：导入我们所需的模块并定义一个简单的 ConvNet，它将在 MNIST 数据集上进行训练。 以下网络是从[`pytorch/examples`仓库](https://github.com/pytorch/examples/tree/master/mnist)中定义的网络中广泛采用的。

```py
import argparse
import os
import time
from threading import Lock

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torchvision import datasets, transforms

# --------- MNIST Network to train, from pytorch/examples -----

class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        print(f"Using {num_gpus} GPUs to train")
        self.num_gpus = num_gpus
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
        print(f"Putting first 2 convs on {str(device)}")
        # Put conv layers on the first cuda device, or CPU if no cuda device
        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
        # Put rest of the network on the 2nd cuda device, if there is one
        if "cuda" in str(device) and num_gpus > 1:
            device = torch.device("cuda:1")

        print(f"Putting rest of layers on {str(device)}")
        self.dropout1 = nn.Dropout2d(0.25).to(device)
        self.dropout2 = nn.Dropout2d(0.5).to(device)
        self.fc1 = nn.Linear(9216, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # Move tensor to next device if necessary
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

```

接下来，让我们定义一些辅助函数，这些函数将对其余脚本有用。 下面使用[`rpc_sync`](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.rpc_sync)和 [RRef](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef) 来定义一个函数，该函数在远程节点上的对象上调用给定方法。 下面，通过`rref`参数指定了对远程对象的句柄，并在其拥有的节点`rref.owner()`上运行它。 在调用者节点上，我们通过使用`rpc_sync`同步运行此命令，这意味着我们将阻塞直到收到响应。

```py
# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods. method could be any matching function, including
# class methods.
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef and passes along the given argument.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

```

现在，我们准备定义参数服务器。 我们将子类化`nn.Module`，并将句柄保存到上面定义的网络中。 我们还将保存一个输入设备，该输入设备将是在调用模型之前将输入传输到的设备。

```py
# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        model = Net(num_gpus=num_gpus)
        self.model = model
        self.input_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

```

接下来，我们将定义前进通道。 请注意，无论模型输出的设备如何，我们都会将输出移至 CPU，因为分布式 RPC 框架当前仅支持通过 RPC 发送 CPU 张量。 由于有可能在调用者/被调用者上使用不同的设备（CPU/GPU），因此我们有意禁用通过 RPC 发送 CUDA 张量，但在将来的版本中可能会支持此功能。

```py
class ParameterServer(nn.Module):
...
    def forward(self, inp):
        inp = inp.to(self.input_device)
        out = self.model(inp)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        out = out.to("cpu")
        return out

```

接下来，我们将定义一些其他函数，可用于训练和验证。 第一个`get_dist_gradients`将采用分布式 Autograd 上下文 ID，并调用`dist_autograd.get_gradients` API，以检索由分布式 Autograd 计算的梯度。 可以在[分布式 Autograd 文档](https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework)中找到更多信息。 请注意，由于该框架当前仅支持通过 RPC 发送张量，因此我们还会迭代生成的字典并将每个张量转换为 CPU 张量。 接下来，`get_param_rrefs`将迭代我们的模型参数，并将它们包装为（本地）[RRef](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef)。 训练者节点将通过 RPC 调用此方法，并将返回要优化的参数列表。 这是[分布式优化器](https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim)的输入，它需要所有必须优化的参数作为`RRef`的列表。

```py
# Use dist autograd to retrieve gradients accumulated for this model.
# Primarily used for verification.
def get_dist_gradients(self, cid):
    grads = dist_autograd.get_gradients(cid)
    # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
    # Tensors must be moved in and out of GPU memory due to this.
    cpu_grads = {}
    for k, v in grads.items():
        k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
        cpu_grads[k_cpu] = v_cpu
    return cpu_grads

# Wrap local parameters in a RRef. Needed for building the
# DistributedOptimizer which optimizes paramters remotely.
def get_param_rrefs(self):
    param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
    return param_rrefs

```

最后，我们将创建用于初始化参数服务器的方法。 请注意，所有过程中只有一个参数服务器实例，并且所有训练器都将与同一参数服务器对话并更新相同的存储模型。 如`run_parameter_server`所示，服务器本身不采取任何独立的操作； 它等待来自训练者的请求（尚未定义），并通过运行所请求的函数对其作出响应。

```py
# The global parameter server instance.
param_server = None
# A lock to ensure we only have one parameter server.
global_lock = Lock()

def get_parameter_server(num_gpus=0):
    """
    Returns a singleton parameter server to all trainer processes
    """
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server

def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")

```

请注意，以上`rpc.shutdown()`不会立即关闭参数服务器。 相反，它将等待所有工作器（在这种情况下为训练人员）也呼唤`rpc.shutdown()`。 这样可以保证参数服务器在所有训练人员（尚未定义）完成训练过程之前不会脱机。

接下来，我们将定义`TrainerNet`类。 这也将是`nn.Module`的子类，并且我们的`__init__`方法将使用`rpc.remote` API 获取对我们的参数服务器的 RRef 或远程引用。 请注意，此处我们没有将参数服务器复制到本地进程，而是可以将`self.param_server_rref`视为指向驻留在单独进程中的参数服务器的分布式共享指针。

```py
# --------- Trainers --------------------

# nn.Module corresponding to the network trained by this trainer. The
# forward() method simply invokes the network on the given parameter
# server.
class TrainerNet(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(num_gpus,))

```

接下来，我们将定义一个名为`get_global_param_rrefs`的方法。 为了激发对这种方法的需求，值得阅读[`DistributedOptimizer`](https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim)上的文档，尤其是 API 签名。 必须向优化器传递与要优化的远程参数相对应的`RRef`列表，因此在这里我们获得了必要的`RRef`。 由于给定`TrainerNet`与之交互的唯一远程工作器是`ParameterServer`，因此我们只需在`ParameterServer`上调用`remote_method`。 我们使用在`ParameterServer`类中定义的`get_param_rrefs`方法。 此方法将`RRef`的列表返回到需要优化的参数。 请注意，在这种情况下，我们的`TrainerNet`没有定义自己的参数； 如果确实如此，我们还需要将每个参数都包装在`RRef`中，并将其包含在`DistributedOptimizer`的输入中。

```py
class TrainerNet(nn.Module):
...
    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params

```

现在，我们准备定义`forward`方法，该方法将调用（同步）RPC 以运行`ParameterServer`上定义的网络的正向传播。 请注意，我们将`self.param_server_rref`（它是`ParameterServer`的远程句柄）传递给 RPC 调用。 该调用将向运行`ParameterServer`的节点发送 RPC，调用`forward`传递，然后返回与模型输出相对应的`Tensor`。

```py
class TrainerNet(nn.Module):
...
    def forward(self, x):
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, x)
        return model_output

```

完全定义好训练器之后，现在该编写我们的神经网络训练循环，该循环将创建我们的网络和优化器，通过网络运行一些输入并计算损失。 训练循环看起来很像本地训练计划，但由于我们的网络在机器之间分布，因此进行了一些修改。

下面，我们初始化`TrainerNet`并构建一个`DistributedOptimizer`。 请注意，如上所述，我们必须传入要优化的所有全局参数（跨参与分布式训练的所有节点）。 另外，我们传入要使用的本地优化器，在这种情况下为 SGD。 请注意，我们可以像创建本地优化器一样配置基础优化器算法-`optimizer.SGD`的所有参数都将正确转发。 例如，我们传入一个自定义学习率，它将用作所有本地优化器的学习率。

```py
def run_training_loop(rank, num_gpus, train_loader, test_loader):
    # Runs the typical nueral network forward + backward + optimizer step, but
    # in a distributed fashion.
    net = TrainerNet(num_gpus=num_gpus)
    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)

```

接下来，我们定义我们的主要训练循环。 我们遍历了 PyTorch 的[`DataLoader`](https://pytorch.org/docs/stable/data.html)提供的可迭代项。 在编写典型的前向/后向/优化器循环之前，我们首先将逻辑包装在[分布式 Autograd 上下文](https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.context)中。 请注意，这需要记录在模型的正向传播中调用的 RPC，以便可以构造一个适当的图，其中包括在后向传递中所有参与的分布式工作器。 分布式 Autograd 上下文返回`context_id`，它用作用于累积和优化与特定迭代对应的梯度的标识符。

与调用典型的`loss.backward()`会启动此本地工作程序的反向传播相反，我们调用`dist_autograd.backward()`并传递我们的`context_id`和`loss`，这是我们希望反向传播从它开始的根。 另外，我们将此`context_id`传递到优化程序调用中，该调用程序必须能够在所有节点上查找由该特定反向传播计算出的相应梯度。

```py
def run_training_loop(rank, num_gpus, train_loader, test_loader):
...
    for i, (data, target) in enumerate(train_loader):
        with dist_autograd.context() as cid:
            model_output = net(data)
            target = target.to(model_output.device)
            loss = F.nll_loss(model_output, target)
            if i % 5 == 0:
                print(f"Rank {rank} training batch {i} loss {loss.item()}")
            dist_autograd.backward(cid, [loss])
            # Ensure that dist autograd ran successfully and gradients were
            # returned.
            assert remote_method(
                ParameterServer.get_dist_gradients,
                net.param_server_rref,
                cid) != {}
            opt.step(cid)

     print("Training complete!")
     print("Getting accuracy....")
     get_accuracy(test_loader, net)

```

与传统的本地模型非常相似，下面的内容只是简单地计算了我们训练后模型的准确率。 但是，请注意，我们在上面传递给此函数的`net`是`TrainerNet`的实例，因此，正向传播以透明方式调用 RPC。

```py
def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible
    device = torch.device("cuda:0" if model.num_gpus > 0
        and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data, -1)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")

```

接下来，类似于我们将`run_parameter_server`定义为负责初始化 RPC 的`ParameterServer`的主循环的方式，让我们为训练者定义一个类似的循环。 所不同的是，我们的训练器必须执行上面定义的训练循环：

```py
# Main loop for trainers.
def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)

    print(f"Worker {rank} done initializing RPC")

    run_training_loop(rank, num_gpus, train_loader, test_loader)
    rpc.shutdown()

```

请注意，类似于`run_parameter_server`，`rpc.shutdown()`默认情况下将等待该节点退出之前，所有训练器和`ParameterServer`的所有工作器都调用`rpc.shutdown()`。 这样可确保节点正常终止，并且没有一个节点脱机，而另一个节点则期望其联机。

现在，我们已经完成了特定于训练器和参数服务器的代码，剩下的就是添加代码以启动训练器和参数服务器。 首先，我们必须接受适用于我们的参数服务器和训练器的各种参数。 `world_size`对应于将参加训练的节点总数，并且是所有训练器和参数服务器的总和。 我们还必须为每个单独的进程传递唯一的`rank`，从 0（将在其中运行单个参数服务器的地方）到`world_size - 1`。 `master_addr`和`master_port`是可用于标识等级 0 进程在何处运行的参数，并且各个节点将使用它们来相互发现。 要在本地测试此示例，只需将`localhost`和相同的`master_port`传递给所有产生的实例。 请注意，出于演示目的，此示例仅支持 0-2 个 GPU，尽管可以扩展该模式以使用其他 GPU。

```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument(
        "rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "num_gpus",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, Currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

```

现在，我们将根据命令行参数创建一个与参数服务器或训练器相对应的过程。 如果传入的等级为 0，我们将创建一个`ParameterServer`，否则，将创建一个`TrainerNet`。 请注意，我们正在使用`torch.multiprocessing`启动与我们要执行的函数相对应的子进程，并使用`p.join()`从主线程等待该进程完成。 在初始化训练器的情况下，我们还使用 PyTorch 的[数据加载器](https://pytorch.org/docs/stable/data.html)来指定 MNIST 数据集上的训练和测试数据加载器。

```py
processes = []
world_size = args.world_size
if args.rank == 0:
    p = mp.Process(target=run_parameter_server, args=(0, world_size))
    p.start()
    processes.append(p)
else:
    # Get data to train on
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True,)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=32,
        shuffle=True,
    )
    # start training worker on this node
    p = mp.Process(
        target=run_worker,
        args=(
            args.rank,
            world_size, args.num_gpus,
            train_loader,
            test_loader))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

```

要在本地运行示例，请在单独的终端窗口中为服务器和要生成的每个工作程序运行以下命令工作程序：`python rpc_parameter_server.py --world_size=WORLD_SIZE --rank=RANK`。 例如，对于世界大小为 2 的主节点，命令为`python rpc_parameter_server.py --world_size=2 --rank=0`。 然后可以在单独的窗口中使用命令`python rpc_parameter_server.py --world_size=2 --rank=1`启动训练器，这将开始使用一台服务器和一台训练器进行训练。 请注意，本教程假定使用 0 到 2 个 GPU 进行训练，并且可以通过将`--num_gpus=N`传递到训练脚本中来配置此参数。

您可以传入命令行参数`--master_addr=ADDRESS`和`--master_port=PORT`来指示主工作器正在监听的地址和端口，例如，以测试在其他机器上运行训练者和主节点的功能。