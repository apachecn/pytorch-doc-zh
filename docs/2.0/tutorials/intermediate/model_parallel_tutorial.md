
 单机模型并行最佳实践
 [¶](#single-machine-model-parallel-best-practices "固定链接到此标题")
=============================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/model_parallel_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html>




**作者** 
 :
 [沉力](https://mrshenli.github.io/)




 模型并行广泛应用于分布式训练技术。之前的文章已经解释了如何使用
 [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
 在多个 GPU 上训练神经网络；此功能将相同的模型复制到所有 GPU，其中每个 GPU 消耗输入数据的不同分区。尽管它可以显着加速训练过程，但对于模型太大而无法适应单个 GPU 的某些用例来说，它不起作用。这篇文章展示了如何通过使用
 **模型并行** 
 来解决该问题，
与
 `DataParallel`
 相比，它将单个模型分割到不同的 GPU 上，
而不是复制整个模型在每个 GPU 上（具体来说，假设模型
 `m`
 包含 10 层：当使用
 `DataParallel`
 时，每个 GPU 将拥有这 10 层中每一层的副本，而使用模型时在两个 GPU 上并行，
每个 GPU 可以托管 5 个层）。




 模型并行的高级思想是将模型的不同子网络
放置到不同的设备上，并相应地实现
 `forward`
 方法
以跨设备移动中间输出。由于模型只有一部分在任何单个设备上运行，因此一组设备可以共同为更大的模型提供服务。在这篇文章中，我们不会尝试构建巨大的模型并将其压缩到
有限数量的 GPU 中。相反，这篇文章的重点是展示模型并行的想法。读者可以将这些想法应用到实际
应用程序中。





 注意




 对于模型跨多个
服务器的分布式模型并行训练，请参阅
 [分布式 RPC 框架入门](rpc_tutorial.html)
 了解示例和详细信息。






 基本用法
 [¶](#basic-usage "此标题的永久链接")
-----------------------------------------------------------



 让我们从包含两个线性层的玩具模型开始。要在两个 GPU 上运行此模型，只需将每个线性层放在不同的 GPU 上，然后移动输入和中间输出以相应地匹配层设备。






```
import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))

```




 请注意，上面
 `ToyModel`
 看起来与在单个 GPU 上
实现它的方式非常相似，
 除了四个
 `to(device)`
 调用（用于放置线性层）
和适当设备上的张量。这是模型中唯一需要更改的地方。 
 `backward()`
 和
 `torch.optim`
 将自动处理梯度，就好像模型位于一个 GPU 上一样。您只需在
调用损失函数时
确保标签与输出位于同一设备上。






```
model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cuda:1')
loss_fn(outputs, labels).backward()
optimizer.step()

```






 将模型并行应用到现有模块
 [¶](#apply-model-parallel-to-existing-modules "永久链接到此标题")
-------------------------------------------------------------------------------------------------------------------------



 还可以在多个 GPU 上运行现有的单 GPU 模块
只需进行几行更改。下面的代码展示了如何将 `torchvision.models.resnet50()` 分解为两个 GPU。这个想法是继承现有的“ResNet”模块，并在构建过程中将层分割到两个 GPU。然后，重写
 `forward`
 方法，通过相应地移动中间输出来缝合两个
子网络。






```
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))

```




 上述实现解决了模型太大而无法适应单个 GPU 的情况。但是，您可能已经注意到，如果您的模型适合，
it 会比在单个 GPU 上运行慢。这是因为，在任何时间点，两个 GPU 中只有一个在工作，而另一个 GPU 则坐在那里什么都不做。由于需要将中间输出从“cuda:0”复制到“layer2”和“layer3”之间，因此性能进一步恶化。 




 让我们运行一个实验来更定量地了解执行时间。
在此实验中，我们通过运行随机输入和标签来训练
 `ModelParallelResNet50`
 和现有的
 `torchvision.models.resnet50()`
。训练结束后，模型不会产生任何有用的预测，
但我们可以对执行时间有一个合理的了解。






```
import torchvision.models as models

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()

```




 上面的
 `train(model)`
 方法使用
 `nn.MSELoss`
 作为损失函数，
 和
 `optim.SGD`
 作为优化器。它模仿对
 `128
 

 X
 

 128`
 图像的训练，这些图像被组织成 3 个批次，其中每个批次包含 120 个
图像。然后，我们使用
 `timeit`
 运行
 `train(model)`
 方法 10 次
并用标准差绘制执行时间。






```
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit

num_repeat = 10

stmt = "train(model)"

setup = "model = ModelParallelResNet50()"
mp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + \
        "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
rn_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)


def 图（意思，stds，[标签]（https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor"），fig_name）：
 Fig，ax = plt.subplots()
 ax.bar(np.arange(len(means)),means, yerr=stds,
align='center', alpha=0.5, ecolor='red', capsize= 10, width=0.6)
 ax.set_ylabel('ResNet50 执行时间(秒)')
 ax.set_xticks(np.arange(len(means)))
 ax.set _xticklabels([labels](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor"))
 ax.yaxis.grid(True)
 plt.tight\ \_layout()
 plt.savefig(fig_name)
 plt.close(fig)


plot([mp_mean, rn_mean],
     [mp_std, rn_std],
     ['Model Parallel', 'Single GPU'],
     'mp_vs_rn.png')

```




![](https://pytorch.org/tutorials/_images/mp_vs_rn.png)


 结果表明，模型并行实现的执行时间比现有单GPU实现长
 `4.02/3.75-1=7%`
。因此我们可以得出结论，在 GPU 之间来回复制张量大约有 7% 的开销。还有改进的空间，因为我们知道
两个 GPU 之一在整个执行过程中处于空闲状态。一种选择是将每个批次进一步划分为一个分割管道，这样当一个分割到达第二个子网络时，下一个分割就可以输入第一个子网络。这样，两个连续的分割就可以在两个
GPU 上同时运行。






 通过流水线输入加速
 [¶](#speed-up-by-pipelined-inputs "永久链接到此标题")
-------------------------------------------------------------------------------------------------



 在下面的实验中，我们进一步将每个 120 个图像批次划分为 
20 个图像分割。由于 PyTorch 异步启动 CUDA 操作，
实现不需要生成多个线程来实现
并发。






```
class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # A. ``s_prev`` runs on ``cuda:1``
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. ``s_next`` runs on ``cuda:0``, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)


setup = "model = PipelineParallelResNet50()"
pp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

plot([mp_mean, rn_mean, pp_mean],
     [mp_std, rn_std, pp_std],
     ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
     'mp_vs_rn_vs_pp.png')

```




 请注意，设备到设备的张量复制操作在源设备和目标设备上的当前流上同步。如果您创建
多个流，则必须确保复制操作
正确同步。在完成复制操作之前写入源张量或读取/写入目标张量可能会导致未定义的行为。
上述实现仅在源设备和目标设备上使用默认流，因此无需强制执行额外的同步。 




![](https://pytorch.org/tutorials/_images/mp_vs_rn_vs_pp.png)


 实验结果表明，将输入流水线化到模型并行
ResNet50 可以将训练过程加快大约
 `3.75/2.51-1=49%`
 。距离理想的 100% 加速还有相当远的距离。由于我们在管道并行实现中引入了一个新的参数“split_sizes”，因此尚不清楚新参数如何影响整体训练时间。直观地说，使用小
 `split_size`
 会导致许多微小的 CUDA 内核启动，
而使用大
 
 `split_size`
 会导致在第一个和最后一个期间相对较长的空闲时间分裂。两者都不是最佳的。对于此特定实验，可能存在最佳
 `split_size`
 配置。让我们尝试通过使用几个不同
 `split_size`
 值运行实验来找到
。






```
means = []
stds = []
split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60]

for split_size in split_sizes:
    setup = "model = PipelineParallelResNet50(split_size=%d)" % split_size
    pp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    means.append(np.mean(pp_run_times))
    stds.append(np.std(pp_run_times))

fig, ax = plt.subplots()
ax.plot(split_sizes, means)
ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
ax.set_ylabel('ResNet50 Execution Time (Second)')
ax.set_xlabel('Pipeline Split Size')
ax.set_xticks(split_sizes)
ax.yaxis.grid(True)
plt.tight_layout()
plt.savefig("split_size_tradeoff.png")
plt.close(fig)

```




![](https://pytorch.org/tutorials/_images/split_size_tradeoff.png)


 结果表明，将`split_size`设置为 12 可以实现最快的
训练速度，从而实现
 `3.75/2.43-1=54%`
 加速。仍有机会进一步加快培训进程。例如，
 `cuda:0` 上的所有操作都放在其默认流上。这意味着下一个分割的计算不能与上一个分割的复制操作重叠。然而，由于
 `上一个`
 和下一个分割是不同的张量，
将一个’s 计算与另一个’s 副本重叠是没有问题的。实现需要在两个GPU上使用多个流，并且不同的子网络结构需要不同的流管理策略。由于没有通用的多流解决方案适用于所有模型并行用例，因此我们不会在本教程中讨论它。




**注：**




 这篇文章展示了一些性能测量。在您自己的计算机上运行相同的代码时，您可能会看到不同的数字，因为结果取决于底层硬件和软件。为了获得
适合您的环境的最佳性能，正确的方法是首先生成曲线以
找出最佳分割大小，然后使用该分割大小来管道
输入。




**脚本总运行时间:** 
 ( 5 分 49.881 秒)
