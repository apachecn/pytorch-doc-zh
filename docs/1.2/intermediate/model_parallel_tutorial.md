# 1.型号并行最佳实践

**作者** ：[沉莉](https://mrshenli.github.io/)

在分布式训练技巧型号并行广泛使用。先前的文章已经解释了如何使用[数据并行](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)培养上多GPU的神经网络;此功能复制相同的模型来所有的GPU，其中每个GPU消耗输入数据的不同分区。虽然它可以显著加快培训过程中，它并不适用于一些使用情况下，该模型是太大，不适合到一个单一的GPU工作。此信息显示了如何通过使用
**模型平行** ，其中，与`数据并行 `，分割一个单一的模式上不同的GPU，而不是复制来解决这个问题每个GPU整个模型（将混凝土，说一个模型`M
`包含10层：使用`数据并行 `时，每个GPU将有这些层10中的一个副本，在两个GPU使用模型时平行而，每个GPU可能拥有5层）。

并行模型的高级别想法是一个模型的不同的子网络放置到不同的设备，并相应地实现了`向前
`的方法来跨设备移动中间输出。作为模型中的一部分的任何单独的设备上操作时，一组设备可以共同用作一个更大的模型。在这篇文章中，我们不会试图构建巨大的模型，并将其挤压到GPU的数量有限。取而代之的是，这篇文章的重点是展示并行模型的想法。它是由读者的想法应用到现实世界的应用。

## 基本用法

让我们先从一个包含两个线性层的玩具模型。上运行两个GPU此模型中，简单地把一个不同的GPU每个线性层，并移动输入和中间输出到层设备相应地匹配。

    
    
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
    

需要注意的是，关于上述`ToyModel`看起来非常相似，一个是如何实现它在单GPU，除了五个`到（设备）HTG6]
`其中放置线性层和张量上适当的设备的呼叫。这是需要改变的模型的唯一地方。的`向后（） `和`torch.optim
`将自动仿佛模型是一个GPU采取梯度的照顾。你只需要确保标签的生产日期相同的设备的输出上调用损失函数时。

    
    
    model = ToyModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    optimizer.zero_grad()
    outputs = model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to('cuda:1')
    loss_fn(outputs, labels).backward()
    optimizer.step()
    

## 应用模型平行于现有模块

也可以用短短几行的变化上运行多个GPU现有的单GPU模块。下面的代码说明了如何分解`torchvision.models.reset50（）
`到两个GPU。我们的想法是从现有`RESNET`模块继承，和施工期间层被划分到两个GPU。然后，重写`向前
`方法通过相应地移动所述中间输出缝合两个子网络。

    
    
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
    

上述实施解决了这个问题。对于其中的模型太大，适合单个GPU的情况。然而，你可能已经注意到，它会比，如果你的模型适合于单一GPU运行更慢。这是因为，在任何时间点，只有两个GPU中的一个工作，而另一种是坐在那里什么都不做。性能进一步恶化作为中间输出需要从`
CUDA复制：0`至`CUDA：1`之间`二层 `和`三层 `。

让我们进行实验获得的执行时间更定量的观点。在这个实验中，我们培养`ModelParallelResNet50`和现有`
torchvision.models.reset50（）
`通过贯穿其中随机输入和标签。培训结束后，该车型将不会产生任何有用的预测，但我们可以得到的执行时间有一定的了解。

    
    
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
    

的`列车（模型） `上述方法使用`nn.MSELoss`作为损失函数，并`optim.SGD`作为优化器。它模仿在`训练128  X
128`，其被组织成3批，其中每个批次包含120张图像的图像。然后，我们使用`timeit`运行`列车（模型）
`方法10次，并用标准偏差绘制的执行时间。

    
    
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    import numpy as np
    import timeit
    
    num_repeat = 10
    
    stmt = "train(model)"
    
    setup = "model = ModelParallelResNet50()"
    # globals arg is only available in Python 3. In Python 2, use the following
    # import __builtin__
    # __builtin__.__dict__.update(locals())
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)
    
    setup = "import torchvision.models as models;" + \
            "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
    rn_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)
    
    
    def plot(means, stds, labels, fig_name):
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(means)), means, yerr=stds,
               align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
        ax.set_ylabel('ResNet50 Execution Time (Second)')
        ax.set_xticks(np.arange(len(means)))
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close(fig)
    
    
    plot([mp_mean, rn_mean],
         [mp_std, rn_std],
         ['Model Parallel', 'Single GPU'],
         'mp_vs_rn.png')
    

![](../_images/mp_vs_rn.png)

结果表明，模型并行实现的执行时间是`4.02 / 3.75-1 = 7％
`比现有的单GPU实现更长。因此，我们可以得出结论存在抄袭张量来回GPU的大约7％的开销。有客房的改进，因为我们知道两个GPU之一是在整个执行闲置。一个选择是每批进一步分成分割，使得当一个分割到达第二子网络，下面的分割可被馈送到第一子网络的一个管道。通过这种方式，两个连续的裂缝可以在两个GPU并行运行。

## 加快以流水线方式输入

在下面的实验中，我们进一步将每个120-图像批次为20个图像分割。作为PyTorch推出CUDA运算asynchronizely，实现不需要生成多个线程来实现并发。

    
    
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
                # A. s_prev runs on cuda:1
                s_prev = self.seq2(s_prev)
                ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
    
                # B. s_next runs on cuda:0, which can run concurrently with A
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
    

请注意，设备到设备的张量复制操作的源和目标设备上的电流流同步。如果创建多个数据流，你必须确保复制操作正确同步。编写源张量或读/完成复制操作可能会导致不确定的行为之前以书面目的地张量。上述实现仅在两个源和目的设备使用默认流，因此，没有必要执行额外同步。

![](../_images/mp_vs_rn_vs_pp.png)

实验结果表明，流水线输入到模型平行ResNet50由大致`3.75 / 2.51-1 = 49％
`加速训练过程。它仍然是相当远从100％理想的加速比。正如我们已经推出了新的参数`split_sizes
`在我们的管道并行实现，目前还不清楚新的参数是如何影响整体的训练时间。直观地说，用小`split_size
`导致许多微小的CUDA内核启动，而在使用大`split_size`结果相对长的空闲时间第一和最后的分裂。无论是最优的。有可能是对于该特定实验的最佳`
split_size`配置。让我们尝试使用几种不同的`split_size`值运行实验来找到它。

    
    
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
    

![](../_images/split_size_tradeoff.png)

结果表明，设置`split_size`至12达到最快的训练速度，这导致`3.75 / 2.43-1 = 54％
`加速。还有机会进一步加速训练过程。例如，在`所有操作CUDA：0
`被放置在它的默认流。这意味着，对下一个分割的计算不能与先前分裂的复制操作重叠。然而，随着上一个和下一个分裂为不同的张量，没有重叠一个人与另一个人的副本计算问题。实现需要两个GPU的使用多个数据流，以及不同的子网络结构需要不同的数据流管理策略。由于没有通用的多流解决方案适用于所有型号并联使用的情况下，我们不会在本教程中讨论它。

**注：**

这篇文章显示了几个性能测量。运行你自己的机器上相同的代码时，您可能会看到不同的数字，因为结果取决于底层硬件和软件上。为了让您的环境中获得最佳性能，正确的做法是先产生曲线找出最佳分割尺寸，然后使用该拆分大小的管道输入。

**脚本的总运行时间：** （5分钟51.519秒）

[`Download Python source code:
model_parallel_tutorial.py`](../_downloads/d961a67e594a77a630ec636c89f84bb8/model_parallel_tutorial.py)

[`Download Jupyter notebook:
model_parallel_tutorial.ipynb`](../_downloads/b882009cab92c6a1d9121b1f8c4108c4/model_parallel_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-orange.svg)](ddp_tutorial.html "2.
Getting Started with Distributed Data Parallel")
[![](../_static/images/chevron-right-orange.svg)
Previous](../advanced/super_resolution_with_onnxruntime.html "4. \(optional\)
Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 1.型号并行最佳实践
    * 基本用法
    * 套用模型并行为现有模块
    * 加快通过流水线输入

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

