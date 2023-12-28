# 新版本: PyTorch 1.11 版本，TorchData和functorch现已可用

> 发布: 2022年03月28日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch-1.11-released/>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V1.11>

**来自 PyTorch团队**

我们很高兴地宣布PyTorch 1.11([发布说明](https://github.com/pytorch/pytorch/releases/tag/v1.11.0))的发布。此版本由自1.10以来的3300多个提交组成，由434个贡献者完成。与1.11一起，我们正在发布TorchData和functorch的测试版。

总结：

*   **TorchData**是一个用于通用模块化数据加载原语的新库，用于轻松构建灵活和高性能的数据管道。[在GitHub上查看](https://github.com/pytorch/data)。
*   **functorch**是一个将可组合函数转换添加到PyTorch的库，现在有测试版。[在GitHub上查看](https://github.com/pytorch/functorch)。
*   分布式数据并行(DDP)静态图优化在稳定中可用。

## 介绍 TorchData

我们很高兴推出[TorchData](https://github.com/pytorch/data)的测试版。这是一个常见的模块化数据加载原语库，用于轻松构建灵活且高性能的数据管道。根据社区反馈，我们发现现有的DataLoader捆绑了太多的功能，可能很难扩展。此外，不同的用例通常必须一遍又一遍地重写相同的数据加载实用程序。这里的目标是通过称为“[DataPipes](https://github.com/pytorch/data#what-are-datapipes)”的可迭代样式和地图样式的构建块实现可组合数据加载，这些构建块与[PyTorch的DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)开箱即用。

`DataPipe` 在Python数据结构上接受一些访问函数，其中 `__iter__` 用于 `IterDataPipe`， `__getitem__` 用于 `MapDataPipe` 并返回一个新的访问函数，并应用一个轻微的转换。您可以将多个 `DataPipe` 链接在一起，以形成执行所有必要数据转换的数据管道。

我们已经实现了50多个DataPipes，这些数据管道提供了不同的核心功能，例如打开文件、解析文本、转换样本、缓存、洗牌和批处理。对于有兴趣连接到云提供商(如Google Drive或AWS S3)的用户，[fsspec](https://pytorch.org/data/0.3.0/torchdata.datapipes.iter.html#io-datapipes)和iopath DataPipes将允许您这样做。该文档提供了每个[IterDataPipe](https://pytorch.org/data/0.3.0/torchdata.datapipes.iter.html)和[MapDataPipe](https://pytorch.org/data/0.3.0/torchdata.datapipes.map.html)的详细解释和使用示例。

在此版本中，一些PyTorch域库已将其数据集迁移到使用DataPipes。在TorchText中，[库提供的流行数据集](https://github.com/pytorch/text/tree/release/0.12/torchtext/datasets)是使用DataPipes实现的，[其SST-2二进制文本分类教程的一部分](https://pytorch.org/text/0.12.0/tutorials/sst2_classification_non_distributed.html#dataset)演示了如何使用DataPipes来预处理模型的数据。在[TorchVision(在夜间版本中提供)](https://github.com/pytorch/vision/tree/main/torchvision/prototype/datasets/_builtin)和[TorchRec中](https://pytorch.org/torchrec/torchrec.datasets.html)，还有其他带有DataPipes的数据集原型实现。你可以[在这里](https://pytorch.org/data/0.3.0/examples.html)找到更[具体的例子](https://pytorch.org/data/0.3.0/examples.html)。

[TorchData](https://pytorch.org/data)的[文档](https://pytorch.org/data)现已上线。它包含一个教程，涵盖了[如何使用DataPipes](https://pytorch.org/data/0.3.0/tutorial.html#using-datapipes)，[将其与DataLoader一起使用](https://pytorch.org/data/0.3.0/tutorial.html#working-with-dataloader)，以及[实现自定义Pipes](https://pytorch.org/data/0.3.0/tutorial.html#implementing-a-custom-datapipe)。与DataLoader相关的常见问题解答和未来计划[在我们的项目的README文件中](https://github.com/pytorch/data#readme)描述。

## 介绍 functorch

我们很高兴地宣布[functorch](https://github.com/pytorch/functorch)的第一个测试版。深受[Google JAX](https://github.com/google/jax)的启发，functorch是一个为PyTorch添加可组合函数转换的库。它旨在提供可组合的vmap(矢量化)和autodiff转换，与PyTorch模块和PyTorch autograd配合使用，具有良好的渴望模式性能。

可组合函数转换可以帮助解决许多当今在PyTorch中棘手的用例：

*   计算每个样本梯度(或其他每个样本的数量)
*   在一台机器上运行模型合奏
*   在MAML的内循环中高效地将任务批处理在一起
*   高效计算雅各比人和黑塞人以及批处理的

编写vmap(矢量化)、vjp(反向模式AD)和jvp(正向模式AD)转换使我们能够毫不费力地表达上述内容，而无需为每个库设计一个单独的库。

有关更多详细信息，请参阅我们的[文档](https://pytorch.org/functorch/)、[教程](https://pytorch.org/functorch)和[安装说明](https://pytorch.org/functorch/stable/install.html)。

## 分布式训练

### (Stable)DDP静态图

DDP静态图假设您的模型在每次迭代中都使用相同的使用/未使用的参数集，因此它可以确定性地知道状态，例如哪个钩子将触发，钩子将触发多少次，以及第一次迭代后梯度计算就绪顺序。静态图在第一次迭代中缓存这些状态，因此它可以支持DDP在以前版本中无法支持的功能，例如，支持同一参数上的多个激活检查点，无论是否有未使用的参数。静态图形功能还在有未使用的参数时应用性能优化，例如，它避免了每次迭代都通过遍历图形来搜索未使用的参数，并启用动态桶式顺序。DDP静态图中的这些优化为一些推荐模型带来了10%的QPS增益。

要启用静态图，只需在DDP API中设置 static_graph=True，如下：

```
ddp_model = DistributedDataParallel(model, static_graph=True)
```

有关更多详细信息，请参阅我们的[文档](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)和[教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)。

感谢您的阅读，如果您对这些更新感兴趣，并想加入PyTorch社区，我们鼓励您加入[讨论论坛](https://discuss.pytorch.org/)并[打开GitHub问题](https://github.com/pytorch/pytorch/issues)。要从PyTorch获取最新消息，请在[Twitter](https://twitter.com/PyTorch)、[Medium](https://medium.com/pytorch)、[YouTube](https://www.youtube.com/pytorch)和[LinkedIn](https://www.linkedin.com/company/pytorch)上关注我们。

Cheers!

PyTorch 团队