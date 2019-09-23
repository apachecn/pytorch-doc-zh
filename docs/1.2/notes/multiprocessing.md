# 多处理器的最佳做法

[ `torch.multiprocessing`](../multiprocessing.html#module-
torch.multiprocessing "torch.multiprocessing")是在更换液滴为Python的[ `多处理 `
](https://docs.python.org/3/library/multiprocessing.html#module-
multiprocessing "\(in Python v3.7\)")模块。它支持完全相同的操作，但它延伸，以便通过[ `
multiprocessing.Queue`
](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
"\(in Python v3.7\)")，将其数据发送的所有张量移入共享存储器和将只发送一个句柄到另一个进程。

注意

当[ `张量 `](../tensors.html#torch.Tensor "torch.Tensor")被发送到另一个的过程中，[ `张量 `
](../tensors.html#torch.Tensor "torch.Tensor")数据是共享。如果[ `torch.Tensor.grad`
](../tensors.html#torch.Tensor.grad "torch.Tensor.grad")不是`无 `，它也共享。后一个[ `张量
`](../tensors.html#torch.Tensor "torch.Tensor")无[ `torch.Tensor.grad`
](../tensors.html#torch.Tensor.grad
"torch.Tensor.grad")字段被发送到其他过程中，它创建了一个标准过程特异性`.grad`[ `张量 `
](../tensors.html#torch.Tensor "torch.Tensor")未自动共享跨越所有流程，不像如何[ `张量的 `
](../tensors.html#torch.Tensor "torch.Tensor")数据已被共享。

这使得实现各种训练方法，如Hogwild，A3C，或需要异步操作的任何其他人。

## CUDA在多处理

CUDA运行时不支持`叉 `启动方法。然而，[ `多处理 `
](https://docs.python.org/3/library/multiprocessing.html#module-
multiprocessing "\(in Python v3.7\)")在Python 2可使用`叉 `仅创建子进程。所以Python 3和任一`菌种
`或`forkserver`启动方法需要在子过程使用CUDA。

Note

start方法则可以通过创建与`multiprocessing.get_context上下文设置（...） `或直接使用`
multiprocessing.set_start_method（...） `。

不同于CPU张量，需要在发送过程中保持原有的张量，只要该接收处理保留了张量的副本。它的引擎盖下实现的，但需要用户遵循最佳实践的程序才能正常运行。例如，发送过程必须只要活着，消费过程中有对张引用，如果消费者通过处理一个致命的信号异常退出的引用计数不能救你。参见[
本节 [HTG3。](../multiprocessing.html#multiprocessing-cuda-sharing-details)

另请参见：[ 使用nn.DataParallel而不是多处理 ](cuda.html#cuda-nn-dataparallel-instead)

## 最佳做法和技巧

### 避免和战斗死锁

有很多事情，当一个新的进程产生，死锁是后台线程的最常见的原因可能出错。如果有持有锁或导入一个模块，`叉
`被称为，这是非常有可能的是，子进程将处于损坏状态，并会死锁或在不同的失败的任何线索方式。请注意，即使你不这样做，Python的内置在图书馆做 -
没有必要进一步看起来比[ `多重处理 `[HTG9。
](https://docs.python.org/3/library/multiprocessing.html#module-
multiprocessing "\(in Python v3.7\)")[ `multiprocessing.Queue`
](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
"\(in Python
v3.7\)")实际上是一个非常复杂的类，会派生用来序列，发送和接收对象的多个线程，并且他们可以也导致上述问题。如果在这样的情况下发现自己尝试使用`
multiprocessing.queues.SimpleQueue`，即不使用任何额外的线程。

我们正在尽我们所能，让您轻松，并确保这些死锁不会发生，但有些事情是我们无法控制的。如果您有无法应付了，而任何问题，请在论坛上伸出，我们会看到，如果它是我们能够解决的问题。

### 重用缓冲器通过队列传递

请记住，每次你把[ `张量 `](../tensors.html#torch.Tensor "torch.Tensor")到[ `
multiprocessing.Queue`
](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
"\(in Python
v3.7\)")它必须被移动到共享存储器中。如果它已经共享的，它是一个空操作，否则会产生额外的内存拷贝，可以减慢整个过程。即使你有发送数据到单一的进程池中，使缓冲区发回
- 这几乎是免费的，可以让发送下一批当你避免副本。

### 异步多进程训练（例如Hogwild）

使用[ `torch.multiprocessing`](../multiprocessing.html#module-
torch.multiprocessing
"torch.multiprocessing")，所以可以以异步方式训练模式，与参数共用所有的时间，或周期性地同步。在第一种情况下，建议发送在整个模型对象，而在后者中，我们建议只发送的[
`state_dict（） `](../nn.html#torch.nn.Module.state_dict
"torch.nn.Module.state_dict")。

我们建议您使用[ `multiprocessing.Queue`
](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
"\(in Python v3.7\)")传递各种PyTorch对象的进程之间。有可能例如使用`叉
`启动方法当继承张量和存储器已经在共享存储器中，但是这是非常容易产生的错误，应该小心使用，并且仅由高级用户。队列，即使他们有时不太优雅的解决方案，将正确地适用于所有情况。

警告

你应该小心为全球性陈述，未与`如果 __name__  ==  '__main__' 把守 [ HTG9。如果超过`不同的启动方法叉
`时，它们将被所有子过程执行。`

#### Hogwild

一个具体的实施Hogwild可以在[实例库](https://github.com/pytorch/examples/tree/master/mnist_hogwild)中找到，但展示的代码的总体结构，也有以下以及一个最小的例子：

    
    
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
        # NOTE: this is required for the ``fork``method to work
        model.share_memory()
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=train, args=(model,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    

[Next ![](../_static/images/chevron-right-orange.svg)](randomness.html
"Reproducibility") [![](../_static/images/chevron-right-orange.svg)
Previous](large_scale_deployments.html "Features for large-scale deployments")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 多处理最佳实践
    * CUDA在多处理
    * 最佳做法和技巧
      * 避免和战斗死锁
      * 重用缓冲器通过队列传递
      * 异步多进程训练（例如Hogwild）
        * Hogwild 

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

