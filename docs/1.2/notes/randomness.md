# 再现性

完全可重复的结果不能跨PyTorch版本中，提交个人或不同的平台保证。此外，结果不需要是CPU和GPU执行之间可重现的，使用相同的种子时也是如此。

然而，为了使计算您在一个特定的平台和PyTorch释放特定问题的确定性，有几个要采取的步骤。

有参与PyTorch 2个伪随机数生成器，您将需要手动种子，使运行重复性。此外，你应该确保所有其他库的代码依赖和使用随机数也使用固定的种子。

## PyTorch

可以使用[ `torch.manual_seed（） `](../torch.html#torch.manual_seed
"torch.manual_seed")种子为所有设备（CPU和CUDA）的RNG：

    
    
    import torch
    torch.manual_seed(0)
    

存在使用CUDA功能，可以是非确定性的源一些PyTorch功能。一类这样的CUDA功能是原子操作，特别是`atomicAdd
`，其中并行加法的为相同的值的顺序是不确定的，并且对于浮点变量，方差的源在结果中。 PyTorch功能，在前进用`atomicAdd`包括[ `
torch.Tensor.index_add_（） `](../tensors.html#torch.Tensor.index_add_
"torch.Tensor.index_add_")，[ `torch.Tensor.scatter_add_（） `
](../tensors.html#torch.Tensor.scatter_add_ "torch.Tensor.scatter_add_")，[ `
torch.bincount（） `](../torch.html#torch.bincount "torch.bincount")。

多个操作的具有向后的是使用`atomicAdd`，特别是[ `torch.nn.functional.embedding_bag（） `
](../nn.functional.html#torch.nn.functional.embedding_bag
"torch.nn.functional.embedding_bag")，[ `torch.nn.functional.ctc_loss（） `
](../nn.functional.html#torch.nn.functional.ctc_loss
"torch.nn.functional.ctc_loss")和池，填充和采样的许多形式。当前有避免这些功能的非确定性的没有简单的方法。

## CuDNN

当在CuDNN后台运行，另外两个选项必须设置：

    
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

警告

确定性模式可以有一个性能的影响，这取决于你的模型。这意味着，由于模型的确定性性质，处理速度（即每秒处理批次项目）可以比当模型是非确定性较低。

## numpy的

如果您或任何你正在使用的库的依赖numpy的，你应该播种numpy的RNG以及。这是可以做到的：

    
    
    import numpy as np
    np.random.seed(0)
    

[Next ![](../_static/images/chevron-right-orange.svg)](serialization.html
"Serialization semantics") [![](../_static/images/chevron-right-orange.svg)
Previous](multiprocessing.html "Multiprocessing best practices")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 重复性
    * PyTorch 
    * CuDNN 
    * numpy的

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

