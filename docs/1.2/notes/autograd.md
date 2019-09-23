# Autograd力学

这说明将呈现autograd是如何工作的，并记录操作的概述。这不是绝对必要了解这一切，但我们建议熟悉它，因为它会帮助你编写更高效，更清洁的程序，可以帮助您进行调试。

## 从不含子图向后

每张量有一个标志：`requires_grad`，其允许从梯度计算子图的细粒排斥和可以提高工作效率。

### `requires_grad`

如果有一个单一的输入操作，需要梯度，它的输出也需要梯度。相反，只有当所有的输入不需要梯度，输出也不会需要它。向后计算是从来没有在子图，所有的张量并不需要梯度进行。

    
    
    >>> x = torch.randn(5, 5)  # requires_grad=False by default
    >>> y = torch.randn(5, 5)  # requires_grad=False by default
    >>> z = torch.randn((5, 5), requires_grad=True)
    >>> a = x + y
    >>> a.requires_grad
    False
    >>> b = a + z
    >>> b.requires_grad
    True
    

当你想冻结模型的一部分，这是非常有用的，或者你事先知道你不打算使用渐变w.r.t.一些参数。例如，如果你想微调预训练CNN，这足以切换`
requires_grad
`标志的冷冻基地，没有中间缓冲区将被保存，直到计算得到最后一个层，其中，所述仿射变换将使用需要梯度权重，并且所述网络的输出也将需要它们。

    
    
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 100)
    
    # Optimize only the classifier
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
    

## autograd如何编码的历史

Autograd是反向自动分化系统。从概念上讲，autograd记录的图形记录所有创建该数据的操作为您执行操作，给你一个向无环图，叶子是输入张量和根输出张量。通过跟踪从根此图的叶子，可以自动计算使用链式法则的梯度。

在内部，autograd表示该图表为`函数 `对象（真表达式）的曲线图，其可以是`申请（） `
ED来计算评价的曲线图的结果。当计算向前传，同时autograd执行所请求的计算和积聚的曲线图表示计算梯度（的`.grad_fn`属性函数中的每个[
`torch.Tensor`](../tensors.html#torch.Tensor
"torch.Tensor")是一个入口点到该图）。当向前传球完成后，我们评估该图在向后传递给计算梯度。

要注意的重要一点是，图形是从头开始，在每次迭代重建，而这正是允许使用任意Python控制流语句，可以在每次迭代改变图形的整体形状和大小。你不必编码所有可能的路径在启动之前的训练
- 你运行的是你区分什么。

## 就地与autograd操作

在autograd支持就地操作是很难的事，我们不鼓励在大多数情况下，它们的使用。
Autograd咄咄逼人的缓冲释放和再利用使得它非常有效，也有极少数场合就地操作任何显著量实际上更低的内存使用率。除非你在重内存压力工作，你可能永远需要使用它们。

有限制就地操作的应用主要有两个原因：

  1. 就地操作可能会覆盖计算梯度所需的值。

  2. 每个就地操作实际上需要执行重写计算图。外的地方版本简单分配新对象，并保持引用旧图，而就地操作，需要更改的所有投入的创作者到`功能 `表示此操作。这可以是棘手的，尤其是如果有引用相同的存储（例如，通过索引或调换创建）许多张量，和就地如果改性输入的存储是通过引用的函数实际上将产生一个错误的任何其他`张量 `。

## 就地正确性检查

每张量保持一个版本计数器，即每递增它标志着在任何操作脏的时间。当一个函数保存任何张量为落后的，其包含的张量的一个版本计数器被保存为好。一旦你进入`
self.saved_tensors
`检查，如果它比保存的值将引发一个错误更大。这确保了如果你使用就地功能，没有看到任何错误，你可以肯定的是，计算的梯度是正确的。

[Next ![](../_static/images/chevron-right-orange.svg)](broadcasting.html
"Broadcasting semantics") [![](../_static/images/chevron-right-orange.svg)
Previous](../index.html "PyTorch documentation")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * Autograd力学
    * 从向后 不包括子图
      * `requires_grad`
    * autograd如何编码的历史
    * [HTG0在就地用autograd操作
    * [HTG0在就地正确性检查

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

