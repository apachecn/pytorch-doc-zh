# 广播语义

许多PyTorch运营支持[ `NumPy的 广播 语义 `
[HTG9。](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#module-
numpy.doc.broadcasting "\(in NumPy v1.17\)")

简言之，如果一个PyTorch操作支持广播，那么它的张量参数可以自动扩展为等于尺寸的（不使数据的副本）。

## 通用语义

二张量“broadcastable”如果以下规则成立：

  * 每个张量至少有一个尺寸。

  * 当迭代的尺寸大小，在开始尾部尺寸，该尺寸大小必须是相等的，它们中的一个是1，或它们中的一个不存在。

例如：

    
    
    >>> x=torch.empty(5,7,3)
    >>> y=torch.empty(5,7,3)
    # same shapes are always broadcastable (i.e. the above rules always hold)
    
    >>> x=torch.empty((0,))
    >>> y=torch.empty(2,2)
    # x and y are not broadcastable, because x does not have at least 1 dimension
    
    # can line up trailing dimensions
    >>> x=torch.empty(5,3,4,1)
    >>> y=torch.empty(  3,1,1)
    # x and y are broadcastable.
    # 1st trailing dimension: both have size 1
    # 2nd trailing dimension: y has size 1
    # 3rd trailing dimension: x size == y size
    # 4th trailing dimension: y dimension doesn't exist
    
    # but:
    >>> x=torch.empty(5,2,4,1)
    >>> y=torch.empty(  3,1,1)
    # x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3
    

如果两个张量`× `，`Y`的“broadcastable”，所得到的张量大小的计算方法如下：

  * 如果`× `和`Y`不等于，在前面加上1用更少的尺寸张量的尺寸，使维数它们相等的长度。

  * 然后，对于每个维度大小，所得到的尺寸大小是`× `和尺寸`Y`沿着该维度的最大值。

For Example:

    
    
    # can line up trailing dimensions to make reading easier
    >>> x=torch.empty(5,1,4,1)
    >>> y=torch.empty(  3,1,1)
    >>> (x+y).size()
    torch.Size([5, 3, 4, 1])
    
    # but not necessary:
    >>> x=torch.empty(1)
    >>> y=torch.empty(3,1,7)
    >>> (x+y).size()
    torch.Size([3, 1, 7])
    
    >>> x=torch.empty(5,2,4,1)
    >>> y=torch.empty(3,1,1)
    >>> (x+y).size()
    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
    

## 就地语义

一个复杂的是，就地操作不允许就地张量改变形状作为广播的结果。

For Example:

    
    
    >>> x=torch.empty(5,3,4,1)
    >>> y=torch.empty(3,1,1)
    >>> (x.add_(y)).size()
    torch.Size([5, 3, 4, 1])
    
    # but:
    >>> x=torch.empty(1,3,1)
    >>> y=torch.empty(3,1,7)
    >>> (x.add_(y)).size()
    RuntimeError: The expanded size of the tensor (1) must match the existing size (7) at non-singleton dimension 2.
    

## 向后兼容

PyTorch的现有版本允许某些逐点函数来执行对具有不同形状的张量，只要在每个张量元素的数量是相等的。逐点操作将随后通过查看每个张量作为1维的来进行。
PyTorch现在支持广播和“一维”逐点行为被废弃了，会产生在张量不broadcastable，但有相同数量的元素的情况下，一个Python警告。

需要注意的是引入广播可能会导致以下情况：张量不具有相同的形状，向后兼容的更改，但broadcastable，并且具有相同数目的元素。例如：

    
    
    >>> torch.add(torch.ones(4,1), torch.randn(4))
    

将先前产生具有尺寸的张量：torch.Size（[4,1]），但现在产生具有尺寸的张量：torch.Size（[4,4]）。为了帮助识别代码的情况下向后通过广播介绍不兼容性可能存在，你可以通过设置
torch.utils.backcompat.broadcast_warning.enabled 至真，这将产生一个python在这种情况下警告。

For Example:

    
    
    >>> torch.utils.backcompat.broadcast_warning.enabled=True
    >>> torch.add(torch.ones(4,1), torch.ones(4))
    __main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, and have the same number of elements.
    Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 1-dimensional.
    

[Next ![](../_static/images/chevron-right-
orange.svg)](cpu_threading_torchscript_inference.html "CPU threading and
TorchScript inference") [![](../_static/images/chevron-right-orange.svg)
Previous](autograd.html "Autograd mechanics")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 广播语义
    * 通用语义
    * [HTG0在就地语义
    * 向后兼容性

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

