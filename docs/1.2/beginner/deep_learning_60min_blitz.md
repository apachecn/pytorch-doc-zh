# 深度学习与PyTorch：60分钟的闪电战

**作者** ：[ Soumith Chintala ](http://soumith.ch)

本教程的目标：

  * 了解PyTorch的张量库和神经网络在较高的水平。
  * 培养一个小神经网络分类图片

_本教程假设你有numpy的一个基本的了解_

注意

请确保您有[火炬](https://github.com/pytorch/pytorch)和[ torchvision
](https://github.com/pytorch/vision)安装的软件包。

![../_images/tensor_illustration_flat.png](../_images/tensor_illustration_flat.png)

[ 什么是PyTorch？  ](blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-
tutorial-py)

![../_images/autodiff.png](../_images/autodiff.png)

[ Autograd：自动微分 ](blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-
autograd-tutorial-py)

![../_images/mnist1.png](../_images/mnist1.png)

[ 神经网络 ](blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-
networks-tutorial-py)

![../_images/cifar101.png](../_images/cifar101.png)

[ 训练分类 ](blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-
py)

![../_images/data_parallel.png](../_images/data_parallel.png)

[ 可选：数据并行 ](blitz/data_parallel_tutorial.html#sphx-glr-beginner-blitz-data-
parallel-tutorial-py)

[Next ![](../_static/images/chevron-right-
orange.svg)](blitz/tensor_tutorial.html "What is PyTorch?")
[![](../_static/images/chevron-right-orange.svg) Previous](../index.html
"Welcome to PyTorch Tutorials")

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

  * 深与PyTorch学习：60分钟闪电

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

