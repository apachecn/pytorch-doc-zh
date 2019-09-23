# 序列化语义

## 最佳实践

### 用于保存模型的推荐方法

有序列化和恢复模型的两种主要方法。

第一个（推荐）保存并只加载模型参数：

    
    
    torch.save(the_model.state_dict(), PATH)
    

再后来：

    
    
    the_model = TheModelClass(*args, **kwargs)
    the_model.load_state_dict(torch.load(PATH))
    

第二保存和加载整个模型：

    
    
    torch.save(the_model, PATH)
    

Then later:

    
    
    the_model = torch.load(PATH)
    

然而，在这种情况下，序列化的数据绑定到特定的类和使用的准确的目录结构，所以在其他项目中使用时，它可以通过各种方式突破，或在一些严重的refactors。

[Next ![](../_static/images/chevron-right-orange.svg)](windows.html "Windows
FAQ") [![](../_static/images/chevron-right-orange.svg)
Previous](randomness.html "Reproducibility")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 序列化语义
    * 最佳实践
      * 用于保存模型推荐的方法

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

