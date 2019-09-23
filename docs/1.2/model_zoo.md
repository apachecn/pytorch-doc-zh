# torch.utils.model_zoo

移动到 torch.hub 。

`torch.utils.model_zoo.``load_url`( _url_ , _model_dir=None_ ,
_map_location=None_ , _progress=True_ )

    

加载在给定的URL火炬序列化对象。

如果对象已存在于 model_dir ，它的反序列化和返回。的URL的文件名部分应遵循命名惯例`的文件名 - & LT。; SHA256 & GT ;
EXT`其中`& LT ; SHA256 & GT ;
`是该文件的内容的散列SHA256的前八个或多个数字。哈希用于确保唯一的名称，并验证该文件的内容。

的 model_dir 默认值是`$ TORCH_HOME /检查点 `其中环境变量`$ TORCH_HOME`默认为`$
XDG_CACHE_HOME /火炬 [HTG13。 `$ XDG_CACHE_HOME`遵循了Linux
filesytem布局的X设计组规范，带有默认值HTG18] 〜/ .cache`如果没有设置。

Parameters

    

  * **URL** （ _串_ ） - 对象的URL下载

  * **model_dir** （ _串_ _，_ _可选_ ） - 目录中保存对象

  * **map_location** （ _可选_ ） - 一个功能或一个字典指定如何重新映射的存储位置（参见torch.load）

  * **进展** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否要显​​示进度条到stderr

例

    
    
    >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    

[Next ![](_static/images/chevron-right-orange.svg)](tensorboard.html
"torch.utils.tensorboard") [![](_static/images/chevron-right-orange.svg)
Previous](dlpack.html "torch.utils.dlpack")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * torch.utils.model_zoo 

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

![](_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

