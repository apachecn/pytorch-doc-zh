# torch.Storage

A `torch.Storage`是一个单一数据类型的连续，一维阵列。

每[ `torch.Tensor`](tensors.html#torch.Tensor
"torch.Tensor")具有相同的数据类型的一个对应的存储。

_class_`torch.``FloatStorage`[[source]](_modules/torch.html#FloatStorage)

    

`bfloat16`()

    

强制转换该存储到bfloat16类型

`bool`()

    

强制转换该存储到布尔类型

`byte`()

    

强制转换该存储到byte型

`char`()

    

强制转换该存储为char类型

`clone`()

    

返回此存储的副本

`copy_`()

    

`cpu`()

    

返回此存储的CPU副本，如果它不是已经在CPU上

`cuda`( _device=None_ , _non_blocking=False_ , _**kwargs_ )

    

返回此对象的CUDA内存的副本。

如果该对象已在CUDA内存和正确的设备上，则没有执行复制操作，并返回原来的对象。

Parameters

    

  * **装置** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 目标GPU ID。默认为当前设备。

  * **non_blocking** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `和源极被固定存储器，复制将是异步相对于主机。另外，参数没有任何影响。

  * **** kwargs** \- 对于相容性，可以含有键`异步 `代替`non_blocking`参数的。

`data_ptr`()

    

`device`

    

`double`()

    

强制转换该存储为double型

`dtype`

    

`element_size`()

    

`fill_`()

    

`float`()

    

强制转换该存储浮动型

_static_`from_buffer`()

    

_static_`from_file`( _filename_ , _shared=False_ , _size=0_ ) → Storage

    

如果分享是真，然后存储器被所有进程之间共享。所有的变化写入文件。如果分享是假，然后在存储所做的更改不会影响文件。

大小是在存储元件的数量。如果分享是假，则文件必须包含至少尺寸*的sizeof（类型）字节（类型是存储类型）。如果分享是真该文件将被如果需要创建。

Parameters

    

  * **文件名** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")） - 文件名映射

  * **分享** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 是否共享存储器

  * 在存储元件的数 - **大小** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")）

`half`()

    

强制转换该存储一半类型

`int`()

    

强制转换该存储为int类型

`is_cuda`_= False_

    

`is_pinned`()

    

`is_shared`()

    

`is_sparse`_= False_

    

`long`()

    

强制转换该存储长型

`new`()

    

`pin_memory`()

    

复制存储到固定的内存，如果它不是已经固定。

`resize_`()

    

`share_memory_`()

    

移动存储到共享存储器中。

这是一个无操作为已在共享内存和CUDA储存仓库，这并不需要移动跨进程共享。在共享存储器中存储装置不能调整大小。

返回：自

`short`()

    

强制转换该存储短型

`size`()

    

`tolist`()

    

返回包含此存储的元素的列表

`type`( _dtype=None_ , _non_blocking=False_ , _**kwargs_ )

    

返回类型，如果 DTYPE 不设置，否则铸件此对象为指定的类型。

如果这是正确的类型已经没有执行复制操作，并返回原来的对象。

Parameters

    

  * **DTYPE** （[ _输入_ ](https://docs.python.org/3/library/functions.html#type "\(in Python v3.7\)") _或_ _串_ ） - 所需的类型

  * **non_blocking** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `，并且源是在固定存储器和目的地是在GPU或反之亦然，副本被相对于所述主机异步地执行。另外，参数没有任何影响。

  * **** kwargs** \- 对于相容性，可以含有键`异步 `代替`non_blocking`参数的。的`异步 `ARG被弃用。

[Next ![](_static/images/chevron-right-orange.svg)](nn.html "torch.nn")
[![](_static/images/chevron-right-orange.svg) Previous](cuda.html
"torch.cuda")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * torch.Storage 

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

