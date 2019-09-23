# 张量属性

每个`torch.Tensor`具有 `torch.dtype`， `炬.device`和 `torch.layout`。

## torch.dtype

_class_`torch.``dtype`

    

A`torch.dtype`是表示的数据类型的对象的[ `torch.Tensor`
](tensors.html#torch.Tensor "torch.Tensor")。 PyTorch有九个不同的数据类型：

数据类型

|

D型

|

张量类型  
  
---|---|---  
  
32位浮点

|

`torch.float32`或`torch.float`

|

`炬。*。FloatTensor` 
  
64位浮点

|

`torch.float64`或`torch.double`

|

`炬。*。DoubleTensor` 
  
16位浮点

|

`torch.float16`或`torch.half`

|

`炬。*。HalfTensor` 
  
8位整数（无符号）

|

`torch.uint8`

|

`炬。*。ByteTensor` 
  
8位整数（签名）

|

`torch.int8`

|

`炬。*。CharTensor` 
  
16位整数（签名）

|

`torch.int16`或`torch.short`

|

`炬。*。ShortTensor` 
  
32位整数（签名）

|

`torch.int32`或`torch.int`

|

`炬。*。IntTensor` 
  
64位整数（签名）

|

`torch.int64`或`torch.long`

|

`炬。*。LongTensor` 
  
布尔

|

`torch.bool`

|

`炬。*。BoolTensor` 
  
以找出是否一个 `torch.dtype`是一个浮点数据类型，属性[ `is_floating_point`
](torch.html#torch.is_floating_point "torch.is_floating_point")可以被使用，它返回`
真如果数据类型是浮点数据类型`。

## torch.device

_class_`torch.``device`

    

A`torch.device`是表示装置的对象在其上[ `torch.Tensor`
](tensors.html#torch.Tensor "torch.Tensor")或将被分配。

的 `torch.device`包含一个设备类型（`'CPU' `或`' CUDA”
`）和用于设备类型任选装置的序号。如果设备序不存在，这个对象将总是代表的设备类型的当前装置中，即使[ `torch.cuda.set_device后（）
`](cuda.html#torch.cuda.set_device "torch.cuda.set_device")被称为;例如，[ ``
](tensors.html#torch.Tensor "torch.Tensor")构造torch.Tensor与设备`'CUDA' `等于`
'CUDA：X' `其中，X是[ `torch.cuda.current_device的（）的结果 `
](cuda.html#torch.cuda.current_device "torch.cuda.current_device")。

A [ `torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")的设备可以通过[ `
Tensor.device`[HTG11被访问]属性。](tensors.html#torch.Tensor.device
"torch.Tensor.device")

A`torch.device`可以通过一个字符串或经由串和设备序号来构成

通过字符串：

    
    
    >>> torch.device('cuda:0')
    device(type='cuda', index=0)
    
    >>> torch.device('cpu')
    device(type='cpu')
    
    >>> torch.device('cuda')  # current cuda device
    device(type='cuda')
    

通过串和设备顺序：

    
    
    >>> torch.device('cuda', 0)
    device(type='cuda', index=0)
    
    >>> torch.device('cpu', 0)
    device(type='cpu', index=0)
    

注意

的 `torch.device`在功能参数可以通常与串取代。这使得代码的快速原型。

    
    
    >>> # Example of a function that takes in a torch.device
    >>> cuda1 = torch.device('cuda:1')
    >>> torch.randn((2,3), device=cuda1)
    
    
    
    >>> # You can substitute the torch.device with a string
    >>> torch.randn((2,3), device='cuda:1')
    

Note

对于传统的原因，一个设备可以经由单个装置序，这将被视为一个CUDA设备来构建。这符合[ `Tensor.get_device（） `
](tensors.html#torch.Tensor.get_device
"torch.Tensor.get_device")，它返回CUDA张量的序和不支持CPU张量。

    
    
    >>> torch.device(1)
    device(type='cuda', index=1)
    

Note

其采取的装置的方法通常会接受一个（适当格式化的）字符串或（传统）整数装置序，即，以下都是等效的：

    
    
    >>> torch.randn((2,3), device=torch.device('cuda:1'))
    >>> torch.randn((2,3), device='cuda:1')
    >>> torch.randn((2,3), device=1)  # legacy
    

## torch.layout

_class_`torch.``layout`

    

A`torch.layout`是代表的存储器布局的对象的[ `torch.Tensor`
](tensors.html#torch.Tensor "torch.Tensor")。目前，我们支持`torch.strided
`（密集张量），并有`torch.sparse_coo  [HTG19（稀疏COO张量）的实验性支持。`

`torch.strided`代表致密张量，并且是最常用的存储器布局。每个跨距张量具有相关联的`torch.Storage
`，其保持它的数据。这些张量提供多维的，[存储的跨距](https://en.wikipedia.org/wiki/Stride_of_an_array)图。跨越式发展是整数的列表：第k个步幅表示从一个元素去的张量的第k个维度上的下一个必要的存储器中的跳跃。这个概念使得它可以有效地进行多张操作。

例：

    
    
    >>> x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)
    
    >>> x.t().stride()
    (1, 5)
    

有关`的更多信息torch.sparse_coo`张量，见[ torch.sparse  ](sparse.html#sparse-docs)。

[Next ![](_static/images/chevron-right-orange.svg)](type_info.html "Type
Info") [![](_static/images/chevron-right-orange.svg) Previous](tensors.html
"torch.Tensor")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 张量属性
    * torch.dtype 
    * torch.device 
    * torch.layout 

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

