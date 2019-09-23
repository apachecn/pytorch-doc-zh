# 键入信息

的数值性质的[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype
"torch.torch.dtype")可通过访问任一 `torch.finfo`或 `torch.iinfo`。

## torch.finfo

_class_`torch.``finfo`

    

A`torch.finfo`是表示一个浮点[ `torch.dtype 的数值性质[对象HTG10]
`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")（即`
torch.float32`，`torch.float64`和`torch.float16`）。这类似于[ numpy.finfo
](https://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html)。

A`torch.finfo`提供以下属性：

名称

|

类型

|

描述  
  
---|---|---  
  
位

|

INT

|

由类型所占用的位数。  
  
EPS

|

浮动

|

可表示的最小数量，使得`1.0  +  EPS  ！=  1.0`。  
  
最大

|

float

|

最大可表示数。  
  
分

|

float

|

可表示的最小数目（典型`-max`）。  
  
小

|

float

|

最小的正表示数。  
  
注意

的构造 `torch.finfo`可以被称为无参数，在这种情况下，对于缺省pytorch D类创建的类（由[ [作为返回HTG7]
torch.get_default_dtype（） ](torch.html#torch.get_default_dtype
"torch.get_default_dtype")）。

## torch.iinfo

_class_`torch.``iinfo`

    

A`torch.iinfo`是表示一个整数[ `torch.dtype  [HTG10的数值属性的对象]
`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")（即`
torch.uint8`，`torch.int8`，`torch.int16`，`torch.int32`和`torch.int64
`）。这类似于[ numpy.iinfo
](https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html)。

A`torch.iinfo`提供以下属性：

Name

|

Type

|

Description  
  
---|---|---  
  
bits

|

int

|

The number of bits occupied by the type.  
  
max

|

int

|

The largest representable number.  
  
min

|

int

|

最小可表示数。  
  
[Next ![](_static/images/chevron-right-orange.svg)](sparse.html
"torch.sparse") [![](_static/images/chevron-right-orange.svg)
Previous](tensor_attributes.html "Tensor Attributes")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 类型信息
    * torch.finfo 
    * torch.iinfo 

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

