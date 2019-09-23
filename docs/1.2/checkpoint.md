# torch.utils.checkpoint

注意

检查点是通过在向后重新运行对每个检查点段向前通段实现。这可能会导致持续的状态就像是先进的RNG状态，他们会比没有检查点。默认情况下，检查点包括逻辑来玩弄的RNG状态，使得检查点的通过利用RNG的（通过差例如）如相对于非检查点通行证具有确定性输出。逻辑藏匿和恢复RNG状态可以承担因设置检查点操作的运行时性能适中命中。如果确定的输出相比，非检查点通行证不是必需的，供应`
preserve_rng_state =假 `至`检查点 `或`checkpoint_sequential
`省略积攒和每个检查点期间恢复所述RNG状态。

在积攒逻辑保存并恢复为当前设备的RNG状态和所有CUDA张量参数到`run_fn`该设备。然而，该逻辑还没有办法预测如果用户将在`run_fn
`本身内张量移动到新设备。因此，如果移动张量，以一个新的装置（“新”的意思不属于集合[当前设备+的张量参数的装置]的）内`run_fn
`，确定性的输出进行比较，以非检查点通行证从不保证。

`torch.utils.checkpoint.``checkpoint`( _function_ , _*args_ , _**kwargs_
)[[source]](_modules/torch/utils/checkpoint.html#checkpoint)

    

检查点模型的模型或部分

检查点的工作原理是交易计算内存。而不是存储整个计算图的所有中间激活用于向后计算，检查点部分不 **不**
保存中间激活，而是重新计算它们向后通。它可以在模型的任何部分被应用。

具体而言，在直传，`函数 `将在`torch.no_grad运行（） `的方式，即不存储中间激活。相反，直传保存的输入元组和`函数
`参数。在向后传送时，保存的输入和`函数 `时retreived，并且直传被计算在`函数 `再次，现在跟踪中间激活，然后梯度使用这些激活值来计算。

警告

检查点不工作[ `torch.autograd.grad（） `](autograd.html#torch.autograd.grad
"torch.autograd.grad")，但只有[ `torch.autograd.backward（ ） `
](autograd.html#torch.autograd.backward "torch.autograd.backward")。

Warning

如果`函数 `在落后的调用做任何事情比一个向前时不同，例如，由于一些全局变量，设立检查点版本将不会是等价的，不幸的是它不能检测。

Parameters

    

  * **函数** \- 介绍如何在模型的模型或部分直传运行。还应该知道如何处理的元组传递的输入。例如，在LSTM，如果用户通过`（活化， 隐藏） `，`函数 `应该正确地使用第一输入为`活化 `和第二输入为`隐藏 `

  * **preserve_rng_state** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ _，_ _默认=真_ ） - 省略积攒和每个检查点期间恢复所述RNG状态。

  * **ARGS** \- 包含元组输入到`函数 `

Returns

    

运行的输出`函数 `在`* ARGS`

`torch.utils.checkpoint.``checkpoint_sequential`( _functions_ , _segments_ ,
_*inputs_ , _**kwargs_
)[[source]](_modules/torch/utils/checkpoint.html#checkpoint_sequential)

    

对于检查点顺序模型辅助函数。

顺序执行模型的模块/功能，以便（顺序地）的列表。因此，我们可以划分在各个分段这样的模型和检查点每个段。除了最后的所有段将在`torch.no_grad（）
`方式运行，即不存储中间激活。每个检查点段的输入端将被保存用于重新运行段在向后通。

参见 `如何检查点检查点工作（） `[HTG5。

Warning

Checkpointing doesn’t work with
[`torch.autograd.grad()`](autograd.html#torch.autograd.grad
"torch.autograd.grad"), but only with
[`torch.autograd.backward()`](autograd.html#torch.autograd.backward
"torch.autograd.backward").

Parameters

    

  * **功能** \- A [ `torch.nn.Sequential`](nn.html#torch.nn.Sequential "torch.nn.Sequential")或模块或功能（包括模型）的列表按顺序运行。

  * **段** \- 组块数量在模型中创建

  * **输入** \- 这被输入到`张量的元组的功能 `

  * **preserve_rng_state** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ _,_ _default=True_ ) – Omit stashing and restoring the RNG state during each checkpoint.

Returns

    

运行`函数的输出 `上依次`*输入 `

例

    
    
    >>> model = nn.Sequential(...)
    >>> input_var = checkpoint_sequential(model, chunks, input_var)
    

[Next ![](_static/images/chevron-right-orange.svg)](cpp_extension.html
"torch.utils.cpp_extension") [![](_static/images/chevron-right-orange.svg)
Previous](bottleneck.html "torch.utils.bottleneck")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * torch.utils.checkpoint 

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

