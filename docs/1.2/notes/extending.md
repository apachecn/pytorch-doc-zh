# 扩展PyTorch

在这份说明中，我们将介绍torch.nn 延长[ `的方式，[ `torch.autograd`
](../autograd.html#module-torch.autograd
"torch.autograd")，并编写利用我们的C库定制的C扩展。`](../nn.html#module-torch.nn "torch.nn")

## 延伸[ `torch.autograd`](../autograd.html#module-torch.autograd
"torch.autograd")

加法运算到[ `autograd`](../autograd.html#module-torch.autograd
"torch.autograd")需要实现一个新的[ `函数 `](../autograd.html#torch.autograd.Function
"torch.autograd.Function")亚类为每个操作。回想一下，[ `功能 `
](../autograd.html#torch.autograd.Function "torch.autograd.Function") S是什么[ `
autograd`](../autograd.html#module-torch.autograd
"torch.autograd")用途计算结果和梯度和编码操作历史。每一个新的功能需要实现2种方法：

  * [ `向前（） `](../autograd.html#torch.autograd.Function.forward "torch.autograd.Function.forward") \- 执行操作的代码。只要你想，可以采取许多参数，其中一些是可选的，如果指定的默认值。各种Python对象都在这里接受。 `张量 `参数跟踪历史（即，与`requires_grad =真 `）将被转换为那些不前的跟踪历史打电话，和他们的使用将在图形注册。请注意，这个逻辑不会遍历列表/类型的字典/任何其他的数据结构，将只考虑`张量 `S是直接的参数调用。可以返回一个单一`张量 `输出，或[ `元组 `](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") `张量的 `S，如果有多个输出。另外，请参考[ `功能的文档 `](../autograd.html#torch.autograd.Function "torch.autograd.Function")找到有用的方法的描述，只能从[ `前被调用（ ） `](../autograd.html#torch.autograd.Function.forward "torch.autograd.Function.forward")。

  * [ `向后（） `](../autograd.html#torch.autograd.Function.backward "torch.autograd.Function.backward") \- 梯度公式。这将被给定为许多`张量 `参数作为有输出，与它们中的每代表梯度w.r.t.该输出。因为有输入则它应该返回尽可能多的`张量 `S，与它们中的每含有梯度w.r.t.其相应的输入。如果输入不要求梯度（HTG14]  needs_input_grad  是指示每个输入是否需要梯度计算布尔值的元组），或者被非`张量 `的对象，就可以返回`无 [HTG25。另外，如果您有可选参数为[ `向前（） `](../autograd.html#torch.autograd.Function.forward "torch.autograd.Function.forward")你可以返回更多的梯度比有投入，只要他们都[ `无 `](https://docs.python.org/3/library/constants.html#None "\(in Python v3.7\)")。`

下面你可以找到的代码为`线性 `功能从[ `torch.nn`](../nn.html#module-torch.nn
"torch.nn")，以补充意见：

    
    
    # Inherit from Function
    class LinearFunction(Function):
    
        # Note that both forward and backward are @staticmethods
        @staticmethod
        # bias is an optional argument
        def forward(ctx, input, weight, bias=None):
            ctx.save_for_backward(input, weight, bias)
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            return output
    
        # This function has only a single output, so it gets only one gradient
        @staticmethod
        def backward(ctx, grad_output):
            # This is a pattern that is very convenient - at the top of backward
            # unpack saved_tensors and initialize all gradients w.r.t. inputs to
            # None. Thanks to the fact that additional trailing Nones are
            # ignored, the return statement is simple even when the function has
            # optional inputs.
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None
    
            # These needs_input_grad checks are optional and there only to
            # improve efficiency. If you want to make your code simpler, you can
            # skip them. Returning gradients for inputs that don't require it is
            # not an error.
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)
    
            return grad_input, grad_weight, grad_bias
    

现在，为了更容易地使用这些定制的OPS，我们建议他们走样`应用 `方法：

    
    
    linear = LinearFunction.apply
    

在这里，我们给由非张量参数的参数化功能的附加示例：

    
    
    class MulConstant(Function):
        @staticmethod
        def forward(ctx, tensor, constant):
            # ctx is a context object that can be used to stash information
            # for backward computation
            ctx.constant = constant
            return tensor * constant
    
        @staticmethod
        def backward(ctx, grad_output):
            # We return as many input gradients as there were arguments.
            # Gradients of non-Tensor arguments to forward must be None.
            return grad_output * ctx.constant, None
    

注意

输入`向后 `，即`grad_output`，也可以是张量的跟踪历史。因此，如果`向后 `与可微操作，实现（例如，另一个定制`函数
`的调用），高阶导数将工作。

你可能想检查是否实际执行的落后方法计算你的函数的导数。它可以通过使用小的有限差与数值近似比较：

    
    
    from torch.autograd import gradcheck
    
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
    test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
    print(test)
    

参见[ 数值梯度检查 ](../autograd.html#grad-check)用于在有限差分梯度比较的更多细节。

## 延伸[ `torch.nn`](../nn.html#module-torch.nn "torch.nn")

[ `NN`](../nn.html#module-torch.nn "torch.nn")出口两种接口 -
模块及其功能的版本。您可以以两种方式扩展它，但我们建议您使用模块的各种层，持有任何参数或缓冲区，并推荐使用函数形式参数的操作较少样激活功能，池等

添加的操作的功能版本已经完全覆盖在上面的部分。

### 添加[ `模块 `](../nn.html#torch.nn.Module "torch.nn.Module")

由于[ `NN`](../nn.html#module-torch.nn "torch.nn")大量利用[ `autograd`
](../autograd.html#module-torch.autograd "torch.autograd")，添加一个新的[ `模块 `
](../nn.html#torch.nn.Module "torch.nn.Module")需要实现一个[ `函数 `
](../autograd.html#torch.autograd.Function
"torch.autograd.Function")执行操作，并且可以计算出梯度。从现在开始，让我们假设我们要实现一个`线性
`模块，我们必须在以上列表中实现的功能。有添加这需要非常少的代码。现在，有一些需要实现两个功能：

  * `__init__`（ _可选_ ） - 发生在参数如内核尺寸，特征的数字等，并初始化参数和缓冲剂。

  * [ `向前（） `](../nn.html#torch.nn.Module.forward "torch.nn.Module.forward") \- 实例化[ `函数 `](../autograd.html#torch.autograd.Function "torch.autograd.Function")，并使用它来执行操作。这是非常类似于上面所示的功能性包装。

这是一个`线性 `模块如何可以实现：

    
    
    class Linear(nn.Module):
        def __init__(self, input_features, output_features, bias=True):
            super(Linear, self).__init__()
            self.input_features = input_features
            self.output_features = output_features
    
            # nn.Parameter is a special kind of Tensor, that will get
            # automatically registered as Module's parameter once it's assigned
            # as an attribute. Parameters and buffers need to be registered, or
            # they won't appear in .parameters() (doesn't apply to buffers), and
            # won't be converted when e.g. .cuda() is called. You can use
            # .register_buffer() to register buffers.
            # nn.Parameters require gradients by default.
            self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(output_features))
            else:
                # You should always register all possible parameters, but the
                # optional ones can be None if you want.
                self.register_parameter('bias', None)
    
            # Not a very smart way to initialize weights
            self.weight.data.uniform_(-0.1, 0.1)
            if bias is not None:
                self.bias.data.uniform_(-0.1, 0.1)
    
        def forward(self, input):
            # See the autograd section for explanation of what happens here.
            return LinearFunction.apply(input, self.weight, self.bias)
    
        def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None
            )
    

## 编写自定义C ++扩展

看到这个[
PyTorch教程[HTG1用于详细说明和实施例。](https://pytorch.org/tutorials/advanced/cpp_extension.html)

单证可在[ torch.utils.cpp_extension  ](../cpp_extension.html)。

## 编写自定义的C扩展

实施例可在[本GitHub的库](https://github.com/pytorch/extension-ffi)。

[Next ![](../_static/images/chevron-right-orange.svg)](faq.html "Frequently
Asked Questions") [![](../_static/images/chevron-right-orange.svg)
Previous](cuda.html "CUDA semantics")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 扩展PyTorch 
    * 扩展`torch.autograd`
    * 扩展`torch.nn`
      * 添加`模块 `
    * 编写自定义C ++扩展
    * 编写自定义的C扩展

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

