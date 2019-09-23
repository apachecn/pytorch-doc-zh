# 创建扩展使用numpy的和SciPy的

**作者** ：[亚当Paszke ](https://github.com/apaszke)

**由** 更新：[亚当Dziedzic的](https://github.com/adam-dziedzic)

在本教程中，我们将通过两个任务去：

  1. 创建不带参数的神经网络层。

>     * 该调用到 **numpy的的** 作为其实现的一部分

  2. 创建具有可学习的权重神经网络层

>     * 该调用到 **SciPy的的** 作为其实现的一部分

    
    
    import torch
    from torch.autograd import Function
    

## 无参数的例子

这层不特别做任何有用的或数学上是正确的。

它恰当地命名为BadFFTFunction

**层实现**

    
    
    from numpy.fft import rfft2, irfft2
    
    
    class BadFFTFunction(Function):
        @staticmethod
        def forward(ctx, input):
            numpy_input = input.detach().numpy()
            result = abs(rfft2(numpy_input))
            return input.new(result)
    
        @staticmethod
        def backward(ctx, grad_output):
            numpy_go = grad_output.numpy()
            result = irfft2(numpy_go)
            return grad_output.new(result)
    
    # since this layer does not have any parameters, we can
    # simply declare this as a function, rather than as an nn.Module class
    
    
    def incorrect_fft(input):
        return BadFFTFunction.apply(input)
    

**所创建的层的实例：**

    
    
    input = torch.randn(8, 8, requires_grad=True)
    result = incorrect_fft(input)
    print(result)
    result.backward(torch.randn(result.size()))
    print(input)
    

日期：

    
    
    tensor([[ 7.4515,  2.2547,  7.6126,  2.8310, 13.8079],
            [12.0822,  6.2765,  3.5208,  2.2051,  3.7491],
            [ 5.7192,  4.8086,  4.7709, 17.7730,  0.6329],
            [ 9.4613,  4.3585,  4.1035,  8.0416,  5.4456],
            [16.4615,  2.8314,  8.1768,  5.0839,  9.6213],
            [ 9.4613, 12.6283,  3.2765,  4.8069,  5.4456],
            [ 5.7192,  4.9651, 18.9993,  9.2646,  0.6329],
            [12.0822,  3.0731,  3.7945, 12.1748,  3.7491]],
           grad_fn=<BadFFTFunctionBackward>)
    tensor([[ 0.1924,  0.7354, -0.0498, -1.2294, -0.2937, -0.1854, -0.8866, -0.5025],
            [-0.4862, -0.2749,  0.7363, -0.3230,  0.6703,  0.2308,  0.5687, -2.1133],
            [ 0.0432, -0.5409, -0.7979,  1.3634, -1.1702,  0.0747,  1.5215,  0.0555],
            [ 1.6646, -0.2177, -0.4921, -0.7097,  0.5300, -1.3457, -1.1927,  0.3836],
            [-0.5561,  2.3293,  0.5014, -1.0231,  2.8309,  1.1796,  1.1218,  0.8208],
            [-0.3520,  0.4791, -1.3561, -2.2878,  0.6373, -0.6391, -0.0277, -0.5974],
            [ 0.5807, -2.2914,  0.9253,  0.8924, -0.7267,  0.5135,  0.0629, -0.9859],
            [-0.1888, -1.5387, -0.2399, -1.1361,  0.7858, -1.0179, -1.3784, -0.7279]],
           requires_grad=True)
    

## 参数化示例

在深学习文献中，这个层被混淆的被称为卷积而实际操作是互相关（唯一的区别是，过滤器被翻转为卷积，这对于互相关的情况下）。

与可学习权重，其中，互相关具有过滤器（内核）表示权重的层的实施方式。

向后过程计算的梯度WRT输入和梯度WRT过滤器。

    
    
    from numpy import flip
    import numpy as np
    from scipy.signal import convolve2d, correlate2d
    from torch.nn.modules.module import Module
    from torch.nn.parameter import Parameter
    
    
    class ScipyConv2dFunction(Function):
        @staticmethod
        def forward(ctx, input, filter, bias):
            # detach so we can cast to NumPy
            input, filter, bias = input.detach(), filter.detach(), bias.detach()
            result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
            result += bias.numpy()
            ctx.save_for_backward(input, filter, bias)
            return torch.as_tensor(result, dtype=input.dtype)
    
        @staticmethod
        def backward(ctx, grad_output):
            grad_output = grad_output.detach()
            input, filter, bias = ctx.saved_tensors
            grad_output = grad_output.numpy()
            grad_bias = np.sum(grad_output, keepdims=True)
            grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
            # the previous line can be expressed equivalently as:
            # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
            grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
            return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)
    
    
    class ScipyConv2d(Module):
        def __init__(self, filter_width, filter_height):
            super(ScipyConv2d, self).__init__()
            self.filter = Parameter(torch.randn(filter_width, filter_height))
            self.bias = Parameter(torch.randn(1, 1))
    
        def forward(self, input):
            return ScipyConv2dFunction.apply(input, self.filter, self.bias)
    

**实例：**

    
    
    module = ScipyConv2d(3, 3)
    print("Filter and bias: ", list(module.parameters()))
    input = torch.randn(10, 10, requires_grad=True)
    output = module(input)
    print("Output from the convolution: ", output)
    output.backward(torch.randn(8, 8))
    print("Gradient for the input map: ", input.grad)
    

Out:

    
    
    Filter and bias:  [Parameter containing:
    tensor([[ 1.7061, -0.1771,  0.6047],
            [-0.5862,  1.0628, -0.3486],
            [ 0.0778,  1.0832, -0.4671]], requires_grad=True), Parameter containing:
    tensor([[-0.0718]], requires_grad=True)]
    Output from the convolution:  tensor([[ 0.2062,  1.9303, -0.7497,  0.4171,  1.5641, -5.1289, -2.3736,  3.7649],
            [ 2.7773, -2.5988,  2.9943, -0.3176,  0.8576, -4.0404,  3.2371, -1.6735],
            [ 2.7032,  2.0101, -1.6930,  5.7152, -2.3599, -1.3598, -0.9169, -0.9801],
            [ 2.2197,  0.6012,  1.2883, -0.9301, -1.2504, -3.7107, -3.8789,  1.2738],
            [-0.1329,  3.8820, -2.1698,  2.1074,  1.1566,  1.0722, -1.4080, -0.8036],
            [ 3.1168,  1.6253,  1.7778,  1.0007, -3.1746, -2.1811, -1.5891, -1.8327],
            [ 0.6647, -2.6461,  1.3050, -4.5868,  3.2904, -0.8035, -1.3580,  0.2333],
            [ 4.1536,  5.0878, -2.0750,  0.8895,  1.0726, -0.7173,  4.1948, -1.1099]],
           grad_fn=<ScipyConv2dFunctionBackward>)
    Gradient for the input map:  tensor([[-5.1065e+00, -4.0534e-01, -3.3325e+00, -1.2726e+00,  2.4771e+00,
             -3.2429e+00, -1.5069e+00,  1.5726e+00, -1.2231e+00,  7.7367e-01],
            [ 1.9218e+00, -5.0568e+00,  1.2643e-02, -2.6672e+00,  1.5748e+00,
              2.0034e+00, -2.7221e+00, -2.6116e+00,  1.0872e+00, -7.6352e-01],
            [ 2.1668e-01, -1.8442e+00, -1.9986e+00, -2.6523e+00, -1.9327e+00,
              5.6665e+00, -3.1624e+00, -2.5519e+00,  1.9742e+00, -7.9616e-01],
            [-9.4981e-01,  2.1246e-02, -2.6692e+00, -2.6164e+00, -1.2055e+00,
              2.8694e+00,  9.6858e-01, -2.8408e+00, -1.8077e-01,  1.2826e-01],
            [-2.0616e+00, -1.2810e+00,  2.4630e+00, -1.6501e+00,  1.6563e+00,
              2.8737e+00,  1.8403e+00,  1.4945e-01, -1.4603e+00,  6.9704e-01],
            [ 4.0963e+00, -3.5031e-01, -2.8858e+00, -7.2992e-01, -3.5853e+00,
              7.3722e-01, -2.0150e+00, -1.2854e+00, -1.0418e+00,  1.9080e-01],
            [-1.1950e+00, -6.5445e-01, -1.3459e+00, -4.7732e-01, -1.3988e+00,
              1.9046e+00, -8.2274e-01,  4.5361e-01,  7.6111e-01,  6.7929e-01],
            [ 1.6332e-01,  5.5551e+00,  2.6393e+00, -1.8013e+00,  1.0392e+00,
              2.8623e+00, -1.1749e-01, -2.0843e+00,  2.6226e+00, -8.2416e-01],
            [-7.9535e-03, -1.0495e+00,  4.9276e-01, -6.3320e-01, -2.2379e-01,
             -1.2710e+00,  2.2692e+00, -5.2379e-01,  1.1417e+00, -5.0747e-01],
            [ 1.3845e-03,  1.6173e-01,  2.0946e+00,  8.4693e-01, -2.9453e-01,
              2.0683e-01,  2.0648e+00, -7.6050e-01, -5.9649e-01,  2.1951e-01]])
    

**检查梯度：**

    
    
    from torch.autograd.gradcheck import gradcheck
    
    moduleConv = ScipyConv2d(3, 3)
    
    input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
    test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
    print("Are the gradients correct: ", test)
    

Out:

    
    
    Are the gradients correct:  True
    

**脚本的总运行时间：** （0分钟3.843秒）

[`Download Python source code:
numpy_extensions_tutorial.py`](../_downloads/f90300e089ec4a4b37bb662251daec65/numpy_extensions_tutorial.py)

[`Download Jupyter notebook:
numpy_extensions_tutorial.ipynb`](../_downloads/36e0b75bb574c654dd2e56581312013b/numpy_extensions_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-orange.svg)](cpp_extension.html
"Custom C++ and CUDA Extensions") [![](../_static/images/chevron-right-
orange.svg) Previous](torch_script_custom_ops.html "Extending TorchScript with
Custom C++ Operators")

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

  * 创建扩展使用numpy的和SciPy的
    * 参数少示例
    * 参数化的实例

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

