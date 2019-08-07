# 用 numpy 和 scipy 创建扩展

> 译者：[cangyunye](https://github.com/cangyunye)
>
> 校对者：[FontTian](https://github.com/fonttian)

**作者**: [Adam Paszke](https://github.com/apaszke)

**修订者**: [Adam Dziedzic](https://github.com/adam-dziedzic)

在这个教程里，我们要完成两个任务:

1.  创建一个无参神经网络层。

    这里需要调用**numpy**作为实现的一部分。

2.  创建一个权重自主优化的神经网络层。

    这里需要调用**Scipy**作为实现的一部分。

```python
import torch
from torch.autograd import Function

```

## 无参数神经网络层示例

这一层并没有特意做什么任何有用的事或者去进行数学上的修正。

它只是被恰当的命名为BadFFTFunction

**本层的实现方式**

```python
from numpy.fft import rfft2, irfft2

class BadFFTFunction(Function):

    def forward(self, input):
        numpy_input = input.detach().numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    def backward(self, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)

# 由于本层没有任何参数，我们可以简单的声明为一个函数，而不是当做 nn.Module 类

def incorrect_fft(input):
    return BadFFTFunction()(input)

```

**创建无参数神经网络层的示例方法:**

```python
input = torch.randn(8, 8, requires_grad=True)
result = incorrect_fft(input)
print(result)
result.backward(torch.randn(result.size()))
print(input)

```

输出:

```python
tensor([[2.2488e-03, 5.1309e+00, 6.4310e+00, 6.0649e+00, 8.1197e+00],
        [3.4379e+00, 1.5772e+00, 1.0834e+01, 5.2234e+00, 1.0509e+01],
        [2.6480e+00, 1.2934e+01, 9.1619e+00, 1.6011e+01, 9.7914e+00],
        [4.0796e+00, 8.6867e+00, 8.8971e+00, 1.0232e+01, 5.7227e+00],
        [1.8085e+01, 5.4060e+00, 5.2141e+00, 3.5451e+00, 5.1584e+00],
        [4.0796e+00, 8.2662e+00, 1.1570e+01, 8.7164e+00, 5.7227e+00],
        [2.6480e+00, 4.5982e+00, 1.1056e+00, 8.8158e+00, 9.7914e+00],
        [3.4379e+00, 6.2059e+00, 5.9354e+00, 3.1194e+00, 1.0509e+01]],
       grad_fn=<BadFFTFunction>)
tensor([[-0.6461,  0.3270, -1.2190, -0.5480, -1.7273, -0.7326,  0.6294, -0.2311],
        [ 0.4305,  1.7503, -0.2914, -0.4237,  0.5441,  1.6597, -0.5645, -0.7901],
        [ 0.4248, -2.5986, -0.9257, -0.8651, -0.1673,  1.5749, -1.1857,  1.2867],
        [-0.5180,  2.3175, -1.9279,  1.2128,  0.7789,  0.0385, -1.1871,  0.3431],
        [ 0.6934,  1.0216, -0.7450,  0.0463, -1.5447, -1.5220,  0.9389, -0.5811],
        [ 1.9286, -1.0957,  0.6878, -0.5469, -0.5505,  0.5088,  0.8965,  0.4874],
        [-0.2699,  0.3370,  0.3749, -0.3639, -0.0599,  0.8904,  0.1679, -1.8218],
        [-0.2963,  0.2246,  0.6617,  1.2258,  0.1530,  0.3114,  0.4568,  0.6181]],
       requires_grad=True)

```

## 参数化示例

在深度学习的文献中，这一层被意外的称作卷积`convolution `，尽管实际操作是交叉-关联性`cross-correlation` (唯一的区别是过滤器`filter`是为了卷积而翻转，而不是为了交叉关联)。

本层的可自优化权重的实现，依赖于交叉-关联`cross-correlation` 一个表示权重的过滤器filter (kernel)。

向后传播的函数`backward`计算的是输入数据的梯度以及过滤器的梯度。

```python
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
        # 上一行可以等效表示为:
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

```

**示例:**

```python
module = ScipyConv2d(3, 3)
print("Filter and bias: ", list(module.parameters()))
input = torch.randn(10, 10, requires_grad=True)
output = module(input)
print("Output from the convolution: ", output)
output.backward(torch.randn(8, 8))
print("Gradient for the input map: ", input.grad)

```

输出：

```python
Filter and bias:  [Parameter containing:
tensor([[-0.8330,  0.3568,  1.3209],
        [-0.5273, -0.9138, -1.0039],
        [-1.1179,  1.3722,  1.5137]], requires_grad=True), Parameter containing:
tensor([[0.1973]], requires_grad=True)]
Output from the convolution:  tensor([[-0.7304, -3.5437,  2.4701,  1.0625, -1.8347,  3.3246,  2.5547, -1.1341],
        [-5.0441, -7.1261,  2.8344,  2.5797, -2.4117, -1.4123, -0.2520, -3.1231],
        [ 1.2296, -0.7957,  1.9413,  1.5257,  0.2727,  6.2466,  2.3363,  2.1833],
        [-2.6944, -3.3933,  2.3844,  0.2523, -2.0322, -3.1275, -0.2472,  1.5382],
        [ 3.6807, -1.1985, -3.9278,  0.8025,  3.3435,  6.6806,  1.1656,  1.3711],
        [-1.7426,  1.3875,  8.2674, -0.8234, -4.7534,  3.0932,  1.3048,  2.1184],
        [ 0.2095,  1.3225,  0.9022,  3.3324,  0.8768, -5.3459, -1.0970, -4.5304],
        [ 2.1688, -1.7967, -0.5568, -9.3585,  0.3259,  5.4264,  2.8449,  6.8120]],
       grad_fn=<ScipyConv2dFunctionBackward>)
Gradient for the input map:  tensor([[ 7.7001e-01, -2.6786e-02, -1.0917e+00, -4.1148e-01,  2.2833e-01,
         -1.7494e+00, -1.4960e+00,  2.3307e-01,  2.2004e+00,  3.1210e+00],
        [ 7.0960e-02,  1.8954e+00,  2.0912e+00, -1.3058e+00, -6.1822e-02,
          3.8630e+00, -5.1720e-01, -6.9586e+00, -2.5478e+00, -1.4459e+00],
        [ 9.3677e-01, -7.5248e-01,  3.0795e-03, -2.1788e+00, -2.6326e+00,
         -3.4089e+00,  2.2524e-01,  4.7127e+00,  3.7717e+00,  2.0393e+00],
        [-2.0010e+00,  2.7616e+00,  4.0060e+00, -2.0298e+00,  1.6074e+00,
          2.3062e+00, -5.4927e+00, -5.3029e+00,  3.5081e+00,  4.5952e+00],
        [ 3.4492e-01, -2.3043e+00, -1.5235e+00, -3.3520e+00, -1.3291e-01,
          1.4629e+00,  1.9298e+00,  4.5369e-01, -1.5986e+00, -2.3851e+00],
        [-2.3929e+00,  5.3965e+00,  5.1353e+00, -1.0269e+00,  2.1031e+00,
         -6.2344e+00, -3.6539e+00, -1.7951e+00, -5.6712e-01,  8.6987e-01],
        [ 1.1006e-01, -1.5961e+00,  1.2179e+00,  3.4799e-01, -7.1710e-01,
          2.5705e+00,  4.5020e-01,  3.8066e+00,  4.8558e+00,  2.1423e+00],
        [-9.9457e-01,  1.5614e+00,  1.3985e+00,  3.6700e+00, -1.9708e+00,
         -2.4845e+00,  2.5387e+00, -1.2250e+00, -4.6877e+00, -3.3492e+00],
        [-4.5289e-01,  2.4210e+00,  3.3681e+00, -2.7785e+00,  1.5472e+00,
         -5.0358e-01, -9.7416e-01,  1.1032e+00,  2.0812e-01,  8.2830e-01],
        [ 1.1052e+00, -2.5233e+00,  2.0461e+00,  1.1886e-01, -4.8352e+00,
          2.4197e-01, -1.5177e-01, -6.9245e-01, -1.8357e+00, -1.5302e+00]])

```

**梯度检查:**

```python
from torch.autograd.gradcheck import gradcheck

moduleConv = ScipyConv2d(3, 3)

input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
print("Are the gradients correct: ", test)

```

输出：

```python
Are the gradients correct:  True

```
