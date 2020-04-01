# 用 numpy 和 scipy 创建扩展

> **作者** ：[Adam Paszke](https://github.com/apaszke)
>
>**修订者**: [Adam Dziedzic](https://github.com/adam-dziedzic)
>
> 译者：[Foxerlee](https://github.com/FoxerLee)、[cangyunye](https://github.com/cangyunye)
> 
> 校验：[Foxerlee](https://github.com/FoxerLee)、[FontTian](https://github.com/fonttian)

在本教程中，我们需要完成两个任务：

1. 创建一个无参数神经网络层。
	- 这里需要调用 **numpy** 包作为实现的一部分。

2.  创建一个权重自主优化的神经网络层。
	- 这里需要调用 **Scipy** 包作为实现的一部分。

```python
import torch
from torch.autograd import Function

```

## 无参数神经网络层示例

该层并没有做任何有用的或数学上正确的事情。

它只是被恰当的命名为 BadFFTFunction

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

# 由于本层没有任何参数，我们可以简单的声明为一个函数，
# 而不是当做 nn.Module 类

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
tensor([[ 0.4073, 11.6080,  7.4098, 18.1538,  3.4384],
        [ 4.9980,  3.5935,  6.9132,  3.8621,  6.1521],
        [ 5.2876,  6.2480,  9.3535,  5.1881,  9.5353],
        [ 4.5351,  2.3523,  6.9937,  4.2700,  2.6574],
        [ 0.7658,  7.8288,  3.9512,  5.2703, 15.0991],
        [ 4.5351,  4.9517,  7.7959, 17.9770,  2.6574],
        [ 5.2876, 11.0435,  4.1705,  0.9899,  9.5353],
        [ 4.9980, 11.1055,  5.8031,  3.1775,  6.1521]],
       grad_fn=<BadFFTFunctionBackward>)
tensor([[-1.4503, -0.6550,  0.0648,  0.2886,  1.9357, -1.2299, -1.7474,  0.6866],
        [-0.2466, -1.0292,  0.3109, -0.4289, -0.3620,  1.1854, -1.3372, -0.2717],
        [ 0.0828,  0.9115,  0.7877, -0.5776,  1.6676, -0.5576, -0.2321, -0.3273],
        [ 0.1632,  0.3835,  0.5422, -0.9144,  0.2871,  0.1441, -1.8333,  1.4951],
        [-0.2183, -0.5220,  0.9151,  0.0540, -1.0642,  0.4409,  0.7906, -1.2262],
        [ 0.4039,  0.3374,  1.0567, -0.8190,  0.7870, -0.6152, -0.2887,  1.3878],
        [ 1.6407,  0.0220,  1.4984, -1.9722,  0.3797, -0.0180, -0.7096, -0.2454],
        [ 0.7194,  2.3345, -0.0780, -0.2043, -0.4576, -0.9087, -2.4926,  0.9283]],
       requires_grad=True)

```

## 参数化示例

在深度学习的文献中，这一层被误解的称作卷积 `convolution`，尽管该层的实际操作是交叉-关联性 `cross-correlation` (唯一的区别是滤波器 `filter` 是为了卷积而翻转，而不是为了交叉关联)。

本层的可自优化权重的实现，依赖于交叉-关联 `cross-correlation` 一个表示权重的滤波器。

后向传播函数 `backward` 计算的是输入数据的梯度以及滤波器的梯度。

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
tensor([[ 0.6693, -0.2222,  0.4118],
        [-0.3676, -0.9931,  0.2691],
        [-0.1429,  1.8659, -0.7335]], requires_grad=True), Parameter containing:
tensor([[-1.3466]], requires_grad=True)]
Output from the convolution:  tensor([[ 0.5250, -4.8840, -0.5804, -0.4413, -0.2209, -5.1590, -2.2587, -3.5415],
        [ 0.1437, -3.4806,  2.8613, -2.5486, -0.6023,  0.8587,  0.6923, -3.9129],
        [-6.2535,  2.7522, -2.5025,  0.0493, -3.2200,  1.2887, -2.4957,  1.6669],
        [ 1.6953, -0.9312, -4.6079, -0.9992, -1.4760,  0.2594, -3.8285, -2.9756],
        [ 1.2716, -5.1037, -0.2461, -1.1965, -1.6461, -0.6712, -3.1600, -0.9869],
        [-2.0643, -1.1037,  1.0145, -0.4984,  1.6899, -1.2842, -3.5010,  0.8348],
        [-2.6977,  0.7242, -5.2932, -2.1470, -4.0301, -2.8247, -1.4165,  0.0572],
        [-1.1560,  0.8500, -3.5242,  0.0686, -1.9708,  0.8417,  2.1091, -4.5537]],
       grad_fn=<ScipyConv2dFunctionBackward>)
Gradient for the input map:  tensor([[ 0.2475, -1.0357,  0.9908, -1.5128,  0.9041,  0.0582, -0.5316,  1.0466,
         -0.4844,  0.2972],
        [-1.5626,  1.4143, -0.3199, -0.9362,  1.0149, -1.6612, -0.1623,  1.0273,
         -0.8157,  0.4636],
        [ 1.1604,  2.5787, -5.6081,  4.6548, -2.7051,  1.4152,  1.0695, -5.0619,
          1.9227, -1.4557],
        [ 0.8890, -5.4601,  5.3478,  0.3287, -3.0955,  1.7628,  1.3722,  0.9022,
          4.6063, -1.7763],
        [ 0.4180, -1.4749,  1.9056, -6.5754,  1.1695, -0.3068, -2.7579, -1.2399,
         -3.2611,  1.7447],
        [-1.5550,  1.0767,  0.5541,  0.5231,  3.7888, -2.4053,  0.4745,  4.5228,
         -5.2254,  0.7871],
        [ 0.8094,  5.9939, -4.4974,  1.9711, -4.6029, -0.7072,  0.8058, -1.0656,
          1.7967, -0.5905],
        [-1.1218, -4.8356, -3.5650,  2.0387,  0.6232,  1.4451,  0.9014, -1.1660,
         -0.5986,  0.7368],
        [ 0.4346,  3.4302,  5.3058, -3.0440,  1.0593, -3.6538, -1.7829, -0.0543,
         -0.4385,  0.2770],
        [ 0.2144, -2.5117, -2.6153,  1.1894, -0.6176,  1.9013, -0.7186,  0.4952,
          0.6256, -0.3308]])

```

**检查梯度:**

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

脚本的总运行时间：(0分钟 4.128秒）
