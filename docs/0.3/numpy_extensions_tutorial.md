# 使用 numpy 和 scipy 创建扩展

> 译者：[@飞龙](https://github.com/wizardforcel)

**作者**: [Adam Paszke](https://github.com/apaszke)

这个教程中, 我们将完成以下两个任务:

1.  创建不带参数的神经网络层

    &gt; *   这会调用 **numpy**, 作为其实现的一部分

2.  创建带有可学习的权重的神经网络层

    &gt; *   这会调用 **SciPy**, 作为其实现的一部分

```py
import torch
from torch.autograd import Function
from torch.autograd import Variable

```

## 无参示例

这一层并不做任何有用的, 或者数学上正确的事情.

它被恰当地命名为 BadFFTFunction

**层的实现**

```py
from numpy.fft import rfft2, irfft2

class BadFFTFunction(Function):

    def forward(self, input):
        numpy_input = input.numpy()
        result = abs(rfft2(numpy_input))
        return torch.FloatTensor(result)

    def backward(self, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return torch.FloatTensor(result)

# 由于这一层没有任何参数, 我们可以
# 仅仅将其声明为一个函数, 而不是 nn.Module 类

def incorrect_fft(input):
    return BadFFTFunction()(input)

```

**所创建的层的使用示例:**

```py
input = Variable(torch.randn(8, 8), requires_grad=True)
result = incorrect_fft(input)
print(result.data)
result.backward(torch.randn(result.size()))
print(input.grad)

```

## 参数化示例

它实现了带有可学习的权重的层.

它使用可学习的核, 实现了互相关.

在深度学习文献中, 它容易和卷积混淆.

反向过程计算了输入和滤波的梯度.

**实现:**

_要注意, 实现作为一个演示, 我们并不验证它的正确性_

```py
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class ScipyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter):
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        ctx.save_for_backward(input, filter)
        return torch.FloatTensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        input, filter = ctx.saved_tensors
        grad_output = grad_output.data
        grad_input = convolve2d(grad_output.numpy(), filter.t().numpy(), mode='full')
        grad_filter = convolve2d(input.numpy(), grad_output.numpy(), mode='valid')

        return Variable(torch.FloatTensor(grad_input)), \
            Variable(torch.FloatTensor(grad_filter))

class ScipyConv2d(Module):

    def __init__(self, kh, kw):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(kh, kw))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter)

```

**示例用法:**

```py
module = ScipyConv2d(3, 3)
print(list(module.parameters()))
input = Variable(torch.randn(10, 10), requires_grad=True)
output = module(input)
print(output)
output.backward(torch.randn(8, 8))
print(input.grad)

```
