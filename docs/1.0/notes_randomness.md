# 再生性

> 译者：[ApacheCN](https://github.com/apachecn)

PyTorch版本，单个提交或不同平台无法保证完全可重现的结果。此外，即使使用相同的种子，也不需要在CPU和GPU执行之间重现结果。

但是，为了在一个特定平台和PyTorch版本上对您的特定问题进行计算确定，需要采取几个步骤。

PyTorch中涉及两个伪随机数生成器，您需要手动播种以使运行可重现。此外，您应该确保您的代码依赖于使用随机数的所有其他库也使用固定种子。

## PyTorch

您可以使用为所有设备(CPU和CUDA）播种RNG：

```
import torch
torch.manual_seed(0)

```

有一些PyTorch函数使用CUDA函数，这些函数可能是非确定性的来源。一类这样的CUDA函数是原子操作，特别是`atomicAdd`，其中对于相同值的并行加法的顺序是未确定的，并且对于浮点变量，是结果中的变化源。在前向中使用`atomicAdd`的PyTorch函数包括，。

许多操作具有向后使用`atomicAdd`，特别是许多形式的池，填充和采样。目前没有简单的方法来避免这些功能中的非确定性。

## CuDNN

在CuDNN后端运行时，必须设置另外两个选项：

```
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

```

警告

确定性模式可能会对性能产生影响，具体取决于您的型号。

## NumPy的

如果您或您使用的任何库依赖于Numpy，您也应该为Numpy RNG播种。这可以通过以下方式完成：

```
import numpy as np
np.random.seed(0)

```