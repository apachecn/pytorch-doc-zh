# 自定义 C++ 和 CUDA 扩展

> **作者** ：[彼得戈尔兹伯勒](https://www.goldsborough.me/)
>
> 译者：[Foxerlee](https://github.com/FoxerLee)
>
> 校验：[Foxerlee](https://github.com/FoxerLee)

PyTorch 提供了大量与神经网络、随机张量代数(arbitrary tensor algebra）、数据整合(data wrangling）以及其他目的相关的操作。但是，您仍然可能会发现自己需要更多自定义操作。例如，您可能想使用在论文中发现的新的激活函数，或者实现您在研究过程中所开发的新的运算。

在 PyTorch 中整合这样的自定义操作最简单的方法是利用 Python 编写扩展的`函数(Funciton）`和`模型(Module）`，如[此处](https://pytorch.apachecn.org/docs/1.2/notes/extending.html)所描写的那样。这让您可以充分地利用自动微分(automatic differentiation）(使你不需要自己编写派生函数）与 Python 在通常情况下的表现力。然而，在有些时候您的一些操作可以使用 C++ 以获得更佳的效果。比如，您的代码在模型当中会被*十分* 频繁地调用，或者即便调用次数较少也会带来昂贵的开销。另一个可能的原因是您的代码依赖于一些 C 和 C++ 库，或者需要与它们交互。为了解决这种情况，PyTorch 提供了一种非常简单的编写自定义 C++ *扩展* 的方法。

C++ 扩展是一种我们开发的以允许用户(您）创建一些*包含的资源* 之外的 PyTorch 运算符，例如，与 PyTorch 后端分离开来。此方法与原生的 PyTorch 操作的实现方式不同。C++ 扩展旨在为您提供大量与 PyTorch 后端集成在一起相关的样板(boilerplate），同时为基于 PyTorch 的项目提供高度的灵活性。但是，一旦将操作定义为 C++ 扩展，将其转换为原生 PyTorch 函数在很大程度上取决于您的代码组织结构，如果您决定在较早阶段进行操作，则可以解决这个问题。


## 动机和例子

本篇文章的其余部分将逐步介绍一个编写和使用 C++(和CUDA）扩展的实际示例。如果您一直在被催促，或者在今天结束前仍未完成该扩展您就会被开除，那么可以跳过本节，直接进入下一部分的实施细节。

假设您想出了一种新型的循环单元，与现有技术相比，它具有更好的性能。该循环单元与 LSTM 相似，但不同之处在于，它没有*遗忘门*，并使用*指数线性单元*(ELU）作为其内部激活函数。由于此单元永远不会忘记，因此我们将其称为 *LLTM* 或*长长期记忆(Long-Long-Term-Memory）*单元。

由于 LLTM 和 LSTM 两者的区别过于明显，以至于我们不能通过修改 PyTorch 中的 `LSTMCell` 来实验我们的目标，因此我们需要创建一个自定义单元。解决这个问题的第一种也是最简单的一种 -- 并且在所有情况下都是最好的一步 -- 是使用 Python 在原生的 PyTorch 中实现我们所需的功能。为此，我们需要继承 `torch.nn.Module` 并实现LLTM的前向传播。 代码如下：


```python
 class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell

```

单元的调用方式如预期那样：

```python
import torch

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

new_h, new_C = rnn(X, (h, C))

```

当然，如果可能的话，您应该使用如下方法扩展 PyTorch。由于 PyTorch 在 [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)，[Intel MKL](https://software.intel.com/en-us/mkl) 或 [NNPACK](https://github.com/Maratyszcza/NNPACK) 等库的支持下对其 CPU 和 GPU 的操作进行了高度优化的实现，因此前述的 PyTorch 代码通常足够快。但是，我们还是可以发现，在某些情况下为什么性能仍然有进一步改进的空间。最明显的原因是 PyTorch 不了解您要实现的算法。它仅知道您用于组成算法的单个操作。因此，PyTorch 必须逐个执行您的操作。由于对操作的实现(或*内核*）的每个单独调用(可能涉及启动CUDA内核）都具有一定的开销，因此该开销在许多函数调用中可能变得十分明显。此外，运行我们代码的 Python 解释器本身也可能会使我们的程序变慢。

一种明显的加速方法是用 C++(或CUDA）重写这部分代码并*融合* 特定的操作组。 融合是指将许多函数的实现组合到一个函数中，这可以从两个方面受益：更少的内核启动，以及在提高全局数据流可见性的情况下执行的其他优化。

让我们看看如何使用 C++ 扩展来实现 LLTM 的*融合* 版本。我们将从使用支持 PyTorch 大部分后端功能的 [ATen](https://github.com/zdevito/ATen) 库以原生 C++ 编写代码开始，然后看看它是如何让我们轻松转换 Python 代码的。然后，我们将模型的各个部分移至 CUDA 内核，以从 GPU 提供的大规模并行处理中受益，从而进一步加快处理速度。

## 编写一个 C++ 扩展

C++ 扩展有两种形式：可以使用 `setuptools` “提前”构建，也可以通过 `torch.utils.cpp_extension.load()` “即时”构建。 我们将从第一种方法开始，稍后再讨论后者。

### 使用 `setuptools` 进行构建

为了实现“提前”构建，我们编写一个 `setup.py` 脚本来构建 C++ 扩展，其使用 setuptools 来编译我们的 C++ 代码。对于 LLTM，脚本十分简单，如下所示：

```python
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
```
    
在这部分代码中，`CppExtension` 是 `setuptools.Extension` 的一个便利的包装器(wrapper），它传递正确的引用路径，并且将扩展包语言设置为 c++。等效的泛化版 `setuptools` 简单代码如下所示：

```python
Extension(
   name='lltm_cpp',
   sources=['lltm.cpp'],
   include_dirs=cpp_extension.include_paths(),
   language='c++')
```
    
`BuildExtension` 执行并检查许多必需的配置步骤，并且在混合使用 C++ / CUDA 扩展的情况下管理混合编译。这就是我们目前真正需要了解的有关构建 C++ 扩展的全部信息！现在让我们看一下 `lltm.cpp` 中的 C++ 扩展的实现。

### 编写 c++ 操作

现在让我们开始利用 c++ 实现 LLTM！我们后向传播需要的一个函数是 Sigmoid 的导数。 这是一小段代码，用于讨论编写 C++ 扩展时可供我们使用的总体环境：


```cpp
#include <torch/extension.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}
```
    
`<torch / extension.h>` 是一站式(one-stop）头文件，其中包括编写 C++ 扩展所有必需的 PyTorch 扩展。 这包括：

  * ATen 库，它是我们张量计算的主要 API，
  * [pybind11](https://github.com/pybind/pybind11)，用于实现我们的 C++ 代码的 Python 衔接方法，
  * 其他管理 ATen 和 pybind11 交互细节的头文件。

`d_sigmoid()` 的实现展示了如何使用 ATen API。PyTorch 的张量和变量接口是由 ATen 库自动生成的，因此我们可以或多或少地实现将 Python 以 1：1 的形式转换为 C++。我们用于所有计算的主要数据类型将是 `torch::Tensor`。它的完整 API 可以在[这里](https://pytorch.org/cppdocs/api/library_root.html)查到。注意，我们可以包含 `<iostream>` 或任何*其他 C 或 C++ 头文件* -- 我们可以使用 C++11 的全部功能。

#### 前向传播

接下来，我们可以将整个前向传播部分移植为 C++ 代码：

```cpp  
#include <vector>

std::vector<at::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}
```    

#### 后向传播

C++ 扩展 API 当前不提供为我们自动生成后向传播函数的方法。因此，我们必须要自己实现 LLTM 的后向传播，其将计算每个前向传播的输入的导数。最终，我们前向传播和后向传播函数加入 `torch.autograd.Function` 中以建立一个不错的 Python 衔接。后向传播的复杂度较高，因此我们不深入研究代码(如果您感兴趣，可以阅读 [Alex Graves 的论文](https://www.cs.toronto.edu/~graves/phd.pdf)，以获得更多有关此方面的信息：

```cpp
// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}
``` 
    
     

### 衔接到 Python

一旦您用 C++ 和 ATen 编写了计算，可以使用 pybind11 以非常简单的方式将 C++ 函数或类衔接到 Python 中。关于 PyTorch 的 C++ 扩展的这一部分的问题或疑问您可以参考 [pybind11](https://pybind11.readthedocs.io/en/master/) 文档来解决。

    
```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}
```
    
这里要注意的一点是宏 `TORCH_EXTENSION_NAME`。torch 的扩展程序构建会将其定义为您在 setup.py 脚本中为扩展程序指定的名称。在本教程中，`TORCH_EXTENSION_NAME` 的值为 “lltm”。这是为了避免在两个位置(构建脚本和您的 C++ 代码）都维护扩展名，因为两者之间的不匹配会导致令人讨厌且难以跟踪的问题。

### 使用您的扩展

现在，我们准备将扩展名导入 PyTorch 中。 此时，目录结构可能如下所示：

```
pytorch/
  lltm-extension/
    lltm.cpp
    setup.py
```

现在，运行 `python setup.py install` 安装你的扩展。终端的输入应该如下：

    
```bash
running install
running bdist_egg
running egg_info
creating lltm_cpp.egg-info
writing lltm_cpp.egg-info/PKG-INFO
writing dependency_links to lltm_cpp.egg-info/dependency_links.txt
writing top-level names to lltm_cpp.egg-info/top_level.txt
writing manifest file 'lltm_cpp.egg-info/SOURCES.txt'
reading manifest file 'lltm_cpp.egg-info/SOURCES.txt'
writing manifest file 'lltm_cpp.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_ext
building 'lltm_cpp' extension
creating build
creating build/temp.linux-x86_64-3.7
gcc -pthread -B ~/local/miniconda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I~/local/miniconda/lib/python3.7/site-packages/torch/include -I~/local/miniconda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -I~/local/miniconda/lib/python3.7/site-packages/torch/include/TH -I~/local/miniconda/lib/python3.7/site-packages/torch/include/THC -I~/local/miniconda/include/python3.7m -c lltm.cpp -o build/temp.linux-x86_64-3.7/lltm.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=lltm_cpp -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11
cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
creating build/lib.linux-x86_64-3.7
g++ -pthread -shared -B ~/local/miniconda/compiler_compat -L~/local/miniconda/lib -Wl,-rpath=~/local/miniconda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/lltm.o -o build/lib.linux-x86_64-3.7/lltm_cpp.cpython-37m-x86_64-linux-gnu.so
creating build/bdist.linux-x86_64
creating build/bdist.linux-x86_64/egg
copying build/lib.linux-x86_64-3.7/lltm_cpp.cpython-37m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
creating stub loader for lltm_cpp.cpython-37m-x86_64-linux-gnu.so
byte-compiling build/bdist.linux-x86_64/egg/lltm_cpp.py to lltm_cpp.cpython-37.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm_cpp.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm_cpp.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm_cpp.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying lltm_cpp.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
zip_safe flag not set; analyzing archive contents...
__pycache__.lltm_cpp.cpython-37: module references __file__
creating 'dist/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing lltm_cpp-0.0.0-py3.7-linux-x86_64.egg
removing '~/local/miniconda/lib/python3.7/site-packages/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg' (and everything under it)
creating ~/local/miniconda/lib/python3.7/site-packages/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg
Extracting lltm_cpp-0.0.0-py3.7-linux-x86_64.egg to ~/local/miniconda/lib/python3.7/site-packages
lltm-cpp 0.0.0 is already the active version in easy-install.pth

Installed ~/local/miniconda/lib/python3.7/site-packages/lltm_cpp-0.0.0-py3.7-linux-x86_64.egg
Processing dependencies for lltm-cpp==0.0.0
Finished processing dependencies for lltm-cpp==0.0.0
```
    

关于编译器的一个小注意事项：由于 ABI 版本问题，用于构建 C++ 扩展的编译器必须与 ABI 兼容，并且这里的编译器是必须是与构建 PyTorch 时采用的编译器一样的。实际上，这意味着您必须在 Linux 上使用 GCC 4.9 及更高版本。 对于 Ubuntu 16.04 和其他较新的 Linux 发行版，这应该已经是默认的编译器。 在最坏的情况下，您可以使用编译器从源代码构建 PyTorch ，然后使用相同的编译器构建扩展。

扩展程序构建完成后，您可以使用在 `setup.py` 脚本中指定的名称，简单地将其导入 Python。只需要确保优先调用 `import torch`，因为这将解析一些动态链接器必须能够看到的标志：
  
```python
In [1]: import torch
In [2]: import lltm_cpp
In [3]: lltm_cpp.forward
Out[3]: <function lltm.PyCapsule.forward>
```
    
 
如果我们对函数或者模块调用 `help()` 函数，我们可以看到，其签名符合我们的 C++ 代码：

    
```python  
In[4] help(lltm_cpp.forward)
forward(...) method of builtins.PyCapsule instance
    forward(arg0: torch::Tensor, arg1: torch::Tensor, arg2: torch::Tensor, arg3: torch::Tensor, arg4: torch::Tensor) -> List[torch::Tensor]

    LLTM forward
```   

由于我们现在能够从 Python 中调用我们的 C++ 函数，我们可以使用 `torch.autograd.Function`和 `torch.nn.Module` 来包装(warp）它们，使它们成为 PyTorch 中的最顶层的类(first class citizens，关键的一部分）：
    
```python
import math
import torch

# Our module!
import lltm_cpp

class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
```

#### 性能比较

现在我们可以使用 PyTorch 调用 C++ 函数，我们可以运行一个小的基准测试，以查看通过用 C++ 重写函数获得的性能。我们将调用 LLTM 的前向传播和后向传播函数几次，并且记录耗时：

    
```python
import time

import torch

batch_size = 16
input_features = 32
state_size = 128

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))
```
    

如果我们使用本文开头用原生 Python 编写的原始 LLTM 来运行此代码，则会得到以下结果(在我的机器上）：

```
Forward: 506.480 us | Backward 444.694 us
```
    

而我们的新的 C++ 版本结果：

    
```    
Forward: 349.335 us | Backward 443.523 us
``` 

我们可以看到前向传播已经有一个明显的速度提升(超过 30%）。对于后向传播，速度提升是可见的，尽管并不是最明显的那个。我上面所写的后向传播没有特别优化，可以肯定代码仍然能够改进。另外，PyTorch 的自动微分引擎可以自动并行化计算图，可以是一个更高效的操作流，并且也可以用 C++ 实现，因此可以预见我们的代码速度能够更快。当然，这已经是一个很好的开始了。

#### GPU 设备上的性能

关于 PyTorch 的 ATen 后端的一个美妙事实是，它可以抽象化您正在运行的计算设备。这意味着我们为 CPU 编写的相同代码也可以在 GPU 上运行，并且各个操作将相应地分派到 GPU 优化的实现。对于某些运算，如矩阵乘法(例如 `mm` 或者 `addmm`），这将会是一个很大的提升。让我们看一下使用 CUDA 张量运行 C++ 代码所获得的性能。 无需更改实现，我们只需要将张量在 Python 中加入 GPU 内存，即可在开始时添加 `device = cuda_device` 参数，或者在创建后使用 `.to(cuda_device)`。

    
```python
import torch

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

batch_size = 16
input_features = 32
state_size = 128

# Note the device=cuda_device arguments here
X = torch.randn(batch_size, input_features, device=cuda_device)
h = torch.randn(batch_size, state_size, device=cuda_device)
C = torch.randn(batch_size, state_size, device=cuda_device)

rnn = LLTM(input_features, state_size).to(cuda_device)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    torch.cuda.synchronize()
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    torch.cuda.synchronize()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))
```

再次将原始的 PyTorch 代码与 C++ 版本(现在都在 CUDA 设备上运行）进行比较，我们又看到了性能提升。 对于 Python / PyTorch：  
    
```
Forward: 187.719 us | Backward 410.815 us
```
    

而 C++ / ATen：

```
Forward: 149.802 us | Backward 393.458 us
```
    

与非 CUDA 代码相比，这可以大大提高整体速度。但是，通过编写自定义 CUDA 内核，我们可以利用 C++ 获得更多性能，我们很快将深入其中。 

### JIT 编译扩展

在此之前，让我们讨论构建 C++ 扩展的另一种方法。在介绍了前者之后，让我们详细介绍后者。在介绍了前者之后，让我们详细介绍后者。JIT 编译机制通过调用 PyTorch API 中一个称为 `torch.utils.cpp_extension.load()` 的简单函数，为您提供了一种动态编译和加载扩展的方式。 对于LLTM，这看起来像这样简单：

```python
from torch.utils.cpp_extension import load

lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"])
```



在这里，我们为函数提供与 `setuptools` 相同的信息。 在后端，其将执行以下操作：

1. 创建一个临时目录 `/tmp/torch_extensions/lltm`，
2. 将 [Ninja](https://ninja-build.org/) 构建文件发送到该临时目录中，
3. 将您的源文件编译到共享库中，
4. 将此共享库导入为 Python 模块。

实际上，如果将变量 `verbose = True` 传递给 `cpp_extension.load()`，该进程会在运行过程中告诉你：

```bash
Using /tmp/torch_extensions as PyTorch extensions root...
Emitting ninja build file /tmp/torch_extensions/lltm_cpp/build.ninja...
Building extension module lltm_cpp...
Loading extension module lltm_cpp...
```

生成的 Python 模块将与 `setuptools` 生成的模块完全相同，但是避免了必须维护单独的 `setup.py` 构建文件的要求。如果您的设置更加复杂，并且确实需要 `setuptools` 的全部功能，你的确可以编写自己的 `setup.py` -- 但在许多情况下，这种 JIT 技术就足够了。第一次运行此行时，将需要一些时间，因为扩展程序是在后台编译的。由于我们使用 Ninja 构建系统来构建您的源代码，重新编译是以增量的形式，因此如果您不更改扩展程序的源文件，那您第二次运行 Python 模块重新加载扩展程序时会十分快捷，开销很低。 

## 编写一个 C++/CUDA 混合扩展

为了将我们的实现提升到一个新的水平，我们可以使用自定义 CUDA 内核来手写向前和向后传递的部分内容。对于 LLTM，其提升空间将会十分明显，因为 LLTM 有大量按顺序进行的逐点计算，所有这些计算都可以在单个 CUDA 内核中融合和并行化。 让我们看看如何编写这种 CUDA 内核，并使用此扩展机制将其与 PyTorch 集成。 

编写 CUDA 扩展的一般策略是先编写一个 C++ 文件，该文件定义将从 Python 调用的函数，然后使用 pybind11 将这些函数衔接到 Python。此外，此文件还将声明在CUDA(`.cu`）文件中定义的函数。然后，C++ 函数将进行一些检查，并最终将其调用转发给 CUDA 函数。在 CUDA 文件中，我们编写实际的 CUDA 内核。然后，`cpp_extension` 包将负责使用 `gcc` 之类的 C++ 编译器来编译 C++ 源代码，并使用 NVIDIA 的 `nvcc` 编译器来编译 CUDA 源代码。这样可以确保每个编译器都编译地它最了解的文件。最终，它们将被链接到一个共享库中，该库可以从 Python 代码中获得。

我们将从 C++ 文件开始，我们将其称为 `lltm_cuda.cpp`，例如：  
    
```cpp
#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> lltm_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(old_h);
  CHECK_INPUT(old_cell);

  return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);

  return lltm_cuda_backward(
      grad_h,
      grad_cell,
      new_cell,
      input_gate,
      output_gate,
      candidate_cell,
      X,
      gate_weights,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward (CUDA)");
  m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
```    

正如您所看到的，它主要是样板文件，检查并转发到我们将在 CUDA 文件中定义的功能。我们将其命名为 `lltm_cuda_kernel.cu`(注意扩展名为 `.cu`！）NVCC 可以聪明地编译 C++11，因此我们仍然可以使用 ATen 和 C++ 标准库(但不提供 `torch.h`）。请注意， `setuptools` 无法处理具有相同名称但扩展名不同的文件，因此，如果您使用 `setup.py` 方法而不是 JIT 方法，则必须为 CUDA 文件指定与 C++ 文件不同的名称(对于 JIT 方法则可以正常区分 `lltm.cpp` 和 `lltm.cu`）。让我们简单看一下该文件： 
    
```cpp
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}
```
    

在这里，我们看到了刚刚提到的头文件，以及我们正在使用的特定于 CUDA 的声明(例如 `__device__` 和 `__forceinline__`）以及函数(例如 `exp`）。让我们继续添加一些我们需要的辅助函数：

    
```cpp
template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}
```    

现在，要真正实现一个函数，我们还需要两个函数：一个函数执行我们不希望手工编写并调用 CUDA 内核的操作，另一个是要加速部分的实际 CUDA 内核。对于前向传播，第一个函数应如下所示：

    
```cpp
std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gates = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.data<scalar_t>(),
        old_cell.data<scalar_t>(),
        new_h.data<scalar_t>(),
        new_cell.data<scalar_t>(),
        input_gate.data<scalar_t>(),
        output_gate.data<scalar_t>(),
        candidate_cell.data<scalar_t>(),
        state_size);
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}
```    

这里的主要关注点是 `AT_DISPATCH_FLOATING_TYPES` 宏和内核启动代码(由 `<<< ... >>>` 表示）。尽管 ATen 提取了我们处理的张量的设备和数据类型，但张量在运行时仍将由具体设备上的具体类型的内存支持。因此，我们需要一种在运行时确定张量是什么类型，然后有选择地调用具有相应正确类型签名的函数的方法。手动完成后，(在概念上）将如下所示：

```cpp   
switch (tensor.type().scalarType()) {
  case torch::ScalarType::Double:
    return function<double>(tensor.data<double>());
  case torch::ScalarType::Float:
    return function<float>(tensor.data<float>());
  ...
}
```
    

`AT_DISPATCH_FLOATING_TYPES` 的目的是为我们处理此调度。它需要一个类型(在我们的例子中是 `gates.type()`），一个名称(用于错误消息）和一个 lambda 函数。在此 lambda 函数中，类型别名 `scalar_t` 可用，并且定义为该上下文中张量实际上在运行时的类型。这样，如果我们有一个模板函数(CUDA 内核将会使用的），则可以使用此 `scalar_t` 别名实例化它，然后将调用正确的函数。在这种情况下，我们还希望检索张量的数据指针作为该 `scalar_t` 类型的指针。 如果想分派所有类型，而不仅是浮点类型(`Float` 和 `Double`），则可以使用 `AT_DISPATCH_ALL_TYPES`。 

请注意，我们使用基本的 ATen 执行一些操作。 这些操作仍将在 GPU 上运行，但使用 ATen 的默认实现。这是有道理的，因为 ATen 将使用高度优化的例程来处理矩阵乘法(例如 `addmm`）或卷积之类的操作，而这些将很难由我们自己实现和改善。

至于内核启动本身，我们在这里指定每个 CUDA 块将具有1024个线程，并且整个 GPU 网格被分成尽可能多的 `1x1024` 线程块，并以一组一个线程的方式填充我们的矩阵。例如，如果我们的状态大小为 2048，批处理大小为 4，则我们将以每个 1024 个线程启动，总共 `4x2=8` 个块。 如果您以前从未听说过 CUDA 的“块”或“网格”，那么这篇[有关 CUDA 的介绍性阅读](https://devblogs.nvidia.com/even-easier-introduction-cuda/)可能会有所帮助。 

实际的 CUDA 内核十分简单(如果您以往编写过 GPU）：

```cpp
template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ old_cell,
    scalar_t* __restrict__ new_h,
    scalar_t* __restrict__ new_cell,
    scalar_t* __restrict__ input_gate,
    scalar_t* __restrict__ output_gate,
    scalar_t* __restrict__ candidate_cell,
    size_t state_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    input_gate[index] = sigmoid(gates[gates_row + column]);
    output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
    candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
    new_cell[index] =
        old_cell[index] + candidate_cell[index] * input_gate[index];
    new_h[index] = tanh(new_cell[index]) * output_gate[index];
  }
}
```  
这里有趣的内容是，我们能够让门矩阵中的每个单独组件完全并行地计算所有逐点操作。想象一下，如果要用一个巨型的 `for` 循环遍历一百万个元素来完成这个操作，您就可以明白为什么这样做会更快了。

### 使用存取器(accessors）

您可以在 CUDA 内核中看到，我们直接处理具有正确类型的指针。实际上，直接在 cuda 内核内部使用高级类型不可知张量是非常低效的。

但是，这是以易于使用和可读性为代价的，特别是对于高维数据。在我们的示例中，我们知道连续门张量具有3个维度：

  1. 批处理，大小为 `batch_size`，步长为 `3*state_size`
  2. 行，大小为 3，步长为 `state_size`
  3. 索引，大小为 `state_size`，步长为 1

那么，我们如何访问内核内部的元素 `gates[n][row][column]`？事实上，您只需要通过一些简单的算法，就可以利用步长访问您的元素。

```cpp
gates.data<scalar_t>()[n*3*state_size + row*state_size + column]
```
    
除了冗长之外，此表达式还需要明确知道步长的值，并且通过参数将其传递给内核函数。 您会发现，在内核函数接受具有不同大小的多个张量的情况下，您将得到很长的参数列表。

幸运的是，ATen 提供了一种可以动态检查张量类型和维度的存取器。它利用一个 API，可以有效地访问张量元素，而不需要转换为单个指针：

```cpp
torch::Tensor foo = torch::rand({12, 12});

// assert foo is 2-dimensional and holds floats.
auto foo_a = foo.accessor<float,2>();
float trace = 0;

for(int i = 0; i < foo_a.size(0); i++) {
  // use the accessor foo_a to get tensor data.
  trace += foo_a[i][i];
}
```
    

访问器对象具有相对较高级别的接口，如 `.size()` 和 `.srtide()` 方法和多维索引。`.accseeor<>` 旨在在 cpu 张量上有效地访问数据。针对 cuda 张量的等效函数是 `packed_accessor64<>` 和 `packed_accessor32<>`，它们分别提供具有 64 位或 32 位整数索引的打包的存取器。

与普通的存取器的根本区别在于，打包的存取器在其结构内部复制大小和跨度数据，而不是指向它。它允许我们将其传递给 CUDA 内核函数并在其中使用其接口。

我们可以设计一个使用打包的存取器而不是指针的函数。   
    
```cpp
__global__ void lltm_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell)
```
    

让我们分解一下这里使用的模板。前两个参数 `scalar_t` 和 `2` 与常规存取器相同。参数 `torch::RestrictPtrTraits` 指示必须使用 `__restrict__` 关键字。请注意，我们使用了 `PackedAccessor32` 变量，该变量将大小和步长存储为 `int32_t` 类型。这一点很重要，因为使用 64 位变量(`PackedAccessor64`）会使内核变慢。

该函数声明变成了

    
```cpp
template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < gates.size(2)){
    input_gate[n][c] = sigmoid(gates[n][0][c]);
    output_gate[n][c] = sigmoid(gates[n][1][c]);
    candidate_cell[n][c] = elu(gates[n][2][c]);
    new_cell[n][c] =
        old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
    new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
  }
}
```
    
该实现更具可读性！然后我们可以通过在主函数内使用 `.packed_accessor32<>` 方法创建打包的存取器来调用此函数。

    
```cpp
std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto gates = gate_weights.reshape({batch_size, 3, state_size});
  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}
```    

向后传播遵循几乎相同的模式，我就不再进一步阐述：

    
```cpp
template <typename scalar_t>
__global__ void lltm_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_old_cell,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_cell,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gate_weights) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < d_gates.size(2)){
    const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
    const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
    const auto d_new_cell =
        d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


    d_old_cell[n][c] = d_new_cell;
    const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
    const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

    d_gates[n][0][c] =
        d_input_gate * d_sigmoid(gate_weights[n][0][c]);
    d_gates[n][1][c] =
        d_output_gate * d_sigmoid(gate_weights[n][1][c]);
    d_gates[n][2][c] =
        d_candidate_cell * d_elu(gate_weights[n][2][c]);
  }
}

std::vector<torch::Tensor> lltm_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gates,
    torch::Tensor weights) {
  auto d_old_cell = torch::zeros_like(new_cell);
  auto d_gates = torch::zeros_like(gates);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        d_gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        grad_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));

  auto d_gate_weights = d_gates.reshape({batch_size, 3*state_size});
  auto d_weights = d_gate_weights.t().mm(X);
  auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gate_weights.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
```    

### 将 C++/CUDA 操作与 PyTorch 集成

同样，将支持 CUDA 的操作与 PyTorch 集成非常简单。 如果要编写 `setup.py` 脚本，它可能看起来像这样：

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm',
    ext_modules=[
        CUDAExtension('lltm_cuda', [
            'lltm_cuda.cpp',
            'lltm_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```
    

现在，我们使用 `CUDAExtension()` 代替 `CppExtension()`。 我们只需要指定 `.cu` 文件和 `.cpp` 文件，该库将替您处理所有麻烦部分。JIT 机制甚至更简单：

    
```python
from torch.utils.cpp_extension import load

lltm = load(name='lltm', sources=['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])
```
    

#### 性能比较

我们希望并行化与融合我们代码与 CUDA 的逐点操作将改善我们的 LLTM 的性能。让我们看看是否成立。我们可以运行在前面列出的代码来进行基准测试。我们之前的最快版本是基于 CUDA 的 C++ 代码：

```    
Forward: 149.802 us | Backward 393.458 us
```    

现在使用我们的自定义 CUDA 内核：

```    
Forward: 129.431 us | Backward 304.641 us
```    

性能得到了更多的提升！

## 结论

你现在应该对 PyTorch 的 C++ 扩展机制以及使用它们的动机有一个很好的大致上的了解了。你可以在[此处](https://github.com/pytorch/extension-cpp)中找到本文中显示的代码示例。如果你有任何疑问，请使用 [PyTorch 论坛](https://discuss.pytorch.org/)。如果你遇到任何问题，请务必查看我们的 [FAQ](https://pytorch.org/cppdocs/notes/faq.html)。


