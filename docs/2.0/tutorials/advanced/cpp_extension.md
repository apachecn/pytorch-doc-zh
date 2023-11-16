


# 自定义 C++ 和 CUDA 扩展 [¶](#custom-c-and-cuda-extensions "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/cpp_extension>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/cpp_extension.html>




**作者** 
 :
 [Peter Goldsborough](https://www.goldsborough.me/)


PyTorch 提供了大量与神经网络、任意张量代数、数据整理和其他目的相关的操作。但是，您可能仍然发现
自己需要更加自定义的操作。例如，您可能想要
使用在论文中发现的新颖激活函数，或实现
您在研究中开发的操作。




 在 PyTorch 中集成此类自定义操作的最简单方法是通过扩展 
 `Function`
 和 
 `Module`
 来用 Python 编写它，如概述
 [此处](https://pytorch.org/docs/master/notes/extending.html) 
 。这为您提供了自动微分的全部功能（使您无需编写派生函数）以及 Python 的通常表达能力。然而，有时
您的操作最好用 C++ 实现。例如，您的代码
可能需要
*真正*
快，因为它在您的模型中被非常频繁地调用
而且即使对于很少的调用也非常昂贵。另一个可能的原因是它依赖于其他 C 或 C++ 库或与其他 C 或 C++ 库交互。为了解决此类情况，
PyTorch 提供了一种非常简单的方法来编写自定义
 *C++ 扩展* 
 。




 C++ 扩展是我们开发的一种机制，允许用户（您）创建
定义的 PyTorch 运算符
 *out-of-source* 
 ，即与 PyTorch
 后端分开。此方法与本机 PyTorch 操作的实现方式不同。 C++ 扩展旨在为您节省大量与 PyTorch’s 后端集成操作相关的样板文件，同时为您基于 PyTorch 的项目提供高度的灵活性。
不过，一旦您定义了将您的操作作为 C++ 扩展，
将其转换为原生 PyTorch 函数很大程度上是代码组织的问题，
如果您决定将您的操作贡献给上游，
您可以在事后解决这个问题。





## 动机和示例 [¶](#motivation-and-example "永久链接到此标题")




 本说明的其余部分将介绍编写和使用 C++（和 CUDA）扩展的实际示例。如果您被追赶，或者如果
您在一天结束前没有
’ 完成该操作，有人会解雇您，您可以跳过本节并
直接进入下一节中的实现细节。\ n



 让’s 假设你’s 已经想出了一种新的循环单元，你发现它
与现有技术相比具有更优越的性能。此循环单元
 与 LSTM 类似，但不同之处在于它缺少
 *遗忘门* 
 并使用
 *指数线性单元* 
 (ELU) 作为其内部激活函数。因为
这个单元永远不会忘记，所以我们’ 将其称为
 *LLTM* 
 或
 *Long-Long-Term-Memory* 
 单元。




 LLTM 与普通 LSTM 的两种不同之处足够重要
我们可以\xe2\x80\x99t 配置 PyTorch\xe2\x80\x99s
 `LSTMCell`
 以满足我们的目的，因此我们\xe2\x80 \x99 必须
创建一个自定义单元格。对于这个 \xe2\x80\x93 来说，第一个也是最简单的方法可能在所有情况下都是一个好的第一步 \xe2\x80\x93 是使用 Python 在普通 PyTorch 中实现我们想要的功能。为此，我们需要子类
 [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "(在 PyTorch 中v2.1)")
 并实现 LLTM 的前向传递。这
看起来像这样：






```
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




 然后我们可以按预期使用：






```
import torch

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

new_h, new_C = rnn(X, (h, C))

```




 当然，如果可能并且合理的话，您应该使用这种方法来扩展 PyTorch。由于 PyTorch 对其 CPU 和 GPU 操作进行了高度优化的实现，由 [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) 等库提供支持，
 [Intel MKL](https://software.intel.com/en-us/mkl) 
 或 
 [NNPACK](https://github.com/Maratyszcza/NNPACK) 
 ，像上面这样的 PyTorch 代码将通常
足够快。然而，我们也可以看出为什么在某些情况下，
还有进一步改进性能的空间。最明显的原因是 PyTorch 不了解您正在实现的算法。它只知道
您用来组成算法的各个操作。因此，PyTorch
必须一个接一个地单独执行您的操作。由于对操作的实现（或
 *kernel* 
）的每个
单独调用（可能涉及CUDA内核的启动）都有一定的开销，因此在许多函数调用中，此
开销可能会变得很大。此外，
运行我们代码的 Python 解释器本身也会减慢我们的程序速度。




 因此，加快速度的一个明确方法是用 C++（或
CUDA）重写部分内容和
 *熔断* 
 特定的操作组。融合意味着
将许多函数的实现组合到一个函数中，
这得益于
更少的内核启动以及
我们可以执行的其他优化，
提高了全局数据流的可见性。




 让’s 看看如何使用 C++ 扩展来实现 
 *fused* 
 版本的
LLTM。我们’ 将首先使用普通 C++ 编写它，使用
 [ATen](https://github.com/zdevito/ATen)
 库，该库为 PyTorch’s\ 的大部分提供支持
 nbackend，看看它如何让我们轻松地翻译 Python 代码。然后，我们’
将模型的部分内容移至 CUDA 内核，
从 GPU 提供的大规模并行性中获益，从而进一步加快速度。





## 编写 C++ 扩展 [¶](#writing-a-c-extension "永久链接到此标题")




 C++ 扩展有两种风格：它们可以使用
 `setuptools`
 提前\xe2\x80\x9c 构建\xe2\x80\x9c，或者及时\xe2\x80\ 构建\xe2\x80\x9c x9d via
 [`torch.utils.cpp_extension.load()`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load "(在 PyTorch v2.1 中) 1)")
.我们\xe2\x80\x99将从第一种方法开始，
稍后讨论后者。




### 使用
 `setuptools`构建 [¶](#building-with-setuptools "永久链接到此标题")



 对于“ahead of time” 风格，我们通过编写
 `setup.py`
 脚本来构建我们的C++ 扩展，该脚本使用setuptools 来编译我们的C++ 代码。对于 LLTM，
看起来就像这样简单：






```
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

```




 在此代码中，
 `CppExtension`
 是 `setuptools.Extension`
 的便捷包装器，它传递正确的包含路径并将
扩展语言设置为 C++。等效的普通
 `setuptools`
 代码就是：






```
Extension(
   name='lltm_cpp',
   sources=['lltm.cpp'],
   include_dirs=cpp_extension.include_paths(),
   language='c++')

```




`BuildExtension`
 执行许多必需的配置步骤和检查，并在混合 C++/CUDA
 扩展的情况下管理混合编译。 ’ 是我们现在真正需要了解的关于构建 C++ 扩展的全部信息！现在让’s 看一下我们的 C++ 扩展的实现，
它进入
 `lltm.cpp`
 。





### 编写 C++ Op [¶](#writing-the-c-op "此标题的永久链接")



 让’s 开始用C++ 实现LLTM！我们’ 向后传递所需的函数之一是 sigmoid 的导数。这是一段足够小的代码，
足以讨论我们在编写 C++
扩展时可用的整体环境：






```
#include <torch/extension.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
 auto s = torch::sigmoid(z);
 return (1 - s) * s;
}

```




`<torch/extension.h>`
 是一站式标头，包含编写 C++ 扩展所需的所有 PyTorch
 位。它包括：



* ATen 库，这是我们用于张量计算的主要 API，
* [pybind11](https://github.com/pybind/pybind11) 
，这是我们为 C++ 代码创建 Python 绑定的方式，
 * 管理 ATen 和 pybind11 之间交互细节的标头。



 `d_sigmoid()`的实现展示了如何使用 ATen API。
PyTorch’s 张量和变量接口是从 ATen 库自动生成的，因此我们可以更多或者更少地将我们的 Python 实现 1:1
 转换为 C++。我们所有计算的主要数据类型将是
 `torch::Tensor`
 。可以在[此处](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html)检查其完整 API。另请注意，我们可以包含
 `<iostream>`
 或
 *任何其他 C 或 C++ 标头* 
 –，我们可以使用
C++11 的全部功能。 




 请注意，在 Windows 上解析 torch/extension.h 时，CUDA-11.5 nvcc 将遇到内部编译器错误。
要解决此问题，请将 python 绑定逻辑移至纯 C++ 文件。
使用示例:






```
#include <ATen/ATen.h>
at::Tensor SigmoidAlphaBlendForwardCuda(....)

```




 而不是：






```
#include <torch/extension.h>
torch::Tensor SigmoidAlphaBlendForwardCuda(...)

```




 当前未解决的 nvcc bug 问题
 [此处](https://github.com/pytorch/pytorch/issues/69460) 
.
完整的解决方法代码示例
 [此处](https://github.com/facebookresearch/pytorch3d/commit/cb170ac024a949f1f9614ffe6af1c38d972f7d48) 
.




#### 正向传递 [¶](#forward-pass "永久链接到此标题")



 接下来我们可以将整个前向传递移植到 C++：






```
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





#### 向后传递 [¶](#backward-pass "永久链接到此标题")



 C++ 扩展 API 目前不提供自动
为我们生成向后函数的方法。因此，我们还必须实现 LLTM 的后向传递，它计算损失相对于前向传递的每个输入的导数。最终，我们将
forward和backward函数放入
 [`torch.autograd.Function`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function "(in PyTorch v2.1)")
 创建
 良好的 Python 绑定。向后函数稍微复杂一些，所以
我们’不会深入研究代码（如果你有兴趣，
 [Alex Graves’ 论文](https://www.cs.toronto.edu/~graves/phd.pdf）
 是一本很好的读物，可了解更多
这方面的信息）：






```
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






### 绑定到 Python [¶](#binding-to-python "永久链接到此标题")



 一旦您用 C++ 和 ATen 编写了操作，您就可以使用 pybind11 以非常简单的方式将您的 C++ 函数或类绑定到 Python 中。
您对 PyTorch C++ 扩展的这一部分的疑问或问题将主要是
由
 [pybind11 文档](https://pybind11.readthedocs.io/en/stable/) 解决
 。




 对于我们的扩展，必要的绑定代码仅跨越四行：






```
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("forward", &lltm_forward, "LLTM forward");
 m.def("backward", &lltm_backward, "LLTM backward");
}

```




 这里需要注意的一点是宏
 `TORCH_EXTENSION_NAME`
 。 torch 扩展
build 会将其定义为您在
 `setup.py`
 脚本中为扩展指定的名称。在这种情况下，
 `TORCH_EXTENSION_NAME`
 的值将为 “lltm_cpp”。
这是为了避免必须维护
在两个地方（构建脚本和 C++ 代码）进行扩展，因为两者之间的不匹配可能导致
令人讨厌且难以跟踪的问题。





### 使用您的扩展 [¶](#using-your-extension "永久链接到此标题")



 我们现在准备在 PyTorch 中导入我们的扩展。此时，您的目录
结构可能如下所示：






```
pytorch/
  lltm-extension/
    lltm.cpp
    setup.py

```




 现在，运行
 `python
 

 setup.py
 

 install`
 来构建并安装您的扩展。这
应该看起来像这样：






```
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




 关于编译器的一个小注意事项：由于 ABI 版本控制问题，用于构建 C++ 扩展的编译器必须
 *ABI 兼容* 
 与构建 PyTorch 时
所使用的编译器
相同。实际上，这意味着您必须在 Linux 上使用 GCC 版本 4.9 及更高版本。
对于 Ubuntu 16.04 和其他更新的 Linux 发行版，这应该已经是
默认编译器。在 MacOS 上，您必须使用 clang（它没有任何 ABI 版本控制问题）。在最坏的情况下，您可以使用编译器从源代码构建 PyTorch，然后使用同一编译器构建扩展。




 构建扩展后，您只需使用在 
 `setup.py` 脚本中指定的
名称将其导入到 Python 中即可。请务必先
 `import
 

 torch`
，因为这将解析动态链接器必须
看到的一些符号：






```
In [1]: import torch
In [2]: import lltm_cpp
In [3]: lltm_cpp.forward
Out[3]: <function lltm.PyCapsule.forward>

```




 如果我们在函数或模块上调用
 `help()`
，我们可以看到它的签名
与我们的 C++ 代码匹配：






```
In[4] help(lltm_cpp.forward)
forward(...) method of builtins.PyCapsule instance
    forward(arg0: torch::Tensor, arg1: torch::Tensor, arg2: torch::Tensor, arg3: torch::Tensor, arg4: torch::Tensor) -> List[torch::Tensor]

    LLTM forward

```


由于我们现在可以从 Python 调用 C++ 函数，因此我们可以用 [`torch.autograd.Function`](https://pytorch.org/docs/stable/autograd.html 包装它们
 #torch.autograd.Function "(在 PyTorch v2.1 中)")
 和 
 [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "(在 PyTorch v2.1)")
 使它们成为 PyTorch 的一等公民：






```
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
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
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




#### 性能比较 [¶](#performance-comparison "永久链接到此标题")


现在我们可以从 PyTorch 使用和调用 C++ 代码，我们可以运行一个小型基准测试来看看我们通过用 C++ 重写操作获得了多少性能。我们’ 将前后运行 LLTM 几次并测量
d持续时间：






```
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

print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))

```




 如果我们使用本文开头用纯 Python 编写的原始 LLTM 运行此代码
，我们会得到以下数字（在我的机器上）：






```
Forward: 506.480 us | Backward 444.694 us

```




 以及我们新的 C++ 版本：






```
Forward: 349.335 us | Backward 443.523 us

```




 我们已经可以看到前向函数的显着加速（超过
30%）。对于后向函数，加速是可见的，尽管不是主要的。
我上面写的后向传递没有特别优化，
肯定可以改进。此外，PyTorch’s 自动微分引擎
可以自动并行化计算图，
可以使用整体上更高效的
操作流程，并且也是用 C++ 实现的，因此’s
预计会很快
 。尽管如此，这是一个好的开始。





#### GPU 设备上的性能 [¶](#performance-on-gpu-devices "此标题的永久链接")



 关于 PyTorch’s
 *ATen* 
 后端的一个奇妙事实是，它抽象
您正在运行的计算设备。这意味着我们为 CPU 编写的相同代码也可以在 GPU 上运行，并且各个操作将相应地分派给 GPU 优化的实现。对于诸如矩阵乘法之类的某些操作（如 
 `mm`
 或 
 `addmm`
 ），这是一个巨大的胜利。让’s 看看我们通过使用 CUDA 张量运行 C++ 代码获得了多少性能。不需要对我们的实现进行任何更改，我们只需将张量从 Python 放入 GPU 内存中，在创建时添加
 `device=cuda_device`
 参数或使用
 `.to (cuda_device)`
 创建后：






```
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




 再次将我们的普通 PyTorch 代码与 C++ 版本进行比较，现在两者都在 CUDA 设备上运行，我们再次看到性能提升。对于 Python/PyTorch：






```
Forward: 187.719 us | Backward 410.815 us

```




 和 C++/ATen:






```
Forward: 149.802 us | Backward 393.458 us

```




 与非 CUDA 代码相比，’ 的整体加速效果非常好。但是，我们可以通过编写自定义 CUDA 内核来提高 C++ 代码的性能，我们很快就会深入研究该内核。在此之前，让’s 讨论一下构建 C++
扩展的另一种方法。






### JIT 编译扩展 [¶](#jit-compiling-extensions "永久链接到此标题")



 之前，我提到有两种构建 C++ 扩展的方法：使用
 `setuptools`
 或即时 (JIT)。介绍完前者后，让’s
e 详细讨论后者。 JIT 编译机制为您提供了一种通过调用 PyTorch’s API 中名为
 [`torch.utils.cpp_extension.load( )`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load "(在 PyTorch v2.1 中)")
 。对于
LLTM，这看起来就像这样简单：






```
from torch.utils.cpp_extension import load

lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"])

```




 在这里，我们为函数提供与 `setuptools` 相同的信息。在后台，这将执行以下操作:



1. 创建临时目录
 `/tmp/torch_extensions/lltm`
 ,
2.将
 [Ninja](https://ninja-build.org/) 
 构建文件发送到该临时目录中，
3.将源文件编译到共享库中，
4.将此共享库导入为 Python 模块。



 事实上，如果您将
 `verbose=True`
 传递给
 `cpp_extension.load()`
 ，您将
被告知该过程：






```
Using /tmp/torch_extensions as PyTorch extensions root...
Emitting ninja build file /tmp/torch_extensions/lltm_cpp/build.ninja...
Building extension module lltm_cpp...
Loading extension module lltm_cpp...

```




 生成的 Python 模块将与 setuptools 生成的完全相同，
但消除了必须维护单独
 `setup.py`
 build
文件的要求。如果您的设置更复杂并且您确实需要 
 `setuptools`
 的全部功能，您
 *可以* 
 编写自己的
 `setup.py`
 \xe2\x80\x93 但在很多情况下，这种 JIT 技术就可以很好地发挥作用。第一次运行此行时，
it 将需要一些时间，因为扩展正在后台编译。由于
我们使用 Ninja 构建系统来构建源代码，因此重新编译是增量的，因此，如果您没有\xe2\x80\，则在第二次运行 Python 模块时
重新加载扩展会很快且开销较低x99t 更改扩展名\xe2\x80\x99s
源文件。





## 编写混合 C++/CUDA 扩展 [¶](#writing-a-mixed-c-cuda-extension "永久链接到此标题")



为了真正将我们的实现提升到一个新的水平，我们可以使用自定义 CUDA 内核手写部分前向和后向传递。对于 LLTM，这有望特别有效，因为存在大量按顺序进行的逐点操作，这些操作都可以在单个 CUDA 内核中融合和并行化。让’s 看看我们如何编写这样的 CUDA 内核并
使用此扩展机制将其与 PyTorch 集成。



编写 CUDA 扩展的一般策略是首先编写一个 C++ 文件，该文件定义将从 Python 调用的函数，并使用 pybind11 将这些函数绑定到 Python。此外，此文件还将声明 CUDA (
 `.cu`
 ) 文件中定义的函数。然后，C++ 函数将执行一些检查并最终将其调用转发给 CUDA 函数。在 CUDA 文件中，我们编写实际的 CUDA 内核。然后，
 `cpp_extension`
 包
将使用 C++ 编译器（如
 `gcc`
）来编译 C++ 源代码，并使用 NVIDIA’s
 `nvcc 来编译 CUDA 源代码`
 编译器。这可确保每个编译器处理它最了解的要编译的文件。最终，它们
将链接到一个共享库，我们可以通过 Python
代码使用该库。




 我们’ll 从 C++ 文件开始，我们’ll 将其称为
 `lltm_cuda.cpp`
 ，例如：






```
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

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
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




 如您所见，它主要是样板文件，检查并转发到我们’ 将在 CUDA 文件中定义的函数。我们’ 将此文件命名为
 `lltm_cuda_kernel.cu`
（注意
 `.cu`
 扩展名！）。 NVCC 可以合理
编译 C++11，因此我们仍然可以使用 ATen 和 C++ 标准库（但不是
 `torch.h`
 ）。请注意，
 `setuptools`
 无法处理
具有相同名称但扩展名不同的文件，因此，如果您使用
 `setup.py`
 方法而不是 JIT 方法，则必须为您的 CUDA 文件指定不同的名称
而不是 C++ 文件的名称（对于 JIT 方法，
 `lltm.cpp`
 和
 `lltm.cu`
 可以正常工作）。让’s 看一下这个文件的样子：






```
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
 return 1.0 / (1.0 + exp(-z));
}

```




 在这里我们看到我刚刚描述的标头，以及我们正在使用
CUDA 特定声明的事实，例如 
 `__device__`
 和 
 ` __forceinline__`
 和
类似
 `exp`
 的函数。让’s 继续使用
我们’ 需要的更多辅助函数：






```
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




 现在要实际实现一个函数，我们’ 将再次需要两件事：一个函数
执行我们不’ 不希望显式手动编写的操作并调用
到 CUDA 内核，然后是我们想要加速的部分的实际 CUDA 内核。对于前向传播，第一个函数应如下所示：






```
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




 这里的主要兴趣点是
 `AT_DISPATCH_FLOATING_TYPES`
 宏和
内核启动（由
 `<<<...>>> 表示） `
)。虽然 ATen 抽象出了我们处理的张量的设备和数据类型，但张量在运行时仍由具体设备上的具体类型的内存支持。因此，我们需要一种在运行时确定张量是什么类型的方法，然后有选择地调用具有相应正确类型签名的函数。手动完成，
这（概念上）看起来像这样：






```
switch (tensor.type().scalarType()) {
 case torch::ScalarType::Double:
 return function<double>(tensor.data<double>());
 case torch::ScalarType::Float:
 return function<float>(tensor.data<float>());
 ...
}

```




 `AT_DISPATCH_FLOATING_TYPES`
 的目的是为我们处理这个调度。它需要一个类型（
 `gates.type()`
 在我们的例子中）、一个名称（用于错误消息）和一个 lambda 函数。在此 lambda 函数内，类型别名
 `scalar_t`
 可用，并被定义为张量在运行时在该上下文中实际存在的类型。因此，如果我们有一个模板函数（我们的 CUDA 内核就是这个函数），我们可以使用这个“scalar_t”别名来实例化它，并调用正确的函数。在这种情况下，我们还想检索张量的数据指针作为该“标量_t”类型的指针。如果您想要分派所有类型而不仅仅是浮点类型（
 `Float`
 和 
 `Double`
 ），您可以使用
 `AT_DISPATCH_ALL_TYPES` 
.




 请注意，我们使用普通 ATen 执行一些操作。这些操作仍将在 GPU 上运行，但使用 ATen’s 默认实现。这是有意义的，因为 ATen 将使用高度优化的例程来处理诸如矩阵乘法（例如，`addmm`）或卷积之类的事情，而这将更难以
实现和改进我们自己。




 至于内核启动本身，我们在这里指定每个 CUDA 块
将有 1024 个线程，并且整个 GPU 网格被分成
多个块
 `1
 

 x
 \每个组件需要 n
 1024`
 个线程来填充我们的矩阵。例如，如果我们的状态大小为 2048，批次大小为 4，则我们’d 总共启动
 `4
 

 x
 

 2
 

 =
 

 8`
 块，每块 1024 个线程。如果
您’之前从未听说过 CUDA “blocks” 或 “grids”，
 [关于 CUDA 的介绍性阅读
] （https://devblogs.nvidia.com/even-easier-introduction-cuda）
 可能
有帮助。




 实际的 CUDA 内核相当简单（如果您’ 以前曾经编写过 GPU）：






```
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




 这里最有趣的是，我们能够为门矩阵中的每个单独组件完全并行地计算所有这些
逐点运算。如果您想象必须使用一个巨大的
 `for`
 循环来连续处理
 百万个元素，您就会明白为什么这会快得多。




### 使用访问器 [¶](#using-accessors "永久链接到此标题")



 您可以在 CUDA 内核中看到，我们直接处理具有正确
类型的指针。事实上，直接使用 cuda
内核中的高级类型不可知张量将非常低效。



然而，这是以易用性和可读性为代价的，特别是对于高维数据。在我们的示例中，我们知道连续
 `gates`
 张量具有 3 个维度：



1. 批次，大小为
 `batch_size`
 和步幅
 `3*state_size`
2.行、大小为
 `3`
 和步长为
 `state_size`
3。索引、
 `state_size` 的大小和 
 `1` 的步幅



 那么我们如何在内核内部访问
 `gates[n][row][column]`
 元素呢？
事实证明，您需要使用一些简单的
arithmetic 来访问元素。






```
gates.data<scalar_t>()[n*3*state_size + row*state_size + column]

```




 除了冗长之外，该表达式还需要明确
已知步幅，从而在其参数中传递给内核函数。您可以看到，
如果内核函数接受不同大小的多个张量，
您最终会得到一个非常长的参数列表。




 对我们来说幸运的是，ATen 提供了通过一次动态检查来创建的访问器，
动态检查张量的类型和维数。
然后访问器公开一个 API，用于有效地访问张量元素
而无需转换为单个指针:






```
torch::Tensor foo = torch::rand({12, 12});

// assert foo is 2-dimensional and holds floats.
auto foo_a = foo.accessor<float,2>();
float trace = 0;

for(int i = 0; i < foo_a.size(0); i++) {
 // use the accessor foo_a to get tensor data.
 trace += foo_a[i][i];
}

```




 访问器对象具有相对较高级别的接口，具有
 `.size()`
 和
 `.stride()`
 方法和多维索引。 
 `.accessor<>`
 接口旨在有效地访问 cpu 张量上的数据。 cuda 张量的等效项是 `packed_accessor64<>`
 和 
 `packed_accessor32<>`
 ，它们生成具有 64 位或 32 位整数索引的打包访问器。 




 与访问器的根本区别在于，打包访问器在其结构内部复制大小
和跨步数据，而不是指向它。它允许我们
将其传递给 CUDA 内核函数并在其中使用其接口。




 我们可以设计一个采用打包访问器而不是指针的函数。






```
__global__ void lltm_cuda_forward_kernel(
 const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
 const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
 torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
 torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
 torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
 torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
 torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell)

```




 让’s 分解这里使用的模板。前两个参数
 `scalar_t`
 和
 `2`
 与常规访问器相同。参数
 `torch::RestrictPtrTraits`
 指示必须
 使用
 `__restrict__`
 关键字。另请注意，我们’使用了
 `PackedAccessor32`
 变体，它将大小和步幅存储在
 
 `int32_t`
 中。这很重要，因为使用 64 位
变体 (
 `PackedAccessor64`
 ) 会使内核变慢。




 函数声明变为






```
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




 实现更具可读性！然后通过使用主机函数中的
 `.packed_accessor32<>`
 方法创建
打包访问器来调用该函数。






```
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




 向后传递遵循大致相同的模式，我不会’ 进一步详细说明
：






```
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

 AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_cuda", ([&] {
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





### 将 C++/CUDA 操作与 PyTorch 集成 [¶](#integrating-a-c-cuda-operation-with-pytorch "永久链接到此标题")



 我们支持 CUDA 的操作与 PyTorch 的集成同样非常简单。
如果您想编写
 `setup.py`
 脚本，它可能如下所示：






```
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




 我们现在使用
 `CUDAExtension()`
 而不是
 `CppExtension()`
 。我们只需
指定
 `.cu`
 文件以及
 `.cpp`
 文件–，库会
处理这给您带来的所有麻烦。 JIT 机制甚至
更简单:






```
from torch.utils.cpp_extension import load

lltm = load(name='lltm', sources=['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])

```




#### 性能比较 [¶](#id4 "此标题的永久链接")



 我们希望将代码的逐点操作与 CUDA 进行并行化和融合
能够提高 LLTM 的性能。让’s 看看这是否成立。
我们可以运行我之前列出的代码来运行基准测试。我们之前最快的
版本是基于 CUDA 的 C++ 代码：






```
Forward: 149.802 us | Backward 393.458 us

```




 现在使用我们的自定义 CUDA 内核：






```
Forward: 129.431 us | Backward 304.641 us

```




 更多性能提升！








## 结论 [¶](#conclusion "永久链接到此标题")




 您现在应该对 PyTorch’s C++ 扩展
机制以及使用它们的动机有一个很好的概述。您可以在[此处](https://github.com/pytorch/extension-cpp)
 找到本说明中显示的代码
示例。如果您有疑问，请使用
 [论坛](https://discuss.pytorch.org) 
 。另请务必查看我们的
 [常见问题解答](https://pytorch.org/cppdocs/notes/faq.html)
，以防遇到任何问题。









