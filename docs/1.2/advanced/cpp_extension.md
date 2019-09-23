# 定制C ++和CUDA扩展

**作者** ：[彼得戈尔兹伯勒](https://www.goldsborough.me/)

PyTorch提供相关的神经网络，任意张量代数，数据扯皮和其他用途的操作过多。但是，你还是会发现自己需要一个更个性化的操作。例如，你可能想使用你发现了一种新的激活功能纸，或者实现你开发了一个操作你的研究的一部分。

在PyTorch整合这样的自定义动作的最简单的方法是通过延长`函数 `和`模块 `所概述[HTG8将它写在Python
]此处[HTG9。这使您可以自动分化的全功率（备件从编写导函数）以及Python的通常的表现。然而，有时可能当你的操作是更好的用C
++实现。例如，你的代码可能需要 _真的_ 快，因为它被称为非常频繁的模型或者甚至几个电话非常昂贵。另一个可行的原因是，它依赖于或与其它的C或C
++库交互。为了解决这样的情况下，提供PyTorch编写自定义 _C ++扩展_ 的一个非常简单的方法。

C ++扩展是我们已经开发以允许用户（你）来创建PyTorch运营商定义的 _外的源_ ，即，从PyTorch后端分离的机构。这种方法
_从本地PyTorch业务的实现方式不同[HTG3。 C
++扩展旨在免去你多用，同时为您提供为基于PyTorch项目具有高度的灵活性集成的操作与PyTorch的后端相关的样板。然而，一旦你已经确定你的操作为C
++的扩展，把它变成一个本地PyTorch功能主要是组织代码，你可以在事后，如果你决定上游有助于您的操作解决的问题。_

## 动机和实施例

本说明的其余部分将通过编写和使用C
++（和CUDA）延伸的一个实际的例子行走。如果你正在追逐或有人会解雇你，如果你没有得到运到这天结束时完成，你可以跳过这一节，并直奔下一节的实施细节。

比方说，你已经来到了一个新的重复单元的，你发现有比现有技术的优异性能。此重复单元是类似于LSTM，但不同之处在于它缺乏一个 _忘记门_ ，并使用
_指数线性单位_ （ELU）作为其内部的激活功能。由于本机永远不会忘记，我们把它叫做 _LLTM_ 或 _长，长期内存_ 单元。

其中LLTMs香草LSTMs不同的两种方式是不够显著，我们不能配置PyTorch的`LSTMCell
`我们的目的，所以我们必须创建一个自定义单元格。在所有情况下的第一步，并有可能 - -
这样做的第一个和最简单的方法是使用Python平原PyTorch来实现我们所期望的功能。对于这一点，我们需要继承`torch.nn.Module
`和实施LLTM的直传。这将是这个样子：

    
    
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
    

然后我们可以使用如预期：

    
    
    import torch
    
    X = torch.randn(batch_size, input_features)
    h = torch.randn(batch_size, state_size)
    C = torch.randn(batch_size, state_size)
    
    rnn = LLTM(input_features, state_size)
    
    new_h, new_C = rnn(X, (h, C))
    

当然，如果在所有可能的和合理的，你应该用这种方式来延长PyTorch。由于PyTorch已高度优化其操作的实现的用于CPU _和_ GPU，搭载库如[
NVIDIA cuDNN ](https://developer.nvidia.com/cudnn)，[英特尔MKL
](https://software.intel.com/en-us/mkl)或[ NNPACK
](https://github.com/Maratyszcza/NNPACK)像上面PyTorch代码往往是速度不够快。但是，我们也可以看到，为什么在某些情况下，有余地进一步的性能提升。最明显的原因是，PyTorch没有了
_算法_ 你正在实施的知识。它知道只有你用它来撰写你的算法个人操作。因此，PyTorch必须单独执行你的操作，一前一后。由于一个操作的实现（或 _内核_
），这可能涉及推出了CUDA核心的每一个人打电话，有一定的开销，这种开销可能会成为跨越许多函数调用显著。此外，运行我们的代码Python解释器本身可以放慢我们的节目。

因此超速东西一个明确的方法是重写份C ++（或CUDA）和 _熔丝操作的_
特定的基团。熔化装置的许多功能的实现合并成一个单一的功能，其利润较少的内核启动，以及我们可以用数据的全球流动的知名度提高进行其他优化。

让我们来看看如何使用C ++的扩展来实现 _融合_ 的LLTM的版本。我们将通过在普通的C ++写它开始，使用[ ATEN
](https://github.com/zdevito/ATen)库，权力多大PyTorch的后端，看看它是如何让我们很容易把我们的Python代码。随后，我们将移动模型的部分CUDA内核从大规模并行GPU的提供有利于更加快速度。

## 写一个C ++扩展

C ++的扩展有两种形式：他们可以“提前”与`setuptools的 `，或“即时”通过`torch.utils.cpp_extension建。负载（）
`。我们将与第一种方法开始，再讨论后者。

### 以建设`setuptools的 `

对于“提前”的味道，我们建立我们的C ++写一个使用setuptools的编译我们的C ++代码`setup.py
`脚本扩展。对于LLTM，它看起来像这样简单：

    
    
    from setuptools import setup, Extension
    from torch.utils import cpp_extension
    
    setup(name='lltm_cpp',
          ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
          cmdclass={'build_ext': cpp_extension.BuildExtension})
    

在该代码中，`CppExtension`是一个方便的包装周围`setuptools.Extension
`认为传递正确的包含路径，并设置扩展的语言于C ++。等效香草`setuptools的 `代码将仅仅是：

    
    
    Extension(
       name='lltm_cpp',
       sources=['lltm.cpp'],
       include_dirs=cpp_extension.include_paths(),
       language='c++')
    

`BuildExtension`执行许多所需的配置步骤和检查，并还管理混合汇编在混合的C ++ /
CUDA扩展的情况下。而这一切，我们真正需要知道的关于建立C ++的扩展，现在！现在让我们来看看我们的C ++的扩展，它进入`lltm.cpp
`的实施。

### 写入C ++作业

让我们开始实现C ++中的LLTM！我们需要为后向通行功能之一是乙状结肠的衍生物。这是一个足够小的一段代码，讨论编写C ++扩展时，提供给我们的整体环境：

    
    
    #include <torch/extension.h>
    
    #include <iostream>
    
    torch::Tensor d_sigmoid(torch::Tensor z) {
      auto s = torch::sigmoid(z);
      return (1 - s) * s;
    }
    

`& LT ;炬/ extension.h & GT ;`是一站式头部以包括所有必要的PyTorch位写C ++的扩展。这包括：

  * 该ATEN库，它是我们的张量计算主要的API，
  * [ pybind11 ](https://github.com/pybind/pybind11)，这是我们如何创造我们的C ++代码的Python绑定，
  * 该管理ATEN和pybind11相互作用的细节头。

`d_sigmoid的执行（） `示出了如何使用阿坦API。
PyTorch的张量和可变接口从ATEN库自动生成的，所以我们可以1或多或少地把我们的Python实现：1到C ++。我们对所有的计算主要数据类型将是`
火炬::张量 [HTG7。它的完整的API可以检查[此处[HTG9。还要注意的是，我们可以包括`& LT ;的iostream & GT ;`或
_任何其他C或C ++头_ - 我们的C ++
11的全部力量在我们的处置。](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html)`

#### 直传

接下来，我们可以端口我们整个直传到C ++：

    
    
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
    

#### 倒推

C
++的扩展API目前不提供自动生成一个向后的功能为我们的方式。因此，我们也必须实现我们的LLTM，其计算损失的衍生物相对于直传的每个输入端的复路。最终，我们将扑通正向和反向功能为`
torch.autograd.Function
`创建一个不错的Python绑定。落后的功能是稍微有点复杂，所以我们将不会深入挖掘代码（如果你有兴趣，[亚历克斯·格雷夫斯论文](https://www.cs.toronto.edu/~graves/phd.pdf)是这方面的更多信息，很好看的）：

    
    
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
    

### 结合到Python

一旦你用C语言编写你的操作++和ATEN，您可以使用pybind11绑定你的C ++函数或类成Python以非常简单的方式。您有任何关于PyTorch C
++的扩展，这部分疑问或问题将在很大程度上被[
pybind11文献](https://pybind11.readthedocs.io/en/master/)解决。

对于我们的扩展，必要的绑定代码仅仅跨越四行：

    
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("forward", &lltm_forward, "LLTM forward");
      m.def("backward", &lltm_backward, "LLTM backward");
    }
    

一位这里需要注意的是宏`TORCH_EXTENSION_NAME  [HTG3。火炬扩展构建将其定义为名字会在`setup.py
`脚本您的分机。在这种情况下，`的值TORCH_EXTENSION_NAME`将是“lltm”。这是为了避免必须在两个地方维护扩展（构建脚本和C
++代码）的名称，因为这两个之间的不匹配会导致讨厌的，难以跟踪的问题。`

### 利用分机

我们现在设置为导入我们在PyTorch扩展。在这一点上，你的目录结构可能会是这个样子：

    
    
    pytorch/
      lltm-extension/
        lltm.cpp
        setup.py
    

现在，运行`巨蟒 setup.py  安装 `构建和安装你的扩展。这应该是这个样子：

    
    
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
    

对编译器的小记：由于ABI版本问题，编译器用于构建你的C ++扩展名必须是 _ABI兼容_ 与编译器PyTorch与建造。在实践中，这意味着你必须使用GCC
4.9版及以上的Linux操作系统。为Ubuntu
16.04和其他更近期的Linux发行版，这应该是默认的编译器了。在MacOS上，您必须使用铛（其中没有任何ABI版本问题）。在最坏的情况下，你可以从你的编译器源代码编译PyTorch，然后建立与相同的编译器扩展。

一旦你的扩展构建，你可以简单地将其导入在Python中，使用您在`setup.py`脚本中指定的名称。只是一定要`进口 火炬
`第一，因为这将解决一些符号动态链接程序必须看到：

    
    
    In [1]: import torch
    In [2]: import lltm_cpp
    In [3]: lltm_cpp.forward
    Out[3]: <function lltm.PyCapsule.forward>
    

如果我们调用`帮助（） `在功能或模块，我们可以看到，其签名符合我们的C ++代码：

    
    
    In[4] help(lltm_cpp.forward)
    forward(...) method of builtins.PyCapsule instance
        forward(arg0: torch::Tensor, arg1: torch::Tensor, arg2: torch::Tensor, arg3: torch::Tensor, arg4: torch::Tensor) -> List[torch::Tensor]
    
        LLTM forward
    

由于我们现在能够从Python中调用我们的C ++函数，我们可以用`他们包裹 torch.autograd.Function`和`
torch.nn.Module`让他们PyTorch的一等公民：

    
    
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
    

#### 性能比较

现在我们已经能够使用和调用我们的C ++从PyTorch代码，我们可以运行一个小的基准测试，看看我们有多少的性能从此改写了我们的运算在C
++中获得的。我们将运行LLTM向前和向后几次，测量时间：

    
    
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
    

如果我们运行这段代码与我们在纯Python这个帖子的开头写的最原始LLTM，我们可以得到下面的数字（我的机器上）：

    
    
    Forward: 506.480 us | Backward 444.694 us
    

并与我们的新的C ++版本：

    
    
    Forward: 349.335 us | Backward 443.523 us
    

我们已经可以看到的转发功能（30％以上），一个显著加速。对于落后的功能的加速是可见的，虽然不是主要的一个。复路我上面写的不是特别的优化，肯定可以得到改善。此外，PyTorch的自动分化引擎可以自动并行计算图形，可以使用操作更有效的流动从整体来看，在C
++中也实现的，所以它的预期要快。然而，这是一个良好的开端。

#### 对GPU设备的性能

关于PyTorch的 _ATEN_ 后端一个美妙的事实是，它抽象您正在运行的计算设备。这意味着可以 _也_
运行在GPU我们为CPU写相同的代码，并单独行动将相应地分派给GPU优化的实现。对于像矩阵乘法的某些操作（例如`毫米 `或`addmm
`），这是一个大的胜利。让我们来看看我们是多么的性能随CUDA张量运行我们的C
++代码获得。没有改变我们的实现是必需的，我们只需要简单地把我们的张量从Python的GPU内存，搭配要么加入`设备= cuda_device
`参数在创建时或使用`。要（cuda_device） [HTG19创建后]：`

    
    
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
    

一旦更多的与我们的C ++版本比较我们的平原PyTorch代码，现在无论在CUDA设备上运行，我们再次看到性能提升。对于Python / PyTorch：

    
    
    Forward: 187.719 us | Backward 410.815 us
    

和C ++ / ATEN：

    
    
    Forward: 149.802 us | Backward 393.458 us
    

这是相对于非CUDA代码有很大的整体加速。但是，我们可以通过编写自定义的CUDA内核，我们将深入很快拉更是表现出我们的C
++代码。在此之前，让我们dicuss建立你的C ++扩展的另一种方式。

### JIT编译扩展

以前，我提到过有建立C ++扩展的方法有两种：使用`setuptools的
`或只是在时间（JIT）。在覆盖了前者，让我们详细阐述了后者。在JIT编译机制提供了编译和通过调用PyTorch的API）一个简单的函数调用`
torch.utils.cpp_extension.load（ `加载在飞行您的扩展方式。对于LLTM，这看起来简单，如下：

    
    
    from torch.utils.cpp_extension import load
    
    lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"])
    

在这里，我们提供相同的信息作为`setuptools的 `功能。在此背景下，这将做到以下几点：

  1. 创建一个临时目录`/ TMP / torch_extensions / lltm`
  2. 发出[忍者](https://ninja-build.org/)建立文件保存到临时目录，
  3. 源文件编译成一个共享库，
  4. 导入此共享库作为一个Python模块。

事实上，如果你通过`详细=真 `至`cpp_extension.load（） `，您作的过程中获悉：

    
    
    Using /tmp/torch_extensions as PyTorch extensions root...
    Emitting ninja build file /tmp/torch_extensions/lltm_cpp/build.ninja...
    Building extension module lltm_cpp...
    Loading extension module lltm_cpp...
    

将得到的Python模块将通过setuptools的产生完全一样的，但移除具有维护一个单独的`setup.py
`建立文件的要求。如果您的设置较为复杂，你需要的`全功率的setuptools`，你 _可以_ 写自己`setup.py`\-
但在很多情况下，这JIT技术会做得很好。您通过这条线首次运行，它需要一定的时间，为扩展在后台编译。由于我们使用的忍者建立系统建立你的源代码，重新编译的增量，从而重新加载扩展当你运行你的Python模块的第二时间快，并具有低开销，如果你没有更改扩展名的源文件。

## 写一个混合的C ++ / CUDA扩展

要真正把我们的实施，一个新的水平，我们可以手工编写我们前进的部件和定制CUDA内核向后传递。对于LLTM，这具有特别有效的前景，因为有大量的序列逐点操作，可所有的融合，并在单个内核的CUDA并行化。让我们来看看，我们怎么能写这样的CUDA核心。它使用PyTorch这个扩展机制整合。

用于写入CUDA扩展的一般策略是先写一个C
++文件，该文件定义了将在Python被调用的功能，并结合这些功能到Python与pybind11。此外，该文件也将 _声明在CUDA（`.CU
`）文件中定义的_功能。然后，C ++函数会做一些检查，并最终推进其调用的CUDA功能。在CUDA的文件，我们写我们的实际CUDA内核。的`
cpp_extension`包将然后采取与C ++编译器编译的C ++源的象`GCC`和CUDA源NVIDIA的[HTG14护理]  NVCC
编译器。这确保了每个编译器负责文件的它知道最好的编译。最终，他们将被链接到这是提供给我们从Python代码一个共享库。

我们将与C ++文件，我们称之为`lltm_cuda.cpp`，例如开始：

    
    
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
    
    #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
    #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
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
    

正如你所看到的，它在很大程度上是样板，检查和转发到，我们将在CUDA文件中定义的功能。我们将其命名该文件`lltm_cuda_kernel.cu
`（注意`.CU`扩展！）。 NVCC可以合理编译C ++ 11，因此，我们仍然有ATEN和C ++提供给我们的标准库（但不是`torch.h
`）。需要注意的是`setuptools的 `不能处理具有相同名称但不同的扩展名的文件，因此，如果您使用`setup.py
`方法而不是JIT方法，你必须给你的CUDA的文件不同的名称比你的C ++文件（JIT的方法，`lltm.cpp`和`lltm.cu
`将正常工作）。让我们来看看这是什么文件将看起来像一个小偷看：

    
    
    #include <torch/extension.h>
    
    #include <cuda.h>
    #include <cuda_runtime.h>
    
    #include <vector>
    
    template <typename scalar_t>
    __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
      return 1.0 / (1.0 + exp(-z));
    }
    

在这里，我们看到我刚才所描述的报头，以及我们使用特定CUDA申述状`__device__`和事实`__forceinline__`和类似的`
功能EXP`。让我们继续，我们将需要几个辅助函数：

    
    
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
    

到现在实际上实现一个功能，我们将再次需要两样东西：执行我们不希望明确手工写操作并调用CUDA内核一个功能，并为部分则实际CUDA内核，我们要加快。对于直传，第一个函数应该是这样的：

    
    
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
    

这里关注的主要点是`AT_DISPATCH_FLOATING_TYPES`宏和内核启动（由表示的`& LT ; & LT ; & LT ; ...
& GT ; & GT ; & GT ;`）
。虽然ATEN抽象了我们处理张量的设备和数据类型，张量会在运行时，仍然可以通过一个具体类型的存储器的具体设备上的支持。因此，我们需要在运行什么类型的一个张量，然后有选择地调用与相应的正确的类型签名功能确定的一种方式。手工完成，这将（概念）是这个样子：

    
    
    switch (tensor.type().scalarType()) {
      case torch::ScalarType::Double:
        return function<double>(tensor.data<double>());
      case torch::ScalarType::Float:
        return function<float>(tensor.data<float>());
      ...
    }
    

AT_DISPATCH_FLOATING_TYPES的`目的 `是为了照顾记者发稿我们。它需要一个类型（`gates.type（）
`在我们的例子），一个名称（错误消息）和lambda函数。这里面lambda函数，所述类型别名`scalar_t
`可用，并且被定义为张量实际上是在这方面的运行时的类型。因此，如果我们有一个模板函数（我们的CUDA内核会），我们可以用这个`scalar_t
`别名实例化，并正确的函数将被调用。在这种情况下，我们也想取回张量的数据指针为`scalar_t
`类型的指针。如果你想派遣了所有类型的并不仅仅是浮点类型（`浴液HTG22] `和`双 `），你可以使用`AT_DISPATCH_ALL_TYPES
`。

请注意，我们执行某些操作与普通ATEN。这些操作仍然会在GPU上运行，但使用ATEN的默认实现。这是有意义的，因为宏正将使用高度优化的例程，用于像矩阵乘法的东西（例如，`
addmm`）或回旋这将是更难实施，提高自己。

作为内核启动本身，我们在这里指定每个CUDA块将具有1024个线程，并且整个GPU网格被分成`1  ×的许多块 1024
`线程如需要填写我们的矩阵，每个部件的一个线程。例如，如果我们的状态大小是2048和我们的批量大小4，我们就启动一个总的`4  × 2  =  8
`块与每个1024个螺纹。如果你从来没有听说过CUDA“块”或“网格”之前已经，一个[入门了解CUDA
](https://devblogs.nvidia.com/even-easier-introduction-cuda)可能解决问题。

实际的CUDA内核是相当简单的（如果你以往的GPU编程）：

    
    
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
    

什么是有趣的，主要是在这里，我们能够计算所有这些逐点操作的完全平行于我们的大门矩阵各个组件。如果你想像一下与 循环遍历序列百万元的巨型`
做到这一点，你可以看到为什么会快得多。`

### 使用访问

您可以在CUDA内核，我们直接与正确类型的指针看到工作。事实上，直接与CUDA内核内的高级别类型不可知的张量的工作效率会很低。

然而，这是以易用性和可读性的成本，尤其是对高维数据。在我们的例子中，我们知道的例子，连续`门 `张量有3个方面：

  1. 的`的batch_size`批次，尺寸和`步幅3 * state_size`
  2. 的`3`行，尺寸和`步幅state_size`
  3. 的`state_size`索引，大小和`1`的步幅

我们怎样才能访问元素`门[N] [行] [列]`，然后在内核里？事实证明，你所需要的进步与一些简单的算术来访问你的元素。

    
    
    gates.data<scalar_t>()[n*3*state_size + row*state_size + column]
    

除了是冗长，这个表达式需要步幅被明确地已知的，因此传递给它的参数内的核函数。你可以看到，在内核函数接受多张量大小不同，你会最终有一个很长的参数列表的情况。

幸运的是，ATEN提供了与单个动态检查，一个张量的类型和尺寸的数量创建访问器。存取然后暴露的API用于有效地访问张量元素，而不必转换为单个指针：

    
    
    torch::Tensor foo = torch::rand({12, 12});
    
    // assert foo is 2-dimensional and holds floats.
    auto foo_a = foo.accessor<float,2>();
    float trace = 0;
    
    for(int i = 0; i < foo_a.size(0); i++) {
      // use the accessor foo_a to get tensor data.
      trace += foo_a[i][i];
    }
    

访问对象具有一个相对高的水平界面，用`.size（） `和`.stride（） `方法和多维索引。的`.accessor & LT ; & GT ;
`接口被设计为高效地访问CPU的张量数据。为CUDA张量相当于是`packed_accessor & LT ; & GT ;`，其产生的盒装访问器。

与访问者的根本区别在于它的结构内部的填充访问器拷贝规模和步幅数据，而不是指向它。它允许我们将它传递给一个CUDA内核函数，并使用它里面它的接口。

我们可以设计一个函数，盒装访问者，而不是指针。

    
    
    __global__ void lltm_cuda_forward_kernel(
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gates,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_cell,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell)
    

让我们分解这里使用的模板。前两个参数`scalar_t`和`2`是相同的规则访问器。的参数`炬:: RestrictPtrTraits
`表示`__restrict__必须使用 `关键字。最后，参数`为size_t`表示的尺寸和进展必须被存储在`为size_t
`整数。这是因为默认情况下`的int64_t 使用`重要，可以使内核速度较慢。

该函数声明成为

    
    
    template <typename scalar_t>
    __global__ void lltm_cuda_forward_kernel(
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gates,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_cell,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell) {
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
    

实施更加可读！该函数然后通过与创建盒装访问者称为`.packed_accessor & LT ; & GT ;`主机功能内的方法。

    
    
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
            gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            old_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }));
    
      return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
    }
    

向后传球遵循几乎相同的模式，我不就可以了进一步阐述：

    
    
    template <typename scalar_t>
    __global__ void lltm_cuda_backward_kernel(
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_old_cell,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_gates,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_h,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_cell,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gate_weights) {
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
            d_old_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            grad_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
      }));
    
      auto d_gate_weights = d_gates.reshape({batch_size, 3*state_size});
      auto d_weights = d_gate_weights.t().mm(X);
      auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);
    
      auto d_X = d_gate_weights.mm(weights);
      auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
      auto d_input = d_X.slice(/*dim=*/1, state_size);
    
      return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
    }
    

### 与PyTorch集成C ++ / CUDA操作

我们与PyTorch支持CUDA运算整合又是非常简单的。如果你想要写一个`setup.py`脚本，它看起来是这样的：

    
    
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
    

相反`CppExtension的（） `，我们现在使用的`CUDAExtension（） [HTG7。我们可以只指定`.CU`与`的.cpp
`文件文件一起 - 库通吃，这需要对麻烦的护理您。 JIT的机制是更简单：`

    
    
    from torch.utils.cpp_extension import load
    
    lltm = load(name='lltm', sources=['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])
    

#### 性能比较

我们的希望是，并行化和融合的我们与CUDA代码的逐点行动将提高我们的LLTM的性能。让我们来看看是否能成立。我们可以跑我前面列出运行基准测试的代码。我们最快的早期版本是基于CUDA的C
++代码：

    
    
    Forward: 149.802 us | Backward 393.458 us
    

现在通过我们的定制内核CUDA：

    
    
    Forward: 129.431 us | Backward 304.641 us
    

更多的性能提升！

## 结论

现在，您应该配备的PyTorch的C
++扩展机制很好的概述，以及使用他们的动机。你可以找到本笔记[此处](https://github.com/pytorch/extension-
cpp)中显示的代码示例。如果您有任何疑问，请使用[论坛[HTG3。此外，一定要检查我们的](https://discuss.pytorch.org)[常见问题](https://pytorch.org/cppdocs/notes/faq.html)如果你遇到的任何问题。

[Next ![](../_static/images/chevron-right-orange.svg)](cpp_frontend.html
"Using the PyTorch C++ Frontend") [![](../_static/images/chevron-right-
orange.svg) Previous](numpy_extensions_tutorial.html "Creating Extensions
Using numpy and scipy")

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

  * 自定义C ++和CUDA扩展
    * 动机与实施例
    * 编写C ++扩展
      * 与`构建setuptools的 `
      * 编写C ++作业
        * 直传
        * 倒推
      * 结合到Python 
      * 利用分机
        * 性能比较
        * 对GPU设备的性能
      * JIT编译扩展
    * 编写混合的C ++ / CUDA扩展
      * 使用访问器
      * 与PyTorch集成一个C ++ / CUDA操作
        * 性能比较
    * 结论

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

