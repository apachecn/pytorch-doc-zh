

# 在C++中加载PYTORCH模型
> 译者：[talengu](https://github.com/talengu)

PyTorch的主要接口为Python。虽然Python有动态编程和易于迭代的优势，但在很多情况下，正是Python的这些属性会带来不利。我们经常遇到的生产环境，要满足低延迟和严格部署要求。对于生产场景而言，C++通常是首选语言，也能很方便的将其绑定到另一种语言，如Java，Rust或Go。本教程将介绍从将PyTorch训练的模型序列化表示，到C++语言_加载_和_执行_的过程。


## 第一步：将PyTorch模型转换为Torch Script
PyTorch模型从Python到C++的转换由[Torch Script](https://pytorch.org/docs/master/jit.html)实现。Torch Script是PyTorch模型的一种表示，可由Torch Script编译器理解，编译和序列化。如果使用基础的“eager”API编写的PyTorch模型，则必须先将模型转换为Torch Script，当然这也是比较容易的。如果已有模型的Torch Script，则可以跳到本教程的下一部分。

将PyTorch模型转换为Torch Script有两种方法。
第一种方法是Tracing。该方法通过将样本输入到模型中一次来对该过程进行评估从而捕获模型结构.并记录该样本在模型中的flow。该方法适用于模型中很少使用控制flow的模型。
第二个方法就是向模型添加显式注释(Annotation)，通知Torch Script编译器它可以直接解析和编译模型代码，受Torch Script语言强加的约束。


> 小贴士
可以在官方的[Torch Script 参考](https://pytorch.org/docs/master/jit.html)中找到这两种方法的完整文档，以及有关使用哪个方法的细节指导。


### 利用Tracing将模型转换为Torch Script
要通过tracing来将PyTorch模型转换为Torch脚本,必须将模型的实例以及样本输入传递给`torch.jit.trace`函数。这将生成一个 `torch.jit.ScriptModule`对象，并在模块的`forward`方法中嵌入模型评估的跟踪：

```py
import torch
import torchvision

# 获取模型实例
model = torchvision.models.resnet18()

# 生成一个样本供网络前向传播 forward()
example = torch.rand(1, 3, 224, 224)

# 使用 torch.jit.trace 生成 torch.jit.ScriptModule 来跟踪
traced_script_module = torch.jit.trace(model, example)

```
现在，跟踪的`ScriptModule`可以与常规PyTorch模块进行相同的计算：

```py
In[1]: output = traced_script_module(torch.ones(1, 3, 224, 224))
In[2]: output[0, :5]
Out[2]: tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)

```

### 通过Annotation将Model转换为Torch Script

在某些情况下，例如，如果模型使用特定形式的控制流，如果想要直接在Torch Script中编写模型并相应地标注(annotate)模型。例如，假设有以下普通的 Pytorch模型：

```py
import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

```

由于此模块的`forward`方法使用依赖于输入的控制流，因此它不适合利用Tracing的方法生成Torch Script。为此,可以通过继承`torch.jit.ScriptModule`并将`@ torch.jit.script_method`标注添加到模型的`forward`中的方法，来将model转换为`ScriptModule`：

```py
import torch

class MyModule(torch.jit.ScriptModule):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    @torch.jit.script_method
    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

my_script_module = MyModule()

```
现在，创建一个新的`MyModule`对象会直接生成一个可序列化的`ScriptModule`实例了。



## 第二步：将Script Module序列化为一个文件

不论是从上面两种方法的哪一种方法获得了`ScriptModule`,都可以将得到的`ScriptModule`序列化为一个文件,然后C++就可以不依赖任何Python代码来执行该Script所对应的Pytorch模型。
假设我们想要序列化前面trace示例中显示的`ResNet18`模型。要执行此序列化，只需在模块上调用 [save](https://pytorch.org/docs/master/jit.html#torch.jit.ScriptModule.save)并给个文件名：

```py
traced_script_module.save("model.pt")

```
这将在工作目录中生成一个`model.pt`文件。现在可以离开Python，并准备跨越到C ++语言调用。

## 第三步:在C++中加载你的Script Module

要在C ++中加载序列化的PyTorch模型，应用程序必须依赖于`PyTorch C ++ API` - 也称为_LibTorch_。_LibTorch发行版_包含一组共享库，头文件和`CMake`构建配置文件。虽然CMake不是依赖LibTorch的要求，但它是推荐的方法，并且将来会得到很好的支持。在本教程中，我们将使用CMake和LibTorch构建一个最小的C++应用程序，加载并执行序列化的PyTorch模型。

### 最小的C++应用程序

以下内容可以做到加载模块：

```py
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  assert(module != nullptr);
  std::cout << "ok\n";
}

```
`<torch/script.h>`头文件包含运行该示例所需的LibTorch库中的所有相关`include`。main函数接受序列化`ScriptModule`的文件路径作为其唯一的命令行参数，然后使用`torch::jit::load()`函数反序列化模块，得到一个指向`torch::jit::script::Module`的共享指针，相当于C ++中的`torch.jit.ScriptModule`对象。最后，我们只验证此指针不为null。我们展示如何在接下来执行它。

### 依赖库LibTorch和构建应用程序

我们将上面的代码保存到名为`example-app.cpp`的文件中。对应的构建它的简单`CMakeLists.txt`为：

```py
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)

```
我们构建示例应用程序的最后一件事是下载LibTorch发行版。从PyTorch网站的下载页面获取最新的稳定版本 [download page](https://pytorch.org/)。如果下载并解压缩最新存档，则有以下目录结构：
```py
libtorch/
  bin/
  include/
  lib/
  share/

```

*   `lib/` 包含含链接的共享库,
*   `include/` 包含程序需要`include`的头文件,
*   `share/`包含必要的CMake配置文件使得 `find_package(Torch)` 。

> 小贴士
在Windows平台上, debug and release builds are not ABI-compatible. 如果要使用debug, 要使用 [源码编译 PyTorch](https://github.com/pytorch/pytorch#from-source)方法。

最后一步是构建应用程序。为此，假设我们的示例目录布局如下：

```py
example-app/
  CMakeLists.txt
  example-app.cpp

```

我们现在可以运行以下命令从`example-app/`文件夹中构建应用程序：

```py
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make

```

其中 `/path/to/libtorch` 应该是解压缩的LibTorch发行版的完整路径。如果一切顺利，它将看起来像这样：

```py
root@4b5a67132e81:/example-app# mkdir build
root@4b5a67132e81:/example-app# cd build
root@4b5a67132e81:/example-app/build# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Configuring done
-- Generating done
-- Build files have been written to: /example-app/build
root@4b5a67132e81:/example-app/build# make
Scanning dependencies of target example-app
[ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
[100%] Linking CXX executable example-app
[100%] Built target example-app

```

如果我们提供前面的序列化`ResNet18`模型的路径给`example-app`，C++输出的结果应该是 OK:

```py
root@4b5a67132e81:/example-app/build# ./example-app model.pt
ok

```

## 在C++代码中运行Script Module

在C ++中成功加载了我们的序列化`ResNet18`后，我们再加几行执行代码，添加到C++应用程序的`main()`函数中：

```py
// Create a vector of inputs.
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({1, 3, 224, 224}));

// Execute the model and turn its output into a tensor.
at::Tensor output = module->forward(inputs).toTensor();

std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

```
前两行设置我们模型的输入。 创建了一个 `torch::jit::IValue` (`script::Module` 对象可接受和返回的一种数据类型) 的向量和添加一个输入。要创建输入张量，我们使用`torch::ones()`(C++ API）和python中的`torch.ones` 一样。 然后我们运行`script::Module`的`forward`方法，传入我们创建的输入向量，返回一个新的`IValue`，通过调用`toTensor()`可将其转换为张量。


>小贴士
更多关于`torch::ones` 和 PyTorch的对应 C++ API的内容 [https://pytorch.org/cppdocs](https://pytorch.org/cppdocs)。PyTorch C++ API 和Python API差不多，可以使你像python 中一样操作处理tensors。


在最后一行中，我们打印输出的前五个条目。由于我们在本教程前面的Python中为我们的模型提供了相同的输入，因此理想情况下我们应该看到相同的输出。让我们通过重新编译我们的应用程序并使用相同的序列化模型运行它来尝试：

```py
root@4b5a67132e81:/example-app/build# make
Scanning dependencies of target example-app
[ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
[100%] Linking CXX executable example-app
[100%] Built target example-app
root@4b5a67132e81:/example-app/build# ./example-app model.pt
-0.2698 -0.0381  0.4023 -0.3010 -0.0448
[ Variable[CPUFloatType]{1,5} ]

```

作为参考，之前Python代码的输出是：

```py
tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)

```

由此可见,C++的输出与Python的输出是一样的,成功啦!

>小贴士
将你的模型放到GPU上，可以写成`model->to(at::kCUDA);`。确保你的输入也在CUDA的存储空间里面，可以使用`tensor.to(at::kCUDA)`检查，这个函数返回一个新的在CUDA里面的tensor。

## 第五步:进阶教程和详细API

本教程希望能使你理解PyTorch模型从python到c++的调用过程。通过上述教程，你能够通过“eager” PyTorch做一个简单模型，转成`ScriptModule`，并序列化保存。然后在C++里面通过 `script::Module`加载运行模型。

当然，还有好多内容我们没有涉及。举个例子，你希望在C++或者CUDA中实现`ScriptModule`中的自定义操作，然后就可以在C++调用运行`ScriptModule`模型。这种是可以做到的，可以参考[this](https://github.com/pytorch/pytorch/tree/master/test/custom_operator)。下面还有一些文档可以参考，比较有帮助：

*   Torch Script 参考: [https://pytorch.org/docs/master/jit.html](https://pytorch.org/docs/master/jit.html)
*   PyTorch C++ API 文档: [https://pytorch.org/cppdocs/](https://pytorch.org/cppdocs/)
*   PyTorch Python API 文档: [https://pytorch.org/docs/](https://pytorch.org/docs/)

如果有任何bug或者问题，可以向社区 [Pytorch forum](https://discuss.pytorch.org/) 或者 [Pytorch GitHub issues](https://github.com/pytorch/pytorch/issues) 寻求帮助。

