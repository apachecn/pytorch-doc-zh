# 在C++中加载 TorchScript 模型

> 译者：[masteryi-0018](https://github.com/masteryi-0018)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/cpp_export>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/cpp_export.html>

# 在C++中加载 TORCHSCRIPT 模型

顾名思义，PyTorch的主要接口是Python程序设计语言。虽然Python是一种适合许多需要动态性和易于迭代的场景的首选语言，但同样有很多是Python的这些属性恰恰不利的情况。后者经常适用的环境是生产——需要低延迟和严格的部署要求。对于生产方案，C++ 通常是首选的语言，即使只是将其绑定到另一种语言中，像Java，Rust或Go这样的语言。以下段落将概述使用现有的Python语言从PyTorch提供的模型到序列化的模型，可以纯粹从C++加载和执行的表示，没有对Python的依赖。

## 步骤 1：将 PyTorch 模型转换为 Torch 脚本

PyTorch模型从Python到C++的过程由[Torch Script](https://pytorch.org/docs/master/jit.html)实现，Torch Script是PyTorch的表示形式。可以通过Torch Script理解、编译和序列化的模型编译器。如果您从编写的现有 PyTorch 模型开始，原版“渴望”API，必须先将模型转换为Torch Script。在最常见的情况，下面讨论，这只需要很少的努力。如果你已经有一个Torch Script模块，你可以跳到下一部分教程。

有两种方法可以将 PyTorch 模型转换为 Torch Script。第一个称为trace，其中模型的结构为通过使用示例输入评估一次并记录这些输出。这适用于有限使用控制流的模型。第二种方法是将显式注释添加到您的Torch Script编译器，以告知 TorchScript 编译器可以根据 Torch Script 语言施加的约束直接解析和编译模型代码。

> 提示
>
> 您可以在官方 [Torch Script](https://pytorch.org/docs/master/jit.html)中找到这两种方法的完整文档以及使用方法的进一步指导。

### 通过Trace转换为TORCHSCRIPT

要将 PyTorch 模型通过trace转换为 Torch Script，必须将模型的实例以及示例输入传递给`torch.jit.trace`函数。 这将产生一个`torch.jit.ScriptModule`对象，并将模型评估的轨迹嵌入到模块的`forward`方法中：

```python
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
```

现在可以对跟踪的`ScriptModule`进行评估，使其与常规 PyTorch 模块相同：

```python
In[1]: output = traced_script_module(torch.ones(1, 3, 224, 224))
In[2]: output[0, :5]
Out[2]: tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)
```

### 通过Script转换为TORCHSCRIPT

在某些情况下，例如，如果您的模型采用特定形式的控制流，你可能想直接用Torch Script编写模型，并且相应地注释您的模型。例如，假设您有以下内容，原始Pytorch 模型：

```python
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

因为此模块的`forward`方法使用取决于输入的控制流，所以它不适合跟踪。相反，我们可以将其转换为`ScriptModule`。为了将模块转换为`ScriptModule`，需要使用`torch.jit.script`编译模块，如下所示：

```python
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

my_module = MyModule(10,20)
sm = torch.jit.script(my_module)
```

如果您需要在`nn.Module`中排除某些方法，因为它们使用的是 TorchScript 不支持的 Python 函数，则可以使用`@torch.jit.ignore`来注释这些方法

`sm`是已准备好进行序列化的`ScriptModule`的实例。

## 步骤 2：将脚本模块序列化为文件

跟踪或注解 PyTorch 模型后，一旦有了`ScriptModule`，就可以将其序列化为文件了。稍后，您将能够使用 C++ 从此文件加载模块并执行它，而无需依赖 Python。假设我们要序列化先前在跟踪示例中显示的`ResNet18`模型。要执行此序列化，只需在模块上调用[`save`](https://pytorch.org/docs/master/jit.html#torch.jit.ScriptModule.save)并为其传递文件名：

```python
traced_script_module.save("traced_resnet_model.pt")
```

这将在您的工作目录中生成一个`traced_resnet_model.pt`文件。 如果您还想序列化`sm`，请使用`sm.save("my_module_model.pt")`。我们现在已经正式离开 Python 领域，并准备跨入 C++ 领域。

## 步骤 3：在C++中加载脚本模块

要在C++中加载序列化的 PyTorch 模型，您的应用程序必须依赖于 PyTorch C++ API —— 也称为 LibTorch。LibTorch 发行版包含共享库、头文件和 CMake 构建的集合配置文件。虽然CMake不是 LibTorch 依赖的要求，这是推荐的方法，并将得到很好的支持前途。在本教程中，我们将C++使用 CMake 和 LibTorch，只需加载并执行序列化的 PyTorch 模型。

### 最少的C++应用

让我们从讨论加载模块的代码开始。以下内容已经完成：

```cpp
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}
```

`<torch/script.h>`标头包含了运行示例所需的 LibTorch 库中的所有相关包含。我们的应用接受序列化的 PyTorch `ScriptModule`的文件路径作为其唯一的命令行参数，然后继续使用`torch::jit::load()`函数对该模块进行反序列化，该函数将该文件路径作为输入。作为回报，我们收到一个`torch::jit::script::Module`对象。我们将稍后讨论如何执行它。

### 依赖于 LibTorch 并构建应用程序

假设我们将以上代码存储到名为`example-app.cpp`的文件中。 最小的`CMakeLists.txt`构建起来看起来很简单：

```cpp
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
```

构建示例应用程序所需的最后一件事是 LibTorch 分配。您可以随时从[下载页面](https://pytorch.org/)中获取最新的稳定版本。如果您下载并解压缩最新的存档，您应该会收到一个包含以下目录的文件夹结构：

```shell
libtorch/
  bin/
  include/
  lib/
  share/
```

*   `lib/`文件夹包含您必须链接的共享库，
*   `include/`文件夹包含程序需要包含的头文件，
*   `share/`文件夹包含必要的 CMake 配置，以启用上面的简单`find_package(Torch)`命令。

> 提示
>
> 在 Windows 上，调试和发行版本不兼容 ABI。如果计划以调试模式构建项目，请尝试使用 LibTorch 的调试版本。另外，请确保在下面的`cmake --build .`行中指定正确的配置。

最后一步是构建应用程序。为此，假设我们的示例 目录布局如下：

```shell
example-app/
  CMakeLists.txt
  example-app.cpp
```

现在，我们可以运行以下命令从`example-app/`文件夹中构建应用：

```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

其中`/path/to/libtorch`应该是解压缩的 LibTorch 发行版的完整路径。如果一切顺利，它将看起来像这样：

```shell
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

如果我们提供到先前创建的跟踪`ResNet18`模型`traced_resnet_model.pt`到生成的`example-app`二进制文件的路径，则应该以友好的“ok”来回报。请注意，如果尝试使用`my_module_model.pt`运行此示例，则会收到一条错误消息，提示您输入的形状不兼容。`my_module_model.pt`期望使用 1D 而不是 4D。

```shell
root@4b5a67132e81:/example-app/build# ./example-app <path_to_model>/traced_resnet_model.pt
ok
```

## 步骤 4：在C++中执行脚本模块

在用 C++ 成功加载序列化的`ResNet18`之后，我们现在离执行它仅几行代码了！ 让我们将这些行添加到 C++ 应用的`main()`函数中：

```cpp
// Create a vector of inputs.
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({1, 3, 224, 224}));

// Execute the model and turn its output into a tensor.
at::Tensor output = module.forward(inputs).toTensor();
std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
```

前两行设置了模型的输入。 我们创建一个`torch::jit::IValue`的向量（类型可擦除的值类型`script::Module`方法接受并返回），并添加单个输入。要创建输入张量，我们使用`torch::ones()`，等效于 python API 中的`torch.ones`。 然后，我们运行`script::Module`的`forward`方法，并将其传递给我们创建的输入向量。作为回报，我们得到了一个新的`IValue`，我们可以通过调用`toTensor()`将其转换为张量。

> 提示
>
> 要总体上了解有关`torch::ones`和 PyTorch C++ API 之类的功能的更多信息，请参阅[文档](https://pytorch.org/cppdocs)上的文档。PyTorch C++ API 提供了与 Python API 几乎相同的功能，使您可以像在 Python 中一样进一步操纵和处理张量。

在最后一行中，我们打印输出的前五个条目。由于我们在本教程前面的 Python 中为我们的模型提供了相同的输入，我们理想情况下，应该看到相同的输出。让我们通过重新编译我们的来尝试一下应用程序并使用相同的序列化模型运行它：

```cpp
root@4b5a67132e81:/example-app/build# make
Scanning dependencies of target example-app
[ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
[100%] Linking CXX executable example-app
[100%] Built target example-app
root@4b5a67132e81:/example-app/build# ./example-app traced_resnet_model.pt
-0.2698 -0.0381  0.4023 -0.3010 -0.0448
[ Variable[CPUFloatType]{1,5} ]
```

作为参考，Python 中的输出以前是：

```python
tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)
```

> 提示
>
> 要将模型移至 GPU 内存，可以编写`model.to(at::kCUDA);`。 通过调用`tensor.to(at::kCUDA)`来确保模型的输入也位于 CUDA 内存中，这将在 CUDA 内存中返回新的张量。

## 步骤 5：获取帮助并探索 API

本教程有望使您对 PyTorch 模型从 Python 到 C++ 的路径有一个大致的了解。 利用本教程中介绍的概念，您应该能够从原始的“急切的” PyTorch 模型，到 Python 中的已编译`ScriptModule`，再到磁盘上的序列化文件，以及–结束循环–到可执行文件`script::Module`在 C++ 中。

当然，有许多我们没有介绍的概念。 例如，您可能会发现自己想要扩展使用 C++ 或 CUDA 实现的自定义运算符来扩展`ScriptModule`，并希望在纯 C++ 生产环境中加载的`ScriptModule`内执行该自定义运算符。 好消息是：这是可能的，并且得到了很好的支持！ 现在，您可以浏览[此文件夹](https://github.com/pytorch/pytorch/tree/master/test/custom_operator)作为示例，我们将很快提供一个教程。目前，以下链接通常可能会有所帮助：

*   [Torch Script参考](https://pytorch.org/docs/master/jit.html)
*   [PyTorch C++ API 文档](https://pytorch.org/cppdocs/)
*   [PyTorch Python API 文档](https://pytorch.org/docs/)

与往常一样，如果您遇到任何问题或疑问，可以使用我们的[论坛](https://discuss.pytorch.org/)或 [GitHub Issues](https://github.com/pytorch/pytorch/issues) 进行联系。