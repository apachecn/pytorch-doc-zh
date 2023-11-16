


 使用 PyTorch C++ 前端
 [¶](#using-the-pytorch-c-frontend "永久链接到此标题")
=============================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/cpp_frontend>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/cpp_frontend.html>



PyTorch C++ 前端是 PyTorch 机器学习框架的纯 C++ 接口。虽然 PyTorch 的主要接口自然是 Python，但此 Python API 位于大量 C++ 代码库之上，提供基础数据结构和功能，例如张量和自动微分。 
C++ 前端公开了一个纯 C++11 API，该 API 使用机器学习训练和推理所需的工具扩展了这个底层 C++ 代码库。这包括用于神经网络建模的通用组件的内置集合；一个 API，用于
使用自定义模块扩展此集合；流行的优化\算法库，例如随机梯度下降；带有 API 的并行数据加载器，
用于定义和加载数据集；序列化例程等。




 本教程将引导您完成使用 C++ 前端训练模型的端到端示例。具体来说，我们将训练
 [DCGAN](https://arxiv.org/abs/1511.06434) 
 – 一种生成模型 – 来生成 MNIST 图像数字。虽然从概念上讲是一个简单的示例，但它应该足以让您对 PyTorch C++ 前端有一个旋风般的概述，并激发您训练更复杂模型的兴趣。我们将从一些
激励性词语开始解释为什么您想要使用 C++ 前端，
然后直接深入定义和训练我们的模型。





 提示




 观看
 [CppCon 2018 的闪电演讲](https://www.youtube.com/watch?v=auRPXMMHJzc)
 了解有关 C++ 前端的快速（幽默）
演示。






 提示




[本说明](https://pytorch.org/cppdocs/frontend.html)
 提供了 C++ 前端’s 组件和设计理念的全面概述。






 提示




 PyTorch C++ 生态系统的文档位于
 <https://pytorch.org/cppdocs>
 。您可以在那里找到高级描述以及
API 级文档。






 动机
 [¶](#motivation "此标题的永久链接")
---------------------------------------------------------------------



 在我们开始令人兴奋的 GAN 和 MNIST 数字之旅之前，让 ’s
 退后一步，讨论一下为什么您想要使用 C++ 前端而不是 
Python 前端。我们（PyTorch 团队）创建了 C++ 前端，
以便在无法使用 Python 或根本不是
适合该工作的工具的环境中进行研究。此类环境的示例包括：



* **低延迟系统** 
 ：您可能希望在具有高每秒帧数和低延迟
要求的纯 C++ 游戏引擎中进行强化学习研究。使用纯 C++ 库比 Python 库更适合这种环境。
 Python 可能根本无法处理，因为
Python 解释器速度缓慢。
* **高度多线程环境** 
 : 由于全局解释器锁
(GIL)，Python 无法运行多个系统线程一次。
多处理是一种替代方案，但可扩展性较差，并且有明显的
缺点。 C++ 没有这样的限制，线程很容易使用和创建。需要大量并行化的模型，例如 [Deep
Neuroevolution](https://eng.uber.com/deep-neuroevolution/) 中使用的模型，可以从中受益。
* **现有 C++ 代码库* * 
 ：您可能是现有 C++ 应用程序的所有者，该应用程序执行从在后端服务器中提供网页到在照片编辑软件中渲染 3D 图形等各种操作，并希望将机器学习方法集成到您的系统中。 C++ 前端允许您继续使用 C++，免去在 Python 和 C++ 之间来回绑定的麻烦，同时保留传统 PyTorch (Python) 体验的灵活性和直观性。



 C++ 前端无意与 Python 前端竞争。它是为了补充它。我们知道研究人员和工程师都喜欢 PyTorch，因为它的简单性、灵活性和直观的 API。我们的目标是确保您可以
在每种可能的环境（包括上述环境）中利用这些核心设计原则。如果这些场景之一很好地描述了您的用例，或者您只是感兴趣或好奇，请跟随我们
在以下段落中详细探索 C++ 前端。





 提示




 C++ 前端尝试提供与 Python 前端尽可能接近的 API。如果您对 Python 前端有经验，并且曾经问
自己 “h 如何使用 C++ 前端执行 X 操作？”，请像在 Python 中
那样编写代码，并且更常见的是C++ 中的函数和方法与 Python 中的函数和方法不同（只需记住用双冒号替换点）。







 编写基本应用程序
 [¶](#writing-a-basic-application "永久链接到此标题")
------------------------------------------------------------------------------------------------------------



 让’s 首先编写一个最小的 C++ 应用程序来验证我们’ 是否位于有关我们的设置和构建环境的同一页面上。首先，您需要
获取
 *LibTorch*
 发行版– 我们已构建好的 zip 存档的副本，该存档
打包了使用 C++ 所需的所有相关头文件、库和 CMake 构建文件
前端。 LibTorch 发行版可在 [PyTorch 网站](https://pytorch.org/get-started/locally/) 上下载，适用于 Linux、MacOS 和 Windows。本教程的其余部分将假设基本的 Ubuntu Linux
环境，但您也可以在 MacOS 或 Windows 上随意进行操作。





 提示




 有关
 [安装 PyTorch C++ 发行版](https://pytorch.org/cppdocs/installing.html) 的说明
 更详细地介绍了以下步骤。






 提示




 在 Windows 上，调试和发布版本不兼容 ABI。如果您打算
在调试模式下构建项目，请尝试 LibTorch 的调试版本。
此外，请确保在
 `cmake
 

 --build
 
\ 中指定正确的配置n.`
 下面一行。





 第一步是通过从 PyTorch 网站获取的链接在本地下载 LibTorch 发行版。对于普通 Ubuntu Linux 环境，
这意味着运行：






```
# If you need e.g. CUDA 9.0 support, please replace "cpu" with "cu90" in the URL below.
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

```




 接下来，让’s 编写一个名为
 `dcgan.cpp`
 的小 C++ 文件，其中包含
 `torch/torch.h`
，现在只需打印出一个 3三个恒等
矩阵:






```
#include <torch/torch.h>
#include <iostream>

int main() {
 torch::Tensor tensor = torch::eye(3);
 std::cout << tensor << std::endl;
}

```




 为了稍后构建这个小型应用程序以及成熟的训练脚本
，我们’ 将使用此
 `CMakeLists.txt`
 文件：






```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)

```





 没有10



 虽然 CMake 是 LibTorch 的推荐构建系统，但它并不是硬性要求。您还可以使用 Visual Studio 项目文件、QMake、普通
Makefile 或任何其他您觉得舒服的构建环境。但是，
我们不为此提供开箱即用的支持。





 记下上述 CMake 文件中的第 4 行：
 `find_package(Torch
 

 REQUIRED)`
 。
这指示 CMake 查找 LibTorch 库的构建配置。 
为了让 CMake 知道
 *在哪里*
 找到这些文件，我们必须在调用
 `cmake`
 时设置
 `CMAKE_PREFIX_PATH`
 。在执行此操作之前，让’s 同意
我们
 `dcgan`
 应用程序的以下目录结构:






```
dcgan/
 CMakeLists.txt
 dcgan.cpp

```




 此外，我将解压后的 LibTorch 发行版的路径称为
 `/path/to/libtorch`
 。请注意，此
 **必须是绝对路径** 
 。特别是，将 `CMAKE_PREFIX_PATH` 设置为 `../../libtorch`
 将会以意想不到的方式中断。相反，请写入
 `$PWD/../../libtorch`
 以获取
相应的绝对路径。现在，我们准备构建我们的应用程序：






```
root@fa350df05ecf:/home# mkdir build
root@fa350df05ecf:/home# cd build
root@fa350df05ecf:/home/build# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
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
-- Found torch: /path/to/libtorch/lib/libtorch.so
-- Configuring done
-- Generating done
-- Build files have been written to: /home/build
root@fa350df05ecf:/home/build# cmake --build . --config Release
Scanning dependencies of target dcgan
[ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
[100%] Linking CXX executable dcgan
[100%] Built target dcgan

```




 上面，我们首先在 `dcgan`
 目录中创建了
 `build`
 文件夹，
进入该文件夹，运行
 `cmake`
 命令来生成必要的构建
(Make) 文件并最终通过运行
 `cmake
 

 --build
 

 成功编译项目。
 

 --config
 

 Release `
 。现在我们已经准备好执行最小的二进制文件
并完成本节的基本项目配置：






```
root@fa350df05ecf:/home/build# ./dcgan
1 0 0
0 1 0
0 0 1
[ Variable[CPUFloatType]{3,3} ]

```




 对我来说看起来像一个单位矩阵！






 定义神经网络模型
 [¶](#defining-the-neural-network-models "永久链接到此标题")
------------------------------------------------------------------------------------------------------------



 现在我们已经配置了基本环境，我们可以深入了解
本教程中更有趣的部分。首先，我们将讨论如何在 C++ 前端中定义
模块并与之交互。我们’将从基本的小型示例模块开始，
然后使用 C++ 前端提供的
广泛的内置模块库实现成熟的 GAN。




### 
 模块 API 基础知识
 [¶](#module-api-basics "此标题的永久链接")



 与 Python 接口一致，基于 C++ 前端的神经网络
由称为
 *modules* 
 的可重用构建块组成。有一个基本模块
类，所有其他模块均从中派生。在 Python 中，此类为
 `torch.nn.Module`
，在 C++ 中为
 `torch::nn::Module`
 。除了实现模块封装的算法的
 `forward()`
 方法之外，模块通常还包含三种子对象中的任意一种：参数、缓冲区
和子模块。




 参数和缓冲区以张量的形式存储状态。参数记录梯度，而缓冲区则不记录。参数通常是神经网络的可训练权重。缓冲区的示例包括批量归一化的均值和方差。为了重用特定的逻辑和状态块，
PyTorch API 允许嵌套模块。嵌套模块称为
 *子模块* 
 。




 参数、缓冲区和子模块必须显式注册。注册后，
 `parameters()`
 或
 `buffers()`
 等方法可用于检索整个（嵌套）模块层次结构中所有参数的容器。
类似地， 
 `to(...)`
 ，例如
 `to(torch::kCUDA)`
 将所有
参数和缓冲区从 CPU 移动到 CUDA 内存，在整个模块
层次结构上工作。




#### 
 定义模块并注册参数
 [¶](#defining-a-module-and-registering-parameters "永久链接到此标题")



 要将这些词放入代码中，让’s 考虑用 Python 接口编写的这个简单模块：






```
import torch

class Net(torch.nn.Module):
  def __init__(self, N, M):
    super(Net, self).__init__()
    self.W = torch.nn.Parameter(torch.randn(N, M))
    self.b = torch.nn.Parameter(torch.randn(M))

  def forward(self, input):
    return torch.addmm(self.b, input, self.W)

```




 在 C++ 中，它看起来像这样：






```
#include <torch/torch.h>

struct Net : torch::nn::Module {
 Net(int64_t N, int64_t M) {
 W = register_parameter("W", torch::randn({N, M}));
 b = register_parameter("b", torch::randn(M));
 }
 torch::Tensor forward(torch::Tensor input) {
 return torch::addmm(b, input, W);
 }
 torch::Tensor W, b;
};

```




 就像在 Python 中一样，我们定义一个名为
 `Net`
 的类（为了简单起见，这里使用 
 `struct`
 而不是 
 `class`
 ）并从模块派生它基类。
在构造函数中，我们使用
 `torch::randn`
 创建张量，就像我们在 Python 中
使用
 `torch.randn`
 一样。一个有趣的区别是我们注册参数的方式。
在 Python 中，我们使用 `torch.nn.Parameter` 类包装张量，而在 C++ 中，我们必须通过 `register_parameter`
 方法传递张量。原因是 Python
API 可以检测到某个属性的类型为 `torch.nn.Parameter`
 并自动注册此类张量。在 C++ 中，反射非常有限，因此
提供了更传统（且不太神奇）的方法。





#### 
 注册子模块并遍历模块层次结构
 [¶](#registering-submodules-and-traversing-the-module-hierarchy "永久链接到此标题")



 就像我们可以注册参数一样，我们也可以注册子模块。在
Python 中，当子模块被指定为模块的属性时，
会自动检测并注册子模块:






```
class Net(torch.nn.Module):
  def __init__(self, N, M):
      super(Net, self).__init__()
      # Registered as a submodule behind the scenes
      self.linear = torch.nn.Linear(N, M)
      self.another_bias = torch.nn.Parameter(torch.rand(M))

  def forward(self, input):
    return self.linear(input) + self.another_bias

```




 例如，这允许使用
 `parameters()`
 方法来递归
访问模块层次结构中的所有参数：






```
>>> net = Net(4, 5)
>>> print(list(net.parameters()))
[Parameter containing:
tensor([0.0808, 0.8613, 0.2017, 0.5206, 0.5353], requires_grad=True), Parameter containing:
tensor([[-0.3740, -0.0976, -0.4786, -0.4928],
 [-0.1434, 0.4713, 0.1735, -0.3293],
 [-0.3467, -0.3858, 0.1980, 0.1986],
 [-0.1975, 0.4278, -0.1831, -0.2709],
 [ 0.3730, 0.4307, 0.3236, -0.0629]], requires_grad=True), Parameter containing:
tensor([ 0.2038, 0.4638, -0.2023, 0.1230, -0.0516], requires_grad=True)]

```




 要在 C++ 中注册子模块，请使用适当命名的
 `register_module()`
 方法
来注册类似
 `torch::nn::Linear` 的模块
 :






```
struct Net : torch::nn::Module {
 Net(int64_t N, int64_t M)
 : linear(register_module("linear", torch::nn::Linear(N, M))) {
 another_bias = register_parameter("b", torch::randn(M));
 }
 torch::Tensor forward(torch::Tensor input) {
 return linear(input) + another_bias;
 }
 torch::nn::Linear linear;
 torch::Tensor another_bias;
};

```





 提示




 您可以找到可用内置模块的完整列表，例如
 `torch::nn::Linear`
 、
 `torch::nn::Dropout`
 或
 `torch::nn ::Conv2d`
 在 
 `torch::nn`
 命名空间的文档中
 [此处](https://pytorch.org/cppdocs/api/namespace_torch__nn.html) 
.





 上述代码的一个微妙之处是为什么子模块是在构造函数’s 初始化列表中创建的，而参数是在构造函数主体内部创建的。这是有充分理由的，我们将在下面的 C++ 前端部分’s
 *所有权模型*
 中进一步讨论这一点。然而，最终
结果是我们可以递归地访问我们的模块树’s参数
就像在Python中一样。调用
 `parameters()`
 返回一个
 `std::vector<torch::Tensor>`
 ，我们可以对其进行迭代：






```
int main() {
 Net net(4, 5);
 for (const auto& p : net.parameters()) {
 std::cout << p << std::endl;
 }
}

```




 打印：






```
root@fa350df05ecf:/home/build# ./dcgan
0.0345
1.4456
-0.6313
-0.3585
-0.4008
[ Variable[CPUFloatType]{5} ]
-0.1647 0.2891 0.0527 -0.0354
0.3084 0.2025 0.0343 0.1824
-0.4630 -0.2862 0.2500 -0.0420
0.3679 -0.1482 -0.0460 0.1967
0.2132 -0.1992 0.4257 0.0739
[ Variable[CPUFloatType]{5,4} ]
0.01 *
3.6861
-10.1166
-45.0333
7.9983
-20.0705
[ Variable[CPUFloatType]{5} ]

```




 具有三个参数，就像在 Python 中一样。为了同时查看这些参数的名称，C++ API 提供了一个
 `named_parameters()`
 方法，该方法返回

 `OrderedDict`
 就像在 Python 中一样：






```
Net net(4, 5);
for (const auto& pair : net.named_parameters()) {
 std::cout << pair.key() << ": " << pair.value() << std::endl;
}

```




 我们可以再次执行以查看输出：






```
root@fa350df05ecf:/home/build# make && ./dcgan 11:13:48
Scanning dependencies of target dcgan
[ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
[100%] Linking CXX executable dcgan
[100%] Built target dcgan
b: -0.1863
-0.8611
-0.1228
1.3269
0.9858
[ Variable[CPUFloatType]{5} ]
linear.weight: 0.0339 0.2484 0.2035 -0.2103
-0.0715 -0.2975 -0.4350 -0.1878
-0.3616 0.1050 -0.4982 0.0335
-0.1605 0.4963 0.4099 -0.2883
0.1818 -0.3447 -0.1501 -0.0215
[ Variable[CPUFloatType]{5,4} ]
linear.bias: -0.0250
0.0408
0.3756
-0.2149
-0.3636
[ Variable[CPUFloatType]{5} ]

```





 没有10



[文档](https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#exhale-class-classtorch-1-1nn-1-1-module)
 for
 `torch::nn: :Module`
 包含在模块层次结构上运行的方法的完整列表。
。






#### 
 以转发模式运行网络
 [¶](#running-the-network-in-forward-mode "永久链接到此标题")



 要在 C++ 中执行网络，我们只需调用我们自己定义的
 `forward()`
 方法：






```
int main() {
 Net net(4, 5);
 std::cout << net.forward(torch::ones({2, 4})) << std::endl;
}

```




 打印如下内容：






```
root@fa350df05ecf:/home/build# ./dcgan
0.8559 1.1572 2.1069 -0.1247 0.8060
0.8559 1.1572 2.1069 -0.1247 0.8060
[ Variable[CPUFloatType]{2,5} ]

```





#### 
 模块所有权
 [¶](#module-ownership "此标题的永久链接")



 至此，我们知道如何用 C++ 定义模块，注册参数，
注册子模块，通过 `parameters()` 这样的方法遍历模块层次结构
，最后运行模块’s 
 `forward()`
 方法。虽然 C++ API 中还有更多方法、类和主题需要了解，但我将向您推荐
 [文档](https://pytorch.org/cppdocs/api/namespace_torch__nn.html)
 for\完整的菜单。当我们在一秒钟内实现 DCGAN 模型和端到端训练管道时，我们’ 还将触及更多概念。在我们这样做之前，
让我简单介绍一下
 *所有权模型* 
 C++ 前端提供
`torch::nn::Module` 的
子类
 。




 对于此讨论，所有权模型是指模块存储
和在 – 周围传递的方式，它决定了谁或什么
 *拥有* 
 特定模块\实例。在 Python 中，对象始终是动态分配的（在堆上）并且
具有引用语义。这非常容易使用且易于理解。
事实上，在 Python 中，您可以在很大程度上忘记对象所在的位置
以及它们如何被引用，而专注于完成工作。




 C++ 作为一种较低级别的语言，在这个领域提供了更多的选择。这会增加复杂性并严重影响 C++ 前端的设计和人体工程学。特别是，对于 C++ 前端中的模块，我们可以选择使用
 *任一* 
 值语义
 *或* 
 引用语义。第一种情况是
最简单的，在迄今为止的示例中已显示：模块对象在堆栈上分配
，并且当传递给函数时，可以复制、移动（使用
 `std::move`
 ）或通过引用或指针获取：






```
struct Net : torch::nn::Module { };

void a(Net net) { }
void b(Net& net) { }
void c(Net* net) { }

int main() {
 Net net;
 a(net);
 a(std::move(net));
 b(net);
 c(&net);
}

```




 对于第二种情况 – 引用语义 – 我们可以使用
 `std::shared_ptr`
 。
引用语义的优点是，就像Python，它减少了
思考如何将模块传递给函数和
如何声明参数的认知开销（假设您在任何地方都使用
 `shared_ptr`
）。






```
struct Net : torch::nn::Module {};

void a(std::shared_ptr<Net> net) { }

int main() {
 auto net = std::make_shared<Net>();
 a(net);
}

```




 根据我们的经验，来自动态语言的研究人员更喜欢引用语义而不是值语义，尽管后者比 C++ 更
“native”。还需要注意的是，
 `torch::nn::Module`
 ’s
设计为了接近 Python API 的人体工程学，依赖于
共享所有权。例如，采用我们之前（这里缩写的）
 `Net`
 的定义：






```
struct Net : torch::nn::Module {
 Net(int64_t N, int64_t M)
 : linear(register_module("linear", torch::nn::Linear(N, M)))
 { }
 torch::nn::Linear linear;
};

```




 为了使用 
 `线性`
 子模块，我们希望将其直接存储在我们的
类中。但是，我们还希望模块基类了解并有权访问该子模块。为此，它必须存储对此子模块的引用。至此，我们已经达到了共享所有权的需求。 `torch::nn::Module` 类和具体的 `Net` 类都需要对子模块的引用。因此，基类将模块存储为
 `shared_ptr`
 ，因此具体类也必须如此。




但是等等！我在上面的代码中没有看到 ’ 提到
 `shared_ptr`
 ！这是为什么
？好吧，因为 
 `std::shared_ptr<MyModule>`
 需要输入很多内容。为了
保持我们的研究人员的生产力，我们提出了一个精心设计的方案来隐藏
`shared_ptr`
 
 – 的提及，这是通常为值语义保留的好处 –\同时保留引用语义。要了解其工作原理，我们可以看一下核心库中 `torch::nn::Linear` 模块的简化定义（完整定义位于 [此处](https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/modules/linear.h) 
 ):






```
struct LinearImpl : torch::nn::Module {
 LinearImpl(int64_t in, int64_t out);

 Tensor forward(const Tensor& input);

 Tensor weight, bias;
};

TORCH_MODULE(Linear);

```




 简而言之：该模块不称为
 `Linear`
 ，而是
 `LinearImpl`
 。然后宏“TORCH_MODULE”定义了实际的“Linear”类。这个 “ generated”
 类实际上是 
 `std::shared_ptr<LinearImpl>`
 的包装器。它是一个包装器，而不是简单的 typedef，因此构造函数仍然按预期工作，即您仍然可以编写 
 `torch::nn::Linear(3,
 

 4) `
 而不是 
 `std::make_shared<LinearImpl>(3,
 

 4)`
 。我们将宏创建的类称为模块
 *holder* 
 。与（共享）指针一样，您可以使用箭头运算符访问
底层对象（如
 `model->forward(...)`
 ）。最终结果是一个与 Python API 非常相似的所有权模型。引用语义成为默认值，但无需额外输入
 `std::shared_ptr`
 或
 `std::make_shared`
 。对于我们的
 `Net`
 ，使用模块
holder API 如下所示：






```
struct NetImpl : torch::nn::Module {};
TORCH_MODULE(Net);

void a(Net net) { }

int main() {
 Net net;
 a(net);
}

```




 这里有一个微妙的问题值得一提。默认构造的
 `std::shared_ptr`
 是 “empty”，即包含一个空指针。什么是默认构造
 `Linear`
 或
 `Net`
 ？嗯，这是一个棘手的选择。我们可以说它
应该是一个空（null）
 `std::shared_ptr<LinearImpl>`
 。但是，请记住
 `Linear(3,
 

 4)`
 与
 `std::make_shared<LinearImpl>(3,
 

 4) 相同`
 。这
意味着，如果我们决定
 `Linear
 

 Linear;`
 应该是一个空指针，
那么就无法构造一个不接受任何
构造函数参数的模块，或默认全部。因此，在当前 API 中，默认构造的模块持有者（例如“Linear()”）会调用底层模块的默认构造函数（“LinearImpl()”）。如果
底层模块没有默认构造函数，则会出现编译器错误。
要改为构造空持有者，可以将
 `nullptr`
 传递给
持有者的构造函数。




 实际上，这意味着您可以像前面所示那样使用子模块，其中
模块在
 *初始化程序列表* 中注册和构造
 :






```
struct Net : torch::nn::Module {
 Net(int64_t N, int64_t M)
 : linear(register_module("linear", torch::nn::Linear(N, M)))
 { }
 torch::nn::Linear linear;
};

```




 或者您可以先用空指针构造持有者，然后在构造函数中分配给它
（Pythonista 更熟悉）：






```
struct Net : torch::nn::Module {
 Net(int64_t N, int64_t M) {
 linear = register_module("linear", torch::nn::Linear(N, M));
 }
 torch::nn::Linear linear{nullptr}; // construct an empty holder
};

```




 结论：您应该使用哪种所有权模型 – 哪种语义 –？ 
C++ 前端’s API 最好地支持模块持有者提供的所有权模型。
此机制的唯一缺点是模块声明下面
多了一行样板文件。也就是说，最简单的模型仍然是 C++ 模块简介中所示的值
语义模型。对于小型、简单的
脚本，您也可能会侥幸逃脱。但您’迟早会发现，由于
技术原因，它并不总是受支持。例如，序列化API（
 `torch::save`
 和
 `torch::load`
 ）仅支持模块持有者（或普通
 `shared_ptr`
 ）。因此，模块持有者 API 是使用 C++ 前端定义模块的推荐方法，
我们将在本教程中
使用此 API。






### 
 定义 DCGAN 模块
 [¶](#defining-the-dcgan-modules "此标题的永久链接")


我们现在有了必要的背景和介绍来定义我们想要在本文中解决的机器学习任务的模块。回顾一下：我们的任务是从 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/) 生成数字图像。我们希望使用
[生成对抗网络（GAN）](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
来解决
这个任务。特别是，我们’将使用
 [DCGAN架构](https://arxiv.org/abs/1511.06434)
–，这是第一个也是最简单的
类型之一，但完全足以完成此任务。





 提示




 您可以找到本教程中提供的完整源代码
 [在此存储库中](https://github.com/pytorch/examples/tree/master/cpp/dcgan) 
.





#### 
 什么是 GAN aGAN？
 [¶](#what-was-a-gan-agan "此标题的永久链接")



 GAN 由两个不同的神经网络模型组成： a
 *generator* 
 和 a
 *discriminator* 
 。生成器接收来自噪声分布的样本，其目标是将每个噪声样本转换为类似于目标分布 xe2x80x93 的图像，在我们的例子中是 MNIST 数据集。鉴别器依次接收来自 MNIST 数据集的
 *真实* 
 图像，或来自
生成器的
 *假* 
 图像。它被要求发出一个概率来判断特定图像的真实性（接近
 `1`
）或假性（接近
 `0`
）。来自鉴别器的关于生成器生成的图像的真实程度的反馈用于训练生成器。关于鉴别器真实性的洞察力的反馈用于优化鉴别器。理论上，生成器和鉴别器之间的微妙平衡使它们协同改进，导致生成器生成的图像与目标分布无法区分，欺骗鉴别器 xe2x80x99s（那时）优秀的眼睛发射
对于真实图像和虚假图像，概率均为
 `0.5`
。对于我们来说，最终结果
是一台接收噪声作为输入并生成
数字的真实图像作为输出的机器。





#### 
 生成器模块
 [¶](#the-generator-module "永久链接到此标题")



 我们首先定义生成器模块，它由一系列
转置 2D 卷积、批量归一化和 ReLU 激活单元组成。
我们在 `forward()` 中的模块之间显式传递输入（以函数方式） 
 我们自己定义的模块的方法:






```
struct DCGANGeneratorImpl : nn::Module {
 DCGANGeneratorImpl(int kNoiseSize)
 : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
 .bias(false)),
 batch_norm1(256),
 conv2(nn::ConvTranspose2dOptions(256, 128, 3)
 .stride(2)
 .padding(1)
 .bias(false)),
 batch_norm2(128),
 conv3(nn::ConvTranspose2dOptions(128, 64, 4)
 .stride(2)
 .padding(1)
 .bias(false)),
 batch_norm3(64),
 conv4(nn::ConvTranspose2dOptions(64, 1, 4)
 .stride(2)
 .padding(1)
 .bias(false))
 {
 // register_module() is needed if we want to use the parameters() method later on
 register_module("conv1", conv1);
 register_module("conv2", conv2);
 register_module("conv3", conv3);
 register_module("conv4", conv4);
 register_module("batch_norm1", batch_norm1);
 register_module("batch_norm2", batch_norm2);
 register_module("batch_norm3", batch_norm3);
 }

 torch::Tensor forward(torch::Tensor x) {
 x = torch::relu(batch_norm1(conv1(x)));
 x = torch::relu(batch_norm2(conv2(x)));
 x = torch::relu(batch_norm3(conv3(x)));
 x = torch::tanh(conv4(x));
 return x;
 }

 nn::ConvTranspose2d conv1, conv2, conv3, conv4;
 nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(DCGANGenerator);

DCGANGenerator generator(kNoiseSize);

```




 我们现在可以在
 `DCGANGGenerator`
 上调用
 `forward()`
 将噪声样本映射到图像。




 选择的特定模块，如
 `nn::ConvTranspose2d`
 和
 `nn::BatchNorm2d`
 ，
遵循前面概述的结构。 `kNoiseSize`
 常数决定输入噪声向量的大小，并设置为
 `100`
 。 
当然，超参数是通过研究生血统发现的。





注意




 没有研究生因超参数的发现而受到伤害。 
定期给它们喂食 Soylent。






 没有10



 简单介绍一下 C++ 前端中选项传递到内置模块（例如 
 `Conv2d`
）的方式：每个模块都有一些必需的选项，例如 
 `BatchNorm2d` 的功能数量
 。如果您只需要配置
所需的选项，则可以将它们直接传递给模块’s 构造函数，例如
 `BatchNorm2d(128)`
 或
 `Dropout(0.5)`
 或
 `Conv2d(8,
 

 4,
 

 2)`
 （用于输入通道数、输出通道数和内核大小）。但是，如果您需要
修改通常默认的其他选项，例如
 `bias`
 for
 `Conv2d`
 ，则需要构造并传递
 *options* 
 对象。 C++ 前端中的每个模块都有一个关联的选项结构，称为“ModuleOptions”，其中“Module”是模块的名称，例如“LinearOptions”代表“Linear” 
 。这就是我们对上面
 `Conv2d`
 模块所做的事情。






#### 
 鉴别器模块
 [¶](#the-discriminator-module "永久链接到此标题")



 鉴别器类似地是一系列卷积、批量归一化
和激活。然而，现在的卷积是常规的，而不是转置的，并且我们使用 alpha 值为 0.2 的泄漏 ReLU，而不是普通的 ReLU。此外，最终的激活变成了 Sigmoid，它将
值压缩到 0 到 1 之间的范围内。然后我们可以将这些压缩值
解释为鉴别器分配给真实图像的概率。




 为了构建鉴别器，我们将尝试一些不同的东西：
 
 顺序
 
 模块。
与 Python 中一样，PyTorch 这里提供了两个用于模型定义的 API：一个函数式 API
其中输入通过连续的函数传递（例如生成器模块示例），
以及一个更加面向对象的模型，我们构建一个
 
 顺序
 
 模块，其中包含
整个模型作为子模块。使用
 
 Sequential
 
 ，鉴别器将如下所示：






```
nn::Sequential discriminator(
 // Layer 1
 nn::Conv2d(
 nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
 nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
 // Layer 2
 nn::Conv2d(
 nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
 nn::BatchNorm2d(128),
 nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
 // Layer 3
 nn::Conv2d(
 nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
 nn::BatchNorm2d(256),
 nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
 // Layer 4
 nn::Conv2d(
 nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
 nn::Sigmoid());

```





 提示




 A
 `Sequential`
 模块仅执行函数组合。第一个子模块的输出
成为第二个子模块的输入，第三个子模块的输出
成为第四个子模块的输入，依此类推。









 正在加载数据
 [¶](#loading-data "此标题的固定链接")
-----------------------------------------------------------------------------------------


现在我们已经定义了生成器和判别器模型，我们需要一些可以用来训练这些模型的数据。 C++ 前端与 Python 前端一样，
配备了强大的并行数据加载器。此数据加载器可以从数据集中读取
批量数据（您可以自己定义），并提供许多
配置旋钮。





 注意




 虽然 Python 数据加载器使用多处理，但 C++ 数据加载器是真正的
多线程，并且不会启动任何新进程。





 数据加载器是 C++ 前端’s
 `data`
 api 的一部分，包含在
 `torch::data::`
 命名空间中。此 API 由几个不同的组件组成：



* 数据加载器类，
* 用于定义数据集的 API，
* 用于定义
 *transforms* 
 的 API，可应用于数据集，
* 用于定义
 *samplers* \ n ，生成用于索引数据集的索引，
* 现有数据集、转换和采样器的库。



 对于本教程，我们可以使用 C++ 前端附带的
 `MNIST`
 数据集。为此，让’s 实例化一个
 `torch::data::datasets::MNIST`
，并
应用两个转换：首先，我们对图像进行归一化，使它们在
的范围内
 n `-1`
 到
 `+1`
 （从原始范围
 `0`
 到
 `1`
 ）。
其次，我们应用
 `Stack` 
*collat​​ion* 
 ，它接受一批张量并
将它们沿第一个维度堆叠成单个张量：






```
auto dataset = torch::data::datasets::MNIST("./mnist")
 .map(torch::data::transforms::Normalize<>(0.5, 0.5))
 .map(torch::data::transforms::Stack<>());

```




 请注意，MNIST 数据集应位于
 `./mnist`
 目录中，
相对于您执行训练二进制文件的位置。您可以使用
 [此
脚本](https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03)
 下载 MNIST 数据集。




 接下来，我们创建一个数据加载器并将此数据集传递给它。要创建新的数据加载器，我们使用 `torch::data::make_data_loader`
 ，它返回正确类型的
 `std::unique_ptr`
 （这取决于
数据集的类型、采样器的类型和一些其他实现细节）：






```
auto data_loader = torch::data::make_data_loader(std::move(dataset));

```




 数据加载器确实有很多选项。您可以在[此处](https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/dataloader_options.h)检查全套
。
例如，为了加快数据加载速度，我们可以增加
worker的数量。默认数字为零，这意味着将使用主线程。
如果我们将 `workers`
 设置为
 `2`
 ，将生成两个线程来同时加载数据。我们还应该将批量大小从默认值
 `1`
 增加到更合理的值，例如
 `64`
 （
 `kBatchSize`
 的值）。因此
让’s 创建一个
 `DataLoaderOptions`
 对象并设置适当的属性:






```
auto data_loader = torch::data::make_data_loader(
 std::move(dataset),
 torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

```




 我们现在可以编写一个循环来加载批量数据，’ 目前仅将其打印到
控制台：






```
for (torch::data::Example<>& batch : *data_loader) {
 std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
 for (int64_t i = 0; i < batch.data.size(0); ++i) {
 std::cout << batch.target[i].item<int64_t>() << " ";
 }
 std::cout << std::endl;
}

```




 在这种情况下，数据加载器返回的类型是 
 `torch::data::Example`
 。
此类型是一个简单的结构体，带有
 `data`
 数据字段和标签的
 `target`
 字段。因为我们之前应用了
 `Stack`
 排序规则，
数据加载器仅返回一个这样的示例。如果我们没有应用
排序规则，数据加载器将生成
 `std::vector<torch::data::Example<>>`
，批处理中的每个示例都有一个元素。




 如果您重建并运行此代码，您应该看到如下内容：






```
root@fa350df05ecf:/home/build# make
Scanning dependencies of target dcgan
[ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
[100%] Linking CXX executable dcgan
[100%] Built target dcgan
root@fa350df05ecf:/home/build# make
[100%] Built target dcgan
root@fa350df05ecf:/home/build# ./dcgan
Batch size: 64 | Labels: 5 2 6 7 2 1 6 7 0 1 6 2 3 6 9 1 8 4 0 6 5 3 3 0 4 6 6 6 4 0 8 6 0 6 9 2 4 0 2 8 6 3 3 2 9 2 0 1 4 2 3 4 8 2 9 9 3 5 8 0 0 7 9 9
Batch size: 64 | Labels: 2 2 4 7 1 2 8 8 6 9 0 2 2 9 3 6 1 3 8 0 4 4 8 8 8 9 2 6 4 7 1 5 0 9 7 5 4 3 5 4 1 2 8 0 7 1 9 6 1 6 5 3 4 4 1 2 3 2 3 5 0 1 6 2
Batch size: 64 | Labels: 4 5 4 2 1 4 8 3 8 3 6 1 5 4 3 6 2 2 5 1 3 1 5 0 8 2 1 5 3 2 4 4 5 9 7 2 8 9 2 0 6 7 4 3 8 3 5 8 8 3 0 5 8 0 8 7 8 5 5 6 1 7 8 0
Batch size: 64 | Labels: 3 3 7 1 4 1 6 1 0 3 6 4 0 2 5 4 0 4 2 8 1 9 6 5 1 6 3 2 8 9 2 3 8 7 4 5 9 6 0 8 3 0 0 6 4 8 2 5 4 1 8 3 7 8 0 0 8 9 6 7 2 1 4 7
Batch size: 64 | Labels: 3 0 5 5 9 8 3 9 8 9 5 9 5 0 4 1 2 7 7 2 0 0 5 4 8 7 7 6 1 0 7 9 3 0 6 3 2 6 2 7 6 3 3 4 0 5 8 8 9 1 9 2 1 9 4 4 9 2 4 6 2 9 4 0
Batch size: 64 | Labels: 9 6 7 5 3 5 9 0 8 6 6 7 8 2 1 9 8 8 1 1 8 2 0 7 1 4 1 6 7 5 1 7 7 4 0 3 2 9 0 6 6 3 4 4 8 1 2 8 6 9 2 0 3 1 2 8 5 6 4 8 5 8 6 2
Batch size: 64 | Labels: 9 3 0 3 6 5 1 8 6 0 1 9 9 1 6 1 7 7 4 4 4 7 8 8 6 7 8 2 6 0 4 6 8 2 5 3 9 8 4 0 9 9 3 7 0 5 8 2 4 5 6 2 8 2 5 3 7 1 9 1 8 2 2 7
Batch size: 64 | Labels: 9 1 9 2 7 2 6 0 8 6 8 7 7 4 8 6 1 1 6 8 5 7 9 1 3 2 0 5 1 7 3 1 6 1 0 8 6 0 8 1 0 5 4 9 3 8 5 8 4 8 0 1 2 6 2 4 2 7 7 3 7 4 5 3
Batch size: 64 | Labels: 8 8 3 1 8 6 4 2 9 5 8 0 2 8 6 6 7 0 9 8 3 8 7 1 6 6 2 7 7 4 5 5 2 1 7 9 5 4 9 1 0 3 1 9 3 9 8 8 5 3 7 5 3 6 8 9 4 2 0 1 2 5 4 7
Batch size: 64 | Labels: 9 2 7 0 8 4 4 2 7 5 0 0 6 2 0 5 9 5 9 8 8 9 3 5 7 5 4 7 3 0 5 7 6 5 7 1 6 2 8 7 6 3 2 6 5 6 1 2 7 7 0 0 5 9 0 0 9 1 7 8 3 2 9 4
Batch size: 64 | Labels: 7 6 5 7 7 5 2 2 4 9 9 4 8 7 4 8 9 4 5 7 1 2 6 9 8 5 1 2 3 6 7 8 1 1 3 9 8 7 9 5 0 8 5 1 8 7 2 6 5 1 2 0 9 7 4 0 9 0 4 6 0 0 8 6
...

```




 这意味着我们能够成功地从 MNIST 数据集中加载数据。






 编写训练循环
 [¶](#writing-the-training-loop "永久链接到此标题")
------------------------------------------------------------------------------------------



 现在让’s 完成示例的算法部分，并实现生成器和鉴别器之间的微妙
dance。首先，我们’将创建两个优化器，一个用于生成器，一个用于鉴别器。我们使用的优化器实现了
 [Adam](https://arxiv.org/pdf/1412.6980.pdf)
 算法：






```
torch::optim::Adam generator_optimizer(
 generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
torch::optim::Adam discriminator_optimizer(
 discriminator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.5)));

```





 没有10



 在撰写本文时，C++ 前端提供了实现 Adagrad、
Adam、LBFGS、RMSprop 和 SGD 的优化器。 
 [文档](https://pytorch.org/cppdocs/api/namespace_torch__optim.html)
 具有最新的列表。





 接下来，我们需要更新我们的训练循环。我们’将添加一个外循环来耗尽
每个时期的数据加载器，然后编写GAN训练代码：






```
for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
 int64_t batch_index = 0;
 for (torch::data::Example<>& batch : *data_loader) {
 // Train discriminator with real images.
 discriminator->zero_grad();
 torch::Tensor real_images = batch.data;
 torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
 torch::Tensor real_output = discriminator->forward(real_images);
 torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
 d_loss_real.backward();

 // Train discriminator with fake images.
 torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1});
 torch::Tensor fake_images = generator->forward(noise);
 torch::Tensor fake_labels = torch::zeros(batch.data.size(0));
 torch::Tensor fake_output = discriminator->forward(fake_images.detach());
 torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
 d_loss_fake.backward();

 torch::Tensor d_loss = d_loss_real + d_loss_fake;
 discriminator_optimizer.step();

 // Train generator.
 generator->zero_grad();
 fake_labels.fill_(1);
 fake_output = discriminator->forward(fake_images);
 torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
 g_loss.backward();
 generator_optimizer.step();

 std::printf(
 "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
 epoch,
 kNumberOfEpochs,
 ++batch_index,
 batches_per_epoch,
 d_loss.item<float>(),
 g_loss.item<float>());
 }
}

```


上面，我们首先在真实图像上评估鉴别器，它应该分配一个高概率。为此，我们使用
 `torch::empty(batch.data.size(0)).uniform_(0.8,
 

 1.0)`
 作为目标
概率。





 注意




 我们在各处选择均匀分布在 0.8 和 1.0 之间的随机值，而不是 1.0
，以便使判别器训练更加稳健。这个技巧称为
 *标签平滑* 
 。



在评估鉴别器之前，我们将其参数的梯度归零。计算损失后，我们通过调用
`d_loss.backward()`
将其通过网络反向传播以计算新的梯度。我们对假图像重复这个说法。我们不使用数据集中的图像，而是让生成器通过输入一批随机噪声来为此创建假图像。然后我们将这些假图像转发给鉴别器。这次，我们希望鉴别器发出低概率，最好是全零。一旦我们计算出了
一批真实图像和一批
假图像的鉴别器损失，我们就可以将鉴别器’s优化器前进一步，
以更新其参数。



为了训练生成器，我们首先将其梯度归零，然后重新评估假图像上的鉴别器。然而，这一次我们希望判别器分配的概率非常接近 1，这表明生成器可以生成图像来欺骗判别器，使其认为它们实际上是真实的（来自数据集）。为此，我们用全 1 填充 
 `fake_labels`
 张量。最后，我们逐步执行生成器’s 优化器来更新
its 参数。




 现在我们应该准备好在 CPU 上训练我们的模型了。我们还没有任何代码来捕获状态或样本输出，但我们稍后会添加它。现在，让’s 观察我们的模型正在做
 *某事* 
 – 我们’
稍后
根据生成的图像验证这是否是有意义。
重新构建并运行应该打印如下内容：






```
root@3c0711f20896:/home/build# make && ./dcgan
Scanning dependencies of target dcgan
[ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
[100%] Linking CXX executable dcgan
[100%] Built target dcga
[ 1/10][100/938] D_loss: 0.6876 | G_loss: 4.1304
[ 1/10][200/938] D_loss: 0.3776 | G_loss: 4.3101
[ 1/10][300/938] D_loss: 0.3652 | G_loss: 4.6626
[ 1/10][400/938] D_loss: 0.8057 | G_loss: 2.2795
[ 1/10][500/938] D_loss: 0.3531 | G_loss: 4.4452
[ 1/10][600/938] D_loss: 0.3501 | G_loss: 5.0811
[ 1/10][700/938] D_loss: 0.3581 | G_loss: 4.5623
[ 1/10][800/938] D_loss: 0.6423 | G_loss: 1.7385
[ 1/10][900/938] D_loss: 0.3592 | G_loss: 4.7333
[ 2/10][100/938] D_loss: 0.4660 | G_loss: 2.5242
[ 2/10][200/938] D_loss: 0.6364 | G_loss: 2.0886
[ 2/10][300/938] D_loss: 0.3717 | G_loss: 3.8103
[ 2/10][400/938] D_loss: 1.0201 | G_loss: 1.3544
[ 2/10][500/938] D_loss: 0.4522 | G_loss: 2.6545
...

```






 移动到 GPU
 [¶](#moving-to-the-gpu "固定链接到此标题")
--------------------------------------------------------------------------



 虽然我们当前的脚本可以在 CPU 上运行得很好，但我们都知道卷积
在 GPU 上要快得多。让’s 快速讨论如何将训练转移到
GPU 上。我们’需要为此做两件事：将GPU设备规范
传递给我们自己分配的张量，并通过
`to()`
方法将任何其他张量显式复制到
GPU上C++ 前端中的张量和模块具有。
实现这两个目标的最简单方法是在训练脚本的顶层创建
 `torch::Device`
 实例，然后将该设备传递给张量
工厂类似
 `torch::zeros`
 的函数以及
 `to()`
 方法。我们可以
从使用 CPU 设备执行此操作开始：






```
// Place this somewhere at the top of your training script.
torch::Device device(torch::kCPU);

```




 新的张量分配，例如






```
torch::Tensor fake_labels = torch::zeros(batch.data.size(0));

```




 应更新为将
 `设备`
 作为最后一个参数:






```
torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);

```




 对于其创建不在我们手中的张量，例如来自 MNIST
数据集的张量，我们必须插入显式
 `to()`
 调用。这意味着






```
torch::Tensor real_images = batch.data;

```




 变为






```
torch::Tensor real_images = batch.data.to(device);

```




 我们的模型参数也应该移动到正确的设备：






```
generator->to(device);
discriminator->to(device);

```





 没有10



 如果张量已经存在于提供给
 `to()`
 的设备上，则该调用是
无操作。不制作额外的副本。





 此时，我们’ 刚刚使之前驻留在 CPU 的代码更加明确。
但是，现在将设备更改为 CUDA 设备也非常容易：






```
torch::Device device(torch::kCUDA)

```


现在，所有张量都将存在于 GPU 上，调用快速 CUDA 内核来执行所有操作，而无需更改任何下游代码。如果我们想要
指定特定的设备索引，可以将其作为第二个参数传递给
`Device`
 构造函数。如果我们希望不同的张量存在于不同的设备上，我们可以传递单独的设备实例（例如，一个在 CUDA 设备 0 上，另一个在 CUDA 设备 1 上）。我们甚至可以动态地
进行此配置，这通常有助于使我们的训练脚本更加可移植：






```
torch::Device device = torch::kCPU;
if (torch::cuda::is_available()) {
 std::cout << "CUDA is available! Training on GPU." << std::endl;
 device = torch::kCUDA;
}

```




 甚至






```
torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

```






 检查点并恢复训练状态
 [¶](#checkpointing-and-recovering-the-training-state "永久链接到此标题")
---------------------------------------------------------------------------------------------------------------------------------------


我们应该对训练脚本进行的最后一个增强是定期保存模型参数的状态、优化器的状态以及一些生成的图像样本。如果我们的计算机在训练过程中
崩溃，前两个将使我们能够恢复训练状态。
对于持久的训练课程，这是绝对必要的。幸运的是，
C++ 前端提供了一个 API 来序列化和反序列化模型和
优化器状态以及各个张量。




 其核心 API 是
 `torch::save(thing,filename)`
 和
 `torch::load(thing,filename)`
 ，其中
 `thing`
 可以是
 `torch::nn::Module`
 子类或优化器实例，例如 
 `Adam`
 对象
我们在训练脚本中拥有。让’s 更新我们的训练循环，以按一定的时间间隔检查
模型和优化器状态：






```
if (batch_index % kCheckpointEvery == 0) {
 // Checkpoint the model and optimizer state.
 torch::save(generator, "generator-checkpoint.pt");
 torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
 torch::save(discriminator, "discriminator-checkpoint.pt");
 torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
 // Sample the generator and save the images.
 torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
 torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
 std::cout << "-> checkpoint " << ++checkpoint_counter << '';
}

```




 其中
 `kCheckpointEvery`
 是一个整数，设置为
 `100`
 之类的值，
每
 `100`
 批次检查一次，并且
 `checkpoint_counter`
 
每次我们创建检查点时，计数器都会发生碰撞。




 要恢复训练状态，您可以在创建所有模型和
优化器之后、训练循环之前添加如下行：






```
torch::optim::Adam generator_optimizer(
 generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
torch::optim::Adam discriminator_optimizer(
 discriminator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));

if (kRestoreFromCheckpoint) {
 torch::load(generator, "generator-checkpoint.pt");
 torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
 torch::load(discriminator, "discriminator-checkpoint.pt");
 torch::load(
 discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
}

int64_t checkpoint_counter = 0;
for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
 int64_t batch_index = 0;
 for (torch::data::Example<>& batch : *data_loader) {

```






 检查生成的图像
 [¶](#inspecting- generated-images "此标题的永久链接")
------------------------------------------------------------------------------------------------



 我们的训练脚本现已完成。我们已准备好训练我们的 GAN，无论是在 CPU 还是 GPU 上。为了检查训练过程的中间输出，我们添加了代码来定期将图像样本保存到“dcgan-sample-xxx.pt”文件中，我们可以编写一个简短的 Python 脚本来加载
张量并使用 matplotlib 显示它们：






```
import argparse

import matplotlib.pyplot as plt
import torch


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--sample-file", required=True)
parser.add_argument("-o", "--out-file", default="out.png")
parser.add_argument("-d", "--dimension", type=int, default=3)
options = parser.parse_args()

module = torch.jit.load(options.sample_file)
images = list(module.parameters())[0]

for index in range(options.dimension * options.dimension):
  image = images[index].detach().cpu().reshape(28, 28).mul(255).to(torch.uint8)
  array = image.numpy()
  axis = plt.subplot(options.dimension, options.dimension, 1 + index)
  plt.imshow(array, cmap="gray")
  axis.get_xaxis().set_visible(False)
  axis.get_yaxis().set_visible(False)

plt.savefig(options.out_file)
print("Saved ", options.out_file)

```




 现在让’s 训练我们的模型大约 30 个时期：






```
root@3c0711f20896:/home/build# make && ./dcgan 10:17:57
Scanning dependencies of target dcgan
[ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
[100%] Linking CXX executable dcgan
[100%] Built target dcgan
CUDA is available! Training on GPU.
[ 1/30][200/938] D_loss: 0.4953 | G_loss: 4.0195
-> checkpoint 1
[ 1/30][400/938] D_loss: 0.3610 | G_loss: 4.8148
-> checkpoint 2
[ 1/30][600/938] D_loss: 0.4072 | G_loss: 4.36760
-> checkpoint 3
[ 1/30][800/938] D_loss: 0.4444 | G_loss: 4.0250
-> checkpoint 4
[ 2/30][200/938] D_loss: 0.3761 | G_loss: 3.8790
-> checkpoint 5
[ 2/30][400/938] D_loss: 0.3977 | G_loss: 3.3315
...
-> checkpoint 120
[30/30][938/938] D_loss: 0.3610 | G_loss: 3.8084

```




 并在图中显示图像：






```
root@3c0711f20896:/home/build# python display.py -i dcgan-sample-100.pt
Saved out.png

```




 看起来应该像这样：




![数字](https://pytorch.org/tutorials/_images/digits.png)


 数字！万岁！现在球在你的场上：你可以改进模型以使
数字看起来更好吗？






 结论
 [¶](#conclusion "此标题的永久链接")
---------------------------------------------------------------------



 本教程希望为您提供 PyTorch C++ 前端的易于理解的摘要。像 PyTorch 这样的机器学习库必然具有非常广泛的 API。因此，有许多概念我们没有时间或空间在这里讨论。不过，我鼓励您尝试该 API，并查阅
 [我们的文档](https://pytorch.org/cppdocs/) 
，特别是
 [库 API](https://pytorch. org/cppdocs/api/library_root.html) 
 当你遇到困难时
部分。另外，请记住，只要我们能够做到这一点，您就可以期望 C++ 前端
遵循 Python 前端的
设计和语义，因此您可以利用这一事实来提高学习速度。





 提示




 您可以找到本教程中提供的完整源代码
 [在此存储库中](https://github.com/pytorch/examples/tree/master/cpp/dcgan) 
.





 一如既往，如果您遇到任何问题或有疑问，可以使用我们的
 [论坛](https://discuss.pytorch.org/) 
 或
 [GitHub 问题](https://github.com/pytorch/pytorch/issues) 
 取得联系。









