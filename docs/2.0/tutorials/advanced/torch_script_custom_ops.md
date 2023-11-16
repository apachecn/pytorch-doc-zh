


 使用自定义 C++ 运算符扩展 TorchScript
 [¶](#extending-torchscript-with-custom-c-operators "固定链接到此标题")
================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/torch_script_custom_ops>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>




 PyTorch 1.0 版本为 PyTorch 引入了一种新的编程模型，称为
 [TorchScript](https://pytorch.org/docs/master/jit.html) 
 。 TorchScript 是 Python 编程语言的子集，可以由 TorchScript 编译器解析、编译和优化。此外，已编译的 TorchScript 模型
可以选择序列化为磁盘文件格式，
您随后可以从纯 C++（以及 Python）加载并运行该文件格式以进行推理。




 TorchScript 支持 `torch`
 包提供的大量操作子集，允许您将多种复杂模型纯粹表达为 PyTorch

一系列张量操作\xe2\x80\x99s \xe2\x80 \x9c标准库\xe2\x80\x9d。尽管如此，有时您可能会发现自己需要使用自定义 C++ 或 CUDA 函数来扩展 TorchScript。虽然我们建议您仅在
您的想法无法（足够有效）表达为简单的 Python 函数时才使用此选项，
我们确实提供了一个非常友好且简单的界面，用于使用
[ATen]定义自定义 C++ 和
CUDA 内核](https://pytorch.org/cppdocs/#aten) 
 ，PyTorch\xe2\x80\x99s 高性能 C++ 张量库。绑定到 TorchScript 后，您可以将这些
自定义内核（或 \xe2\x80\x9cops\xe2\x80\x9d）嵌入到您的 TorchScript 模型中，并在
Python 中执行它们，并直接在 C++ 中以序列化形式执行它们。




 以下段落给出了编写 TorchScript 自定义操作以调用 
 [OpenCV](https://www.opencv.org) 
 的示例，这是一个用 C++ 编写的计算机视觉库。我们将讨论如何在 C++ 中使用张量、如何有效
将它们转换为第三方张量格式（在本例中为 OpenCV
 `Mat`
 ）、
如何向 TorchScript 运行时注册您的运算符以及最后如何
编译该运算符并在 Python 和 C++ 中使用它。





 在 C++ 中实现自定义运算符
 [¶](#implementing-the-custom-operator-in-c "固定链接到此标题")
---------------------------------------------------------------------------------------------------------------------



 对于本教程，我们’ 将公开
 [warpPerspective](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective) 
 函数，它将透视变换应用于图像，从 OpenCV 到
TorchScript 作为自定义运算符。第一步是用 C++ 编写自定义运算符的实现。让’s 调用此实现的文件
 `op.cpp`
 并使其如下所示：






```
torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
 // BEGIN image_mat
 cv::Mat image_mat(/*rows=*/image.size(0),
 /*cols=*/image.size(1),
 /*type=*/CV_32FC1,
 /*data=*/image.data_ptr<float>());
 // END image_mat

 // BEGIN warp_mat
 cv::Mat warp_mat(/*rows=*/warp.size(0),
 /*cols=*/warp.size(1),
 /*type=*/CV_32FC1,
 /*data=*/warp.data_ptr<float>());
 // END warp_mat

 // BEGIN output_mat
 cv::Mat output_mat;
 cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{8, 8});
 // END output_mat

 // BEGIN output_tensor
 torch::Tensor output = torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{8, 8});
 return output.clone();
 // END output_tensor
}

```




 该运算符的代码非常短。在文件的顶部，我们包含 OpenCV 头文件，
 `opencv2/opencv.hpp`
 ，以及
 `torch/script.h`
 头文件，它公开了 PyTorch 中所有必要的功能\ xe2\x80\x99s C++ API，我们需要编写自定义 TorchScript 运算符。我们的函数
 `warp_perspective`
 有两个参数：一个输入
 `image`
 和
 我们希望应用于图像的
 `warp`
 变换矩阵。这些输入的类型是
 `torch::Tensor`
 、
PyTorch\xe2\x80\x99s C++ 中的张量类型（这也是 Python 中所有张量的基础类型）。我们的
 `warp_perspective`
 函数的返回类型也将是
 `torch::Tensor`
 。





 提示




 请参阅
 [本说明](https://pytorch.org/cppdocs/notes/tensor_basics.html) 
 了解有关 ATen 的更多信息，该库提供
 `Tensor`
 类
 nPyTorch。此外，
 [本教程](https://pytorch.org/cppdocs/notes/tensor_creation.html)
 描述了如何在 C++ 中
分配和初始化新的张量对象（此运算符
不需要）。






 注意




 TorchScript 编译器可以识别固定数量的类型。只有这些类型
可以用作自定义运算符的参数。目前这些类型是：
 `torch::Tensor`
 、
 `torch::Scalar`
 、
 `double`
 、
 `int64_t`
 和
 `std ::vector`
 这些类型。请注意，
 *仅* 
`double`
 和
 *不* 
`float`
 和
 *仅* 
`int64_t`
 和
 *不* 
 支持其他整型，例如
 `int`
 、
 `short`
 或
 `long`
。





 在我们的函数内部，我们需要做的第一件事是将 PyTorch
张量转换为 OpenCV 矩阵，如 OpenCV’s
 `warpPerspective`
 需要 
 `cv::Mat` 
 对象作为输入。幸运的是，有一种方法可以做到这一点
 **无需复制
任何** 
 数据。在前几行中，






```
 cv::Mat image_mat(/*rows=*/image.size(0),
 /*cols=*/image.size(1),
 /*type=*/CV_32FC1,
 /*data=*/image.data_ptr<float>());

```




 我们正在调用 OpenCV
 `Mat`
 类的
 [此构造函数](https://docs.opencv.org/trunk/d3/d63/classcv_1_1Mat.html#a922de793eabcec705b3579c5f95a643e)将我们的张量转换为
 `Mat`
 对象。我们传递
原始
 `image`
张量的行数和列数、数据类型
（我们’将在本例中将其修复为
 `float32`
），以及最后是指向底层数据 – a
 `float*`
 的原始指针。 `Mat` 类的这个构造函数的特殊之处在于它不复制输入数据。相反，它会简单地引用此内存以执行在 `Mat` 上执行的所有操作。如果对 
 `image_mat`
 执行就地操作，这将反映在
原始
 `image`
 张量中（反之亦然）。这允许我们使用库’s 本机矩阵类型调用
后续 OpenCV 例程，即使
我们’ 实际上将数据存储在 PyTorch 张量中。我们重复此过程，将
 `warp`
 PyTorch 张量转换为
 `warp_mat`
 OpenCV 矩阵：






```
 cv::Mat warp_mat(/*rows=*/warp.size(0),
 /*cols=*/warp.size(1),
 /*type=*/CV_32FC1,
 /*data=*/warp.data_ptr<float>());

```




 接下来，我们准备调用我们非常想在 TorchScript 中使用的 OpenCV 函数：
 `warpPerspective`
 。为此，我们向 OpenCV 函数传递
 `image_mat`
 和
 `warp_mat`
 矩阵，以及一个空的输出矩阵
称为
 `output_mat`\名词我们还指定了我们想要的输出矩阵（图像）的大小。对于此示例，它被硬编码为
 `8
 

 x
 

 8`
:






```
 cv::Mat output_mat;
 cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{8, 8});

```




 自定义运算符实现的最后一步是将 
 `output_mat`
 转换回 PyTorch 张量，以便我们可以在 
PyTorch 中进一步使用它。这与我们之前在另一个方向上进行的转换惊人地相似。在这种情况下，PyTorch 提供了
 `torch::from_blob`
 方法。在这种情况下，A
 *blob* 
 旨在表示一些不透明的、指向内存的平面指针，
我们希望将其解释为 PyTorch 张量。对
`torch::from_blob`的调用
看起来像这样：






```
 torch::Tensor output = torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{8, 8});
 return output.clone();

```




 我们使用 OpenCV
 `Mat`
 类上的
 `.ptr<float>()`
 方法来获取指向底层数据的原始
指针（就像
 `.data _ptr<float>()`
 用于之前的 PyTorch
张量）。我们还指定张量的输出形状，
将其硬编码为
 `8
 

 x
 

 8`
 。 
 `torch::from_blob`
 的输出是
 `torch::Tensor`
 ，指向 OpenCV 矩阵拥有的内存。




 在从运算符实现返回此张量之前，我们必须在张量上调用
 `.clone()`
 以执行底层数据的内存复制。原因是 `torch::from_blob` 返回一个不拥有数据的张量。此时，数据仍归 OpenCV 矩阵所有。然而，这个 OpenCV 矩阵将超出范围并在函数结束时被释放。如果我们按原样返回“输出”张量，那么当我们在函数外部使用它时，它会指向无效的内存。调用
 `.clone()`
 返回
新张量以及新张量本身拥有的原始数据的副本。
因此可以安全地返回到外部世界。






 使用 TorchScript 注册自定义运算符
 [¶](#registering-the-custom-operator-with-torchscript "永久链接到此标题")
-----------------------------------------------------------------------------------------------------------------------------------------



 现在已经在 C++ 中实现了我们的自定义运算符，我们需要
 *注册* 
 它
与 TorchScript 运行时和编译器。这将允许 TorchScript
编译器解析 TorchScript 代码中对我们的自定义运算符的引用。
如果您曾经使用过 pybind11 库，我们的注册语法
与 pybind11 语法非常相似。要注册单个函数，
我们编写：






```
TORCH_LIBRARY(my_ops, m) {
 m.def("warp_perspective", warp_perspective);
}

```




 位于我们
 `op.cpp`
 文件顶层的某个位置。 
 `TORCH_LIBRARY`
 宏创建一个在程序启动时调用的函数。您的库的名称 (
 `my_ops`
 ) 作为第一个参数给出（不应包含在引号中）。第二个参数 (
 `m`
 ) 定义类型为
 `torch::Library`
 的变量，它是注册运算符的主接口。
方法
 `Library::def`
实际上创建了一个名为
 `warp_perspective`
 的运算符，
将其暴露给 Python 和 TorchScript。您可以通过多次调用
 `def`
 来定义任意数量的运算符。




 在幕后，
 `def`
 函数实际上做了相当多的工作：
 正在使用模板元编程来检查
函数的类型签名，并将其转换为指定运算符的运算符模式
在 TorchScript’s 类型系统中键入。






 构建自定义运算符
 [¶](#building-the-custom-operator "永久链接到此标题")
------------------------------------------------------------------------------------------------



 现在我们已经在 C++ 中实现了自定义运算符并编写了其
注册代码，是时候将该运算符构建到（共享）库中
我们可以将其加载到 Python 中进行研究和实验，或者加载到 C++ 中进行
推理在没有Python的环境中。有多种方法可以使用纯 CMake 或 Python 替代方案（例如 `setuptools`）来构建我们的运算符。
为简洁起见，下面的段落仅讨论 CMake 方法。本教程的附录
深入探讨了其他替代方案。




### 
 环境设置
 [¶](#environment-setup "永久链接到此标题")



 我们需要安装 PyTorch 和 OpenCV。获取两者的最简单且最独立于平台\的方法是通过 Conda：






```
conda install -c pytorch pytorch
conda install opencv

```





### 
 使用 CMake 构建
 [¶](#building-with-cmake "永久链接到此标题")



 要使用 [CMake](https://cmake.org)
 构建系统将我们的自定义运算符构建到共享库中，我们需要编写一个简短的
 `CMakeLists.txt`
 文件并放置与我们之前的
 `op.cpp`
 文件一起使用。为此，让’s 同意如下所示的目录结构：






```
warp-perspective/
  op.cpp
  CMakeLists.txt

```




 我们的
 `CMakeLists.txt`
 文件的内容应如下所示：






```
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(warp_perspective)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Define our library target
add_library(warp_perspective SHARED op.cpp)
# Enable C++14
target_compile_features(warp_perspective PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(warp_perspective "${TORCH_LIBRARIES}")
# Link against OpenCV
target_link_libraries(warp_perspective opencv_core opencv_imgproc)

```




 现在要构建我们的运算符，我们可以从 
 `warp_perspective`
 文件夹运行以下命令：






```
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
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
-- Found torch: /libtorch/lib/libtorch.so
-- Configuring done
-- Generating done
-- Build files have been written to: /warp_perspective/build
$ make -j
Scanning dependencies of target warp_perspective
[ 50%] Building CXX object CMakeFiles/warp_perspective.dir/op.cpp.o
[100%] Linking CXX shared library libwarp_perspective.so
[100%] Built target warp_perspective

```




 它将在
 `build`
 文件夹中放置
 `libwarp_perspective.so`
 共享库文件。在上面的
 `cmake`
 命令中，我们使用帮助器
变量
 `torch.utils.cmake_prefix_path`
 来方便地告诉我们 PyTorch 安装的 cmake 文件在哪里。 




 我们将在下面进一步详细探讨如何使用和调用我们的运算符，但为了
尽早获得成功，我们可以尝试在
Python 中运行以下代码：






```
import torch
torch.ops.load_library("build/libwarp_perspective.so")
print(torch.ops.my_ops.warp_perspective)

```




 如果一切顺利，应该打印如下内容：






```
<built-in method my_ops::warp_perspective of PyCapsule object at 0x7f618fc6fa50>

```




 这是我们稍后将用来调用自定义运算符的 Python 函数。







 在 Python 中使用 TorchScript 自定义运算符
 [¶](#using-the-torchscript-custom-operator-in-python "永久链接到此标题")
---------------------------------------------------------------------------------------------------------------------------------------



 一旦我们的自定义运算符构建到共享库中，我们就可以在 Python 中的 TorchScript 模型中使用此运算符。此操作分为两部分：
首先将运算符加载到 Python 中，然后在
TorchScript 代码中使用运算符。




 您已经了解了如何将运算符导入 Python：
 `torch.ops.load_library()`
 。此函数获取包含自定义运算符的共享库的路径，并将其加载到当前进程中。加载共享库还将执行
 `TORCH_LIBRARY`
 块。这将向 TorchScript 编译器注册我们的自定义运算符，并允许我们在 TorchScript 代码中使用该运算符。




 您可以将加载的运算符引用为
 `torch.ops.<namespace>.<function>`
 ，
其中
 `<namespace>`
 是运算符名称的命名空间部分，并且\ n `<function>`
 运算符的函数名称。对于我们上面编写的操作符，命名空间是 `my_ops`
 和函数名称
 `warp_perspective`
 ，
这意味着我们的操作符可以作为
 `torch.ops 使用.my_ops.warp_perspective`
 。
虽然此函数可以在脚本化或跟踪的 TorchScript 模块中使用，但我们
也可以在普通的 eager PyTorch 中使用它并传递常规 PyTorch
张量：






```
import torch
torch.ops.load_library("build/libwarp_perspective.so")
print(torch.ops.my_ops.warp_perspective(torch.randn(32, 32), torch.rand(3, 3)))

```




 产生:






```
tensor([[0.0000, 0.3218, 0.4611,  ..., 0.4636, 0.4636, 0.4636],
      [0.3746, 0.0978, 0.5005,  ..., 0.4636, 0.4636, 0.4636],
      [0.3245, 0.0169, 0.0000,  ..., 0.4458, 0.4458, 0.4458],
      ...,
      [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
      [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
      [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000]])

```





 没有10



 幕后发生的事情是，当您第一次在 Python 中访问
 `torch.ops.namespace.function`
 时，TorchScript 编译器（在 C++
land 中）将查看是否有一个函数
 `namespace::函数`
 已注册，
如果是这样，则返回此函数的 Python 句柄，我们随后可以使用该句柄从 Python 调用
我们的 C++ 运算符实现。这是 TorchScript 自定义运算符和 C++ 扩展之间的一个值得注意的区别：C++ 扩展是使用 pybind11 手动绑定的，而 TorchScript 自定义操作是由 PyTorch 本身动态绑定的。 Pybind11 为您提供了更大的灵活性，
您可以将哪些类型和类绑定到 Python 中，因此
推荐用于纯粹的 eager 代码，但 TorchScript
ops 不支持它。





 从这里开始，您可以在脚本或跟踪代码中使用自定义运算符，就像 
 `torch`
 包中的其他函数一样。事实上，“standard
library” 函数如
 `torch.matmul`
 与自定义操作符的注册路径基本相同
，这使得自定义操作符
真正成为一等公民当谈到如何以及在何处可以在TorchScript 中使用它们时。
 （但是，一个区别是标准库函数
具有与
 `torch.ops`
 参数解析不同的自定义编写的 Python 参数解析逻辑。）




### 
 使用自定义运算符进行跟踪
 [¶](#using-the-custom-operator-with-tracing "永久链接到此标题")



 让’s 首先将我们的运算符嵌入到跟踪函数中。回想一下，对于
跟踪，我们从一些普通的 Pytorch 代码开始：






```
def compute(x, y, z):
    return x.matmul(y) + torch.relu(z)

```




 然后对其调用
 `torch.jit.trace`。我们进一步传递`torch.jit.trace`一些示例输入，它将转发到我们的实现以记录输入流经它时发生的操作序列。结果
这实际上是急切 PyTorch 程序的 “frozen” 版本，
TorchScript 编译器可以进一步分析、优化和序列化：






```
inputs = [torch.randn(4, 8), torch.randn(8, 5), torch.randn(4, 5)]
trace = torch.jit.trace(compute, inputs)
print(trace.graph)

```




 制作：






```
graph(%x : Float(4:8, 8:1),
      %y : Float(8:5, 5:1),
      %z : Float(4:5, 5:1)):
  %3 : Float(4:5, 5:1) = aten::matmul(%x, %y) # test.py:10:0
  %4 : Float(4:5, 5:1) = aten::relu(%z) # test.py:10:0
  %5 : int = prim::Constant[value=1]() # test.py:10:0
  %6 : Float(4:5, 5:1) = aten::add(%3, %4, %5) # test.py:10:0
  return (%6)

```




 现在，令人兴奋的发现是，我们可以简单地将自定义运算符放入
我们的 PyTorch 跟踪中，就像
 `torch.relu`
 或任何其他
 `torch`
 函数一样：






```
def compute(x, y, z):
    x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
    return x.matmul(y) + torch.relu(z)

```




 然后像以前一样跟踪它：






```
inputs = [torch.randn(4, 8), torch.randn(8, 5), torch.randn(8, 5)]
trace = torch.jit.trace(compute, inputs)
print(trace.graph)

```




 制作：






```
graph(%x.1 : Float(4:8, 8:1),
      %y : Float(8:5, 5:1),
      %z : Float(8:5, 5:1)):
  %3 : int = prim::Constant[value=3]() # test.py:25:0
  %4 : int = prim::Constant[value=6]() # test.py:25:0
  %5 : int = prim::Constant[value=0]() # test.py:25:0
  %6 : Device = prim::Constant[value="cpu"]() # test.py:25:0
  %7 : bool = prim::Constant[value=0]() # test.py:25:0
  %8 : Float(3:3, 3:1) = aten::eye(%3, %4, %5, %6, %7) # test.py:25:0
  %x : Float(8:8, 8:1) = my_ops::warp_perspective(%x.1, %8) # test.py:25:0
  %10 : Float(8:5, 5:1) = aten::matmul(%x, %y) # test.py:26:0
  %11 : Float(8:5, 5:1) = aten::relu(%z) # test.py:26:0
  %12 : int = prim::Constant[value=1]() # test.py:26:0
  %13 : Float(8:5, 5:1) = aten::add(%10, %11, %12) # test.py:26:0
  return (%13)

```




 将 TorchScript 自定义操作集成到跟踪的 PyTorch 代码中就这么简单！





### 
 将自定义运算符与脚本一起使用
 [¶](#using-the-custom-operator-with-script "永久链接到此标题")



 除了跟踪之外，获得 PyTorch 程序的 TorchScript 表示的另一种方法是直接编写代码
 *in* 
 TorchScript。 TorchScript 很大程度上是 Python 语言的子集，有一些限制使得 TorchScript 编译器更容易推理程序。您可以将
常规 PyTorch 代码转换为 TorchScript，方法是使用
 `@torch.jit.script`
 注释它（对于自由函数），
 `@torch.jit.script_method`
 对于类中的方法
 （也必须源自
 `torch.jit.ScriptModule`
 ）。有关 TorchScript 注释的更多详细信息，请参阅
 [此处](https://pytorch.org/docs/master/jit.html)。




 使用 TorchScript 而不是跟踪的一个特殊原因是跟踪无法捕获 PyTorch 代码中的控制流。因此，让我们考虑这个
确实使用控制流的函数：






```
def compute(x, y):
  if bool(x[0][0] == 42):
      z = 5
  else:
      z = 10
  return x.matmul(y) + z

```




 要将此函数从普通 PyTorch 转换为 TorchScript，我们用
注释它
`@torch.jit.script`
 :






```
@torch.jit.script
def compute(x, y):
  if bool(x[0][0] == 42):
      z = 5
  else:
      z = 10
  return x.matmul(y) + z

```




 这将及时将
 `compute`
 函数编译为
图形表示，我们可以在
 `compute.graph`
 属性中检查
：






```
>>> compute.graph
graph(%x : Dynamic
 %y : Dynamic) {
 %14 : int = prim::Constant[value=1]()
 %2 : int = prim::Constant[value=0]()
 %7 : int = prim::Constant[value=42]()
 %z.1 : int = prim::Constant[value=5]()
 %z.2 : int = prim::Constant[value=10]()
 %4 : Dynamic = aten::select(%x, %2, %2)
 %6 : Dynamic = aten::select(%4, %2, %2)
 %8 : Dynamic = aten::eq(%6, %7)
 %9 : bool = prim::TensorToBool(%8)
 %z : int = prim::If(%9)
 block0() {
 -> (%z.1)
 }
 block1() {
 -> (%z.2)
 }
 %13 : Dynamic = aten::matmul(%x, %y)
 %15 : Dynamic = aten::add(%13, %z, %14)
 return (%15);
}

```




 现在，就像以前一样，我们可以像脚本代码中的任何其他
函数一样使用自定义运算符：






```
torch.ops.load_library("libwarp_perspective.so")

@torch.jit.script
def compute(x, y):
  if bool(x[0] == 42):
      z = 5
  else:
      z = 10
  x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
  return x.matmul(y) + z

```




 当 TorchScript 编译器看到对
 `torch.ops.my_ops.warp_perspective`
 的引用时，它将找到我们
通过
 `TORCH_LIBRARY` 注册的实现
 C++ 中的函数，并将其编译为其
图形表示形式:






```
>>> compute.graph
graph(%x.1 : Dynamic
 %y : Dynamic) {
 %20 : int = prim::Constant[value=1]()
 %16 : int[] = prim::Constant[value=[0, -1]]()
 %14 : int = prim::Constant[value=6]()
 %2 : int = prim::Constant[value=0]()
 %7 : int = prim::Constant[value=42]()
 %z.1 : int = prim::Constant[value=5]()
 %z.2 : int = prim::Constant[value=10]()
 %13 : int = prim::Constant[value=3]()
 %4 : Dynamic = aten::select(%x.1, %2, %2)
 %6 : Dynamic = aten::select(%4, %2, %2)
 %8 : Dynamic = aten::eq(%6, %7)
 %9 : bool = prim::TensorToBool(%8)
 %z : int = prim::If(%9)
 block0() {
 -> (%z.1)
 }
 block1() {
 -> (%z.2)
 }
 %17 : Dynamic = aten::eye(%13, %14, %2, %16)
 %x : Dynamic = my_ops::warp_perspective(%x.1, %17)
 %19 : Dynamic = aten::matmul(%x, %y)
 %21 : Dynamic = aten::add(%19, %z, %20)
 return (%21);
 }

```




 特别注意图表末尾
 处对
 `my_ops::warp_perspective` 的引用。





注意




 TorchScript 图形表示仍可能发生变化。不要依赖
它看起来像这样。





 在 Python 中使用我们的自定义运算符时，’ 确实如此。简而言之，您使用
 `torch.ops.load_library`
 导入包含运算符的库，并像任何其他
 `torch`
 运算符一样从跟踪或调用您的自定义操作脚本化 TorchScript 代码。







 在 C++ 中使用 TorchScript 自定义运算符
 [¶](#using-the-torchscript-custom-operator-in-c "永久链接到此标题")
--------------------------------------------------------------------------------------------------------------------------------



 TorchScript 的一项有用功能是能够将模型序列化为\非磁盘文件。该文件可以通过线路发送、存储在文件系统中，或者更重要的是，可以动态反序列化和执行，而无需保留原始源代码。这在 Python 中是可能的，但在 C++ 中也是如此。为此，PyTorch 提供了[纯 C++ API](https://pytorch.org/cppdocs/) 用于反序列化以及执行 TorchScript 模型。如果您还没有 ’t，
请阅读
 [在 C++ 中加载和运行序列化 TorchScript 模型的教程](https://pytorch.org/tutorials/advanced/cpp_export.html) 
 , 
接下来的几段内容将以此为基础。




 简而言之，自定义运算符可以像常规
 `torch`
 运算符
 一样执行，甚至可以从文件反序列化并在 C++ 中运行。唯一的要求是将我们之前构建的自定义运算符共享库与执行模型的 C++ 应用程序链接起来。在 Python 中，只需调用
 `torch.ops.load_library`
 即可。在 C++ 中，您需要将共享库与
您使用的任何构建系统中的主应用程序链接起来。以下
示例将使用 CMake 展示这一点。





 注意




 从技术上讲，您还可以在运行时将共享库动态加载到 C++
应用程序中，就像我们在 Python 中所做的那样。在 Linux 上，
 [您可以使用 dlopen 执行此操作](https://tldp.org/HOWTO/Program-Library-HOWTO/dl-libraries.html) 
 。其他平台上存在
等效项。





 以上面链接的 C++ 执行教程为基础，让’s 从一个文件中的最小
C++ 应用程序开始，
 `main.cpp`
 与我们的自定义运算符位于不同的文件夹中，加载并执行序列化的 TorchScript 模型:






```
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>


int main(int argc, const char* argv[]) {
 if (argc != 2) {
 std::cerr << "usage: example-app <path-to-exported-script-module>";
 return -1;
 }

 // Deserialize the ScriptModule from a file using torch::jit::load().
 torch::jit::script::Module module = torch::jit::load(argv[1]);

 std::vector<torch::jit::IValue> inputs;
 inputs.push_back(torch::randn({4, 8}));
 inputs.push_back(torch::randn({8, 5}));

 torch::Tensor output = module.forward(std::move(inputs)).toTensor();

 std::cout << output << std::endl;
}

```




 以及一个小
 `CMakeLists.txt`
 文件：






```
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(example_app)

find_package(Torch REQUIRED)

add_executable(example_app main.cpp)
target_link_libraries(example_app "${TORCH_LIBRARIES}")
target_compile_features(example_app PRIVATE cxx_range_for)

```




 此时，我们应该能够构建应用程序：






```
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
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
-- Found torch: /libtorch/lib/libtorch.so
-- Configuring done
-- Generating done
-- Build files have been written to: /example_app/build
$ make -j
Scanning dependencies of target example_app
[ 50%] Building CXX object CMakeFiles/example_app.dir/main.cpp.o
[100%] Linking CXX executable example_app
[100%] Built target example_app

```




 并在不传递模型的情况下运行它：






```
$ ./example_app
usage: example_app <path-to-exported-script-module>

```




 接下来，让’s 序列化我们之前编写的使用自定义运算符的脚本函数：






```
torch.ops.load_library("libwarp_perspective.so")

@torch.jit.script
def compute(x, y):
  if bool(x[0][0] == 42):
      z = 5
  else:
      z = 10
  x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
  return x.matmul(y) + z

compute.save("example.pt")

```




 最后一行将脚本函数序列化到名为
“example.pt” 的文件中。如果我们随后将此序列化模型传递给我们的 C++ 应用程序，
我们可以立即运行它：






```
$ ./example_app example.pt
terminate called after throwing an instance of 'torch::jit::script::ErrorReport'
what():
Schema not found for node. File a bug report.
Node: %16 : Dynamic = my_ops::warp_perspective(%0, %19)

```




 或者也许不是。也许还没有。当然！我们还没有’t 将自定义
ooperator 库与我们的应用程序链接起来。让’s 立即执行此操作，并
正确执行此操作，让’s 稍微更新我们的文件组织，如下所示：






```
example_app/
  CMakeLists.txt
  main.cpp
  warp_perspective/
    CMakeLists.txt
    op.cpp

```




 这将允许我们将 
 `warp_perspective`
 库 CMake 目标添加为 
 应用程序目标的子目录。 
 `example_app`
 文件夹中的
 顶层
 `CMakeLists.txt`
 应如下所示：






```
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(example_app)

find_package(Torch REQUIRED)

add_subdirectory(warp_perspective)

add_executable(example_app main.cpp)
target_link_libraries(example_app "${TORCH_LIBRARIES}")
target_link_libraries(example_app -Wl,--no-as-needed warp_perspective)
target_compile_features(example_app PRIVATE cxx_range_for)

```




 这个基本的 CMake 配置看起来很像以前，除了我们添加 
 `warp_perspective`
 CMake build 作为子目录。 CMake 代码运行后，我们会将我们的
 `example_app`
 应用程序与
 
 `warp_perspective`
 共享库链接起来。





注意



上面的示例中嵌入了一个关键细节：链接行的“-Wl,--no-as-needed”前缀。这是必需的，因为我们实际上不会在应用程序代码中调用来自“warp_perspective”共享库的任何函数。我们只需要
 `TORCH_LIBRARY`
 函数即可运行。不方便的是，这会使链接器感到困惑，并使其认为可以完全跳过对库的链接。在 Linux 上，
 `-Wl,--no-as-needed`
 标志强制链接发生（注意：此标志特定于 Linux！）。 
对此还有其他解决方法。最简单的方法是在运算符库中定义
*某些函数*
，您需要从主应用程序调用该函数。这可以像在某个标头中声明的
函数
 `void
 

 init();`
 一样简单，然后定义为
 `void
 

 init()\ n 

 {
 

 }`
 在运算符库中。在主应用程序中调用此
 `init()`
 函数会给链接器留下这是一个值得链接的库的印象。不幸的是，这超出了我们的控制范围，
我们宁愿让您知道原因和简单的解决方法
，而不是给您一些不透明的宏来放入您的代码中。





 现在，由于我们在顶层找到了
 `Torch`
 包，因此
 `warp_perspective`
 子目录中的
 `CMakeLists.txt`
 文件可以是
缩短了一点。它应该看起来像这样：






```
find_package(OpenCV REQUIRED)
add_library(warp_perspective SHARED op.cpp)
target_compile_features(warp_perspective PRIVATE cxx_range_for)
target_link_libraries(warp_perspective PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(warp_perspective PRIVATE opencv_core opencv_photo)

```




 让’s 重新构建我们的示例应用程序，该应用程序也将与自定义运算符
库链接。在顶级
 `example_app`
 目录中:






```
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
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
-- Found torch: /libtorch/lib/libtorch.so
-- Configuring done
-- Generating done
-- Build files have been written to: /warp_perspective/example_app/build
$ make -j
Scanning dependencies of target warp_perspective
[ 25%] Building CXX object warp_perspective/CMakeFiles/warp_perspective.dir/op.cpp.o
[ 50%] Linking CXX shared library libwarp_perspective.so
[ 50%] Built target warp_perspective
Scanning dependencies of target example_app
[ 75%] Building CXX object CMakeFiles/example_app.dir/main.cpp.o
[100%] Linking CXX executable example_app
[100%] Built target example_app

```




 如果我们现在运行
 `example_app`
 二进制文件并将我们的序列化模型交给它，
我们应该会得到一个圆满的结局：






```
$ ./example_app example.pt
11.4125 5.8262 9.5345 8.6111 12.3997
 7.4683 13.5969 9.0850 11.0698 9.4008
 7.4597 15.0926 12.5727 8.9319 9.0666
 9.4834 11.1747 9.0162 10.9521 8.6269
10.0000 10.0000 10.0000 10.0000 10.0000
10.0000 10.0000 10.0000 10.0000 10.0000
10.0000 10.0000 10.0000 10.0000 10.0000
10.0000 10.0000 10.0000 10.0000 10.0000
[ Variable[CPUFloatType]{8,5} ]

```




成功！您现在已准备好进行推断。






 结论
 [¶](#conclusion "此标题的永久链接")
---------------------------------------------------------------------



 本教程向您介绍了如何在 C++ 中实现自定义 TorchScript 运算符、如何将其构建到共享库、如何在 Python 中使用它来定义
TorchScript 模型以及最后如何将其加载到 C++ 应用程序中
推理工作负载。现在，您可以使用与第三方 C++ 库交互的 C++ 运算符来扩展您的 TorchScript 模型，编写自定义高性能 CUDA 内核，或实现需要 Python、TorchScript 和 C++ 之间的顺畅融合的任何其他用例。 




 一如既往，如果您遇到任何问题或有疑问，可以使用我们的
 [论坛](https://discuss.pytorch.org/) 
 或
 [GitHub issues](https://github.com/pytorch/pytorch/issues) 
 取得联系。此外，我们的
 [常见问题 (FAQ) 页面](https://pytorch.org/cppdocs/notes/faq.html)
 可能包含有用的信息。






 附录 A：构建自定义运算符的更多方法
 [¶](#appendix-a-more-ways-of-building-custom-operators "永久链接到此标题")
---------------------------------------------------------------------------------------------------------------------------------------------------------



 “构建自定义运算符”部分解释了如何使用 CMake 将自定义
运算符构建到共享库中。本附录概述了另外两种
编译方法。它们都使用Python作为编译过程的“driver”或
“接口”。此外，两者都重新使用
 [现有\基础设施](https://pytorch.org/docs/stable/cpp_extension.html) 
 PyTorch
提供
 [*C++ 扩展*]( https://pytorch.org/tutorials/advanced/cpp_extension.html) 
 ，这是依赖于 
 [pybind11](https://github.com/pybind/pybind11) 
 用于将 C++ 函数

“explicit” 绑定到 Python 中。




 第一种方法使用 C++ 扩展’
 [方便的即时 (JIT)
编译接口](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) 
 以便在您第一次运行 PyTorch 脚本时
在后台编译您的代码。第二种方法依赖于古老的
 `setuptools`
 包并涉及
 编写单独的
 `setup.py`
 文件。这允许更高级的
配置以及与其他
基于`setuptools`
的项目的集成。
我们将在下面详细探讨这两种方法。




### 
 使用 JIT 编译进行构建
 [¶](#building-with-jit-compilation "永久链接到此标题")


PyTorch C++ 扩展工具包提供的 JIT 编译功能允许将自定义运算符的编译直接嵌入到 Python 代码中，例如在训练脚本的顶部。





 注意




 “JIT 编译” 这里与 TorchScript 编译器中进行的 JIT 编译无关，以优化你的程序。它只是意味着
您的自定义运算符 C++ 代码将在您第一次导入时编译到系统’s
 
 /tmp
 
 目录下的文件夹中，就好像您已编译它一样\事先你自己。





 这个 JIT 编译功能有两种风格。在第一个中，您仍然将运算符实现保存在单独的文件 (
 `op.cpp`
 ) 中，然后使用
 `torch.utils.cpp_extension.load()`
 进行编译你的分机。通常，此函数将返回公开 C++ 扩展的 Python 模块。然而，由于我们没有将自定义运算符编译到它自己的 Python 模块中，因此我们只想编译一个普通的共享库。幸运的是，
 `torch.utils.cpp_extension.load()`
 有一个参数
 `is_python_module`
，
我们可以将其设置为
 `False`
表明我们只对构建
共享库感兴趣，而不是 Python 模块。
 `torch.utils.cpp_extension.load()`
 然后将编译并将共享库加载到当前进程中， 
就像
 `torch.ops.load_library`
 之前所做的那样：






```
import torch.utils.cpp_extension

torch.utils.cpp_extension.load(
    name="warp_perspective",
    sources=["op.cpp"],
    extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
    is_python_module=False,
    verbose=True
)

print(torch.ops.my_ops.warp_perspective)

```




 这应该大约打印：






```
<built-in method my_ops::warp_perspective of PyCapsule object at 0x7f3e0f840b10>

```




 JIT 编译的第二种风格允许您将自定义 TorchScript 运算符的源代码作为字符串传递。为此，请使用
 `torch.utils.cpp_extension.load_inline`
 :






```
import torch
import torch.utils.cpp_extension

op_source = """
#include <opencv2/opencv.hpp>
#include <torch/script.h>

torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
 cv::Mat image_mat(/*rows=*/image.size(0),
 /*cols=*/image.size(1),
 /*type=*/CV_32FC1,
 /*data=*/image.data<float>());
 cv::Mat warp_mat(/*rows=*/warp.size(0),
 /*cols=*/warp.size(1),
 /*type=*/CV_32FC1,
 /*data=*/warp.data<float>());

 cv::Mat output_mat;
 cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{64, 64});

 torch::Tensor output =
 torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{64, 64});
 return output.clone();
}

TORCH_LIBRARY(my_ops, m) {
 m.def("warp_perspective", &warp_perspective);
}
"""

torch.utils.cpp_extension.load_inline(
    name="warp_perspective",
    cpp_sources=op_source,
    extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
    is_python_module=False,
    verbose=True,
)

print(torch.ops.my_ops.warp_perspective)

```




 当然，如果您的源代码相当短，最好只使用
 `torch.utils.cpp_extension.load_inline`
。




 请注意，如果您’ 在 Jupyter Notebook 中使用此功能，则不应多次执行
具有注册的单元，因为每次执行都会
注册新库并重新注册自定义运算符。如果需要重新执行，
请先重启笔记本的Python内核。





### 
 使用安装工具构建
 [¶](#building-with-setuptools "此标题的永久链接")



 仅从 Python 构建自定义运算符的第二种方法是
使用
 `setuptools`
 。这样做的优点是
 `setuptools`
 有一个相当
强大且广泛的接口，用于构建用 C++ 编写的 Python 模块。
但是，
 
 `setuptools` 实际上是用于构建 Python 模块，而不是简单的共享库（没有 Python
期望的模块所需的入口点），这条路线可能有点奇怪。也就是说，您
需要的只是一个
 `setup.py`
 文件来代替
 `CMakeLists.txt`
，它看起来像
这样：






```
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="warp_perspective",
    ext_modules=[
        CppExtension(
            "warp_perspective",
            ["example_app/warp_perspective/op.cpp"],
            libraries=["opencv_core", "opencv_imgproc"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)

```




 请注意，我们在底部的
 `BuildExtension`
 中启用了
 `no_python_abi_suffix`
 选项。这指示
 `setuptools`
 在生成的共享库的名称中省略任何
Python-3 特定的 ABI 后缀。
否则，例如在 Python 3.7 上，该库可能被称为
 `warp_perspective。 cpython-37m-x86_64-linux-gnu.so`
 其中
 `cpython-37m-x86_64-linux-gnu`
 是 ABI 标记，但我们实际上只是希望它
称为
 `warp_perspective.so`




 如果我们现在在终端中的
文件夹中运行
 `python
 

 setup.py
 

 build
 

development`
 `setup.py`
 已定位，我们应该看到类似以下内容：






```
$ python setup.py build develop
running build
running build_ext
building 'warp_perspective' extension
creating build
creating build/temp.linux-x86_64-3.7
gcc -pthread -B /root/local/miniconda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include/torch/csrc/api/include -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include/TH -I/root/local/miniconda/lib/python3.7/site-packages/torch/lib/include/THC -I/root/local/miniconda/include/python3.7m -c op.cpp -o build/temp.linux-x86_64-3.7/op.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=warp_perspective -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
creating build/lib.linux-x86_64-3.7
g++ -pthread -shared -B /root/local/miniconda/compiler_compat -L/root/local/miniconda/lib -Wl,-rpath=/root/local/miniconda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/op.o -lopencv_core -lopencv_imgproc -o build/lib.linux-x86_64-3.7/warp_perspective.so
running develop
running egg_info
creating warp_perspective.egg-info
writing warp_perspective.egg-info/PKG-INFO
writing dependency_links to warp_perspective.egg-info/dependency_links.txt
writing top-level names to warp_perspective.egg-info/top_level.txt
writing manifest file 'warp_perspective.egg-info/SOURCES.txt'
reading manifest file 'warp_perspective.egg-info/SOURCES.txt'
writing manifest file 'warp_perspective.egg-info/SOURCES.txt'
running build_ext
copying build/lib.linux-x86_64-3.7/warp_perspective.so ->
Creating /root/local/miniconda/lib/python3.7/site-packages/warp-perspective.egg-link (link to .)
Adding warp-perspective 0.0.0 to easy-install.pth file

Installed /warp_perspective
Processing dependencies for warp-perspective==0.0.0
Finished processing dependencies for warp-perspective==0.0.0

```




 这将生成一个名为
 `warp_perspective.so`
 的共享库，我们可以
将其传递给
 `torch.ops.load_library`
 就像我们之前所做的那样我们的运算符
对 TorchScript 可见：






```
>>> import torch
>>> torch.ops.load_library("warp_perspective.so")
>>> print(torch.ops.my_ops.warp_perspective)
<built-in method custom::warp_perspective of PyCapsule object at 0x7ff51c5b7bd0>

```










