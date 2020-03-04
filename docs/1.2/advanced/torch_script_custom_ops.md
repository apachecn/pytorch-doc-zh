# 使用自定义C ++算延伸TorchScript

该PyTorch 1.0版本中引入的一种新的编程模型PyTorch称为[ TorchScript [HTG1。
TorchScript是可解析的，编译和优化由TorchScript编译Python编程语言的子集。此外，编译TorchScript模型有被序列化到磁盘上的文件格式，它可以随后加载和从纯C
++(以及Python）的用于推理运行选项。](https://pytorch.org/docs/master/jit.html)

TorchScript支持由`Torch 提供
`包操作的相当大的一部分，让你表达多种复杂模型的纯粹从PyTorch的“标准库”等一系列张量操作。不过，也有可能是时候，你需要有一个自定义的C
++或CUDA功能扩展TorchScript的发现自己。虽然我们建议您只能求助于这个选项，如果你的想法不能被表达(足够有效），作为一个简单的Python函数，我们提供了一个非常友好和简单的界面使用定义自定义C
++和CUDA内核[ ATEN ](https://pytorch.org/cppdocs/#aten) ，PyTorch的高性能C
++库张。一旦绑定到TorchScript，您可以嵌入这些自定义内核(或“OPS”）到您的TorchScript模型，无论是在Python和直接的序列化的形式在C
++中执行。

下面的段落给出写TorchScript定制运算来调入[的OpenCV ](https://www.opencv.org)，计算机视觉库用C
++编写的一个例子。我们将讨论如何在C ++中，张量工作，如何有效地将它们转换为第三方张量格式(在这种情况下，OpenCV的 ``
Mat``s），如何注册与TorchScript运行，最后如何您的运营商编译操作和Python和C ++使用它。

## 实施自定义操作员在C ++

对于本教程，我们将暴露[ warpPerspective
](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective)函数，它适用于透视变换的图像，从到的OpenCV作为TorchScript自定义操作符。第一步是写我们在C
++运营商定制的实现。让我们把这个实现`op.cpp`，使它看起来像这样的文件：

    
    
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
      cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{8, 8});
    
      torch::Tensor output = torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{8, 8});
      return output.clone();
    }
    

这个操作符的代码很短。在该文件的顶部，我们包括OpenCV的头文件，`opencv2 / opencv.hpp`，沿着`torch/ script.h
`头部暴露从PyTorch的C ++ API所需的所有东西，我们需要编写自定义TorchScript运营商。我们的函数`warp_perspective
`采用两个参数：输入`图像 `和`经线 `变换矩阵我们希望应用到图像。的类型的这些输入是`torch::张量 `，在C
PyTorch的张量类型++(其也是基础类型在Python所有张量）。我们的`返回类型warp_perspective`功能也将是一个`Torch ::张量
[HTG31。`

小费

参见[本说明](https://pytorch.org/cppdocs/notes/tensor_basics.html)约ATEN，它提供了`张量
`类PyTorch库的更多信息。此外，[本教程](https://pytorch.org/cppdocs/notes/tensor_creation.html)描述了如何分配和用C初始化新张量对象++(不需要这个操作符）。

注意

所述编译器TorchScript理解的类型的固定号码。只有这些类型可以作为参数传递给您的自定义操作。目前，这些类型是：Torch ::张量 ，`
Torch ::标量 `，`双 `，`的int64_t`和`的std ::矢量 `这些类型的第需要注意的是 _只有_ `双 `和 _不是_ `
浴液HTG30] `和 _只有_ `的int64_t`和 _不_ 其他整数类型如`INT`，`短 `或`长 `的支持。

里面我们的功能，我们需要做的第一件事就是转变我们的PyTorch张量到OpenCV的矩阵，为的OpenCV的`warpPerspective`预计`
CV ::垫 `对象作为输入。幸运的是，有一种方法来做到这一点 **而不复制任何** 数据。在第几行，

    
    
    cv::Mat image_mat(/*rows=*/image.size(0),
                      /*cols=*/image.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/image.data<float>());
    

我们呼吁[此构造](https://docs.opencv.org/trunk/d3/d63/classcv_1_1Mat.html#a922de793eabcec705b3579c5f95a643e)
OpenCV的`垫 `类的给我们的张量转换为`垫 `对象。我们通过它最初的`图像 `张量，数据类型(我们定为`FLOAT32
`为行数和列数本实施例中），最后一个原始指针到底层数据 - A `浮动*`。有什么特别之处这个构造函数`垫
`类的是，它不会复制的输入数据。相反，它会简单地引用该内存上的`垫 `执行的所有操作。如果在`image_mat`进行就地操作，这将反映原始`图像
`张量(反之亦然在）。这使我们可以调用随后OpenCV的程序与库的本地矩阵型，即使我们实际上是存储在PyTorch张量的数据。我们重复这个过程将`经 `
PyTorch张量转换为`warp_mat`OpenCV的矩阵：

    
    
    cv::Mat warp_mat(/*rows=*/warp.size(0),
                     /*cols=*/warp.size(1),
                     /*type=*/CV_32FC1,
                     /*data=*/warp.data<float>());
    

接下来，我们准备调用，我们是如此渴望在TorchScript使用OpenCV的函数：`warpPerspective
[HTG3。为此，我们通过OpenCV的函数中的`image_mat`和`warp_mat`矩阵，以及被称为空输出矩阵`output_mat
`。我们还指定大小`DSIZE`我们要输出矩阵(图像）是。它是硬编码为`8  × 8`在这个例子中：`

    
    
    cv::Mat output_mat;
    cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{8, 8});
    

在我们的运营商定制实现的最后一步是转换的`output_mat
[HTG3重新站到PyTorch张量，这样我们就可以进一步PyTorch使用它。这是惊人地相似，我们先前做在其他方向转换。在这种情况下，PyTorch提供了`
torch:: from_blob`方法。在这种情况下，A _一滴_ 是指一些不透明的，平坦的内存指针，我们要解释成PyTorch张量。为`Torch ::
from_blob`调用看起来是这样的：`

    
    
    torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{8, 8})
    

我们使用`.ptr & LT ;浮子& GT ;(） `上OpenCV的`垫[方法HTG6] `类来获得原始指针到底层数据(就像`。数据& LT
;浮子& GT ;(） `为PyTorch张量更早）。我们还指定张量的输出的形状，这我们硬编码为`8  × 8`。的`输出torch:: from_blob
`于是为`torch::张量 `，指向由OpenCV的基质所拥有的存储器。

从我们的运营商实现返回，这个张量之前，我们必须调用`.clone(） `对张进行基础数据的内存拷贝。这样做的原因是，`Torch :: from_blob
`返回没有自己的数据的张量。在这一点上，该数据仍然由OpenCV的矩阵拥有。然而，这OpenCV的矩阵将走出去的范围，并在函数结束时被释放。如果我们返回`
输出 `张量-是，它会指向由我们使用它的功能之外的时间无效的内存。调用`.clone(）
`返回与新张拥有自身的原始数据的副本，新的张量。因此安全回到外面的世界。

## 注册运营商定制与TorchScript

现在，已经在C ++中实现我们的运营商定制，我们需要 _与TorchScript运行时和编译注册_
它。这将允许TorchScript编译器来解决我们在TorchScript代码运营商定制引用。注册非常简单。对于我们的情况，我们需要这样写：

    
    
    static auto registry =
      torch::RegisterOperators("my_ops::warp_perspective", &warp_perspective);
    

在某处我们的`op.cpp`文件的全局范围。这将创建一个全局变量`注册表
`，这将在其构造函数注册我们的TorchScript操作(即只出现一次，每个程序）。我们指定的经营者的名称和一个指向它的实现(我们前面写的函数）。名称由两个部分组成：一个
_命名空间_ (`my_ops`），用于我们正在注册的特定运营商和一个名称(`warp_perspective
`）。命名空间和运营商名称是由两个冒号(`::`）分离。

Tip

如果你想注册多个运营商，您可以链接调用`.OP(） `构造函数后：

    
    
    static auto registry =
      torch::RegisterOperators("my_ops::warp_perspective", &warp_perspective)
      .op("my_ops::another_op", &another_op)
      .op("my_ops::and_another_op", &and_another_op);
    

在幕后，`RegisterOperators`将执行一些相当复杂的C ++模板元编程魔术推断函数指针的参数和返回值类型，我们把它传递(`&安培;
warp_perspective`）。该信息被用于形成 _功能架构_ 为我们的运营商。函数模式是运营商的结构化表示 - 一种“签名”或“原型”的 -
使用的TorchScript编译器来验证TorchScript程序的正确性。

## 构建自定义操作

现在，我们已经实现了我们的运营商定制的C
++及书面登记代码，它是时间来建立操作成(共享）库，我们可以在任何的Python的研究和实验加载到Python或成C
++的推理环境。存在多种方式来打造我们的运营商，使用纯CMake的，或Python的替代品如`setuptools的
[HTG3。为了简便起见，下面仅段落讨论CMake的方法。本教程的附录潜入基于Python的替代品。`

### 与CMake的构建

为了建立我们的运营商定制到一个共享库使用[ CMake的](https://cmake.org)构建系统，我们需要写一个简短的`
的CMakeLists.txt`文件，并与我们以前的[放置HTG6]  op.cpp  文件。对于这一点，让我们在一个目录结构，看起来像这样一致认为：

    
    
    warp-perspective/
      op.cpp
      CMakeLists.txt
    

此外，请一定要抓住最新版本的LibTorch分布，包PyTorch的C ++库和CMake的构建文件中，在[ pytorch.org
[HTG1。请将解压分布在文件系统中的某个地方访问。下面的段落将参考该位置为`/路径/到/ libtorch`。我们的`
的CMakeLists.txt`文件应该然后是以下内容：](https://pytorch.org/get-started/locally)

    
    
    cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
    project(warp_perspective)
    
    find_package(Torch REQUIRED)
    find_package(OpenCV REQUIRED)
    
    # Define our library target
    add_library(warp_perspective SHARED op.cpp)
    # Enable C++11
    target_compile_features(warp_perspective PRIVATE cxx_range_for)
    # Link against LibTorch
    target_link_libraries(warp_perspective "${TORCH_LIBRARIES}")
    # Link against OpenCV
    target_link_libraries(warp_perspective opencv_core opencv_imgproc)
    

警告

这种设置使一些假设关于构建环境，特别是什么属于安装的OpenCV。上述`的CMakeLists.txt`文件被运行Ubuntu Xenial与`
libopencv-dev的 `通过`[HTG9安装一个泊坞容器内测试]易于 `。如果它不为你工作，你觉得卡住，请使用`Dockerfile
`中的[伴随教程库](https://github.com/pytorch/extension-
script)建立一个隔离的，可重复的环境在其中扮演周围从本教程中的代码。如果碰上进一步的麻烦，请在本教程的库文件中的一个问题或张贴在[我们的论坛](https://discuss.pytorch.org/)的问题。

到现在建立我们的运营商，我们可以从`warp_perspective`文件夹中运行以下命令：

    
    
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
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
    

这将放置在`构建 `文件夹中的`libwarp_perspective.so`共享库文件。在上面的`cmake的 `命令，则应更换`/路径/到/
libtorch`与路径解压缩后的LibTorch分布。

我们将探讨如何使用和下面进一步呼吁我们的运营商中的细节，但要获得成功的早期感觉，我们可以尝试在Python运行下面的代码：

    
    
    >>> import torch
    >>> torch.ops.load_library("/path/to/libwarp_perspective.so")
    >>> print(torch.ops.my_ops.warp_perspective)
    

在这里，`/path/to/libwarp_perspective.so`应到`libwarp_perspective.so
`共享库的相对或绝对路径我们只是建成。如果一切顺利的话，这应该打印像

    
    
    <built-in method my_ops::warp_perspective of PyCapsule object at 0x7f618fc6fa50>
    

这是Python的功能，我们将在以后使用调用我们的自定义操作。

## 在Python使用TorchScript运营商定制

一旦我们的运营商定制内置共享库，我们准备在我们在Python
TorchScript车型使用此运算符。有两个部分，以这样的：第一加载操作到Python和第二使用TorchScript代码操作。

你已经看到了如何导入您的运营商引入Python：`torch.ops.load_library(）
[HTG3。此功能将路径包含运营商定制的共享库，并将其加载到当前进程。加载共享库也将执行全局`RegisterOperators
`对象，就放到我们的运营商定制实现文件的构造函数。这将注册我们的运营商定制与TorchScript编译器，并允许我们使用该运营商在TorchScript代码。`

你可以参考你的加载运营商为`torch.ops & LT []命名空间& GT ; [ - ] LT ;。函数[ - - ] GT ;`，其中`&
LT ;命名空间& GT ;`是命名空间的一部分您的操作员姓名，以及`& LT ;函数& GT ;
`您的操作者的功能名称。对于我们上面写的操作，命名空间为`my_ops`和函数名`warp_perspective`，这意味着我们的运营商可为`
torch.ops.my_ops.warp_perspective
`。虽然这个功能可以在脚本或跟踪TorchScript模块一起使用，我们也可以只用它在香草渴望PyTorch并将其传递规律PyTorch张量：

    
    
    >>> import torch
    >>> torch.ops.load_library("libwarp_perspective.so")
    >>> torch.ops.my_ops.warp_perspective(torch.randn(32, 32), torch.rand(3, 3))
    tensor([[0.0000, 0.3218, 0.4611,  ..., 0.4636, 0.4636, 0.4636],
          [0.3746, 0.0978, 0.5005,  ..., 0.4636, 0.4636, 0.4636],
          [0.3245, 0.0169, 0.0000,  ..., 0.4458, 0.4458, 0.4458],
          ...,
          [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
          [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000],
          [0.1862, 0.1862, 0.1692,  ..., 0.0000, 0.0000, 0.0000]])
    

注意

会发生什么幕后是你第一次访问`torch.ops.namespace.function`在Python中，TorchScript编译器(在C
++的土地）可以看到，如果一个函数`命名空间::函数
`已经被注册，如果是这样，则返回一个Python句柄这个功能，我们可以随后使用调入从Python中我们的C
++运算符实现。这是TorchScript运营商定制和C ++扩展之间的一个显着的差异：C
++扩展结合使用pybind11手动，而TorchScript定制OPS是在由PyTorch本身飞约束。
Pybind11为您提供了更多的灵活性，以什么样的类型和类可以绑定到Python和因此建议对纯渴望代码的问候，但它不支持TorchScript欢声笑语。

从这里开始，您可以使用脚本或代码追踪您的自定义操作，就像你从`Torch 等功能 `包。事实上，“标准库”的功能，如`torch.matmul
`经历大致相同的注册路径作为运营商定制，这使得运营商定制真正一流的公民，当谈到如何和在那里他们可以TorchScript使用。

### 使用自定义操作与跟踪

让我们在跟踪功能嵌入我们的运营商开始。回想一下，跟踪，我们先从一些香草Pytorch代码：

    
    
    def compute(x, y, z):
        return x.matmul(y) + torch.relu(z)
    

然后在其上调用`torch.jit.trace`。我们进一步通过`torch.jit.trace
`例如一些投入，它将转发给我们的实现记录为输入流过它发生的操作顺序。这样做的结果是有效的渴望PyTorch程序，其中TorchScript编译器可以进一步分析，优化和序列化的“冻结”的版本：

    
    
    >>> inputs = [torch.randn(4, 8), torch.randn(8, 5), torch.randn(4, 5)]
    >>> trace = torch.jit.trace(compute, inputs)
    >>> print(trace.graph)
    graph(%x : Float(4, 8)
        %y : Float(8, 5)
        %z : Float(4, 5)) {
      %3 : Float(4, 5) = aten::matmul(%x, %y)
      %4 : Float(4, 5) = aten::relu(%z)
      %5 : int = prim::Constant[value=1]()
      %6 : Float(4, 5) = aten::add(%3, %4, %5)
      return (%6);
    }
    

现在，激动人心的启示是，我们可以简单的丢弃我们的运营商定制到我们PyTorch痕迹，好像它是`torch.relu`或任何其他`Torch  `函数：

    
    
    torch.ops.load_library("libwarp_perspective.so")
    
    def compute(x, y, z):
        x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
        return x.matmul(y) + torch.relu(z)
    

然后跟踪它像以前一样：

    
    
    >>> inputs = [torch.randn(4, 8), torch.randn(8, 5), torch.randn(8, 5)]
    >>> trace = torch.jit.trace(compute, inputs)
    >>> print(trace.graph)
    graph(%x.1 : Float(4, 8)
        %y : Float(8, 5)
        %z : Float(8, 5)) {
        %3 : int = prim::Constant[value=3]()
        %4 : int = prim::Constant[value=6]()
        %5 : int = prim::Constant[value=0]()
        %6 : int[] = prim::Constant[value=[0, -1]]()
        %7 : Float(3, 3) = aten::eye(%3, %4, %5, %6)
        %x : Float(8, 8) = my_ops::warp_perspective(%x.1, %7)
        %11 : Float(8, 5) = aten::matmul(%x, %y)
        %12 : Float(8, 5) = aten::relu(%z)
        %13 : int = prim::Constant[value=1]()
        %14 : Float(8, 5) = aten::add(%11, %12, %13)
        return (%14);
      }
    

整合TorchScript定制OPS成追溯到PyTorch代码，因为这容易！

### 使用自定义操作与脚本

除了跟踪，另一种方式在PyTorch程序的TorchScript表示到达是直接写你的 TorchScript代码[HTG0。
TorchScript主要是Python语言的一个子集，有一些限制，使得它更容易为TorchScript编译器推理程序。您可以通过使用`//
@标注它torch.jit.script`免费功能和`@ torch.jit.script_method
[关闭你的常规PyTorch代码到TorchScript HTG9一种用于在类方法(其也必须从`torch.jit.ScriptModule派生
`）。参见[此处](https://pytorch.org/docs/master/jit.html)关于TorchScript注释的更多细节。`

而不是使用跟踪TorchScript一个特别的原因是追踪无法捕捉PyTorch代码控制流。因此，让我们考虑这个功能，不使用控制流：

    
    
    def compute(x, y):
      if bool(x[0][0] == 42):
          z = 5
      else:
          z = 10
      return x.matmul(y) + z
    

从香草PyTorch到TorchScript转换这个功能，我们用`将其标注为@ torch.jit.script`：

    
    
    @torch.jit.script
    def compute(x, y):
      if bool(x[0][0] == 42):
          z = 5
      else:
          z = 10
      return x.matmul(y) + z
    

这将刚刚在时间编译`计算 `函数成图形表示，这是我们可以在`compute.graph`属性检查：

    
    
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
    

而现在，就像之前，我们可以用我们的运营商定制等我们的脚本代码中任何其他功能：

    
    
    torch.ops.load_library("libwarp_perspective.so")
    
    @torch.jit.script
    def compute(x, y):
      if bool(x[0] == 42):
          z = 5
      else:
          z = 10
      x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
      return x.matmul(y) + z
    

当TorchScript编译器看到参考`torch.ops.my_ops.warp_perspective`，它会找到我们通过`
RegisterOperators注册的实现 `对象在C ++中，并将其编译成其图形表示：

    
    
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
    

特别是通知所述参照`my_ops :: warp_perspective`在图的结尾。

Attention

所述TorchScript图表示仍然可能发生变化。不要依赖于它看起来像这样。

这就是真正的它，当它涉及到使用Python中我们的运营商定制。总之，你导入使用`torch.ops.load_library
`包含您的运营商(S）的图书馆，并呼吁像任何其他`Torch 自定义运算 `从您的追溯或脚本代码TorchScript操作。

## 在C使用自定义TorchScript算++

TorchScript的一个有用的功能是序列化模型到磁盘上的文件的能力。该文件可以通过线路被发送，存储在文件系统，或者更重要的是，动态地解串行化和执行，而无需保留原始源代码周围。这是可能在Python，而且在C
++。对于这一点，PyTorch提供[纯C ++ API
[HTG1用于反串行化以及执行TorchScript模型。如果你还没有，请阅读](https://pytorch.org/cppdocs/)[在C对加载和运行的系列化TorchScript模型教程++
](https://pytorch.org/tutorials/advanced/cpp_export.html)，在其未来数段将建成。

总之，运营商定制可以像从文件反序列化，即使和用C ++运行规则`torch `运营商来执行。这个唯一的要求就是我们前面在我们执行模型中的C
++应用程序构建运营商定制共享库链接。在Python中，这个工作只是调用`torch.ops.load_library  [HTG7。在C
++中，你需要的共享库，在任何的构建系统使用的是主应用程序链接。下面的例子将展示这一点使用CMake的。`

Note

从技术上讲，你还可以动态加载共享库复制到运行时你的C ++应用程序中的多，我们这样做是在Python一样。在Linux上，[你可以使用dlopen
](https://tldp.org/HOWTO/Program-Library-HOWTO/dl-
libraries.html)做到这一点。存在着在其他平台上的等价物。

上面链接的C ++执行教程的基础上，让我们开始用最小的C ++应用程序在一个文件中，`的main.cpp
`从我们的运营商定制不同的文件夹，即加载并执行一个序列化TorchScript模型：

    
    
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
    
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(torch::randn({4, 8}));
      inputs.push_back(torch::randn({8, 5}));
    
      torch::Tensor output = module->forward(std::move(inputs)).toTensor();
    
      std::cout << output << std::endl;
    }
    

随着小`的CMakeLists.txt`文件：

    
    
    cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
    project(example_app)
    
    find_package(Torch REQUIRED)
    
    add_executable(example_app main.cpp)
    target_link_libraries(example_app "${TORCH_LIBRARIES}")
    target_compile_features(example_app PRIVATE cxx_range_for)
    

在这一点上，我们应该能够构建应用程序：

    
    
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
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
    

而没有通过模型只是还没有运行它：

    
    
    $ ./example_app
    usage: example_app <path-to-exported-script-module>
    

接下来，让我们序列化，我们写的脚本函数较早使用我们的自定义操作：

    
    
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
    

最后一行将序列化脚本函数到一个名为“example.pt”文件。如果我们再通过这个序列化的模型来我们的C ++应用程序，我们可以运行它立刻：

    
    
    $ ./example_app example.pt
    terminate called after throwing an instance of 'torch::jit::script::ErrorReport'
    what():
    Schema not found for node. File a bug report.
    Node: %16 : Dynamic = my_ops::warp_perspective(%0, %19)
    

或者可能不是。也许不是现在。当然！我们没有链接与我们的应用运营商定制库呢。现在，让我们这样做的权利，并做正确，让我们稍微更新我们的文件组织，如下所示：

    
    
    example_app/
      CMakeLists.txt
      main.cpp
      warp_perspective/
        CMakeLists.txt
        op.cpp
    

这将允许我们添加`warp_perspective`库CMake的目标作为我们的应用目标的子目录。顶层`的CMakeLists.txt`中的`
example_app`文件夹应该是这样的：

    
    
    cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
    project(example_app)
    
    find_package(Torch REQUIRED)
    
    add_subdirectory(warp_perspective)
    
    add_executable(example_app main.cpp)
    target_link_libraries(example_app "${TORCH_LIBRARIES}")
    target_link_libraries(example_app -Wl,--no-as-needed warp_perspective)
    target_compile_features(example_app PRIVATE cxx_range_for)
    

这个基本的CMake的配置看起来很像之前，除了我们添加`warp_perspective`
CMake的建设作为一个子目录。一旦它的CMake的代码运行时，我们用`warp_perspective`共享库链接我们的`example_app
`应用。

Attention

有嵌入在上面的例子中一个关键的细节：`-Wl， - 无按需 `前缀到`warp_perspective
`链接线。这是必需的，因为我们实际上不会调用在我们的应用程序代码中的`warp_perspective`共享库的任何功能。我们只需要在全球`
RegisterOperators`对象的构造函数运行。麻烦的是，这混淆了连接器，并使其认为它可以只是完全跳过链接到的库。在Linux上，`轮候册，
- 无按需 `标记强制发生的链接(注：这个标志是具体到Linux！）。还有其他的变通办法此。最简单的就是定义 _一些函数_
在您需要从主应用程序调用操作库。这可能是作为简单的函数`作废 的init(）[]`在一些头，然后将其定义为`空隙声明 初始化(） { }
`在操作库。调用此`的init(）
`在主应用程序的功能将会给连接器的印象，这是值得链接到的库。不幸的是，这是我们无法控制的，我们宁可让你知道原因和简单的解决方法为这个比交给你一些不透明宏在代码噗通。

现在，因为我们现在找到`Torch  `包在最顶层，在`的CMakeLists.txt`文件中的`warp_perspective
`子目录可以缩短一个位。它应该是这样的：

    
    
    find_package(OpenCV REQUIRED)
    add_library(warp_perspective SHARED op.cpp)
    target_compile_features(warp_perspective PRIVATE cxx_range_for)
    target_link_libraries(warp_perspective PRIVATE "${TORCH_LIBRARIES}")
    target_link_libraries(warp_perspective PRIVATE opencv_core opencv_photo)
    

让我们重新构建我们的示例应用程序，这也将与运营商定制库链接。在顶层`example_app`目录：

    
    
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
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
    

如果我们现在运行`example_app`二进制，并把它我们序列化模型，我们应该在一个快乐的结局到达：

    
    
    $ ./example_app example.pt
    11.4125   5.8262   9.5345   8.6111  12.3997
     7.4683  13.5969   9.0850  11.0698   9.4008
     7.4597  15.0926  12.5727   8.9319   9.0666
     9.4834  11.1747   9.0162  10.9521   8.6269
    10.0000  10.0000  10.0000  10.0000  10.0000
    10.0000  10.0000  10.0000  10.0000  10.0000
    10.0000  10.0000  10.0000  10.0000  10.0000
    10.0000  10.0000  10.0000  10.0000  10.0000
    [ Variable[CPUFloatType]{8,5} ]
    

成功！您现在可以推论了。

## 结论

本教程走你扔了如何实现在C
++中的自定义TorchScript运营商，如何将它建设成一个共享库，如何在Python中使用它来定义TorchScript模型，最后如何将其加载到用于推断工作量C
++应用程序。您现在可以使用C ++运算符与第三方C
++库接口扩展您的TorchScript模型，编写自定义的高性能CUDA内核，或实现需要Python，TorchScript和C
++之间的界限顺利融入任何其他使用情况。

与往常一样，如果您遇到任何问题或有任何疑问，您可以使用我们的[论坛](https://discuss.pytorch.org/)或[
GitHub的问题](https://github.com/pytorch/pytorch/issues)取得联系。此外，我们的[常见问题(FAQ）页](https://pytorch.org/cppdocs/notes/faq.html)可能有帮助的信息。

## 附录A：建筑运营商定制的更多方法

“建设运营商定制”一节中介绍如何构建一个运营商定制成使用CMake的共享库。本附录概述了编译另外两个方法。他们都使用Python作为“驾驶员”或“接口”的编译过程。此外，两个重复使用[现有的基础设施](https://pytorch.org/docs/stable/cpp_extension.html)
PyTorch提供[ * C ++扩展*
](https://pytorch.org/tutorials/advanced/cpp_extension.html)，它们是香草(渴望）PyTorch等效TorchScript运营商定制的依赖于[
pybind11 ](https://github.com/pybind/pybind11)为选自C的函数“明确的”结合++成Python。

第一种方法使用C
++的扩展[方便刚刚在实时(JIT）编译接口](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load)编译代码在你PyTorch脚本的后台运行它的第一次。第二种方法依赖于古老`
setuptools的 `包和涉及编写单独的`setup.py`文件。这允许更高级的配置以及整合与其他`setuptools的
`为基础的项目。我们将探讨在下面详细两种方法。

### 与JIT编译馆

由PyTorch C ++扩展工具包中提供的JIT编译特征允许嵌入自定义操作的汇编直接进入Python代码，例如在你的训练脚本的顶部。

Note

“JIT编译”这里没有什么做的JIT编译发生在TorchScript编译器优化你的程序。它只是意味着你的运营商定制的C ++代码将一个文件夹在你的系统的 /
tmp目录目录下的第一次导入它，就好像你自己事先编编就。

这JIT编译功能有两种形式。在第一个，你还是留着你的运营商实现在一个单独的文件(`op.cpp`），然后用`
torch.utils.cpp_extension.load(）`编译你的扩展。通常情况下，这个函数将返回Python模块暴露你的C
++的扩展。然而，由于我们没有编制我们的运营商定制到自己的Python模块，我们只是编译一个普通的共享库。幸运的是，`
torch.utils.cpp_extension.load(） `有一个参数`is_python_module`，我们可以设置为`假
`表明，我们只建立一个共享库，而不是一个Python模块感兴趣。 `torch.utils.cpp_extension.load(）
`将然后编译和共享库也加载到当前进程，就象`torch.ops.load_library`以前那样：

    
    
    import torch.utils.cpp_extension
    
    torch.utils.cpp_extension.load(
        name="warp_perspective",
        sources=["op.cpp"],
        extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
        is_python_module=False,
        verbose=True
    )
    
    print(torch.ops.my_ops.warp_perspective)
    

这应该大约打印：

    
    
    <built-in method my_ops::warp_perspective of PyCapsule object at 0x7f3e0f840b10>
    

JIT编译的第二香味可以让你通过源代码为您定制TorchScript运营商作为一个字符串。对于这一点，使用`
torch.utils.cpp_extension.load_inline`：

    
    
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
    
    static auto registry =
      torch::RegisterOperators("my_ops::warp_perspective", &warp_perspective);
    """
    
    torch.utils.cpp_extension.load_inline(
        name="warp_perspective",
        cpp_sources=op_source,
        extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
        is_python_module=False,
        verbose=True,
    )
    
    print(torch.ops.my_ops.warp_perspective)
    

当然，最好的做法是只使用`torch.utils.cpp_extension.load_inline`如果你的源代码是相当短的。

请注意，如果你在一个Jupyter笔记本电脑用这个，你不应该因为每次执行注册一个新的图书馆，并重新注册运营商定制执行与登记多次的细胞。如果您需要重新执行它，请事先重新启动笔记本的Python的内核。

### 与setuptools的构建

专门从Python的建设我们的运营商定制的第二种方法是使用`setuptools的 [HTG3。这具有`setuptools的 `具有用于建筑用C
++编写Python模块一个相当强大的和广泛的接口的优点。然而，由于`setuptools的
`真的打算用于建筑Python模块和非纯共享库(不具有必要的入口点Python从一个模块期望），这条路线可以稍微古怪。这就是说，你需要的是到位的，看起来像这样的`
的CMakeLists.txt`A `setup.py`文件：`

    
    
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
    

请注意，我们启用了`no_python_abi_suffix`中的`BuildExtension`在底部的选项。这指示`setuptools的
`省略任何的Python-3特异性ABI后缀在所产生的共享库的名称。否则，关于Python 3.7例如，库可以被称为`
warp_perspective.cpython-37m-x86_64-linux-gnu.so`其中`
CPython的-37M-x86_64的-linux-GNU`是ABI标签，但我们真的只是希望它被称为`warp_perspective.so`

如果我们现在运行`巨蟒 setup.py  建 从文件夹内发展 `在终端中`setup.py`坐落，我们应该看到：

    
    
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
    

这将产生所谓的`共享库warp_perspective.so`，其中我们可以通过`torch.ops.load_library
正如我们前面所做为`让我们的运营商看到TorchScript：

    
    
    >>> import torch
    >>> torch.ops.load_library("warp_perspective.so")
    >>> print(torch.ops.custom.warp_perspective)
    <built-in method custom::warp_perspective of PyCapsule object at 0x7ff51c5b7bd0>
    

[Next ![](../_static/images/chevron-right-
orange.svg)](numpy_extensions_tutorial.html "Creating Extensions Using numpy
and scipy") [![](../_static/images/chevron-right-orange.svg)
Previous](../beginner/aws_distributed_training_tutorial.html "4. \(advanced\)
PyTorch 1.0 Distributed Trainer with Amazon AWS")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * 使用自定义C ++扩展算TorchScript 
    * 用C实现运营商定制++ 
    * 与TorchScript注册自定义操作
    * 构建自定义操作
      * 与CMake的构建
    * 在Python使用TorchScript运营商定制
      * 使用运营商定制与跟踪
      * 使用运营商定制与脚本
    * 在C使用自定义TorchScript算++ 
    * 结论
    * [HTG0附录A：建筑运营商定制的更多方法
      * 与JIT编译馆
      * 与setuptools的构建

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



