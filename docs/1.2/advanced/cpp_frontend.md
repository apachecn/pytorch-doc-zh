# 使用PyTorch C ++前端

所述PyTorch C ++前端是一个纯粹的C
++接口到PyTorch机器学习框架。而到PyTorch主接口自然是Python中，这个Python的API上面坐大幅C
++代码库提供基本的数据结构和功能性，诸如张量和自动分化。 C ++的前端暴露出扩展与机器学习训练和推理所需的工具这一潜在的C ++代码库纯C ++ 11
API。这包括神经网络建模的通用组件的内置集合;一个API来扩展此集合与自定义模块;的流行优化算法如随机梯度下降这样的库;与API并行数据加载定义和数据集负载;系列化例程等等。

本教程将指导您完成训练模式与C ++前端的终端到终端的例子。具体而言，我们将训练[ DCGAN
](https://arxiv.org/abs/1511.06434) \- 一种生成模式 -
生成的数字MNIST图像。虽然概念上一个简单的例子，它应该足够给你PyTorch C
++前端的旋风概述和湿你的胃口训练更复杂的模型。我们将与你为什么会想使用C ++前端开始与一些激励的话开始，然后潜水直接进入定义和训练我们的模型。

小费

腕表[从CppCon 2018 ](https://www.youtube.com/watch?v=auRPXMMHJzc)这个闪电谈话对C
++的前端快速(幽默）的介绍。

Tip

[本说明](https://pytorch.org/cppdocs/frontend.html)提供C ++前端的部件和设计理念的清扫概述。

Tip

为PyTorch C ++生态系统文档可在[https://pytorch.org/cppdocs](https://pytorch.org/cppdocs)。在那里，你可以找到高水平的描述以及API级文档。

## 动机

我们踏上我们的甘斯和MNIST数字令人兴奋的旅程之前，让我们退后一步，并讨论你为什么会想使用C
++前端，而不是Python的一个开始。我们(PyTorch队）创造了C
++前端，以使研究中的Python不能使用，或者是根本就没有为工作的工具环境。对于这样的环境的例子包括：

  * **低延迟系统** ：您可能想要做的强化学习研究在高帧每秒和低延迟要求一个纯C ++游戏引擎。使用纯C ++库是一个更好的拟合比Python库这样的环境。蟒蛇可能不听话，因为在所有的Python解释器的缓慢的。
  * **高多线程环境** ：由于全局解释器锁(GIL），Python不能上同时运行多个系统线程。多是一种选择，但不作为可扩展的，具有显著的缺点。 C ++有没有这样的约束和线程易于使用和创造。需要重并行化，像那些在[使用的模型深Neuroevolution ](https://eng.uber.com/deep-neuroevolution/)，可以受益于这种。
  * **现有的C ++代码库** ：您可以是现有的C ++应用程序在后端服务器网页服务中的照片编辑软件渲染3D图形做任何事情的所有者，并希望机器学习方法集成到系统中。 C ++的前端可以让你留在C ++和饶了自己的结合来回Python和C ++之间的麻烦，同时保留大部分的传统PyTorch(Python）的经验，灵活性和直观性。

C
++的前端不打算使用Python前端竞争。它的目的是补充。我们知道，研究人员和工程师都喜欢PyTorch它的简单性，灵活性和直观的API。我们的目标是确保你能在每一个可能的环境中充分利用这些核心设计原理的优势，包括上述的那些。如果这些情况之一描述你的使用情况很好，或者如果你对此有兴趣或好奇，就像我们在下面的段落中探索C
++详细前端跟随一起。

Tip

C ++的前端试图提供尽可能接近到了Python前端的API。如果您正在使用Python的前端有经验和不断问自己“我怎么做X与C
++前端？”，写你的代码在Python会的方式，往往不是同一个函数和方法将在C ++中如在Python(只记得，以取代双冒号点）。

## 编写基本应用

首先，让我们写一个最小的C ++应用程序来验证我们对我们的设置在同一页上，并建立环境。首先，你需要抢 _LibTorch_ 分布的副本 -
我们是包中的所有相关标题，库和CMake的构建使用C ++前端所需的文件准备建造的zip压缩包。该LibTorch分布可供下载[
PyTorch网站](https://pytorch.org/get-
started/locally/)适用于Linux，MacOS和窗户上。本教程将承担基本的Ubuntu
Linux操作系统环境的其余部分，但是你可以自由沿在Mac OS或Windows跟随了。

Tip

上[安装PyTorch的C
++分布的说明](https://pytorch.org/cppdocs/installing.html)更详细地描述的以下步骤。

Tip

在Windows中，调试和发布版本ABI不兼容。如果您计划建立在调试模式下你的项目，请尝试LibTorch的调试版本。

第一步是在本地下载LibTorch分布，通过从PyTorch网站检索到的链接。对于香草Ubuntu Linux操作系统的环境中，这意味着运行：

    
    
    # If you need e.g. CUDA 9.0 support, please replace "cpu" with "cu90" in the URL below.
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    unzip libtorch-shared-with-deps-latest.zip
    

接下来，让我们写所谓的`dcgan.cpp`一个微小的C ++文件，其中包括`Torch / torch.h`现在来看只是打印出来三乘三个矩阵：

    
    
    #include <torch/torch.h>
    #include <iostream>
    
    int main() {
      torch::Tensor tensor = torch::eye(3);
      std::cout << tensor << std::endl;
    }
    

要构建这个小应用程序，以及我们全面的训练脚本，稍后我们将使用这个`的CMakeLists.txt`文件：

    
    
    cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
    project(dcgan)
    
    find_package(Torch REQUIRED)
    
    add_executable(dcgan dcgan.cpp)
    target_link_libraries(dcgan "${TORCH_LIBRARIES}")
    set_property(TARGET dcgan PROPERTY CXX_STANDARD 11)
    

注意

虽然CMake的是LibTorch推荐的构建系统，它不是一个硬性要求。您还可以使用Visual
Studio项目文件，QMAKE，普通的Makefile或者你觉得舒服的任何其他构建环境。但是，我们不提供这个外的现成支持。

记在上述文件的CMake线4：`find_package(Torch  REQUIRED）
`。这CMake的指示找到了LibTorch库的构建配置。为了让CMake的了解 _，其中_ 找到这些文件，必须设定`CMAKE_PREFIX_PATH
`当调用`cmake的 `。我们这样做之前，让我们对我们的`dcgan`应用下面的目录结构达成一致意见：

    
    
    dcgan/
      CMakeLists.txt
      dcgan.cpp
    

此外，我将提到的路径，解压缩后的LibTorch分发`/路径/到/ libtorch`。请注意，这 **必须是绝对路径[HTG5。特别是，设置`
CMAKE_PREFIX_PATH`喜欢的东西`../../libtorch`会以意想不到的方式打破。相反，写`$ PWD /../../
libtorch`来获得相应的绝对路径。现在，我们准备建立我们的应用程序：**

    
    
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
    root@fa350df05ecf:/home/build# make -j
    Scanning dependencies of target dcgan
    [ 50%] Building CXX object CMakeFiles/dcgan.dir/dcgan.cpp.o
    [100%] Linking CXX executable dcgan
    [100%] Built target dcgan
    

以上，我们首先创建了`dcgan`目录内`建 `文件夹，进入该文件夹，跑`cmake的 `命令产生必要的建立(MAKE）文件，并最终通过运行`让
-j`编制的项目成功。我们目前都在集中执行我们最小的二进制和完成基本的项目配置本节：

    
    
    root@fa350df05ecf:/home/build# ./dcgan
    1  0  0
    0  1  0
    0  0  1
    [ Variable[CPUFloatType]{3,3} ]
    

看起来像一个单位矩阵给我！

## 定义神经网络模型

现在，我们已经配置了基本的环境中，我们可以潜入本教程的更有趣的部分。首先，我们将讨论如何定义，并与在C ++前端模块交互。我们将使用由C
++前端提供内置模块的扩展库基本的，小规模的例子模块开始，然后实现一个完整的甘。

### 模块API基础

与Python接口线的基础上，C ++前端神经网络是由所谓的 _模块_ 可重复使用的构建块。存在来自所有其他模块来源的基本模块类。在Python，这个类是`
torch.nn.Module`和在C ++中它是`torch::ン::模块 `。除了一个`向前(）
`实施模块封装，模块通常包含任何三种子对象的算法方法：参数，缓冲剂和子模块。

参数和缓冲区存储在张量的形式状态。参数的记录梯度，而缓冲器不会。参数通常是你的神经网络训练的权重。缓冲剂的实例包括装置和用于批标准化变化。为了重复使用逻辑和状态的特定块时，PyTorch
API允许模块被嵌套。嵌套模块被称为 _子模块_ 。

参数，缓冲区和模块必须明确登记。一旦注册，如`参数）的方法 (`或`缓冲剂(） `可用于检索所有参数的容器，在整个(嵌套的）模块的层次结构。类似地，如`
方法(......） `，其中例如`至(torch:: kCUDA） `移动的所有参数和缓冲器从CPU到CUDA存储器，工作对整个模块的层次结构。

#### 定义模块和注册参数

为了把这些话转换成代码，让我们考虑用Python接口这个简单的模块：

    
    
    import torch
    
    class Net(torch.nn.Module):
      def __init__(self, N, M):
        super(Net, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(N, M))
        self.b = torch.nn.Parameter(torch.randn(M))
    
      def forward(self, input):
        return torch.addmm(self.b, input, self.W)
    

在C ++中，它应该是这样的：

    
    
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
    

就像在Python中，我们定义了一个名为`网 `(为简单起见这里类，而不是`A `结构 `类 `），并从模块基类派生它。构造函数中，我们创建一个使用`
Torch 张量:: randn`就像我们使用`torch.randn
`在Python。一个有趣的差异是我们如何注册的参数。在Python，我们包裹张量与`torch.nn.Parameter`类，而在C
++我们通过传递张量的`register_parameter`方法来代替。这样做的原因是，Python API中可以检测到一个属性是类型`
torch.nn.Parameter`的和自动注册这样张量。在C ++中，反射是非常有限的，因此提供了一种更传统的(和更小神奇）的方法。

#### 注册子模和遍历模块层次结构

以同样的方式，我们可以注册参数，我们还可以注册子模块。在Python，子模块被自动检测和注册时它们被分配作为一个模块的属性：

    
    
    class Net(torch.nn.Module):
      def __init__(self, N, M):
          super(Net, self).__init__()
          # Registered as a submodule behind the scenes
          self.linear = torch.nn.Linear(N, M)
          self.another_bias = torch.nn.Parameter(torch.rand(M))
    
      def forward(self, input):
        return self.linear(input) + self.another_bias
    

这允许，例如，使用`参数(） `方法递归地访问在我们的模块层次中的所有参数：

    
    
    >>> net = Net(4, 5)
    >>> print(list(net.parameters()))
    [Parameter containing:
    tensor([0.0808, 0.8613, 0.2017, 0.5206, 0.5353], requires_grad=True), Parameter containing:
    tensor([[-0.3740, -0.0976, -0.4786, -0.4928],
            [-0.1434,  0.4713,  0.1735, -0.3293],
            [-0.3467, -0.3858,  0.1980,  0.1986],
            [-0.1975,  0.4278, -0.1831, -0.2709],
            [ 0.3730,  0.4307,  0.3236, -0.0629]], requires_grad=True), Parameter containing:
    tensor([ 0.2038,  0.4638, -0.2023,  0.1230, -0.0516], requires_grad=True)]
    

在C ++寄存器子模块，使用恰当地命名为`register_module(） `方法注册等`torch的模块::ン::线性 `：

    
    
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
    

Tip

你可以找到可用的内置模块一样的完整列表`Torch :: NN ::线性 `，`Torch :: NN ::差 `或`Torch :: NN :: Conv2d`中的`
Torch 的文档:: NN
`命名空间[这里](https://pytorch.org/cppdocs/api/namespace_torch__nn.html)。

关于上述代码的一个微妙之处就是为什么子模块是在构造函数的初始化列表中创建的，而参数是在构造函数体内部创建的。有一个很好的理由，我们将在对C ++前端的
_所有权模式_ 下面进一步的部分在此碰。最终的结果，但是，我们可以递归访问我们的模块树的参数，就像在Python。主叫`参数(） `返回`的std
::矢量& LT ;torch::张量& GT ;`，我们可以遍历：

    
    
    int main() {
      Net net(4, 5);
      for (const auto& p : net.parameters()) {
        std::cout << p << std::endl;
      }
    }
    

其打印：

    
    
    root@fa350df05ecf:/home/build# ./dcgan
    0.0345
    1.4456
    -0.6313
    -0.3585
    -0.4008
    [ Variable[CPUFloatType]{5} ]
    -0.1647  0.2891  0.0527 -0.0354
    0.3084  0.2025  0.0343  0.1824
    -0.4630 -0.2862  0.2500 -0.0420
    0.3679 -0.1482 -0.0460  0.1967
    0.2132 -0.1992  0.4257  0.0739
    [ Variable[CPUFloatType]{5,4} ]
    0.01 *
    3.6861
    -10.1166
    -45.0333
    7.9983
    -20.0705
    [ Variable[CPUFloatType]{5} ]
    

有三个参数，就像在Python。还看到这些参数的名称，在C ++ API提供了`named_pa​​rameters(） `方法，它返回一个`
OrderedDict`就像在Python ：

    
    
    Net net(4, 5);
    for (const auto& pair : net.named_parameters()) {
      std::cout << pair.key() << ": " << pair.value() << std::endl;
    }
    

我们可以再次执行看到的输出：

    
    
    root@fa350df05ecf:/home/build# make && ./dcgan                                                                                                                                            11:13:48
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
    linear.weight:  0.0339  0.2484  0.2035 -0.2103
    -0.0715 -0.2975 -0.4350 -0.1878
    -0.3616  0.1050 -0.4982  0.0335
    -0.1605  0.4963  0.4099 -0.2883
    0.1818 -0.3447 -0.1501 -0.0215
    [ Variable[CPUFloatType]{5,4} ]
    linear.bias: -0.0250
    0.0408
    0.3756
    -0.2149
    -0.3636
    [ Variable[CPUFloatType]{5} ]
    

Note

[的文档](https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#exhale-
class-classtorch-1-1nn-1-1-module)为`torch::ン::模块 `包含的，关于模块的层次结构进行操作的方法的完整列表。

#### 在正向模式下运行的网络

为了执行在C ++中的网络中，我们只需调用`向前(） `方法中，我们定义自己：

    
    
    int main() {
      Net net(4, 5);
      std::cout << net.forward(torch::ones({2, 4})) << std::endl;
    }
    

它打印是这样的：

    
    
    root@fa350df05ecf:/home/build# ./dcgan
    0.8559  1.1572  2.1069 -0.1247  0.8060
    0.8559  1.1572  2.1069 -0.1247  0.8060
    [ Variable[CPUFloatType]{2,5} ]
    

#### 模块所有权

在这一点上，我们知道如何定义在C ++模块，注册参数，注册子模块，通过像`参数）的方法 (`穿越模块的层次结构，并最终运行模块的`向前(）
`方法。虽然还有更多的方法，类和主题的C ++
API中吞噬，我会向您推荐[文档](https://pytorch.org/cppdocs/api/namespace_torch__nn.html)完整的菜单。我们也将触及一些概念，我们实现在短短一秒钟DCGAN模型和终端到终端的训练渠道。在这样做之前，让当
_所有权模式_ C ++的前端提供了`Torch 的子类，我简要地谈谈:: NN ::模块 [HTG15。`

为了便于讨论，所有权模式是指模块存储并通过周围的方式 - 它决定谁或什么 _拥有_
特定模块实例。在Python，对象总是动态分配(在堆上），并且具有引用语义。这是非常易于使用和易于理解。事实上，在Python中，你可以在很大程度上忘记对象居住在哪里以及如何他们得到引用，并专注于做事情。

C ++，作为一个较低级别的语言，提供了在此领域的更多选项。这增加了复杂性和严重影响了设计和C ++前端的人体工程学设计。特别地，对于在C
++前端模块，我们有使用 _为_ 值语义 _或_
引用语义的选项。第一种情况是最简单的和实施例中迄今为止被证明：模块对象被分配在栈上，并传递给函数时，既可以复制，移动(与`的std ::移动
`）或采取引用或指针：

    
    
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
    

对于第二种情况 - 参考语义 - 我们可以使用`的std :: shared_ptr的
`。引用传递的好处是，像在Python，它减少想着模块必须如何传递给函数和参数必须如何申报的认知开销(假设你使用`shared_ptr的 `到处）。

    
    
    struct Net : torch::nn::Module {};
    
    void a(std::shared_ptr<Net> net) { }
    
    int main() {
      auto net = std::make_shared<Net>();
      a(net);
    }
    

在我们的经验中，研究人员从动态语言来非常喜欢引用语义过值语义，即使后者更是“原生”到C ++。同样重要的是要注意，`Torch :: NN
::模块的设计，以贴近了Python API的人体工程学设计，依赖于共享所有权 [HTG3。例如，利用`网 `我们先前的(这里缩短）的定义：`

    
    
    struct Net : torch::nn::Module {
      Net(int64_t N, int64_t M)
        : linear(register_module("linear", torch::nn::Linear(N, M)))
      { }
      torch::nn::Linear linear;
    };
    

为了使用`线性
`子模块，我们希望直接存储在我们班。但是，我们也希望在模块的基类来了解并有机会获得这个子模块。为此，它必须保存到该子模块的参考。在这一点上，我们已经到达了需要共享所有权。两者`
torch::ン::模块 `类和混凝土`净 `类需要到子模块的引用。由于这个原因，基类存储模块为`的shared_ptr`S，因此，混凝土类必须太。

可是等等！我没有看到的`在上面的代码中的shared_ptr`任何提及！这是为什么？那么，因为`的std :: shared_ptr的& LT ;
MyModule的& GT ;`是很多类型的地狱。为了使我们的研究人员生产力，我们想出了一个精心设计的方案，以隐藏`提的shared_ptr`
\- 一个好处通常保留值语义 - 同时保持引用语义。要理解这是如何工作的，我们可以看看在`Torch 的简化定义:: NN ::线性
`模块中的核心库(完整的定义是[在这里](https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/modules/linear.h)）：

    
    
    struct LinearImpl : torch::nn::Module {
      LinearImpl(int64_t in, int64_t out);
    
      Tensor forward(const Tensor& input);
    
      Tensor weight, bias;
    };
    
    TORCH_MODULE(Linear);
    

简而言之：将模块不叫`线性 `，但`LinearImpl`。宏，`TORCH_MODULE`然后定义实际`线性
`类。这个“而生成”类实际上是在一个包装一`的std :: shared_ptr的& LT ; LinearImpl & GT ;
`。这是一个包装，而不是一个简单的typedef，这样，除其他事项外，还构造如预期，即你仍然可以写`Torch :: NN ::线性(3， 4 ) `而非`
的std :: make_shared & LT ; LinearImpl & GT ;(3， 4） `。我们呼吁由宏模块 _持有者_
创建的类。像(共享的）指针，则使用箭头操作者访问底层对象(如`模型 - & GT ;向前(......）
`）。最终的结果是所有权模式，类似于Python的API的颇有渊源。引用语义成为默认，但没有`额外的输入的std :: shared_ptr的 `或`
的std :: make_shared  [HTG45。对于我们的`网 `，使用模块底座API看起来是这样的：`

    
    
    struct NetImpl : torch::nn::Module {};
    TORCH_MODULE(Net);
    
    void a(Net net) { }
    
    int main() {
      Net net;
      a(net);
    }
    

有一个值得在这里提到一个微妙的问题。默认构造`的std :: shared_ptr的 `是“空的”，即，包含一个空指针。在默认构造什么`线性 `或`网
`？嗯，这是一个棘手的选择。我们可以说，它应该是一个空(NULL）`的std :: shared_ptr的& LT ; LinearImpl & GT ;
[HTG15。然而，回想`线性(3， 4） `是与`的std :: make_shared & LT [; ] LinearImpl & GT ;(3，
4） `。这意味着，如果我们已经决定，`线性 线性;
`应该是一个空指针，则就没有办法来构造的模块不采取任何构造函数的参数，或者默认所有的人。出于这个原因，目前的API中，默认的构建模块保持器(如`线性(）
`）调用底层模块的默认构造(`LinearImpl(）
`）。如果底层模块不会有一个默认的构造函数，你得到一个编译错误。为了构建，而不是空的持有人，你可以通过`nullptr`到支架的构造。`

在实践中，这意味着你可以使用子模块要么喜欢早些时候，当模块注册，并在 _初始化列表_ 构造图所示：

    
    
    struct Net : torch::nn::Module {
      Net(int64_t N, int64_t M)
        : linear(register_module("linear", torch::nn::Linear(N, M)))
      { }
      torch::nn::Linear linear;
    };
    

或者你可以先建立持有人一个空指针，然后分配给它的构造器(用于Pythonistas比较熟悉）：

    
    
    struct Net : torch::nn::Module {
      Net(int64_t N, int64_t M) {
        linear = register_module("linear", torch::nn::Linear(N, M));
      }
      torch::nn::Linear linear{nullptr}; // construct an empty holder
    };
    

结论：哪个所有制模式 - 其语义 - 你应该使用？在C
++前端的API最好的支持模块保持所提供的所有权模式。这个机制的唯一缺点是模块声明以下样板的一个额外的行。这就是说，最简单的模型是静止在介绍C
++模块中示出的值语义模型。对于小型，简单的脚本，你可以逃脱它。但你会发现早晚，由于技术原因，它并不总是支持。例如，串行化API(`torch::保存 `和`
torch::负载 `）仅支持模块保持器(或纯`shared_ptr的 `）。这样，模块保持器API是定义与C
++前端模块的推荐的方法，我们将在本教程此后使用该API。

### 限定DCGAN模块

我们现在有必要的背景介绍和定义，我们希望在这个岗位，解决了机器学习任务模块。要回顾一下：我们的任务是生成从[
MNIST数据集](http://yann.lecun.com/exdb/mnist/)的数字图像。我们要使用[生成对抗网络(GAN）HTG3]来解决这个任务。特别是，我们将使用](https://papers.nips.cc/paper/5423-generative-
adversarial-nets.pdf)[ DCGAN架构](https://arxiv.org/abs/1511.06434) \-
第一，它的那种简单的，但完全足以完成这个任务之一。

Tip

您可以在此存储库在本教程[提出了完整的源代码。](https://github.com/pytorch/examples/tree/master/cpp/dcgan)

#### 什么是甘阿甘？

甲GAN包括两个不同的神经网络模型：一个 _发生器_ 和a _鉴别_ 。发电机从噪声分布接收样本，且其目的是将每个噪声采样转变成类似于那些目标分布的图像 -
在我们的情况下，MNIST数据集。反过来鉴别从所述数据集MNIST接收任一 _真实_ 的图像，或者从发电机 _假_
图像。它被要求发射的概率判断如何真实(越接近`1`）或假(越接近`0
`）的特定图像是。从如何真正由发电机产生的图像被用来训练发生器鉴别反馈。如何很好的真实性眼睛的鉴别已经被用于优化鉴别反馈。从理论上讲，在发电机和鉴别器之间的微妙平衡使得它们在串联提高，导致发生器产生从目标分布没有区别的图像，欺骗鉴别的(通过随后）优异的眼成发光的`
0.5的概率 `两个真假图像。对我们来说，最终的结果是接收噪声作为输入并产生数字作为其输出逼真的图像的机器。

#### 所述发生器模块

我们首先定义发生器模块，它由一系列换位2D卷积，一批归一化和激活RELU单位的。像在Python，PyTorch这里提供了一种用于模型定义两个API：功能性的，其中输入通过连续函数过去了，更之一，我们构建含有`
序贯 `模块的面向对象的整个模型的子模块。让我们来看看我们的发电机的外观与任何API，你可以自己决定你更喜欢哪一个。首先，使用`序贯 `：

    
    
    using namespace torch;
    
    nn::Sequential generator(
        // Layer 1
        nn::Conv2d(nn::Conv2dOptions(kNoiseSize, 256, 4)
                       .with_bias(false)
                       .transposed(true)),
        nn::BatchNorm(256),
        nn::Functional(torch::relu),
        // Layer 2
        nn::Conv2d(nn::Conv2dOptions(256, 128, 3)
                       .stride(2)
                       .padding(1)
                       .with_bias(false)
                       .transposed(true)),
        nn::BatchNorm(128),
        nn::Functional(torch::relu),
        // Layer 3
        nn::Conv2d(nn::Conv2dOptions(128, 64, 4)
                       .stride(2)
                       .padding(1)
                       .with_bias(false)
                       .transposed(true)),
        nn::BatchNorm(64),
        nn::Functional(torch::relu),
        // Layer 4
        nn::Conv2d(nn::Conv2dOptions(64, 1, 4)
                       .stride(2)
                       .padding(1)
                       .with_bias(false)
                       .transposed(true)),
        nn::Functional(torch::tanh));
    

Tip

A `序贯 `模块简单地执行功能的组合物。该第一子模块的输出变成第二输入，第三的输出变为第四等的输入。

特定模块选择，如`NN :: Conv2d`和`NN :: BatchNorm`，如下前面概括的结构。的`kNoiseSize
`常数决定输入噪声向量的大小和设置为`100`。还要注意的是，我们使用`Torch :: NN ::功能 `模块为我们的激活功能，通过它`Torch ::
RELU`为内层和`torch::的tanh`作为最终活化。超参数进行，当然，通过研究生血统找到。

Note

Python的前端具有用于每个激活功能一个模块，如`torch.nn.ReLU`或`torch.nn.Tanh`。在C ++中，我们不是仅提供`
功能 `模块中，向其中可以传递任何C ++函数，将内部被称为`功能 `“S `向前 `(）方法。

注意

没有研究生在超参数的发现受到伤害。他们定期喂食Soylent。

对于第二种方法，我们明确地通过模块之间的输入(以功能性方式）在`向前(） `的模块的方法，我们定义自己：

    
    
    struct GeneratorImpl : nn::Module {
      GeneratorImpl(int kNoiseSize)
          : conv1(nn::Conv2dOptions(kNoiseSize, 256, 4)
                      .with_bias(false)
                      .transposed(true)),
            batch_norm1(256),
            conv2(nn::Conv2dOptions(256, 128, 3)
                      .stride(2)
                      .padding(1)
                      .with_bias(false)
                      .transposed(true)),
            batch_norm2(128),
            conv3(nn::Conv2dOptions(128, 64, 4)
                      .stride(2)
                      .padding(1)
                      .with_bias(false)
                      .transposed(true)),
            batch_norm3(64),
            conv4(nn::Conv2dOptions(64, 1, 4)
                      .stride(2)
                      .padding(1)
                      .with_bias(false)
                      .transposed(true)),
            batch_norm4(64),
            conv5(nn::Conv2dOptions(64, 1, 4)
                      .stride(2)
                      .padding(1)
                      .with_bias(false)
                      .transposed(true))
     {
       // register_module() is needed if we want to use the parameters() method later on
       register_module("conv1", conv1);
       register_module("conv2", conv2);
       register_module("conv3", conv3);
       register_module("conv4", conv4);
       register_module("batch_norm1", batch_norm1);
       register_module("batch_norm2", batch_norm1);
       register_module("batch_norm3", batch_norm1);
     }
    
     torch::Tensor forward(torch::Tensor x) {
       x = torch::relu(batch_norm1(conv1(x)));
       x = torch::relu(batch_norm2(conv2(x)));
       x = torch::relu(batch_norm3(conv3(x)));
       x = torch::tanh(conv4(x));
       return x;
     }
    
     nn::Conv2d conv1, conv2, conv3, conv4;
     nn::BatchNorm batch_norm1, batch_norm2, batch_norm3;
    };
    TORCH_MODULE(Generator);
    
    Generator generator;
    

我们使用哪种方法，我们现在可以调用`向前(） `关于`发生器 `映射一个噪声样本的图像。

Note

在途中选项的简要字被传递给内置模块等`Conv2d`在C ++前端：每个模块具有一些所需的选项，如特征为`[数HTG5] BatchNorm
`。如果你只需要配置所需选项，你可以直接将它们传递到模块的构造，如`BatchNorm(128） `或`降(0.5） `或`Conv2d(8， 4，
2） `(用于输入信道数，输出信道数，和内核大小）。但是，如果你需要修改的其他选项，这通常是默认，如`with_bias`对`Conv2d
`，你需要构建并传递一个 _项_ 对象。在C ++前端每个模块都有一个相关联的选项结构，称为`ModuleOptions`其中`模块
`是模块的名称，如`LinearOptions`为`线性 `。这就是我们的`Conv2d`上述模块做。

#### 所述鉴别器模块

鉴别器是同样的卷积，批次归一和激活的序列。然而，盘旋现在一些约定俗成的，而不是换位，我们用一个漏水的RELU为0.2，而不是香草RELU的alpha值。另外，最终活化变为乙状结肠，其南瓜值成范围在0和1之间然后，我们可以解释这些压扁值作为鉴别器分配给图像是真实的概率：

    
    
    nn::Sequential discriminator(
      // Layer 1
      nn::Conv2d(
          nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).with_bias(false)),
      nn::Functional(torch::leaky_relu, 0.2),
      // Layer 2
      nn::Conv2d(
          nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).with_bias(false)),
      nn::BatchNorm(128),
      nn::Functional(torch::leaky_relu, 0.2),
      // Layer 3
      nn::Conv2d(
          nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).with_bias(false)),
      nn::BatchNorm(256),
      nn::Functional(torch::leaky_relu, 0.2),
      // Layer 4
      nn::Conv2d(
          nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).with_bias(false)),
      nn::Functional(torch::sigmoid));
    

Note

当该功能我们通过`功能 `需要更多的参数比单个张量，我们可以将它们传递到`功能 `构造，这将它们转发到每个函数调用。对于上面的泄漏RELU，这意味着`
torch:: leaky_relu(previous_output_tensor， 0.2） `是调用。

## 载入数据

现在，我们已经定义了发电机和鉴别模型，我们需要一些数据，我们可以一起训练这些模型。 C
++的前端，诸如Python之一，带有一个强大的并行数据加载器。该数据加载器可以读取一个数据集的数据批次(你可以自己定义），并提供了许多配置旋钮。

Note

虽然Python数据加载程序使用并行处理中，C ++数据加载器是真正的多线程和不启动任何新的过程。

数据加载是C ++前端的`数据 `API，包含在`torch::数据::`命名空间的一部分。这个API是由几个不同的部分组成：

  * 数据加载器类，
  * 用于定义数据集的API，
  * 用于限定 _变换_ 的API，其可以被应用到数据集，
  * 用于限定 _取样_ ，它产生与数据集编索引的索引的API，
  * 现有数据集，转换器和采样库。

在本教程中，我们可以使用自带的C ++前端的`MNIST`数据集。让我们来实例化一个`Torch ::数据::数据集:: MNIST
`对于这一点，并应用两个转变：一是标准化的图像，使它们在范围`-1-`至`+1`(从原始范围`至`[0 ``HTG21] 1
）。其次，我们应用`堆栈 `_整理_ ，这需要一批张量，并将它们堆叠成沿着第一维度单一张量：

    
    
    auto dataset = torch::data::datasets::MNIST("./mnist")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    

需要注意的是MNIST数据集应位于`./mnist
`相对于无论你执行从训练二进制文件目录。您可以使用[这个脚本](https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03)下载MNIST数据集。

接下来，我们创建了一个数据加载器，并通过它这个数据集。为了使新的数据加载，我们使用`Torch ::数据:: make_data_loader
`，它返回的一个`的std ::的unique_ptr`正确的类型(这取决于数据集，采样器和其他一些实施细节的类型的类型）：

    
    
    auto data_loader = torch::data::make_data_loader(std::move(dataset));
    

数据装载的确有很多的选择。您可以检查整套[此处[HTG1。例如，为了加快数据加载，我们可以增加工人的数量。默认号码是零，这意味着主线程将被使用。如果我们设置`
工人 `至`2`，两个线程将同时催生该负载数据。我们还应该从它的`默认增加批量大小1 HTG12] `的东西比较合理，喜欢`64`(值`
kBatchSize`）。因此，让我们创建一个`DataLoaderOptions
`对象，并设置相应的属性：](https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/dataloader_options.h)

    
    
    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));
    

现在，我们可以写一个循环来加载数据，我们将只打印到控制台现在的批次：

    
    
    for (torch::data::Example<>& batch : *data_loader) {
      std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
      for (int64_t i = 0; i < batch.data.size(0); ++i) {
        std::cout << batch.target[i].item<int64_t>() << " ";
      }
      std::cout << std::endl;
    }
    

由数据装入程序在这种情况下返回的类型是`torch::数据::实施例 `。这种类型是一个简单的结构与用于数据的`数据 `字段和一个标签`目标
`字段。因为我们应用了`堆栈 `核对之前，则数据加载器仅返回一个这样的例子。如果我们没有施加核对，数据加载器将产生`的std ::矢量& LT
;torch::数据::实施例& LT ; [ - - ] GT ; & GT ;`代替，以在每批次例如一个元素。

如果重建并运行这段代码，你会看到这样的事情：

    
    
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
    

这意味着我们能够成功地从MNIST数据集加载数据。

## 写作训练循环

现在，让我们完成我们的例子中的算法部分和实施发电机和鉴别之间微妙的舞蹈。首先，我们将创建两个优化，一个发电机和一个用于鉴别。在优化我们使用实现[亚当](https://arxiv.org/pdf/1412.6980.pdf)算法：

    
    
    torch::optim::Adam generator_optimizer(
        generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
    torch::optim::Adam discriminator_optimizer(
        discriminator->parameters(), torch::optim::AdamOptions(5e-4).beta1(0.5));
    

Note

在撰写本文时，C
++的前端提供了实施Adagrad，亚当，LBFGS，RMSprop和SGD优化。在[文档](https://pytorch.org/cppdocs/api/namespace_torch__optim.html)有向上的最新名单。

接下来，我们需要更新我们的训练循环。我们将增加一个外环用尽数据加载每个时间段，然后写GAN训练码：

    
    
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
    

上面，我们首先评估真实图像，它应该指定一个高概率的鉴别。对于这一点，我们使用`torch::空(batch.data.size(0））。uniform_(0.8，
1.0） `作为目标概率。

Note

我们挑选到处都在为了使鉴别训练更强大的0.8和1.0，而不是1.0之间均匀分布的随机值。这一招被称为 _标签平滑[HTG1。_

评估鉴别之前，我们归零其参数的梯度。计算损失后，我们通过网络通过调用`d_loss.backward(）
`来计算新的梯度回传播。我们重复这个高谈阔论的假像。而不是使用图片来自数据集的，我们让发电机通过喂养它了一批随机噪声的创建这种假像。然后，我们这些假图像转发到鉴别。这一次，我们要鉴别发出低概率，最好全部为零。一旦我们计算了两个批次的真实与虚假批图像的鉴别损失，我们可以以更新其参数一步进展鉴别的优化。

为了训练发电机，我们再次先零的梯度，然后重新评估的假图像鉴别。然而，这一次我们要鉴别分配的概率非常接近，这表明该发生器可以产生这种欺骗鉴别以为他们实际上是(从数据集）的真实图像。为此，我们填补`
fake_labels`全部为一张量。我们终于步发电机的优化也更新其参数。

现在，我们应该准备训练CPU上我们的模型。我们没有任何代码尚未捕获状态或样品产出，但我们会在短短的时刻添加此。现在，就让我们看到，我们的模型是做 _东西_
\- 我们将根据生成的图像这个东西是否是有意义的再验证。重新构建和运行应打印是这样的：

    
    
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
    

## 移动到GPU

虽然我们当前的脚本可以运行在CPU上就好了，大家都知道卷积都在GPU快了很多。让我们快速讨论如何我们的训练走上了GPU。我们需要为这个做两件事情：一个GPU设备规范传递给我们分配自己张量，并明确通过`
以(） `法任何其他张量复制到所有GPU张量和模块在C ++前端有。实现这两个最简单的方法是在我们的训练脚本的顶层创建`Torch ::设备
`的实例，然后将该设备传递给张厂的功能，如`torch::零 `以及`至(） `方法。我们可以通过与CPU设备这样开始：

    
    
    // Place this somewhere at the top of your training script.
    torch::Device device(torch::kCPU);
    

新张量分配像

    
    
    torch::Tensor fake_labels = torch::zeros(batch.data.size(0));
    

应该被更新为取`装置 `作为最后一个参数：

    
    
    torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
    

对于张量其创建是不在我们手里，像那些从MNIST数据集的到来，我们必须插入明确的`以(） `通话。这意味着

    
    
    torch::Tensor real_images = batch.data;
    

变

    
    
    torch::Tensor real_images = batch.data.to(device);
    

而且我们的模型参数应该被移动到正确的设备：

    
    
    generator->to(device);
    discriminator->to(device);
    

Note

如果张量已经住供应到`在设备上(） `，该呼叫是一个空操作。无需额外的副本。

在这一点上，我们只是做我们以前的CPU-居住代码更加明确。不过，现在也很容易对设备更改为CUDA设备：

    
    
    torch::Device device(torch::kCUDA)
    

而现在所有的张量将住在GPU上，调入快速CUDA内核的所有操作，不用我们无需更改任何代码的下游。如果我们想要指定一个特定的设备索引，它可以作为第二个参数`
设备
`构造函数传递。如果我们想要不同张量住在不同设备上，我们可以(CUDA装置0和其他CUDA装置1上例如一种）通过单独的装置的实例。我们甚至可以这样做动态配置，这往往是有益的，使我们的训练脚本更便于携带：

    
    
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! Training on GPU." << std::endl;
      device = torch::kCUDA;
    }
    

甚至

    
    
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    

## 检查点和恢复训练状况

最后的增强，我们应该对我们的训练脚本定期保存我们的模型参数，我们优化的状态，以及一些生成的图像样本的状态。如果我们的电脑是在训练过程中间崩溃，前两个将使我们能够恢复训练状态。对于长期的训练课程，这是绝对必要的。幸运的是，C
++前端提供了一个API来序列和反序列化两者模型和优化器状态，​​以及个别张量。

核心API因为这是`Torch ::保存(的东西，文件名）HTG2] `和`Torch ::负载(的东西，文件名）HTG6] `其中`事情 `可以是`
torch::ン::模块 `亚类或类似的`亚当的优化实例 `对象，我们在我们的训练讲稿。让我们来更新我们的训练循环检查点在一定的时间间隔模型和优化状态：

    
    
    if (batch_index % kCheckpointEvery == 0) {
      // Checkpoint the model and optimizer state.
      torch::save(generator, "generator-checkpoint.pt");
      torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
      torch::save(discriminator, "discriminator-checkpoint.pt");
      torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
      // Sample the generator and save the images.
      torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
      torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
      std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
    }
    

其中`kCheckpointEvery`是一个整数设置为类似`100`检查点每`100`批次和`checkpoint_counter
`是一个反撞我们每一个检查点的时间。

要恢复训练状态，可以将所有的模型之后添加这样的诗句和优化创建，但训练循环之前：

    
    
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
    

## 检查生成的图像

我们的训练剧本现已完成。我们准备训练我们的甘，无论是在CPU或GPU。要检查我们的训练过程的中介输出，为此我们添加的代码，以图像样本定期保存到`
“dcgan-sample-xxx.pt”`文件，我们可以写一个小Python脚本加载张量，并与matplotlib显示它们：

    
    
    from __future__ import print_function
    from __future__ import unicode_literals
    
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
    

现在让我们来训练我们的模型大约30时期：

    
    
    root@3c0711f20896:/home/build# make && ./dcgan                                                                                                                                10:17:57
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
    

并显示出图的图像：

    
    
    root@3c0711f20896:/home/build# python display.py -i dcgan-sample-100.pt
    Saved out.png
    

这应该是这个样子：

![digits](img/digits.png)

数字！万岁！现在球在你的场内：您可以改进模型，使数字更好看？

## 结论

本教程希望能够给您的PyTorch C
++前端的消化消化。机器学习库像PyTorch必然具有非常广阔和丰富的API。因此，有很多的概念，我们没有时间或空间在这里讨论。不过，我鼓励你尝试了API，当你遇到问题请教[我们的文档](https://pytorch.org/cppdocs/)，特别是[库API
](https://pytorch.org/cppdocs/api/library_root.html)部分。此外，请记住，你可以期望的C
++前端遵循的设计和Python的前端的语义，只要我们能做到这一点，那么你可以利用这一点来提高你的学习速度。

Tip

You can find the full source code presented in this tutorial [in this
repository](https://github.com/pytorch/examples/tree/master/cpp/dcgan).

与往常一样，如果您遇到任何问题或有任何疑问，您可以使用我们的[论坛](https://discuss.pytorch.org/)或[
GitHub的问题](https://github.com/pytorch/pytorch/issues)取得联系。

[![](../_static/images/chevron-right-orange.svg) Previous](cpp_extension.html
"Custom C++ and CUDA Extensions")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * 使用PyTorch C ++前端
    * 动机
    * 编写基本应用
    * 定义神经网络模型
      * 模块API基础
        * 定义模块和注册参数
        * 注册子模和遍历模块层次结构
        * 在正向模式中运行的网络
        * 模块所有权
      * 定义DCGAN模块
        * 什么是甘阿甘？ 
        * [HTG0所述发生器模块
        * [HTG0所述鉴别器模块
    * 加载数据
    * 写作训练循环
    * 移动到GPU 
    * 和点校验恢复训练状况
    * 检查生成的图像
    * 结论

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



