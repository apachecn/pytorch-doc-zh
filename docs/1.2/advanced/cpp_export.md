# 3.装载在C TorchScript模型++

**本教程更新与PyTorch 1.2工作**

正如它的名字所暗示的，以PyTorch主接口是Python编程语言。虽然Python是许多场景需要活力和易于迭代的合适的和优选的语言，有同样多的地方正是这些的Python性质是不利的情况。其中后者往往应用于一个环境中是
_生产_ \- 低延迟和严格的部署要求的土地。对于生产情景，C
++是经常选择的语言，即使只将其绑定到像Java，生锈或转到另一种语言。以下段落将概括PyTorch提供从现有的Python模型去能够 _加载_ 和 _从C
++执行_ 纯粹序列化表示，与Python中没有依赖关系的路径。

## 第1步：转换您的PyTorch模型火炬脚本

从Python来C
++甲PyTorch模型的旅程是由[启用火炬脚本](https://pytorch.org/docs/master/jit.html)，可被理解的，编译的和由火炬脚本编译器序列化PyTorch模型的表示。如果你是从写在香草“渴望”
API现有PyTorch模型开始了，你必须首先转换模型火炬脚本。在最常见的情况下，下面讨论的，这个只需要很少的努力。如果你已经有了一个火炬脚本模块，可以跳过本教程的下一节。

存在一个PyTorch模型转换为火炬脚本的方法有两种。第一被称为 _追踪_
，其中，所述模型的结构是通过评估一次使用实施例的输入，并记录这些输入通过模型的流动捕获的机制。本品适用于模型制作有限使用控制流。第二种方法是明确的注释添加到您的模型，通知火炬脚本编译器，它可以直接解析和编译你的模型代码，受火炬脚本语言所带来的限制。

小费

你可以找到这两种方法的完整文档，以及在其上使用，在官方[火炬脚本参考](https://pytorch.org/docs/master/jit.html)进一步的指导。

### 通过跟踪转换为火炬脚本

要通过跟踪一个PyTorch模型转换为火炬脚本，你必须用一个例子输入转达您的模型的实例到`torch.jit.trace`功能。这将产生一个`
torch.jit.ScriptModule`对象与你的模型评估的嵌入模块的`转发 `方法跟踪：

    
    
    import torch
    import torchvision
    
    # An instance of your model.
    model = torchvision.models.resnet18()
    
    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 224, 224)
    
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)
    

所追踪的`ScriptModule`现在可以被相同地评价，以常规PyTorch模块：

    
    
    In[1]: output = traced_script_module(torch.ones(1, 3, 224, 224))
    In[2]: output[0, :5]
    Out[2]: tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)
    

### 通过注释转换为火炬脚本

在某些情况下，例如，如果您的模型采用控制流的特定形式，你可能想直接写脚本火炬模型，并相应地标注模型。例如，假设您有以下香草Pytorch模型：

    
    
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
    

因为这个模块的`向前 `方法使用控制流依赖于输入，它是不适合于跟踪。相反，我们可以将其转换为`ScriptModule  [HTG7。为了将模块转换为`
ScriptModule`，需要编译`torch.jit.script`如下模块：`

    
    
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
    

如果您需要排除一些方法你`nn.Module`因为他们使用Python的特点是TorchScript尚不支持，你可以注释那些`@torch
.jit.ignore`

`my_module`是ScriptModule的`一个实例 `认为是准备好进行序列化。

## 步骤2：序列化脚本模块到一个文件

一旦你有一个`ScriptModule
`在你的手中，无论是从跟踪或标注一个PyTorch模型，您准备将其序列化到一个文件中。后来，你就可以从该文件加载模块在C
++中，没有关于Python任何依赖性执行它。假设我们想要序列早些时候跟踪示例所示`ResNet18
`模式。为了执行该序列，只需调用[保存](https://pytorch.org/docs/master/jit.html#torch.jit.ScriptModule.save)在模块上，并通过它一个文件名：

    
    
    traced_script_module.save("traced_resnet_model.pt")
    

这将在您的工作目录中的`traced_resnet_model.pt`文件。如果您也想连载`my_module`，调用`
my_module.save（ “my_module_model.pt”）HTG10] `我们现在已经正式离开的Python境界并准备跨越到C ++的球体。

## 步骤3：加载脚本模块在C ++

要加载在C ++的序列化PyTorch模型，应用程序必须依赖于PyTorch C ++ API - 也被称为 _LibTorch
[HTG1。所述LibTorch分布包括共享库，头文件和CMake的建立配置文件的集合。虽然CMake的不是取决于LibTorch的要求，这是推荐的方法，将很好地支持未来。在本教程中，我们将建立使用CMake和LibTorch简单地装载一个最小的C
++应用程序和执行序列化PyTorch模型。_

### 最小的C ++应用

让我们先来讨论代码加载一个模块。下面将已经这样做了：

    
    
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
    

的`& LT ;炬/ script.h & GT ;
`报头包括从运行示例所必需的库LibTorch所有有关包括。我们的应用程序接受的文件路径的串行化PyTorch `ScriptModule
`作为其唯一的命令行参数，然后进行使用`炬:: JIT反序列化模块::负载（） `函数，该函数此文件路径作为输入。作为回报，我们收到`火炬:: JIT
::脚本::模块 `对象。我们将研究如何在某一时刻执行。

### 根据LibTorch和构建应用

假设我们存储在上面的代码到一个名为`例如-app.cpp`文件。一个最小`的CMakeLists.txt`构建它可能看起来简单：

    
    
    cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
    project(custom_ops)
    
    find_package(Torch REQUIRED)
    
    add_executable(example-app example-app.cpp)
    target_link_libraries(example-app "${TORCH_LIBRARIES}")
    set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
    

我们需要构建示例应用程序的最后一件事是LibTorch分布。您可以随时抓住从[下载页面](https://pytorch.org/)在PyTorch网站上最新的稳定版本。如果您下载并解压缩最新存档，您会收到与下面的目录结构的文件夹：

    
    
    libtorch/
      bin/
      include/
      lib/
      share/
    

  * 在`的lib /`文件夹中包含您必须对链接的共享库，
  * 在`包括/`文件夹中包含头文件你的程序将需要包括，
  * 的`份额/`文件夹中包含的必要CMake的配置来使简单`find_package（火炬） `上述命令。

Tip

在Windows中，调试和发布版本ABI不兼容。如果您计划建立在调试模式下你的项目，请尝试LibTorch的调试版本。

最后一步是构建应用程序。对于这一点，假设我们的例子中的目录布局如下：

    
    
    example-app/
      CMakeLists.txt
      example-app.cpp
    

现在，我们可以运行下面的命令来从`示例应用内/`文件夹内生成应用程序：

    
    
    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
    make
    

其中`/路径/到/ libtorch`应该是完整路径解压LibTorch分布。如果一切顺利，这将是这个样子：

    
    
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
    

如果我们提供的路径，跟踪`ResNet18`模型`traced_resnet_model.pt 我们前面向所得`示例中创建`-app
`二，我们应该有一个友好的“OK”奖励。请注意，如果尝试运行与`这个例子my_module_model.pt
`你会得到一个错误，指出你的输入是不兼容的形状。 `my_module_model.pt`预计1D代替4D。

    
    
    root@4b5a67132e81:/example-app/build# ./example-app <path_to_model>/traced_resnet_model.pt
    ok
    

## 第4步：用C执行脚本模块++

在成功地加载我们的连载`ResNet18`在C ++中，我们现在的代码只是一对夫妇线远离执行它！让我们这些行添加到我们的C ++应用程序的`主（）
`功能：

    
    
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));
    
    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    

前两行设置输入到我们的模型。我们创造`炬:: JIT :: IValue`（一种类型的擦除值类型的矢量`脚本::模块
`方法接受和返回），并添加一个输入。来创建输入张量，我们使用`炬::那些（） `时，等价于`的C ++ API在torch.ones
`。然后，我们运行`脚本::模块 `的`转发 `的方法，通过它我们创建了输入向量。作为回报，我们得到一个新的`IValue`，这是我们通过调用`
toTensor（） `转换为张量。

Tip

要了解更多关于像`功能火炬::者 `和一般的PyTorch C ++ API，请参阅其文档在[ https://pytorch.org/cppdocs
](https://pytorch.org/cppdocs) 。该PyTorch C ++ API提供附近使用Python
API功能奇偶校验，让您进一步的操作和处理张量就像在Python。

在最后一行，我们打印输出的前五个条目。由于我们在Python在本教程中提供的相同输入到我们的模型前，我们应该看到理想相同的输出。让我们尝试一下通过重新编译我们的应用程序，并使用相同的序列化模式运行它：

    
    
    root@4b5a67132e81:/example-app/build# make
    Scanning dependencies of target example-app
    [ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
    [100%] Linking CXX executable example-app
    [100%] Built target example-app
    root@4b5a67132e81:/example-app/build# ./example-app traced_resnet_model.pt
    -0.2698 -0.0381  0.4023 -0.3010 -0.0448
    [ Variable[CPUFloatType]{1,5} ]
    

作为参考，在Python输出先前是：

    
    
    tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)
    

看起来像一个很好的比赛！

Tip

要将你的模型GPU内存，你可以写`model.to（在:: kCUDA）[]  [HTG3。请确保输入到模型也住在CUDA内存通过调用`
tensor.to（在:: kCUDA）HTG6] `，这将返回CUDA内存一个新的张量。`

## 第5步：获取帮助和探索API

本教程希望您配备了PyTorch模型从Python来C ++路径的一个大致的了解。在本教程中描述的概念，你应该能够从香草去，“渴望”
PyTorch模型，为编译`ScriptModule`在Python中，磁盘上的序列化的文件和 - 关闭循环 - 一个可执行`脚本::模块 `在C
++中。

当然，也有我们没有涵盖的许多概念。例如，你可能会发现自己想扩展您的`ScriptModule`用C
++实现的运营商定制++或CUDA，并执行该运营商定制您的`ScriptModule内 `装入在纯C
++的生产环境。好消息是：这是可能的，并得到广泛支持！现在，你可以探索[对于这个例子](https://github.com/pytorch/pytorch/tree/master/test/custom_operator)文件夹，我们会跟进的教程不久。在时间之中，下面的链接可能是一般有所帮助：

  * 火炬脚本参考：[ https://pytorch.org/docs/master/jit.html ](https://pytorch.org/docs/master/jit.html)
  * 所述PyTorch C ++ API文档：[ https://pytorch.org/cppdocs/ ](https://pytorch.org/cppdocs/)
  * 所述PyTorch Python的API文档：[ https://pytorch.org/docs/ ](https://pytorch.org/docs/)

与往常一样，如果您遇到任何问题或有任何疑问，您可以使用我们的[论坛](https://discuss.pytorch.org/)或[
GitHub的问题](https://github.com/pytorch/pytorch/issues)取得联系。

[Next ![](../_static/images/chevron-right-
orange.svg)](super_resolution_with_onnxruntime.html "4. \(optional\) Exporting
a Model from PyTorch to ONNX and Running it using ONNX Runtime")
[![](../_static/images/chevron-right-orange.svg)
Previous](../beginner/Intro_to_TorchScript_tutorial.html "2. Introduction to
TorchScript")

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

  * 3.装载++一个TorchScript模型在C 
    * 步骤1：转换您PyTorch模型火炬脚本
      * 通过跟踪转换为火炬脚本
      * 经由注释转换为火炬脚本
    * [HTG0步骤2：序列化脚本模块到一个文件
    * 步骤3：加载脚本模块在C ++ 
      * 最小的C ++应用
      * 根据LibTorch和构建应用
    * [HTG0步骤4：执行脚本模块在C ++ 
    * [HTG0步骤5：获取帮助和探索API 

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

