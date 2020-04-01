# 对于大规模部署的特点

  * 舰队宽操作者剖析

  * API使用记录

  * 将元数据附加到保存TorchScript模型

  * 构建环境的考虑

  * 普通的扩展点

本说明有关一个更大的系统内运行PyTorch或操作在一个更大的组织使用PyTorch多个系统时可能有用的几个扩展点和技巧讲座。

它不包括生产部署模型的主题。检查[ `torch.jit`](../jit.html#module-torch.jit
"torch.jit")或相应的教程。

该说明假定您无论是从源组织中的建PyTorch或有静态链接用于PyTorch时加载额外的代码的能力。因此，许多钩的公开为C
++的API，可以一次在一个集中的地方来触发，例如在静态初始化代码。

## 舰队宽操作仿形

PyTorch自带`torch.autograd.profiler
`能够测量按需采取个体经营者时间的。人们可以使用相同的机制做“永远在线”的三围运行PyTorch任何进程。这可能是收集有关在给定的过程或整组机器的运行PyTorch工作负荷的信息是有用的。

对于任何运营商调用回调新可以用`添加Torch :: autograd ::探查:: pushCallback  [HTG3。钩将与`被称为Torch ::
autograd ::探查:: RecordFunction`结构描述调用上下文(例如，名称）。如果启用，`RecordFunction ::输入(）
`包含表示为`torch:: IValue`变体类型的函数的自变量。请注意，该输入记录是比较昂贵的，因此，必须明确启用。`

调用回调增加了一些开销，所以通常它只是随机采样操作调用有用。这可以在每个回调基础上通过Torch :: autograd ::探查::
setSamplingProbability 中指定的全局采样率启用。

请注意，`pushCallback`和`setSamplingProbability
`不是线程安全的，没有PyTorch操作运行，只有当可以被调用。通常情况下，这是一个好主意，在初始化时调用了它们一次。

下面是一个例子：

    
    
    // Called somewhere in the program beginning
    void init() {
        // Sample one in a hundred operator runs randomly
        torch::autograd::setSamplingProbability(0.01);
        pushCallback(
            &onFunctionEnter,
            &onFunctionExit,
            /* needs_inputs */ true,
            /* sampled */ true
        );
    }
    
    void onFunctionEnter(const RecordFunction& fn) {
        std::cerr << "Before function " << fn.name()
                  << " with " << fn.inputs().size() << " inputs" << std::endl;
    }
    
    void onFunctionExit(const RecordFunction& fn) {
        std::cerr << "After function " << fn.name();
    }
    

##  API使用登录

当在更广泛的生态系统中运行，例如在管理作业调度程序，但是这是要跟踪的二进制文件调用特定的API
PyTorch。存在于触发一个给定的回调几个重要的API点注射简单的仪器。因为通常PyTorch是一次性的Python脚本调用，回调火灾只有一次针对每个API的规定的处理。

`C10 :: SetAPIUsageHandler`可用于注册API使用的仪器处理程序。传递的参数将是一个“API密钥”识别用于点，例如`
python.import  [HTG7用于PyTorch延长进口或`torch.script.compile
[HTG11如果TorchScript编译被触发。``

    
    
    SetAPIUsageLogger([](const std::string& event_name) {
        std::cerr << "API was used: " << event_name << std::endl;
    });
    

注意为开发新的API的触发点可以在代码被添加与`C10_LOG_API_USAGE_ONCE (“my_api”） `在C ++或`
torch._C._log_api_usage_once(“我的。 API“） `在Python。

## 将元数据附加到保存TorchScript模型

TorchScript模块可以保存为捆绑串行化参数和模块代码作为TorchScript存档文件(见[ `torch.jit.save(） `
](../jit.html#torch.jit.save
"torch.jit.save")）。这是很方便的与该模型一起捆绑的其他信息，例如，模型制作者或辅助的工件的描述。

它可以通过使`_extra_files`参数为[ `torch.jit.save(） `](../jit.html#torch.jit.save
"torch.jit.save")和[HTG10来实现] Torch :: JIT ::负载
存储和保存过程中检索任意的二进制块。由于TorchScript文件定期ZIP档案，额外的信息被存储作为普通的文件归档的`额外/`目录内。

还有一个全局钩子允许额外的文件附加到当前进程产生的任何TorchScript存档。这可能是与制片人的元数据，类似于数码相机产生的JPEG元数据标签模型有用。用法示例可能类似于：

    
    
    SetExportModuleExtraFilesHook([](const script::Module&) {
        script::ExtraFilesMap files;
        files["producer_info.json"] = "{\"user\": \"" + getenv("USER") + "\"}";
        return files;
    });
    

## 构建环境的考虑

TorchScript的编译需要访问原来的Python文件，因为它使用python的`inspect.getsource
`通话。在某些生产环境，可能需要与预编译`.pyc文件 `沿着明确部署`的.py`文件。

## 普通的扩展点

PyTorch的API通常是松散耦合的，很容易与专门的版本替换部件。常见的扩展点包括：

  * 用C语言实现运营商定制++ - 参见[了解详情](https://pytorch.org/tutorials/advanced/cpp_extension.html)教程。

  * 自定义数据读取经常可以通过调用相应的Python库直接集成。的[ `现有功能torch.utils.data`](../data.html#module-torch.utils.data "torch.utils.data")可以通过扩展被利用[ `数据集 `](../data.html#torch.utils.data.Dataset "torch.utils.data.Dataset")或[ `IterableDataset`](../data.html#torch.utils.data.IterableDataset "torch.utils.data.IterableDataset")。

[Next ![](../_static/images/chevron-right-orange.svg)](multiprocessing.html
"Multiprocessing best practices") [![](../_static/images/chevron-right-
orange.svg) Previous](faq.html "Frequently Asked Questions")

* * *

©版权所有2019年，Torch 贡献者。
