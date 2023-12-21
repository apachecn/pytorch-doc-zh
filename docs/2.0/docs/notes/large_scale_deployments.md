# 大规模部署的功能 [¶](#features-for-large-scale-deployments "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/large_scale_deployments>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/large_scale_deployments.html>



* [舰队范围的操作员分析](#fleet-wide-operator-profiling)
* [API 使用日志记录](#api-usage-logging)
* [将元数据附加到保存的 TorchScript 模型](#attaching-metadata-to-saved-torchscript-models)
* [构建环境注意事项](#build-environment-considerations)
* [通用扩展点](#common-extension-points)


 本说明讨论了在较大系统中运行 PyTorch 或在较大组织中使用 PyTorch 操作多个系统时可能有用的几个扩展点和技巧。


 它不涵盖将模型部署到生产的主题。检查 [`torch.jit`](../jit.html#module-torch.jit "torch.jit") 或相应的教程之一。


 本说明假设您从组织中的源代码构建 PyTorch，或者能够静态链接使用 PyTorch 时加载的其他代码。因此，许多钩子都作为 C++ API 公开，可以在集中位置触发一次，例如在静态初始化代码中。


## [舰队范围的操作员分析](#id1) [¶](#fleet-wide-operator-profiling "永久链接到此标题")


 PyTorch 附带了“torch.autograd.profiler”，能够根据需要测量各个操作员所花费的时间。人们可以使用相同的机制对运行 PyTorch 的任何进程进行“始终开启”测量。它对于收集有关在给定进程中或整个机器上运行的 PyTorch 工作负载的信息可能很有用。


 任何运算符调用的新回调都可以使用 `torch::addGlobalCallback` 添加。钩子将使用描述调用上下文(例如 name )的 `torch::RecordFunction` 结构来调用。如果启用，`RecordFunction::inputs()` 包含表示为 `torch::IValue` 变体类型的函数参数。请注意，输入日志记录相对昂贵，因此必须显式启用。


 运算符回调还可以访问 `c10::ThreadLocalDebugInfo::get()` 接口，该接口返回指向保存调试信息的结构的指针。可以使用 `at::DebugInfoGuard` 对象提前设置此调试信息。调试信息通过前向(包括异步“fork”任务)和后向传递进行传播，并且可用于将有关执行环境的一些额外信息(例如模型 ID)从应用程序的较高层传递到操作员回调。


 调用回调会增加一些开销，因此通常随机采样运算符调用很有用。这可以在每个回调的基础上启用，并将可选的采样率传递给 `torch::addGlobalCallback` 。


 请注意，“addGlobalCallback”不是线程安全的，只能在 noPyTorch 运算符运行时调用。通常，在初始化期间调用它们一次是个好主意。


 这是一个例子：


```
// Called somewhere in the program beginning
void init() {
 // Sample one in a hundred operator runs randomly
 addGlobalCallback(
 RecordFunctionCallback(
 &onFunctionEnter,
 &onFunctionExit)
 .needsInputs(true)
 .samplingProb(0.01)
 );
 // Note, to enable observers in the model calling thread,
 // call enableRecordFunction() in the thread before running a model
}

void onFunctionEnter(const RecordFunction& fn) {
 std::cerr << "Before function " << fn.name()
 << " with " << fn.inputs().size() << " inputs" << std::endl;
}

void onFunctionExit(const RecordFunction& fn) {
 std::cerr << "After function " << fn.name();
}

```


## [API 使用日志记录](#id2) [¶](#api-usage-logging "此标题的永久链接")


 当在更广泛的生态系统中运行时，例如在托管作业调度程序中，跟踪哪些二进制文件调用特定的 PyTorch API 通常很有用。在几个重要的 API 点注入了简单的检测，可以触发给定的回调。因为 PyTorch 通常是在一次性 python 脚本中调用的，所以对于每个 API 的给定进程，回调仅触发一次。


`c10::SetAPIUsageHandler` 可用于注册 API 使用检测处理程序。传递的参数将是标识使用点的“api key”，例如用于 PyTorch 扩展导入的“python.import”或如果触发了 TorchScript 编译则为“torch.script.compile”。


```
SetAPIUsageLogger([](const std::string& event_name) {
 std::cerr << "API was used: " << event_name << std::endl;
});

```


 开发者请注意：新的 API 触发点可以通过 C++ 中的 `C10_LOG_API_USAGE_ONCE("my_api")` 或 `torch._C._log_api_usage_once( Python 中的“my.api”)`。


## [将元数据附加到保存的 TorchScript 模型](#id3) [¶](#attaching-metadata-to-saved-torchscript-models "永久链接到此标题")


 TorchScript 模块可以保存为归档文件，该文件将序列化参数和模块代码捆绑为 TorchScript(请参阅 [`torch.jit.save()`](../generated/torch.jit.save.html#torch.jit.save " torch.jit.save") )。将附加信息与模型捆绑在一起通常很方便，例如模型生成器或辅助工件的描述。


 它可以通过将 `_extra_files` 参数传递给 [`torch.jit.save()`](../generated/torch.jit.save.html#torch.jit.save "torch.jit. save") 和 `torch::jit::load` 在保存过程中存储和检索任意二进制 blob。由于 TorchScript 文件是常规 ZIP 存档，因此额外信息将作为常规文件存储在存档的“extra/”目录中。


 还有一个全局挂钩，允许将额外的文件附加到当前进程中生成的任何 TorchScriptarchive。使用生产者元数据标记模型可能很有用，类似于数码相机生成的 JPEG 元数据。用法示例可能如下所示：


```
SetExportModuleExtraFilesHook([](const Module&) {
 ExtraFilesMap files;
 files["producer_info.json"] = "{"user": "" + getenv("USER") + ""}";
 return files;
});

```


## [构建环境注意事项](#id4) [¶](#build-environment-considerations "永久链接到此标题")


 TorchScript 的编译需要访问原始 python 文件，因为它使用 python 的“inspect.getsource”调用。在某些生产环境中，可能需要显式部署“.py”文件以及预编译的“.pyc”。


## [常用扩展点](#id5) [¶](#common-extension-points "永久链接到此标题")


 PyTorch API 通常是松散耦合的，并且很容易用专用版本替换组件。常见的扩展点包括：



* 用 C++ 实现的自定义运算符 
- 请参阅[教程了解更多详细信息](https://pytorch.org/tutorials/advanced/cpp_extension.html)。
* 自定义数据读取通常可以通过调用相应的 python 库直接集成。可以通过扩展 [`Dataset`](../data.html#torch.utils.data.Dataset "torch.utils.data.Dataset") 或 [`IterableDataset`](../data.html#torch.utils.data.IterableDataset "torch.utils.data.IterableDataset" )。