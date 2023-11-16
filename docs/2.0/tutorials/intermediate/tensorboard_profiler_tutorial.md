# 带有 TensorBoard 的 PyTorch Profiler [¶](#pytorch-profiler-with-tensorboard "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/tensorboard_profiler_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html>




 本教程演示如何将 TensorBoard 插件与 PyTorch Profiler 结合使用
来检测模型的性能瓶颈。





## 简介 [¶](#introduction "此标题的永久链接")




 PyTorch 1.8 包含更新的探查器 API，能够
记录 CPU 端操作以及在 GPU 端启动的 CUDA 内核。
探查器可以在 TensorBoard 插件中可视化此信息并提供性能瓶颈分析。




 在本教程中，我们将使用一个简单的 Resnet 模型来演示如何
使用 TensorBoard 插件来分析模型性能。





## 设置 [¶](#setup "此标题的永久链接")




 要安装
 `torch`
 和
 `torchvision`
 请使用以下命令:






```
pip install torch torchvision

```





## 步骤 [¶](#steps "此标题的永久链接")



1. 准备数据和模型
2.使用探查器记录执行事件
3.运行探查器
4.使用 TensorBoard 查看结果并分析模型性能
5.在探查器的帮助下提高性能
6.使用其他高级功能分析性能



### 1. 准备数据和模型 [¶](#prepare-the-data-and-model "永久链接到此标题")



 首先，导入所有必需的库：






```
import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

```




 然后准备输入数据。在本教程中，我们使用 CIFAR10 数据集。
将其转换为所需格式并使用
 `DataLoader`
 加载每个批次。






```
transform = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

```




 接下来，创建 Resnet 模型、损失函数和优化器对象。
要在 GPU 上运行，请将模型和损失移动到 GPU 设备。






```
device = torch.device("cuda:0")
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()

```




 定义每批输入数据的训练步骤。






```
def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```





### 2. 使用探查器记录执行事件 [¶](#use-profiler-to-record-execution-events "Permalink to this header")



 探查器通过上下文管理器启用并接受多个参数，
其中一些最有用的参数是：



* `schedule`
 - 可调用，将步骤 (int) 作为单个参数
并返回要在每个步骤执行的探查器操作。




 在此示例中，
 `wait=1,
 

 Warmup=1,
 

 active=3,
 

 Repeat=1`
 ，
分析器将跳过第一步/迭代，
开始第二步预热，
记录以下三个迭代，
之后跟踪将变得可用，并调用 on_trace_ready（设置时）。
总共，该循环重复一次。每个周期在 TensorBoard 插件中称为 “span”。




 在
 `wait`
 步骤期间，探查器被禁用。
在
 `warmup`
 步骤期间，探查器开始跟踪，但结果被丢弃。
这是为了减少分析开销。
开销分析开始时的值很高，很容易给分析结果带来偏差。
在
 `active`
 步骤期间，分析器工作并记录事件。
* `on_trace_ready`
 -在每个周期结束时调用的可调用函数；
在此示例中，我们使用
 `torch.profiler.tensorboard_trace_handler`
 为 TensorBoard 生成结果文件。
分析后，将保存结果文件进入
 `./log/resnet18`
目录。
将此目录指定为
 `logdir`
参数以分析TensorBoard中的配置文件。
* `record_shapes`
 - 是否记录运算符输入的形状。
* `profile_memory`
 - 跟踪张量内存分配/释放。注意，对于 1.10 之前版本的旧版本 pytorch，如果您的分析时间过长，请禁用它或升级到新版本。
* `with_stack`
 - 记录源信息（文件和行号） 
如果 TensorBoard 在 VS Code 中启动 (
 [参考](https://code.visualstudio.com/docs/datascience/pytorch-support#_tensorboard-integration) 
 )，
单击堆栈框架将导航到特定的代码行。





```
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        if step >= 1 + 1 + 3:
            break
        train(batch_data)

```




 或者，也支持以下非上下文管理器启动/停止。






```
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)
prof.start()
for step, batch_data in enumerate(train_loader):
    prof.step()
    if step >= 1 + 1 + 3:
        break
    train(batch_data)
prof.stop()

```





### 3. 运行探查器 [¶](#run-the-profiler "永久链接到此标题")



 运行上面的代码。分析结果将保存在
 `./log/resnet18`
 目录下。





### 4. 使用 TensorBoard 查看结果并分析模型性能 [¶](#use-tensorboard-to-view-results-and-analyze-model-performance "Permalink to这个标题”）




 注意




 TensorBoard 插件支持已被弃用，因此其中一些功能可能无法
像以前一样工作。请看一下替代品，
 [HTA](https://github.com/pytorch/kineto/tree/main#holistic-trace-analysis)
 。





 安装 PyTorch Profiler TensorBoard 插件。






```
pip install torch_tb_profiler

```




 启动 TensorBoard。






```
tensorboard --logdir=./log

```




 在 Google Chrome 浏览器或 Microsoft Edge 浏览器中打开 TensorBoard 配置文件 URL（
 **不支持 Safari** 
 ）。






```
http://localhost:6006/#pytorch_profiler

```




 您可以看到如下所示的 Profiler 插件页面。



* 概述


[![https://pytorch.org/tutorials/_static/img/profiler_overview1.png](https://pytorch.org/tutorials/_static/img/profiler_overview1.png)](https://pytorch.org/tutorials/_static/img/profiler_overview1.png)

 概述显示高级模型性能摘要。




 “GPU 摘要” 面板显示 GPU 配置、GPU 使用情况和张量核心使用情况。
在此示例中，GPU 利用率较低。
这些指标的详细信息为
 [此处](https://github.com/pytorch/kineto/blob/main/tb_plugin/docs/gpu_utilization.md) 
 。




 “Step Time Breakdown” 显示不同执行类别中每个步骤所花费的时间分布。
在此示例中，您可以看到
 `DataLoader`
 开销非常大.




 底部“性能建议” 使用分析数据
自动突出显示可能的瓶颈，
并为您提供可操作的优化建议。




 您可以在左侧 “Views” 下拉列表中更改视图页面。



![](https://pytorch.org/tutorials/_static/img/profiler_views_list.png)
* 操作员视图



 运算符视图显示在主机或设备上执行的每个 PyTorch 运算符
的性能。



[![https://pytorch.org/tutorials/_static/img/profiler_operator_view.png](https://pytorch.org/tutorials/_static/img/profiler_operator_view.png)](https://pytorch.org/tutorials/_static/img/profiler_operator_view.png)

 “Self ” 持续时间不包括其子运算符’ 时间。
“Total” 持续时间包括其子运算符’ 时间。



* 查看调用堆栈



 单击某个运算符的
 `查看
 

 调用堆栈`
，将显示同名但不同调用堆栈的运算符。
然后单击
 `查看
 

 调用堆栈`
 在此子表中，将显示调用堆栈帧。



[![https://pytorch.org/tutorials/_static/img/profiler_callstack.png](https://pytorch.org/tutorials/_static/img/profiler_callstack.png)](https://pytorch.org/tutorials/_static/img/profiler_callstack.png)

 如果 TensorBoard 在 VS 内启动代码
(
 [启动指南](https://devblogs.microsoft.com/python/python-in-visual-studio-code-february-2021-release/#tensorboard-integration)
 ),
单击调用堆栈帧将导航到特定的代码行。



[![https://pytorch.org/tutorials/_static/img/profiler_vscode.png](https://pytorch.org/tutorials/_static/img/profiler_vscode.png)](https://pytorch.org/tutorials/_static/img/profiler_vscode.png)
* 内核视图



 GPU 内核视图显示所有内核’ 在 GPU 上花费的时间。



[![https://pytorch.org/tutorials/_static/img/profiler_kernel_view.png](https://pytorch.org/tutorials/_static/img/profiler_kernel_view.png)](https://pytorch.org/tutorials/_static/img/profiler_kernel_view.png)

 使用的张量核心:
是否内核使用 Tensor Core。




 每个 SM 的平均块数：
每个 SM 的块数 = 该内核的块数 /该 GPU 的 SM 编号。
如果该数字小于 1，则表明 GPU 多处理器未充分利用。
“Mean每个 SM” 的块数是此内核名称的所有运行的加权平均值，使用每个运行’s 的持续时间作为权重。




 平均估计值已达到入住率：
预计。达到的占用率在此列中定义’s 工具提示。
对于大多数情况（例如内存带宽有限内核），越高越好。
“Mean Est。达到的占用率” 是此内核名称的所有运行的加权平均值，
使用每个运行’s 的持续时间作为权重。



* 轨迹视图



 跟踪视图显示分析运算符和 GPU 内核的时间线。
您可以选择它来查看详细信息，如下所示。



[![https://pytorch.org/tutorials/_static/img/profiler_trace_view1.png](https://pytorch.org/tutorials/_static/img/profiler_trace_view1.png)](https://pytorch.org/tutorials/_static/img/profiler_trace_view1.png)

 您可以移动图形并缩放在右侧工具栏的帮助下输入/输出。
键盘也可用于在时间轴内缩放和移动。
\xe2\x80\x98w\xe2\x80\x99 和 \xe2\x80\x98s\xe2 \x80\x99 键以鼠标为中心放大，
\xe2\x80\x98a\xe2\x80\x99 和 \xe2\x80\x98d\xe2\x80\x99 键左右移动时间线。
您可以可以多次敲击这些键，直到看到可读的表示。




 如果向后运算符’s “Incoming Flow” 字段的值为“forward 对应于向后”，
您可以单击文本以获取其启动前向运算符。



[![https://pytorch.org/tutorials/_static/img/profiler_trace_view_fwd_bwd.png](https://pytorch.org/tutorials/_static/img/profiler_trace_view_fwd_bwd.png)](https://pytorch.org/tutorials/_static/img/profiler_trace_view_fwd_bwd.png)

 在此示例中，我们可以看到前缀为
 `enumerate(DataLoader)`的事件
需要花费大量时间。
并且在这期间，GPU大部分时间处于空闲状态。
因为该函数正在主机端加载数据并转换数据，
在此期间GPU 资源被浪费。





### 5. 在分析器的帮助下提高性能 [¶](#improve-performance-with-the-help-of-profiler "Permalink to this header")



 在 “Overview” 页面底部，“PerformanceRecommendation” 中的建议提示瓶颈是
 `DataLoader`
 。
 PyTorch
 `DataLoader`
 默认使用单进程。
用户可以通过设置参数启用多进程数据加载
 `num_workers`
 。
 [此处](https://pytorch. org/docs/stable/data.html#single-and-multi-process-data-loading) 
 是更多详细信息。




 在此示例中，我们遵循 “ 性能建议” 并设置
 `num_workers`
 如下，
传递不同的名称，例如
 `./log /resnet18_4workers`
 到
 `tensorboard_trace_handler`
 ，然后再次运行。






```
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

```




 然后让’s 在左侧 “Runs” 下拉列表中选择最近分析的运行。



[![https://pytorch.org/tutorials/_static/img/profiler_overview2.png](https://pytorch.org/tutorials/_static/img/profiler_overview2.png)](https://pytorch.org/tutorials/_static/img/profiler_overview2.png)

 从上面的视图中，我们可以发现与之前运行’s 132ms相比，步骤时间减少到约76ms，
主要是`DataLoader`
的时间减少。



[![https://pytorch.org/tutorials/_static/img/profiler_trace_view2.png](https://pytorch.org/tutorials/_static/img/profiler_trace_view2.png)](https://pytorch.org/tutorials/_static/img/profiler_trace_view2.png)

 从上面的视图中，我们可以看到
 `enumerate(DataLoader)`
 的运行时间减少了，
GPU 利用率增加了。





### 6. 使用其他高级功能分析性能 [¶](#analyze-performance-with-other-advanced-features "永久链接到此标题")


* 内存视图



 要分析内存，
 必须在 `torch.profiler.profile` 的参数中将 `profile_memory`
 设置为
 `True`
 。




 您可以使用 Azure 上的现有示例进行尝试






```
pip install azure-storage-blob
tensorboard --logdir=https://torchtbprofiler.blob.core.windows.net/torchtbprofiler/demo/memory_demo_1_10

```




 分析器记录分析期间的所有内存分配/释放事件和 allocator’s 内部状态。
内存视图由三个组件组成，如下所示。



[![https://pytorch.org/tutorials/_static/img/profiler_memory_view.png](https://pytorch.org/tutorials/_static/img/profiler_memory_view.png)](https://pytorch.org/tutorials/_static/img/profiler_memory_view.png)

 组件是内存曲线图，从上到下分别是内存事件表和内存统计表。




 内存类型可以在 \xe2\x80\x9cDevice\xe2\x80\x9d 选择框中选择。
例如，\xe2\x80\x9cGPU0\xe2\x80\x9d 表示下表仅显示每个算子\xe2 \x80\x99s GPU 0 上的内存使用情况，不包括 CPU 或其他 GPU。




 内存曲线显示内存消耗趋势。 “Allocation” 曲线显示实际使用的总内存，例如张量。在 PyTorch 中，CUDA 分配器和其他一些分配器采用了缓存机制。 
“Reserved” 曲线显示分配器保留的总内存。您可以在图表上单击鼠标左键并拖动
以选择所需范围内的事件:



[![https://pytorch.org/tutorials/_static/img/profiler_memory_curve_selecting.png](https://pytorch.org/tutorials/_static/img/profiler_memory_curve_selecting.png)](https://pytorch.org/tutorials/_static/img/profiler_memory_curve_selecting.png)

 选择后，三个组件将在有限的时间范围内进行更新，以便您可以获得
更多相关信息。通过重复此过程，您可以放大非常细粒度的细节。右键单击图表
会将图表重置为初始状态。



[![https://pytorch.org/tutorials/_static/img/profiler_memory_curve_single.png](https://pytorch.org/tutorials/_static/img/profiler_memory_curve_single.png)](https://pytorch.org/tutorials/_static/img/profiler_memory_curve_single.png)

 在内存事件表中，分配和释放事件配对成一个条目。 “operator” 列显示导致分配的直接 ATen 运算符。请注意，在 PyTorch 中，ATen 运算符通常使用 `aten::empty` 来分配内存。例如，
 `aten::ones`
 实现为
 `aten::empty`
 后跟
 `aten::fill_`
 。仅仅将操作符名称显示为
 `aten::empty`
 没有什么帮助。在这种特殊情况下，它将显示为
 `aten::ones
 

 (aten::empty)`
。 “分配时间”、“释放时间” 和 “持续时间”
列’ 数据如果事件发生在时间范围之外，则可能会丢失。




 在内存统计表中，“SizeIncrease” 列汇总了所有分配大小并减去所有内存
释放大小，即该运算符之后的内存使用净增加量。 “Self SizeIncrease” 列与“SizeIncrease” 类似，但它不计算子运算符’ 分配。关于 ATen 运算符’
实现细节，某些运算符可能会调用其他运算符，因此内存分配可能发生在
调用堆栈的任何级别。也就是说，“Self Size Improve” 仅计算当前调用堆栈级别的内存使用量增加。
最后，“Allocation Size” 列求和所有分配都不考虑内存释放。



* 分布式视图



 该插件现在支持以 NCCL/GLOO 作为后端分析 DDP 的分布式视图。




 您可以使用 Azure 上的现有示例进行尝试：






```
pip install azure-storage-blob
tensorboard --logdir=https://torchtbprofiler.blob.core.windows.net/torchtbprofiler/demo/distributed_bert

```



[![https://pytorch.org/tutorials/_static/img/profiler_distributed_view.png](https://pytorch.org/tutorials/_static/img/profiler_distributed_view.png)](https://pytorch.org/tutorials/_static/img/profiler_distributed_view.png)

 \xe2\x80 \x9c计算/通信概述\xe2\x80\x9d显示计算/通信比率及其重叠程度。
从这个视图中，用户可以找出worker之间的负载平衡问题。
例如，如果一个worker的计算+重叠时间为比其他人大得多，
可能存在负载平衡问题，或者该工作人员可能是掉队的。




“同步/通信概述”显示通信效率。
“数据传输时间”是实际数据交换的时间。
 “Synchronizing Time” 是等待并与其他工作人员同步的时间。




 如果一个worker’s “同步时间”比其他worker’短很多，
这个worker可能是一个掉队者，可能有更多计算工作量高于其他worker’。




 “Communication Operations Stats” 汇总了每个工作线程中所有通信操作的详细统计信息。





## 了解更多 [¶](#learn-more "此标题的永久链接")




 请查看以下文档以继续学习，
并随时提出问题
 [此处](https://github.com/pytorch/kineto/issues) 
.



* [PyTorch TensorBoard Profiler Github](https://github.com/pytorch/kineto/tree/master/tb_plugin)
* [torch.profiler API](https://pytorch.org/docs/master/profiler. html)
* [HTA](https://github.com/pytorch/kineto/tree/main#holistic-trace-analysis)



**脚本的总运行时间:** 
 ( 0 分 0.000 秒)
