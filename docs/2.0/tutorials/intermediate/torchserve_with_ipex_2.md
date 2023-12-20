


# 从第一原理了解 PyTorch Intel CPU 性能（第 2 部分） [¶](#grokking-pytorch-intel-cpu-performance-from-first-principles-part-2"此标题的永久链接")
 

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/torchserve_with_ipex_2>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/torchserve_with_ipex_2.html>




 作者：
 [Min Jean Cho](https://github.com/min-jean-cho) 
 ,
 [Jing Xu](https://github.com/jingxu10) 
 ,\ n [马克·萨鲁菲姆](https://github.com/msaroufim)




 在 [Grokking PyTorch Intel CPU Performance From First Principles](https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html) 
 教程
 中，我们介绍了如何调整 CPU 运行时配置、如何分析它们，以及如何将它们集成到
 [TorchServe](https://github.com/pytorch/serve)
 以优化 CPU 性能。




 在本教程中，我们将演示通过 [Intel® Extension for PyTorch* Launcher](https://github.com/intel/intel-extension-for-pytorch) 使用内存分配器提升性能/blob/master/docs/tutorials/performance_tuning/launch_script.md) 
 ，并通过
 [Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch) 
 ，并将其应用到 TorchServe，结果显示 ResNet50 的吞吐量加速为 7.71 倍，BERT 的吞吐量加速为 2.20 倍。




[![https://pytorch.org/tutorials/_images/1.png](https://pytorch.org/tutorials/_images/1.png)](https://pytorch.org/tutorials/_images/1.png)


## 先决条件 [¶](#preconditions "永久链接到此标题")




 在本教程中，我们将使用
 [自顶向下微架构分析 (TMA)](https://www.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html) 
 分析并显示后端限制（内存限制、核心限制）通常是未优化或调整不足的深度学习工作负载的主要瓶颈，以及演示通过 Intel® Extension for PyTorch* 改进后端绑定的优化技术。我们将使用 
 [toplev](https://github.com/andikleen/pmu-tools/wiki/toplev-manual) 
 ，它是 
 [pmu-tools](https://github.com/pmu-tools) 的工具部分。 com/andikleen/pmu-tools) 
 构建于 [Linux perf](https://man7.org/linux/man-pages/man1/perf.1.html) 
 之上，用于 TMA。\ n



 我们还将使用
 [Intel® VTune™ Profiler’s 仪表和跟踪技术 (ITT)](https://github.com/pytorch/pytorch/issues /41001) 
 以更精细的粒度进行分析。




### 自顶向下微架构分析方法 (TMA) [¶](#top-down-microarchitecture-analysis-method-tma "永久链接到此标题")



 当调整 CPU 以获得最佳性能时，’ 了解瓶颈在哪里非常有用。大多数 CPU 内核都有片上性能监控单元 (PMU)。 PMU 是 CPU 内核中的专用逻辑块，用于对系统上发生的特定硬件事件进行计数。这些事件的示例可以是缓存未命中或分支预测错误。 PMU 用于自上而下的微架构分析 (TMA) 来识别瓶颈。 TMA 由层次结构级别组成，如下所示：




[![https://pytorch.org/tutorials/_images/26.png](https://pytorch.org/tutorials/_images/26.png)](https://pytorch.org/tutorials/_images/26.png)


 顶层，1 级，指标收集
 *退休* 
 、
 *不良推测* 
 、
 *前端绑定* 
 、
 *后端绑定* 
 。 CPU的管道在概念上可以简化并分为两部分：前端和后端。 
 *前端* 
 负责获取程序代码并将其解码为称为微操作 (uOps) 的低级硬件操作。然后，uOps 在称为分配的过程中被馈送到
 *后端* 
。一旦分配，后端负责在可用的执行单元中执行uOp。 uOp’s 执行完成称为
 *retirement* 
 。相比之下，
 *糟糕的推测* 
 是指推测获取的 uOp 在退出之前被取消，例如在错误预测的分支的情况下。这些指标中的每一个都可以在后续级别中进一步细分，以查明瓶颈。




#### 调整后端绑定 [¶](#tune-for-the-back-end-bound "永久链接到此标题")



 大多数未经调整的深度学习工作负载将受到后端限制。解决后端限制通常是解决导致退役时间超过必要时间的延迟来源。如上所示，后端绑定有两个子指标 – 核心绑定和内存绑定。




 内存限制停顿的原因与内存子系统有关。例如，最后一级高速缓存（LLC 或 L3 高速缓存）未命中导致对 DRAM 的访问。扩展深度学习模型通常需要大量计算。高计算利用率要求当执行单元需要数据来执行 uOp 时数据可用。这需要预取数据并重用缓存中的数据，而不是从主内存多次获取相同的数据，这会导致返回数据时执行单元处于饥饿状态。在本教程中，我们将展示更高效的内存分配器、运算符融合、内存布局格式优化，通过更好的缓存局部性减少内存绑定的开销。




 核心绑定停顿表示可用执行单元的使用未达到最佳状态，同时没有未完成的内存访问。例如，连续竞争融合乘加 (FMA) 或点积 (DP) 执行单元的多个通用矩阵-矩阵乘法 (GEMM) 指令可能会导致 Core Bound 停顿。包括 DP 内核在内的关键深度学习内核已通过
 [oneDNN 库](https://github.com/oneapi-src/oneDNN)
 (oneAPI 深度神经网络库) 进行了很好的优化，减少了内核的开销已绑定。




 GEMM、卷积、反卷积等操作属于计算密集型操作。虽然池化、批量标准化、ReLU 等激活函数等操作都是受内存限制的。






### Intel® VTune™ Profiler’s 检测和跟踪技术 (ITT) [¶](#intel-vtune-profiler-s-instrumentation -and-tracing-technology-itt"此标题的永久链接")



 Intel® VTune Profiler 的 ITT API 是一个有用的工具，用于注释工作负载区域，以便以更精细的注释粒度进行跟踪、分析和可视化 – OP/函数/子函数粒度。通过按 PyTorch 模型’s OP 的粒度进行注释，Intel® VTune Profiler’s ITT 可实现操作级分析。 Intel® VTune Profiler’s ITT 已集成到
 [PyTorch Autograd Profiler](https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html#autograd-profiler)
.
 1



1.该功能必须通过
 *with torch.autograd.profiler.emit_itt()* 
 显式启用。





## TorchServe 与 PyTorch 的英特尔® 扩展* [¶](#torchserve-with-intel-extension-for-pytorch "永久链接到此标题")




[Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch) 
 是一个 Python 包，用于扩展 PyTorch，并进行优化以进一步提升性能英特尔硬件。




 PyTorch* 的 Intel® 扩展已集成到 TorchServe 中，以提高开箱即用的性能。
 2 
 对于自定义处理程序脚本，我们建议添加
 *intel_extension _for_pytorch* 
 打包进来。



2. 必须通过在
 *config.properties* 
 中设置
 *ipex_enable=true* 
 显式启用该功能。



 在本节中，我们将展示后端限制通常是优化不足或调整不足的深度学习工作负载的主要瓶颈，并演示通过 Intel® Extension for PyTorch* 改进后端的优化技术绑定，它有两个子指标 - 内存绑定和核心绑定。更高效的内存分配器、运算符融合、内存布局格式优化改善了 Memory Bound。理想情况下，可以通过优化的运算符和更好的缓存局部性将内存限制改进为核心限制。卷积、矩阵乘法、点积等关键深度学习原语已通过 Intel® Extension for PyTorch* 和 oneDNN 库进行了很好的优化，从而改进了 Core Bound。




### 利用高级启动器配置：内存分配器 [¶](#leveraging-advanced-launcher-configuration-memory-allocator "永久链接到此标题")



 从性能角度来看，内存分配器起着重要作用。更有效的内存使用可以减少不必要的内存分配或销毁的开销，从而加快执行速度。对于实践中的深度学习工作负载，尤其是在 TorchServe、TCMalloc 或 JeMalloc 等大型多核系统或服务器上运行的工作负载，通常可以获得比默认 PyTorch 内存分配器 PTMalloc 更好的内存使用率。




#### TCMalloc、JeMalloc、PTMalloc [¶](#tcmalloc-jemalloc-ptmalloc "永久链接到此标题")


TCMalloc 和 JeMalloc 都使用线程本地缓存来减少线程同步的开销，并分别使用自旋锁和每线程竞技场来减少锁争用。 TCMalloc 和 JeMalloc 减少了不必要的内存分配和释放的开销。两个分配器都按大小对内存分配进行分类，以减少内存碎片的开销。




 使用启动器，用户可以通过选择三个启动器旋钮之一来轻松尝试不同的内存分配器
 *–enable_tcmalloc* 
 (TCMalloc),
 *–enable _jemalloc* 
 (JeMalloc),
 *–use_default_allocator* 
 (PTMalloc)。




###### 练习 [¶](#exercise "永久链接到此标题")



 让’s 配置 PTMalloc 与 JeMalloc。




 我们将使用启动器来指定内存分配器，并将工作负载绑定到第一个插槽的物理核心，以避免任何 NUMA 复杂性 – 仅分析内存分配器的效果。




 以下示例测量 ResNet50 的平均推理时间：






```
import torch
import torchvision.models as models
import time

model = models.resnet50(pretrained=False)
model.eval()
batch_size = 32
data = torch.rand(batch_size, 3, 224, 224)

# warm up
for _ in range(100):
    model(data)

# measure
# Intel® VTune Profiler's ITT context manager
with torch.autograd.profiler.emit_itt():
    start = time.time()
    for i in range(100):
   # Intel® VTune Profiler's ITT to annotate each step
        torch.profiler.itt.range_push('step_{}'.format(i))
        model(data)
        torch.profiler.itt.range_pop()
    end = time.time()

print('Inference took {:.2f} ms in average'.format((end-start)/100*1000))

```




 让’s 收集 1 级 TMA 指标。




[![https://pytorch.org/tutorials/_images/32.png](https://pytorch.org/tutorials/_images/32.png)](https://pytorch.org/tutorials/_images/32.png)


 1 级 TMA 显示 PTMalloc 和 JeMalloc 均受后端限制。超过一半的执行时间被后端停滞了。让’s 更深入一层。




[![https://pytorch.org/tutorials/_images/41.png](https://pytorch.org/tutorials/_images/41.png)](https://pytorch.org/tutorials/_images/41.png)


 Level-2 TMA 显示后端限制是由内存限制引起的。让’s 更深入一层。




[![https://pytorch.org/tutorials/_images/51.png](https://pytorch.org/tutorials/_images/51.png)](https://pytorch.org/tutorials/_images/51.png)


 内存限制下的大多数指标都确定从 L1 缓存到主内存的内存层次结构的哪一级是瓶颈。限制在给定级别的热点表明大部分数据是从该缓存或内存级别检索的。优化应侧重于使数据更接近核心。 3 级 TMA 显示 PTMalloc 受到 DRAM Bound 的瓶颈。另一方面，JeMalloc 受到 L1 Bound – 的瓶颈。JeMalloc 将数据移近核心，从而加快执行速度。




 让’s 查看 Intel® VTune Profiler ITT 跟踪。在示例脚本中，我们注释了推理循环的每个
 *step_x* 
。




[![https://pytorch.org/tutorials/_images/61.png](https://pytorch.org/tutorials/_images/61.png)](https://pytorch.org/tutorials/_images/61.png)


 每个步骤都在时间线图中追踪。最后一步（步骤_99）的模型推理持续时间从 304.308 毫秒减少到 261.843 毫秒。





##### 使用 TorchServe 进行练习 [¶](#exercise-with-torchserve "此标题的永久链接")



 让’s 使用 TorchServe 分析 PTMalloc 与 JeMalloc。




 我们将使用
 [TorchServe apache-bench 基准测试](https://github.com/pytorch/serve/tree/master/benchmarks#benchmarking-with-apache-bench) 
 与 ResNet50 FP32，批量大小 32 ，并发数 32，请求 8960。所有其他参数与
 [默认参数](https://github.com/pytorch/serve/tree/master/benchmarks#benchmark-parameters) 相同
 。




 与之前的练习一样，我们将使用启动器来指定内存分配器，并将工作负载绑定到第一个套接字的物理核心。为此，用户只需在 [config.properties](https://pytorch.org/serve/configuration.html#config-properties-file) 中添加几行 
 :




 PTMalloc






```
cpu_launcher_enable=true
cpu_launcher_args=--node_id 0 --use_default_allocator

```




 JeMalloc






```
cpu_launcher_enable=true
cpu_launcher_args=--node_id 0 --enable_jemalloc

```




 让’s 收集 1 级 TMA 指标。




[![https://pytorch.org/tutorials/_images/71.png](https://pytorch.org/tutorials/_images/71.png)](https://pytorch.org/tutorials/_images/71.png)


 让’s 更深入一层。




[![https://pytorch.org/tutorials/_images/81.png](https://pytorch.org/tutorials/_images/81.png)](https://pytorch.org/tutorials/_images/81.png)


 让’s使用Intel® VTune Profiler ITT来注释
 [TorchServe推理范围](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py# L188) 
 以推理级粒度进行分析。 As
 [TorchServe Architecture](https://github.com/pytorch/serve/blob/master/docs/internals.md#torchserve-architecture) 
 由几个子组件组成，包括用于处理请求的 Java 前端/response，以及用于在模型上运行实际推理的 Python 后端，使用 Intel® VTune Profiler ITT 来限制推理级别跟踪数据的收集会很有帮助。




[![https://pytorch.org/tutorials/_images/9.png](https://pytorch.org/tutorials/_images/9.png)](https://pytorch.org/tutorials/_images/9.png)


 每个推理调用都在时间线图中进行跟踪。最后一次模型推理的持续时间从 561.688 毫秒减少到 251.287 毫秒 - 加速了 2.2 倍。




[![https://pytorch.org/tutorials/_images/101.png](https://pytorch.org/tutorials/_images/101.png)](https://pytorch.org/tutorials/_images/101.png)


 可以扩展时间线图以查看操作级分析结果。 
 *aten::conv2d* 的持续时间从 16.401 毫秒减少到 6.392 毫秒 - 加速了 2.6 倍。




 在本节中，我们演示了 JeMalloc 可以提供比默认 PyTorch 内存分配器 PTMalloc 更好的性能，并通过高效的线程本地缓存改进后端绑定。







### PyTorch 的英特尔® 扩展* [¶](#id1"此标题的永久链接")



 三大
 [Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch)
 优化技术，Operator、Graph、Runtime、如图所示：









| 
 Intel® PyTorch 扩展* 优化技术
 |
| --- |
| 
 运算符
 | 
 图表
 | 
 运行时
 |
| * 矢量化和多线程
* 低精度 BF16/INT8 计算
* 数据布局优化以获得更好的缓存局部性
 | * 不断折叠以减少计算
* 运算融合以实现更好的缓存局部性
 | * 线程关联
* 内存缓冲池
* GPU 运行时
* 启动器
 |



#### 运算符优化 [¶](#operator-optimization "永久链接到此标题")



 优化的算子和内核通过 PyTorch 调度机制注册。这些运算符和内核是通过英特尔硬件的本机矢量化功能和矩阵计算功能进行加速的。在执行过程中，Intel® Extension for PyTorch* 会拦截 ATen 运算符的调用，并用这些优化的运算符替换原始运算符。卷积、线性等热门算子已在 Intel® Extension for PyTorch* 中进行了优化。




###### 练习 [¶](#id2 "此标题的永久链接")



 让’s 使用 Intel® Extension for PyTorch* 配置优化的运算符。我们将比较代码更改和未更改的情况。




 与前面的练习一样，我们将工作负载绑定到第一个套接字的物理核心。






```
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(16, 33, 3, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

model = Model()
model.eval()
data = torch.rand(20, 16, 50, 100)

#################### code changes ####################
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)
######################################################

print(model)

```




 该模型由两个操作—Conv2d 和 ReLU 组成。通过打印模型对象，我们得到以下输出。




[![https://pytorch.org/tutorials/_images/11.png](https://pytorch.org/tutorials/_images/11.png)](https://pytorch.org/tutorials/_images/11.png)


 让’s 收集 1 级 TMA 指标。




[![https://pytorch.org/tutorials/_images/121.png](https://pytorch.org/tutorials/_images/121.png)](https://pytorch.org/tutorials/_images/121.png)


 请注意，后端边界从 68.9 减少到 38.5 – 1.8 倍加速。




 此外，让’s 使用 PyTorch Profiler 进行分析。




[![https://pytorch.org/tutorials/_images/131.png](https://pytorch.org/tutorials/_images/131.png)](https://pytorch.org/tutorials/_images/131.png)


 请注意，CPU 时间从 851 us 减少到 310 us – 2.7 倍加速。






#### 图形优化 [¶](#graph-optimization "此标题的永久链接")



 强烈建议用户利用 Intel® Extension for PyTorch* 和
 [TorchScript](https://pytorch.org/docs/stable/jit.html) 
 进一步了解图形优化。为了进一步优化 TorchScript 的性能，Intel® Extension for PyTorch* 支持常用 FP32/BF16 运算符模式的 oneDNN 融合，例如 Conv2D+ReLU、Linear+ReLU 等，以减少运算符/内核调用开销，并且为了更好的缓存局部性。一些运算符融合允许维护临时计算、数据类型转换、数据布局，以获得更好的缓存局部性。与 INT8 一样，Intel® Extension for PyTorch* 具有内置量化配方，可为流行的深度学习工作负载（包括 CNN、NLP 和推荐模型）提供良好的统计准确性。然后使用 oneDNN 融合支持优化量化模型。




###### 练习 [¶](#id3 "此标题的永久链接")



 让’s 使用 TorchScript 分析 FP32 图形优化。




 与前面的练习一样，我们将工作负载绑定到第一个套接字的物理核心。






```
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(16, 33, 3, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

model = Model()
model.eval()
data = torch.rand(20, 16, 50, 100)

#################### code changes ####################
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)
######################################################

# torchscript
with torch.no_grad():
    model = torch.jit.trace(model, data)
    model = torch.jit.freeze(model)

```




 让’s 收集 1 级 TMA 指标。




[![https://pytorch.org/tutorials/_images/141.png](https://pytorch.org/tutorials/_images/141.png)](https://pytorch.org/tutorials/_images/141.png)


 请注意，后端边界从 67.1 减少到 37.5 – 1.8 倍加速。




 此外，让’s 使用 PyTorch Profiler 进行分析。




[![https://pytorch.org/tutorials/_images/151.png](https://pytorch.org/tutorials/_images/151.png)](https://pytorch.org/tutorials/_images/151.png)


 请注意，使用 Intel® Extension for PyTorch* Conv + ReLU 运算符进行融合，CPU 时间从 803 us 减少到 248 us – 3.2 倍加速。 oneDNN eltwise post-op 可以将基元与元素基元融合。这是最流行的融合类型之一：带有前面的卷积或内积的 eltwise（通常是激活函数，例如 ReLU）。查看下一节中显示的 oneDNN 详细日志。






#### 通道最后内存格式 [¶](#channels-last-memory-format "永久链接到此标题")



 在模型上调用
 *ipex.optimize* 
 时，Intel® Extension for PyTorch* 会自动将模型转换为优化的内存格式，最后是通道。 Channels Last是一种对Intel架构更加友好的内存格式。与 PyTorch 默认通道优先 NCHW（批量、通道、高度、宽度）内存格式相比，通道最后 NHWC（批量、高度、宽度、通道）内存格式通常可以通过更好的缓存局部性来加速卷积神经网络。




 需要注意的一件事是转换内存格式的成本很高。因此，最好在部署之前转换一次内存格式，并在部署期间保持最小的内存格式转换。当数据通过 model’s 层传播时，通道最后的内存格式将通过连续通道最后支持的层（例如，Conv2d -> ReLU -> Conv2d）保留，并且仅在通道最后不支持的层之间进行转换。请参阅
 [内存格式传播](https://www.intel.com/content/www/us/en/develop/documentation/onednn-developer-guide-and-reference/top/programming-model/memory-format -propagation.html) 
 了解更多详细信息。




###### 练习 [¶](#id4 "此标题的永久链接")



 让’s 演示通道上次优化。






```
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(16, 33, 3, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

model = Model()
model.eval()
data = torch.rand(20, 16, 50, 100)

import intel_extension_for_pytorch as ipex
############################### code changes ###############################
ipex.disable_auto_channels_last() # omit this line for channels_last (default)
############################################################################
model = ipex.optimize(model)

with torch.no_grad():
    model = torch.jit.trace(model, data)
    model = torch.jit.freeze(model)

```




 我们将使用
 [oneDNN 详细模式](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html) 
 ，一个帮助收集 oneDNN 图级别信息（例如算子融合）的工具，执行 oneDNN 原语所花费的内核执行时间。有关更多信息，请参阅
 [oneDNN 文档](https://oneapi-src.github.io/oneDNN/index.html)
 。




[![https://pytorch.org/tutorials/_images/161.png](https://pytorch.org/tutorials/_images/161.png)](https://pytorch.org/tutorials/_images/161.png)


[![https://pytorch.org/tutorials/_images/171.png](https://pytorch.org/tutorials/_images/171.png)](https://pytorch.org/tutorials/_images/171.png)


 以上是来自频道的 oneDNN 详细信息。我们可以验证权重和数据是否进行了重新排序，然后进行计算，最后将输出重新排序。




[![https://pytorch.org/tutorials/_images/181.png](https://pytorch.org/tutorials/_images/181.png)](https://pytorch.org/tutorials/_images/181.png)


 以上是最后一个频道的 oneDNN 详细信息。我们可以验证通道最后的内存格式以避免不必要的重新排序。







### 使用适用于 PyTorch 的 Intel® 扩展实现性能提升* [¶](#performance-boost-with-intel-extension-for-pytorch "永久链接到此标题" ）



 下面总结了 TorchServe 与 Intel® Extension for PyTorch* for ResNet50 和 BERT-base-uncased 的性能提升。




[![https://pytorch.org/tutorials/_images/191.png](https://pytorch.org/tutorials/_images/191.png)](https://pytorch.org/tutorials/_images/191.png)



### 使用 TorchServe 进行练习 [¶](#id5 "此标题的永久链接")



 让’s 使用 TorchServe 配置 Intel® Extension for PyTorch* 优化。




 我们将使用
 [TorchServe apache-bench 基准测试](https://github.com/pytorch/serve/tree/master/benchmarks#benchmarking-with-apache-bench) 
 与 ResNet50 FP32 TorchScript，批量大小32，并发数 32，请求 8960。所有其他参数与
 [默认参数](https://github.com/pytorch/serve/tree/master/benchmarks#benchmark-parameters) 相同
 。




 与上一个练习一样，我们将使用启动器将工作负载绑定到第一个套接字的物理核心。为此，用户只需在 [config.properties](https://github.com/pytorch/serve/tree/master/benchmarks#benchmark-parameters) 中添加几行 
 :






```
cpu_launcher_enable=true
cpu_launcher_args=--node_id 0

```




 让’s 收集 1 级 TMA 指标。




[![https://pytorch.org/tutorials/_images/20.png](https://pytorch.org/tutorials/_images/20.png)](https://pytorch.org/tutorials/_images/20.png)


 1 级 TMA 显示两者均受后端限制。如前所述，大多数未经调整的深度学习工作负载将受到后端限制。请注意，后端边界从 70.0 减少到 54.1。让’s 更深入一层。




[![https://pytorch.org/tutorials/_images/211.png](https://pytorch.org/tutorials/_images/211.png)](https://pytorch.org/tutorials/_images/211.png)


 如前所述，后端绑定有两个子指标 – 内存绑定和核心绑定。内存限制表示工作负载未优化或未充分利用，理想情况下，可以通过优化 OP 和改进缓存局部性将内存限制操作改进为核心限制。 2 级 TMA 显示后端限制从内存限制改进为核心限制。让’s 更深入一层。




[![https://pytorch.org/tutorials/_images/221.png](https://pytorch.org/tutorials/_images/221.png)](https://pytorch.org/tutorials/_images/221.png)


 在 TorchServe 等模型服务框架上扩展深度学习模型以进行生产需要高计算利用率。这要求当执行单元需要数据来执行uOps时，可以通过预取和重用缓存中的数据来获得数据。 3 级 TMA 显示后端内存限制从 DRAM 限制改进为核心限制。




 与之前使用 TorchServe 的练习一样，让’s 使用 Intel® VTune Profiler ITT 来注释
 [TorchServe 推理范围](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L188) 
 以推理级粒度进行分析。




[![https://pytorch.org/tutorials/_images/231.png](https://pytorch.org/tutorials/_images/231.png)](https://pytorch.org/tutorials/_images/231.png)


 每个推理调用都在时间线图中进行跟踪。最后一次推理调用的持续时间从 215.731 毫秒减少到 95.634 毫秒 - 加速了 2.3 倍。




[![https://pytorch.org/tutorials/_images/241.png](https://pytorch.org/tutorials/_images/241.png)](https://pytorch.org/tutorials/_images/241.png)


 可以扩展时间线图以查看操作级分析结果。请注意，Conv + ReLU 已融合，持续时间从 6.393 ms + 1.731 ms 减少到 3.408 ms - 加速提高了 2.4 倍。





## 结论 [¶](#conclusion "此标题的永久链接")




 在本教程中，我们使用自上而下的微架构分析 (TMA) 和 Intel® VTune™ Profiler’s 仪表和跟踪技术 (ITT) 来演示



* 通常，未优化或调整不足的深度学习工作负载的主要瓶颈是后端限制，它有两个子指标：内存限制和核心限制。
* 英特尔提供的更高效的内存分配器、运算符融合、内存布局格式优化® PyTorch 扩展 * 改善内存限制。
* 关键深度学习原语，如卷积、矩阵乘法、点积等已由 Intel® Extension for PyTorch* 进行了很好的优化和 oneDNN 库，改进了 Core Bound。
* Intel® Extension for PyTorch* 已通过易于使用的 API 集成到 TorchServe 中。
* TorchServe 具有 Intel® Extension for PyTorch * 显示 ResNet50 的吞吐量加速为 7.71 倍，BERT 的吞吐量加速为 2.20 倍。





## 相关阅读 [¶](#lated-readings "此标题的固定链接")




[自上而下的微架构分析方法](https://www.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html)




[自上而下的性能分析方法](https://easyperf.net/blog/2019/02/09/Top-Down-performance-analysis-methodology)




[使用 PyTorch 的 Intel® 扩展加速 PyTorch*](https://medium.com/pytorch/acceleating-pytorch-with-intel-extension-for-pytorch-3aef51ea3722)





## 确认 [¶](#acknowledgement "永久链接到此标题")




 我们要感谢 Ashok Emani（英特尔）和 Jiong Kong（英特尔）在本教程的许多步骤中提供的巨大指导和支持以及全面的反馈和审查。我们还要感谢 Hamid Shojanazeri (Meta) 和 Li Ning (AWS) 在代码审查和教程中提供的有用反馈。









