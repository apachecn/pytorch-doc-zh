


# 从第一原理了解 PyTorch Intel CPU 性能 [¶](#grokking-pytorch-intel-cpu-performance-from-first-principles "固定链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/torchserve_with_ipex>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html>




 关于使用 [Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch) 优化的 TorchServe 推理框架的案例研究 
.\ n



 作者：Min Jean Cho、Mark Saroufim




 Reviewers: Ashok Emani, Jiong Gong




 在 CPU 上获得强大的深度学习开箱即用性能可能很棘手，但如果您’了解影响性能的主要问题以及如何衡量它们，那么’会更容易以及如何解决这些问题。




 TL；博士









| 
 问题
 | 
 如何测量
 | 
 解决方案
 |
| 
 GEMM 执行单元存在瓶颈
 | * [不平衡或串行旋转](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/reference/cpu-metrics-reference/spin-time/imbalance- or-serial-spinning-1.html)
* [前端绑定](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/reference/cpu-metrics-reference/front-end-bound.html)
* [核心绑定](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/参考/cpu-metrics-reference/back-end-bound.html)
 | 
 通过核心固定设置与物理核心的线程关联性，避免使用逻辑核心
 |
| 
 非统一内存访问 (NUMA)
 | * 本地与远程内存访问
* [UPI 利用率](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/reference/cpu-metrics-reference /memory-bound/dram-bound/upi-utilization-bound.html)
* 内存访问延迟
* 线程迁移
 | 
 通过核心固定将线程亲和力设置到特定套接字，避免跨套接字计算
 |



*GEMM（通用矩阵乘法）* 
 在融合乘加 (FMA) 或点积 (DP) 执行单元上运行，这将成为瓶颈并导致线程等待延迟/
 *同步旋转* \ n 启用
 *超线程* 
 时的障碍 - 因为使用逻辑核心会导致所有工作线程的并发性不足，因为每个逻辑线程
 *争用相同的核心资源* 
 。相反，如果我们为每个物理核心使用 1 个线程，就可以避免这种争用。因此，我们通常建议
 *通过将 CPU
 *线程亲和力* 
 设置为物理核心，
 *避免逻辑核心* 
 通过
 *核心固定* 
 。




 多插槽系统具有
 *非统一内存访问 (NUMA)* 
 这是一种共享内存架构，描述了主内存模块相对于处理器的布局。但如果进程不支持 NUMA，则在运行时
 *线程迁移* 
 通过跨套接字
 *英特尔超级路径互连 (UPI)* 
 时会频繁访问缓慢
 *远程内存* 
 。我们通过通过
 *核心固定* 
 将 CPU
 *线程关联* 
 设置到特定套接字来解决此问题。




 了解这些原则后，正确的 CPU 运行时配置可以显着提高开箱即用的性能。




 在本博客中，我们’ 将引导您了解您应该从 [CPU 性能调优指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide) 中了解的重要运行时配置.html#cpu-specific-optimizations) 
 ，解释它们如何工作、如何分析它们以及如何将它们集成到模型服务框架中，例如
 [TorchServe](https://github.com/pytorch/serve) 
 通过易于使用
 [启动脚本](https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md) 
 我们’ve
 [集成](https://github.com/pytorch/serve/pull/1354) 
1 
 原生。




 我们’ 将解释所有这些想法
 **直观地** 
 来自
 **第一原则** 
 以及大量
 **配置文件** 
 并向您展示如何我们运用所学知识来提高 TorchServe 上开箱即用的 CPU 性能。



1. 必须通过在
 *config.properties* 
 中设置
 *cpu_launcher_enable=true* 
 显式启用该功能。


## 避免深度学习的逻辑核心 [¶](#avoid-logic-cores-for-deep-learning "永久链接到此标题")




 避免深度学习工作负载的逻辑核心通常可以提高性能。为了理解这一点，让我们回顾一下 GEMM。




**优化 GEMM 可以优化深度学习**




 深度学习训练或推理的大部分时间都花在了 GEMM 的数百万次重复操作上，GEMM 是全连接层的核心。自从多层感知器 (MLP)
 [被证明是任何连续函数的通用逼近器](https://en.wikipedia.org/wiki/Universal_approximation_theorem) 
 以来，全连接层已经使用了几十年。任何 MLP 都可以完全表示为 GEMM。甚至可以使用 [Toepliz 矩阵](https://en.wikipedia.org/wiki/Toeplitz_matrix) 将卷积表示为 GEMM。
 。



 回到最初的主题，大多数 GEMM 运算符受益于使用非超线程，因为深度学习训练或推理的大部分时间都花在运行在融合乘加 (FMA) 或点上的 GEMM 的数百万次重复运算上- 超线程核心共享的产品 (DP) 执行单元。启用超线程后，OpenMP 线程将竞争相同的 GEMM 执行单元。




[![https://pytorch.org/tutorials/_images/1_.png](https://pytorch.org/tutorials/_images/1_.png)](https://pytorch.org/tutorials/_images/1_.png)


 如果两个逻辑线程同时运行 GEMM，它们将共享相同的核心资源，从而导致前端限制，这样前端限制的开销大于同时运行两个逻辑线程的收益。\ n



 因此，我们通常建议避免将逻辑核心用于深度学习工作负载，以实现良好的性能。默认情况下，启动脚本仅使用物理核心；但是，用户只需切换
 `--use_逻辑_core`
 启动脚本旋钮即可轻松试验逻辑核心与物理核心。




**锻炼**




 我们’ 将使用以下输入 ResNet50 虚拟张量的示例：






```
import torch
import torchvision.models as models
import time

model = models.resnet50(pretrained=False)
model.eval()
data = torch.rand(1, 3, 224, 224)

# warm up
for _ in range(100):
    model(data)

start = time.time()
for _ in range(100):
    model(data)
end = time.time()
print('Inference took {:.2f} ms in average'.format((end-start)/100*1000))

```




 在整个博客中，我们’ 将使用
 [Intel® VTune™ Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html#gs.v4egjg) 
 用于分析和验证优化。我们’ 将在具有两个 Intel(R) Xeon(R) Platinum 8180M CPU 的机器上运行所有练习。 CPU信息如图2.1所示。




 环境变量
 `OMP_NUM_THREADS`
 用于设置并行区域的线程数。我们’ 将比较
 `OMP_NUM_THREADS=2`
 与 (1) 使用逻辑核心和 (2) 仅使用物理核心。



1. 两个 OpenMP 线程都尝试利用超线程核心共享的相同 GEMM 执行单元 (0, 56)



 我们可以通过在 Linux 上运行
 `htop`
 命令来可视化这一点，如下所示。




[![https://pytorch.org/tutorials/_images/2.png](https://pytorch.org/tutorials/_images/2.png)](https://pytorch.org/tutorials/_images/2.png)


[![https://pytorch.org/tutorials/_images/3.png](https://pytorch.org/tutorials/_images/3.png)](https://pytorch.org/tutorials/_images/3.png)


 我们注意到旋转时间被标记，其中大部分是不平衡或串行旋转造成的 - 总共 8.982 秒中的 4.980 秒。使用逻辑核心时出现不平衡或串行旋转是由于工作线程并发性不足，因为每个逻辑线程都争用相同的核心资源。




 执行摘要的热门热点部分表明
 `__kmp_fork_barrier`
 花费了 4.589 秒的 CPU 时间 - 在 9.33% 的 CPU 执行时间中，线程只是在旋转由于线程同步而处于此障碍。



2. 每个 OpenMP 线程利用各自物理内核 (0,1) 中的 GEMM 执行单元



[![https://pytorch.org/tutorials/_images/4.png](https://pytorch.org/tutorials/_images/4.png)](https://pytorch.org/tutorials/_images/4.png)


[![https://pytorch.org/tutorials/_images/5.png](https://pytorch.org/tutorials/_images/5.png)](https://pytorch.org/tutorials/_images/5.png)


 我们首先注意到，通过避免逻辑核心，执行时间从 32 秒下降到 23 秒。虽然’s 仍然存在一些不可忽略的不平衡或串行旋转，但我们注意到从 4.980 秒到 3.887 秒的相对改进。




 通过不使用逻辑线程（而是每个物理核心使用 1 个线程），我们可以避免逻辑线程争用相同的核心资源。 “热门热点”部分还表明
 `__kmp_fork_barrier`
 时间从 4.589 秒相对改进到 3.530 秒。





## 本地内存访问始终比远程内存访问快 [¶](#local-memory-access-is-always-faster-than-remote-memory-access "永久链接到此标题")




 我们通常建议将进程绑定到本地套接字，这样进程就不会跨套接字迁移。一般来说，这样做的目标是利用本地内存上的高速缓存并避免远程内存访问，远程内存访问速度可能会慢约 2 倍。




[![https://pytorch.org/tutorials/_images/6.png](https://pytorch.org/tutorials/_images/6.png)](https://pytorch.org/tutorials/_images/6.png)


 图 1. 两个插槽配置




 图 1 显示了典型的双插槽配置。请注意，每个套接字都有自己的本地内存。插槽通过英特尔超级路径互连 (UPI) 相互连接，允许每个插槽访问另一个插槽的本地内存（称为远程内存）。本地内存访问始终比远程内存访问快。




[![https://pytorch.org/tutorials/_images/7.png](https://pytorch.org/tutorials/_images/7.png)](https://pytorch.org/tutorials/_images/7.png)


 图 2.1。 CPU 信息




 用户可以通过在 Linux 计算机上运行 
 `lscpu`
 命令来获取其 CPU 信息。图 2.1。显示了在具有两个 Intel(R) Xeon(R) Platinum 8180M CPU 的计算机上执行
 `lscpu`
 的示例。请注意，每个插槽有 28 个核心，每个核心有 2 个线程（即启用了超线程）。换句话说，除了 28 个物理核心之外，还有 28 个逻辑核心，每个插槽总共有 56 个核心。并且有 2 个插槽，总共 112 个核心 (
 `Thread(s)
 

 per
 

 core`
 x
 `Core(s)
 

每个
 

 套接字`
 x
 `套接字`
 )。




[![https://pytorch.org/tutorials/_images/8.png](https://pytorch.org/tutorials/_images/8.png)](https://pytorch.org/tutorials/_images/8.png)


 图 2.2。 CPU 信息




 2 个套接字分别映射到 2 个 NUMA 节点（NUMA 节点 0、NUMA 节点 1）。物理核心的索引优先于逻辑核心。如图 2.2 所示，第一个插槽上的前 28 个物理核心 (0-27) 和前 28 个逻辑核心 (56-83) 位于 NUMA 节点 0 上。而第二个 28 个物理核心 (28-55) 和第二个插槽上的第二个 28 个逻辑核心 (84-111) 位于 NUMA 节点 1 上。同一插槽上的核心共享本地内存和末级缓存 (LLC)，这比通过 Intel UPI 的跨插槽通信快得多。




 现在我们了解了 NUMA、跨套接字 (UPI) 流量、多处理器系统中的本地与远程内存访问，让’s 分析并验证我们的理解。




**锻炼**




 我们’ 将重用上面的 ResNet50 示例。




 由于我们没有将线程固定到特定套接字的处理器核心，因此操作系统会定期调度位于不同套接字的处理器核心上的线程。




[![https://pytorch.org/tutorials/_images/9.gif](https://pytorch.org/tutorials/_images/9.gif)](https://pytorch.org/tutorials/_images/9.gif)


 图 3. 非 NUMA 感知应用程序的 CPU 使用情况。启动了 1 个主工作线程，然后在所有核心（包括逻辑核心）上启动了物理核心数量 (56) 的线程。




 （旁白：如果线程数未通过 [torch.set_num_threads](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html) 设置
 ，默认的线程数是启用超线程的系统中的物理核心数。这可以通过
 [torch.get_num_threads](https://pytorch.org/docs/stable/generated /torch.get_num_threads.html) 
 。因此我们在上面看到大约一半的核心忙于运行示例脚本。)




[![https://pytorch.org/tutorials/_images/10.png](https://pytorch.org/tutorials/_images/10.png)](https://pytorch.org/tutorials/_images/10.png)


 图 4. 非均匀内存访问分析图




 图 4. 比较本地与远程内存访问随时间的变化。我们验证远程内存的使用情况，这可能会导致性能不佳。




**设置线程关联性以减少远程内存访问和跨套接字 (UPI) 流量**




 将线程固定到同一套接字上的核心有助于维护内存访问的局部性。在此示例中，我们’ll 固定到第一个 NUMA 节点 (0-27) 上的物理核心。使用启动脚本，用户只需切换
 `--node_id`
 启动脚本旋钮即可轻松试验 NUMA 节点配置。




 现在让’s 可视化 CPU 使用情况。




[![https://pytorch.org/tutorials/_images/11.gif](https://pytorch.org/tutorials/_images/11.gif)](https://pytorch.org/tutorials/_images/11.gif)


 图 5. NUMA 感知应用程序的 CPU 使用情况




 启动了 1 个主工作线程，然后在第一个 numa 节点上的所有物理核心上启动了线程。




[![https://pytorch.org/tutorials/_images/12.png](https://pytorch.org/tutorials/_images/12.png)](https://pytorch.org/tutorials/_images/12.png)


 图 6. 非均匀内存访问分析图




 如图6所示，现在几乎所有的内存访问都是本地访问。





## 通过核心固定实现多工作线程推理的高效 CPU 使用率 [¶](#efficient-cpu-usage-with-core-pinning-for-multi-worker-inference "永久链接到此标题")
- 



 运行多工作线程推理时，工作线程之间的核心重叠（或共享），导致 CPU 使用效率低下。为了解决此问题，启动脚本将可用核心数除以工作线程数，以便每个工作线程在运行时都固定到分配的核心。




**使用 TorchServe 进行锻炼**




 对于本练习，让’s 将我们迄今为止讨论的 CPU 性能调优原则和建议应用到
 [TorchServe apache-bench 基准测试](https://github.com/pytorch/服务/树/master/benchmarks#benchmarking-with-apache-bench) 
.




 我们’将使用ResNet50，有4个worker，并发100，请求10,000。所有其他参数（例如，batch_size、输入等）与
 [默认参数](https://github.com/pytorch/serve/blob/master/benchmarks/benchmark-ab.py #L18) 
.




 我们’ 将比较以下三种配置：



1. 默认 TorchServe 设置（无核心固定）
2. [torch.set_num_threads](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html) 
 =
 `数量
 

 的
 
 
 物理
 

 核心
 

 /
 

 数量
 

 
 

 工作人员`
（无核心固定）
3.通过启动脚本固定核心（所需 Torchserve>=0.6.1）



 经过此练习，我们’ll 已验证，我们更喜欢避免逻辑核心，并且更喜欢通过真实 TorchServe 用例的核心固定进行本地内存访问。





## 1. 默认 TorchServe 设置（无核心固定） [¶](#default-torchserve-setting-no-core-pinning "永久链接到此标题")




 [base_handler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py)
 没有’t 显式设置
 [ torch.set_num_threads](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html) 
 。因此，默认线程数是物理 CPU 核心数，如[此处](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#runtime-api)所述。用户可以通过base_handler中的
 [torch.get_num_threads](https://pytorch.org/docs/stable/generated/torch.get_num_threads.html)检查线程数。 4个主工作线程分别启动物理核心数（56）个线程，总共启动56x4 = 224个线程，超过核心总数112。因此保证核心与高逻辑核心重度重叠利用率 - 多个工作人员同时使用多个核心。此外，由于线程不关联到特定的 CPU 核心，操作系统会定期将线程调度到位于不同套接字的核心。



1.CPU使用率



[![https://pytorch.org/tutorials/_images/13.png](https://pytorch.org/tutorials/_images/13.png)](https://pytorch.org/tutorials/_images/13.png)


 启动了 4 个主工作线程，然后每个线程在所有核心（包括逻辑核心）上启动了物理核心数量 (56) 的线程。



2. Core Bound 档位



[![https://pytorch.org/tutorials/_images/14.png](https://pytorch.org/tutorials/_images/14.png)](https://pytorch.org/tutorials/_images/14.png)


 我们观察到 Core Bound 停顿非常高，高达 88.4%，从而降低了管道效率。核心绑定停顿表示 CPU 中可用执行单元的使用未达到最佳状态。例如，连续竞争超线程核心共享的融合乘加 (FMA) 或点积 (DP) 执行单元的多个 GEMM 指令可能会导致核心绑定停顿。正如上一节所述，逻辑核心的使用加剧了这个问题。




[![https://pytorch.org/tutorials/_images/15.png](https://pytorch.org/tutorials/_images/15.png)](https://pytorch.org/tutorials/_images/15.png)


[![https://pytorch.org/tutorials/_images/16.png](https://pytorch.org/tutorials/_images/16.png)](https://pytorch.org/tutorials/_images/16.png)


 未填充微操作 (uOps) 的空管道槽归因于停顿。例如，如果没有核心固定，CPU 使用率可能不会有效地用于计算，而是用于其他操作，例如 Linux 内核的线程调度。我们在上面看到
 `__sched_yield`
 贡献了大部分旋转时间。



3. 线程迁移



 如果没有核心固定，调度程序可能会将在一个核心上执行的线程迁移到另一个核心。线程迁移可以使线程与已提取到缓存中的数据解除关联，从而导致更长的数据访问延迟。当线程跨套接字迁移时，这个问题在 NUMA 系统中会加剧。已提取到本地内存上的高速缓存的数据现在变成远程内存，速度要慢得多。




[![https://pytorch.org/tutorials/_images/17.png](https://pytorch.org/tutorials/_images/17.png)](https://pytorch.org/tutorials/_images/17.png)


 一般来说，线程总数应小于或等于核心支持的线程总数。在上面的示例中，我们注意到在 core_51 上执行大量线程，而不是预期的 2 个线程（因为 Intel(R) Xeon(R) Platinum 8180 CPU 中启用了超线程）。这表明线程迁移。




[![https://pytorch.org/tutorials/_images/18.png](https://pytorch.org/tutorials/_images/18.png)](https://pytorch.org/tutorials/_images/18.png)


 此外，请注意线程 (TID:97097) 正在大量 CPU 核心上执行，这表明发生了 CPU 迁移。例如，该线程在 cpu_81 上执行，然后迁移到 cpu_14，然后迁移到 cpu_5，依此类推。此外，请注意，该线程多次来回跨套接字迁移，导致内存访问效率非常低。例如，此线程在 cpu_70（NUMA 节点 0）上执行，然后迁移到 cpu_100（NUMA 节点 1），然后迁移到 cpu_24（NUMA 节点 0）。



4. 非均匀内存访问分析



[![https://pytorch.org/tutorials/_images/19.png](https://pytorch.org/tutorials/_images/19.png)](https://pytorch.org/tutorials/_images/19.png)


 比较本地与远程内存访问随时间的变化。我们观察到大约一半（51.09%）的内存访问是远程访问，这表明 NUMA 配置不是最优的。





## 2. torch.set_num_threads =
 `数量
 

 的
 

 物理
 

 核心
 

 /
 

 
 

 个工作线程的数量`
（无核心固定） [¶](#torch-set-num-threads-number-of-physical-cores-number -of-workers-no-core-pinning"此标题的永久链接")




 为了与启动器’s 核心固定进行苹果之间的比较，我们’ 将线程数设置为核心数除以工作线程数（启动器在内部执行此操作） 。在
 [base_handler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py)中添加以下代码片段
 :






```
torch.set_num_threads(num_physical_cores/num_workers)

```




 与之前一样，没有核心固定，这些线程不会关联到特定的 CPU 核心，从而导致操作系统定期在位于不同套接字的核心上调度线程。



1.CPU使用率



[![https://pytorch.org/tutorials/_images/20.gif](https://pytorch.org/tutorials/_images/20.gif)](https://pytorch.org/tutorials/_images/20.gif)


 启动了 4 个主工作线程，然后每个线程在所有核心（包括逻辑核心）上启动
 `num_physical_cores/num_workers`
 个 (14) 个线程。



2. Core Bound 档位



[![https://pytorch.org/tutorials/_images/21.png](https://pytorch.org/tutorials/_images/21.png)](https://pytorch.org/tutorials/_images/21.png)


 虽然 Core Bound 停顿的百分比从 88.4% 下降到 73.5%，但 Core Bound 仍然很高。




[![https://pytorch.org/tutorials/_images/22.png](https://pytorch.org/tutorials/_images/22.png)](https://pytorch.org/tutorials/_images/22.png)


[![https://pytorch.org/tutorials/_images/23.png](https://pytorch.org/tutorials/_images/23.png)](https://pytorch.org/tutorials/_images/23.png)

3.线程迁移



[![https://pytorch.org/tutorials/_images/24.png](https://pytorch.org/tutorials/_images/24.png)](https://pytorch.org/tutorials/_images/24.png)


 与之前类似，没有核心固定线程（TID：94290）在大量CPU核心上执行，表明CPU迁移。我们再次注意到跨套接字线程迁移，导致内存访问效率非常低。例如，此线程在 cpu_78（NUMA 节点 0）上执行，然后迁移到 cpu_108（NUMA 节点 1）。



4. 非均匀内存访问分析



[![https://pytorch.org/tutorials/_images/25.png](https://pytorch.org/tutorials/_images/25.png)](https://pytorch.org/tutorials/_images/25.png)


 虽然比原来的 51.09% 有所提高，但仍有 40.45% 的内存访问是远程的，这表明 NUMA 配置不是最佳的。





## 3. 启动器核心固定 [¶](#launcher-core-pinning "永久链接到此标题")




 Launcher内部会将物理核心平均分配给worker，并绑定到每个worker上。提醒一下，默认情况下启动器仅使用物理核心。在此示例中，启动器将工作线程 0 绑定到核心 0-13（NUMA 节点 0），将工作线程 1 绑定到核心 14-27（NUMA 节点 0），将工作线程 2 绑定到核心 28-41（NUMA 节点 1），将工作线程 3 绑定到核心 28-41（NUMA 节点 1）。核心 42-55（NUMA 节点 1）。这样做可以确保工作线程之间的核心不会重叠，并避免逻辑核心使用。



1.CPU使用率



[![https://pytorch.org/tutorials/_images/26.gif](https://pytorch.org/tutorials/_images/26.gif)](https://pytorch.org/tutorials/_images/26.gif)


 启动了 4 个主工作线程，然后每个线程启动
 `num_physical_cores/num_workers`
 个与指定物理核心关联的线程数 (14)。



2. Core Bound 档位



[![https://pytorch.org/tutorials/_images/27.png](https://pytorch.org/tutorials/_images/27.png)](https://pytorch.org/tutorials/_images/27.png)


 核心绑定停顿已从原来的 88.4% 显着减少到 46.2% - 几乎提高了 2 倍。




[![https://pytorch.org/tutorials/_images/28.png](https://pytorch.org/tutorials/_images/28.png)](https://pytorch.org/tutorials/_images/28.png)


[![https://pytorch.org/tutorials/_images/29.png](https://pytorch.org/tutorials/_images/29.png)](https://pytorch.org/tutorials/_images/29.png)


 我们验证了通过核心绑定，大部分 CPU 时间都有效地用于计算 - 自旋时间为 0.256 秒。



3. 线程迁移



[![https://pytorch.org/tutorials/_images/30.png](https://pytorch.org/tutorials/_images/30.png)](https://pytorch.org/tutorials/_images/30.png)


 我们验证
 
 OMP 主线程 #0
 
 已绑定到分配的物理核心 (42-55)，并且未跨插槽迁移。



4. 非均匀内存访问分析



[![https://pytorch.org/tutorials/_images/31.png](https://pytorch.org/tutorials/_images/31.png)](https://pytorch.org/tutorials/_images/31.png)


 现在几乎所有（89.52%）内存访问都是本地访问。





## 结论 [¶](#conclusion "此标题的永久链接")




 在此博客中，我们’ 展示了正确设置 CPU 运行时配置可以显着提高开箱即用的 CPU 性能。




 我们已经介绍了一些通用的 CPU 性能调整原则和建议：



* 在启用超线程的系统中，仅通过核心固定将线程关联设置为物理核心，从而避免逻辑核心。
* 在具有 NUMA 的多插槽系统中，通过将线程关联设置为特定插槽来避免跨插槽远程内存访问核心钉扎。



 我们从第一原理直观地解释了这些想法，并通过分析验证了性能提升。最后，我们将所有学到的知识应用到 TorchServe 中，以提高开箱即用的 TorchServe CPU 性能。




 这些原则可以通过易于使用的启动脚本自动配置，该脚本已集成到 TorchServe 中。




 有兴趣的读者，请查看以下文档：



* [CPU 特定优化](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#cpu-specific-optimizations)
* [最大化 Intel® 软件优化 PyTorch 的性能* CPU](https://www.intel.com/content/www/us/en/developer/articles/technical/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html) 
* [性能调优指南](https://intel.github.io/intel-extension-for-pytorch/tutorials/performance_tuning/tuning_guide.html)
* [启动脚本使用指南](https://intel.github.io/intel-extension-for-pytorch/tutorials/performance_tuning/launch_script.html)
* 【自顶向下微架构分析方法】(https://www.intel.com/content/www/us/en /develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html)
* [配置 oneDNN 进行基准测试](https://oneapi-src.github.io/oneDNN/dev_guide_performance_settings.html#benchmarking-settings)
* [Intel® VTune™ Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html#gs.tcbgpa)
* [Intel® VTune™ Profiler 用户指南](https://www.intel.com/content/www/us/en/develop /documentation/vtune-help/top.html）



 请继续关注有关 CPU 优化内核的后续文章
 [Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch ) 
 以及高级启动器配置，例如内存分配器。





## 确认 [¶](#acknowledgement "永久链接到此标题")




 我们要感谢 Ashok Emani（英特尔）和 Jiong Kong（英特尔）在本博客的许多步骤中给予的大力指导和支持以及全面的反馈和审查。我们还要感谢 Hamid Shojanazeri (Meta)、李宁 (AWS) 和 Jing Xu (Intel) 在代码审查方面提供的有用反馈。 Suraj Subramanian (Meta) 和 Geeta Chauhan (Meta) 在博客上提供了有用的反馈。









