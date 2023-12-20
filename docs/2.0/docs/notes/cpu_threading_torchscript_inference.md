# CPU 线程和 TorchScript 推理 [¶](#cpu-threading-and-torchscript-inference "此标题的固定链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/cpu_threading_torchscript_inference>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html>


 PyTorch 允许在 TorchScript 模型推理期间使用多个 CPU 线程。下图显示了在非典型应用程序中会发现的不同级别的并行性：


[![https://pytorch.org/docs/stable/_images/cpu_threading_torchscript_inference.svg](https://pytorch.org/docs/stable/_images/cpu_threading_torchscript_inference.svg)](https://pytorch.org/docs/stable/_images/cpu_threading_torchscript_inference.svg) 一个或多个推理线程对给定输入执行模型的前向传递。每个推理线程调用 JIT 解释器，该解释器逐一执行内联模型的操作。模型可以利用“fork”TorchScript 原语来启动异步任务。一次分叉多个操作会导致并行执行任务。 `fork` 运算符返回一个 `Future` 对象，可用于稍后同步，例如：


```
@torch.jit.script
def compute_z(x):
    return torch.mm(x, self.w_z)

@torch.jit.script
def forward(x):
    # launch compute_z asynchronously:
    fut = torch.jit._fork(compute_z, x)
    # execute the next operation in parallel to compute_z:
    y = torch.mm(x, self.w_y)
    # wait for the result of compute_z:
    z = torch.jit._wait(fut)
    return y + z

```


 PyTorch 使用单个线程池来实现操作间并行性，该线程池由应用程序进程中分叉的所有推理任务共享。


 除了操作间并行性之外，PyTorch 还可以在操作内利用多个线程(操作内并行性)。这在许多情况下都很有用，包括大张量上的逐元素运算、卷积、GEMM、嵌入查找等。


## 构建选项 [¶](#build-options "此标题的永久链接")


 PyTorch 使用内部 ATen 库来实现操作。除此之外，PyTorch 还可以在外部库的支持下构建，例如 [MKL](https://software.intel.com/en-us/mkl) 和 [MKL-DNN](https://github.com/intel/mkl-dnn)，以加快 CPU 上的计算速度。


 ATen、MKL 和 MKL-DNN 支持操作内并行性，并依赖以下并行化库来实现：



* [OpenMP](https://www.openmp.org/) - 一个标准(和一个库，通常随编译器一起提供)，广泛用于外部库；
* [TBB](https://github.com/intel/tbb) - 一个针对基于任务的并行性和并发环境进行优化的新型并行化库。


 OpenMP 历史上已被大量库使用。它以相对易于使用以及支持基于循环的并行性和其他原语而闻名。


 TBB 在外部库中使用较少，但同时针对并发环境进行了优化。 PyTorch 的 TBB 后端保证有一个单独的、单个的、每个进程的操作内线程池，供应用程序中运行的所有操作使用。


 根据用例，人们可能会发现一个或另一个并行化库是其应用程序中更好的选择。


 PyTorch 允许在构建时使用以下构建选项选择 ATen 和其他库使用的并行化后端：


| 	 Library	  | 	 Build Option	  | 	 Values	  | 	 Notes	  |
| --- | --- | --- | --- |
| 	 ATen	  | 	`ATEN_THREADING`	 | 	`OMP`	 (default),	 `TBB`	 |  |
| 	 MKL	  | 	`MKL_THREADING`	 | 	 (same)	  | 	 To enable MKL use	 `BLAS=MKL`	 |
| 	 MKL-DNN	  | 	`MKLDNN_CPU_RUNTIME`	 | 	 (same)	  | 	 To enable MKL-DNN use	 `USE_MKLDNN=1`	 |


 建议不要在一个版本中混合使用 OpenMP 和 TBB。


 上述任何“TBB”值都需要“USE_TBB=1”构建设置(默认值：OFF)。OpenMP 并行性需要单独的设置“USE_OPENMP=1”(默认值：ON)。


## 运行时 API [¶](#runtime-api "此标题的永久链接")


 以下 API 用于控制线程设置：


| 	 Type of parallelism	  | 	 Settings	  | 	 Notes	  |
| --- | --- | --- |
| 	 Inter-op parallelism	  | 	`at::set_num_interop_threads`	 ,	 `at::get_num_interop_threads`	 (C++)	 		`set_num_interop_threads`	 ,	 `get_num_interop_threads`	 (Python,	 [`torch`](../torch.html#module-torch "torch")	 module)	  | 	 Default number of threads: number of CPU cores.	  |
| 	 Intra-op parallelism	  | 	`at::set_num_threads`	 ,	 `at::get_num_threads`	 (C++)	 `set_num_threads`	 ,	 `get_num_threads`	 (Python,	 [`torch`](../torch.html#module-torch "torch")	 module)	 		 Environment variables:	 `OMP_NUM_THREADS`	 and	 `MKL_NUM_THREADS`	 |


 对于操作内并行度设置，`at::set_num_threads` 、 `torch.set_num_threads` 始终优先于环境变量，`MKL_NUM_THREADS` 变量优先于 `OMP_NUM\ _线程`。


## 调整线程数 [¶](#tuning-the-number-of-threads "永久链接到此标题")


 以下简单脚本显示了矩阵乘法的运行时间如何随线程数变化：


```
import timeit
runtimes = []
threads = [1] + [t for t in range(2, 49, 2)]
for t in threads:
    torch.set_num_threads(t)
    r = timeit.timeit(setup = "import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=100)
    runtimes.append(r)
# ... plotting (threads, runtimes) ...

```


 在具有 24 个物理 CPU 核心(Xeon E5-2680、MKL 和 OpenMP 基于构建)的系统上运行脚本会产生以下运行时间：


[![https://pytorch.org/docs/stable/_images/cpu_threading_runtimes.svg](https://pytorch.org/docs/stable/_images/cpu_threading_runtimes.svg)](https://pytorch.org/docs/stable/_images/cpu_threading_runtimes.svg) 

在调整帧内和帧间数量时应考虑以下注意事项-操作线程：



* 在选择线程数量时，需要避免超额订阅(使用太多线程，会导致性能下降)。例如，在使用大型应用程序线程池或严重依赖操作间并行性的应用程序中，人们可能会发现禁用操作内并行性作为一种可能的选择(即通过调用“set_num_threads(1)”)；*在典型的应用程序中，人们可能会遇到延迟(处理推理请求所花费的时间)和吞吐量(每单位时间完成的工作量)之间的权衡。调整线程数量可能是一种有用的工具，可以以某种方式调整这种权衡。例如，在延迟关键的应用程序中，人们可能希望增加操作内线程的数量以尽可能快地处理每个请求。同时，操作的并行实现可能会增加额外的开销，从而增加每个请求完成的工作量，从而降低总体吞吐量。


!!! warning "警告"

    OpenMP 不保证应用程序中将使用单个每个进程的操作内线程池。相反，两个不同的应用程序或操作线程间可能会使用不同的 OpenMP 线程池来进行操作内工作。这可能会导致应用程序使用大量线程。在调整线程数量时需要格外小心，以避免过度订阅OpenMP 案例中的多线程应用程序。


!!! note "笔记"

    预构建的 PyTorch 版本是使用 OpenMP 支持进行编译的。


!!! note "笔记"

    `parallel_info` 实用程序打印有关线程设置的信息，并可用于调试。在 Python 中也可以通过 `torch.__config__.parallel_info()` 调用获得类似的输出。