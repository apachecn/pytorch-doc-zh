# CPU线程和TorchScript推理

PyTorch允许TorchScript模型推理过程中使用多个CPU线程。下图显示了不同程度的并行人们会发现在一个典型的应用：

[![../_images/cpu_threading_torchscript_inference.svg](../_images/cpu_threading_torchscript_inference.svg)](../_images/cpu_threading_torchscript_inference.svg)

一个或多个线程推断在给定的输入，执行一个模型的直传。每个推理线程调用JIT解释执行模型内嵌的OPS，一个接一个。模型可以利用一个`叉 `
TorchScript原语发起异步任务。在一次分叉几个操作导致在并行执行的任务。的`叉 `操作者返回一个未来`可用于稍后同步上，例如 `对象：

    
    
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
    

PyTorch使用单个线程池的-OP间并行性，这个线程池是由在应用过程中的分叉的所有任务进行推演共享。

除了-
OP间并行性，也PyTorch可以利用OPS（帧内运算并行）内的多个线程。这可能是在许多情况下，包括大张量等元素方面的OPS，卷积，GEMMS，嵌入查找和有用的。

## 构建选项

PyTorch使用内部ATEN库来实现欢声笑语。除此之外，PyTorch还可以与支持外部库，如[ MKL
](https://software.intel.com/en-us/mkl)和[ MKL-DNN
](https://github.com/intel/mkl-dnn)，加快对CPU计算的建造。

ATEN，MKL和MRL-DNN支持内部运算的并行和取决于以下并行库来实现它：

>   * [的OpenMP ](https://www.openmp.org/) \- 一个标准（和图书馆，通常有一个编译器运），广泛用于外部库;

>

>   * [ TBB ](https://github.com/intel/tbb) \- 一个较新的并行库基于任务的并行性和并发环境优化。

>

>

OpenMP的历史已经使用了大量的库。它是著名的相对易用性和支持基于循环的并行性和其他原语。与此同时OpenMP是不知道与应用程序使用的其他线程库一个良好的互操作性。特别是，OpenMP的并不能保证一个每个进程内部运算的线程池会在应用程序中使用。相反，两个不同的运算线程间可能会使用内部的运算工作不同OpenMP的线程池。这可能会导致大量的应用程序所使用的线程。

TBB是用来在外部库在较小程度上，但，在同一时间，为并发环境进行了优化。
PyTorch的TBB后端保证有一个独立的，单一的，每个进程内部运算线程通过所有在运行的应用程序的OPS的使用池。

根据不同的使用情况下，可能会发现一个或另一个并行库在他们的应用程序更好的选择。

PyTorch允许在构建时用下面的生成选项使用宏正和其他库并行后端的选择：

图书馆

|

构建选项

|

值

|

笔记  
  
---|---|---|---  
  
ATEN

|

`ATEN_THREADING`

|

`OMP`（默认），`TBB`

|  
  
MKL

|

`MKL_THREADING`

|

（相同）

|

为了使MKL使用`BLAS = MKL` 
  
MRL-DNN

|

`MKLDNN_THREADING`

|

(same)

|

为了使MKL-DNN用`USE_MKLDNN = 1` 
  
强烈建议不要一个构建中混合使用OpenMP和TBB。

任何`TBB`值的上述要求`USE_TBB = 1`建立设定（缺省值：OFF）。一个单独的设置`USE_OPENMP = 1
`（默认值：ON）需要将OpenMP并行。

## 运行时API

下面的API是用来控制线的设置：

并行的类型

|

设置

|

Notes  
  
---|---|---  
  
跨运算的并行

|

`在:: set_num_interop_threads`，`在:: get_num_interop_threads`（C ++）

`set_num_interop_threads`，`get_num_interop_threads`（Python中，[ `炬 `
](../torch.html#module-torch "torch")模块）

|

`设定*`功能只能使用一次，并且只有在启动期间调用时，实际的运算符之前运行;

线程的默认编号：CPU核心数量。  
  
内部运算的并行

|

`在:: set_num_threads`，`在:: get_num_threads`（C ++）`set_num_threads``
get_num_threads`（Python中，[ `炬 `](../torch.html#module-torch "torch")模块）

环境变量：`OMP_NUM_THREADS`和`MKL_NUM_THREADS` 
  
对于内部运算的并行设置，`在:: set_num_threads`，`torch.set_num_threads`总是优先于环境变量，`
MKL_NUM_THREADS`变量优先于`OMP_NUM_THREADS`。

注意

`可用于调试parallel_info`关于线程设置和工具打印信息。类似的输出也可以在Python与`
炬.__配置得到__。parallel_info`（）调用。

[Next ![](../_static/images/chevron-right-orange.svg)](cuda.html "CUDA
semantics") [![](../_static/images/chevron-right-orange.svg)
Previous](broadcasting.html "Broadcasting semantics")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * CPU线程和TorchScript推理
    * 编译选项
    * 运行时API 

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

