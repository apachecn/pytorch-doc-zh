# torch.utils.bottleneck [¶](#module-torch.utils.bottleneck "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/bottleneck>
>
> 原始地址：<https://pytorch.org/docs/stable/bottleneck.html>


 torch.utils.bottleneck 是一个工具，可用作调试程序中瓶颈的初始步骤。它使用 Python 分析器和 PyTorch 的 autograd 分析器总结了脚本的运行情况。


 在命令行上运行它


```
python -m torch.utils.bottleneck /path/to/source/script.py [args]

```


 其中 [args] 是 script.py 的任意数量的参数，或运行 `python -m torch.utils.bottleneck -h` 以获取更多使用说明。


!!! warning "警告"

     因为您的脚本将被分析，所以请确保它在有限的时间内退出。


!!! warning "警告"

     由于 CUDA 内核的异步特性，当针对 CUDA 代码运行时，cProfile 输出和 CPU 模式自动分级分析器可能无法显示正确的计时：报告的 CPU 时间报告用于启动内核的时间量，但不包括内核执行所花费的时间在 GPU 上，除非该操作进行同步。在常规 CPU 模式分析器下，进行同步的操作似乎非常昂贵。在这些计时不正确的情况下，CUDA 模式自动梯度分析器可能会有所帮助。


!!! note "笔记"

    要决定查看哪种(仅 CPU 模式或 CUDA 模式)autograd profiler 输出，您应该首先检查您的脚本是否受 CPU 限制(“CPU 总时间远大于 CUDA 总时间”)。如果是 CPU -bound，查看 CPU 模式 autogradprofiler 的结果会有所帮助。另一方面，如果您的脚本大部分时间都在 GPU 上执行，那么开始在 CUDA 模式 autograd 分析器的输出中寻找负责任的 CUDA 运算符是有意义的。


 当然，现实要复杂得多，您的脚本可能不是这两个极端之一，具体取决于您正在评估的模型的部分。如果探查器输出没有帮助，您可以尝试查看 [`torch.autograd.profiler.emit_nvtx()`](autograd.html#torch.autograd.profiler.emit_nvtx "torch.autograd.profiler. emit_nvtx") 和 `nvprof` 。但是，请考虑到 NVTX 开销非常高，并且通常会导致时间线严重倾斜。同样，“英特尔® VTune™ Profiler”可通过 [`torch.autograd.profiler.emit_itt()`](autograd.html#torch.autograd.profiler.emit_itt "torch.autograd. profiler.emit_itt"​​) 。


!!! warning "警告"

     如果您正在分析 CUDA 代码，“bottleneck”运行的第一个分析器 (cProfile) 将在其时间报告中包括 CUDA 启动时间(CUDA 缓冲区分配成本)。如果您的瓶颈导致代码比 CUDA 启动时间慢得多，这应该无关紧要。


 有关分析器的更复杂用法(例如在多 GPU 情况下)，请参阅 <https://docs.python.org/3/library/profile.html> 或 [`torch.autograd.profiler.profile() `](autograd.html#torch.autograd.profiler.profile "torch.autograd.profiler.profile") 了解更多信息。