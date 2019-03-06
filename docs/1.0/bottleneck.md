
# torch.utils.bottleneck
> 译者:  [belonHan](https://github.com/belonHan)

`torch.utils.bottleneck`是 调试瓶颈`bottleneck`时首先用到的工具.它总结了python分析工具与PyTorch自动梯度分析工具在脚本运行中情况.

在命令行运行如下命令

```py
python -m torch.utils.bottleneck /path/to/source/script.py [args]

```

其中 `[args]` 是`script.py`脚本的参数(任意个数).运行`python -m torch.utils.bottleneck -h`命令获取更多帮助说明.

警告

请确保脚本在分析时能够在有限时间内退出.

警告

当运行CUDA代码时，由于CUDA内核的异步特性, cProfile的输出 和cpu模式的autograd分析工具可能无法显示正确的计时: 报告的CPU时间 是用于启动内核的时间,不包括在GPU上执行的时间。 在常规cpu模式分析器下，同步操作是非常昂贵的。在这种无法准确计时的情况下，可以使用cuda模式的autograd分析工具。

注意

选择查看哪个分析工具的输出结果(CPU模式还是CUDA模式) ,首先应确定脚本是不是CPU密集型`CPU-bound`(“CPU总时间远大于CUDA总时间”)。如果是cpu密集型，选择查看cpu模式的结果。相反，如果大部分时间都运行在GPU上，再查看CUDA分析结果中相应的CUDA操作。

当然，实际情况取决于您的模型，可能会更复杂，不属于上面两种极端情况。除了分析结果之外,可以尝试使用`nvprof`命令查看[`torch.autograd.profiler.emit_nvtx()`](autograd.html#torch.autograd.profiler.emit_nvtx "torch.autograd.profiler.emit_nvtx")的结果.然而需要注意NVTX的开销是非常高的,时间线经常会有严重的偏差。


警告

如果您在分析CUDA代码, `bottleneck`运行的第一个分析工具 (cProfile),它的时间中会包含CUDA的启动(CUDA缓存分配)时间。当然，如果CUDA启动时间远小于代码的中瓶颈,这就被可以忽略。

更多更复杂关于分析工具的使用方法(比如多GPU),请点击[https://docs.python.org/3/library/profile.html](https://docs.python.org/3/library/profile.html) 或者 [`torch.autograd.profiler.profile()`](autograd.html#torch.autograd.profiler.profile "torch.autograd.profiler.profile").

