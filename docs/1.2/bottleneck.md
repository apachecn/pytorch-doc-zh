# torch.utils.bottleneck

torch.utils.bottleneck
是可以用作用于在程序调试的瓶颈中的初始步骤的工具。它总结了Python的探查和PyTorch的autograd探查脚本的运行。

它运行在命令行上

    
    
    python -m torch.utils.bottleneck /path/to/source/script.py [args]
    

其中，[参数]是任意数量的参数script.py ，或运行`蟒 -m  torch.utils.bottleneck  -h
[HTG11为更多的使用的指令。`

警告

因为你的脚本将被异形，请确保它在退出的时间有限。

Warning

由于CUDA内核的异步特性，对CUDA代码运行时，CPROFILE输出和CPU模式autograd廓线仪可能无法显示正确的时序：报告的CPU时间报告的时间用于启动的内核数量，但不包括时间除非操作做了同步内核花在GPU执行。那些同步出现行动是在常规CPU模式廓线仪非常昂贵。在这些情况下的定时不正确，则CUDA模式autograd分析器可以是有帮助的。

注意

要决定哪些（仅CPU模式或CUDA模式）autograd探查器输出看，应先检查，如果你的脚本是CPU绑定（“CPU总时间比CUDA总时间要大得多”）。如果是CPU密集型的，看着CPU模式autograd分析器可以帮助的结果。如果在另一方面你的脚本花费大量的时间在GPU上执行的，则是有意义的开始寻找负责CUDA运营商在CUDA模式autograd探查器的输出。

当然，现实情况要复杂得多，你的脚本可能不是取决于你正在评估模型的一部分这两个极端中的一个。如果探查器输出不帮忙，你可以尝试寻找[ `
torch.autograd.profiler.emit_nvtx的结果（） `
](autograd.html#torch.autograd.profiler.emit_nvtx
"torch.autograd.profiler.emit_nvtx")与`nvprof
`。但是，请考虑到该NVTX开销是非常高的，往往给人一种严重扭曲的时间表。

Warning

如果您正在配置CUDA代码，第一个分析器，`瓶颈
`运行（CPROFILE）将在其报告时间在CUDA启动时间（CUDA缓冲区分配费用）。如果您的系统瓶颈导致代码比CUDA启动时间慢得多这不应该的问题。

对于（在多GPU情况下等）的廓线的更复杂的应用，请参见[ https://docs.python.org/3/library/profile.html
](https://docs.python.org/3/library/profile.html)或[ `
torch.autograd.profiler.profile（） `
](autograd.html#torch.autograd.profiler.profile
"torch.autograd.profiler.profile")获得更多信息。

[Next ![](_static/images/chevron-right-orange.svg)](checkpoint.html
"torch.utils.checkpoint") [![](_static/images/chevron-right-orange.svg)
Previous](random.html "torch.random")

* * *

©版权所有2019年，Torch 贡献者。