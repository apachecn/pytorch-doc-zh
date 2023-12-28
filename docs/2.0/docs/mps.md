# torch.mps [¶](#module-torch.mps "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/mps>
>
> 原始地址：<https://pytorch.org/docs/stable/mps.html>


 该软件包支持在 Python 中访问 MPS(Metal Performance Shaders)后端的接口。Metal 是 Apple 用于对 Metal GPU(图形处理器单元)进行编程的 API。使用 MPS 意味着可以通过在金属 GPU 上运行工作来提高性能。有关更多详细信息，请参阅 <https://developer.apple.com/documentation/metalperformanceshaders>。


|  |  |
| --- | --- |
| [`synchronize`](generated/torch.mps.synchronize.html#torch.mps.synchronize "torch.mps.synchronize") |等待 MPS 设备上所有流中的所有内核完成。 |
| [`get_rng_state`](generated/torch.mps.get_rng_state.html#torch.mps.get_rng_state "torch.mps.get_rng_state") |以 ByteTensor 形式返回随机数生成器状态。 |
| [`set_rng_state`](generated/torch.mps.set_rng_state.html#torch.mps.set_rng_state "torch.mps.set_rng_state") |设置随机数生成器状态。 |
| [`手册_seed`](generated/torch.mps.manual_seed.html#torch.mps.manual_seed "torch.mps.manual_seed") |设置用于生成随机数的种子。 |
| [`seed`](generated/torch.mps.seed.html#torch.mps.seed "torch.mps.seed") |将生成随机数的种子设置为随机数。 |
| [`empty_cache`](generated/torch.mps.empty_cache.html#torch.mps.empty_cache "torch.mps.empty_cache") |释放缓存分配器当前持有的所有未占用的缓存内存，以便这些内存可以在其他 GPU 应用程序中使用。 |
| [`set_per_process_memory_fraction`](generated/torch.mps.set_per_process_memory_fraction.html#torch.mps.set_per_process_memory_fraction“torch.mps.set_per_process_memory_fraction”) |设置内存分数以限制 MPS 设备上进程的内存分配。 |
| [`当前_分配_内存`](generated/torch.mps.current_allocated_memory.html#torch.mps.current_allocated_memory "torch.mps.current_allocated_memory") |返回tensor当前占用的 GPU 内存(以字节为单位)。 |
| [`驱动_分配_内存`](generated/torch.mps.driver_allocated_memory.html#torch.mps.driver_allocated_memory "torch.mps.driver_allocated_memory") |返回 Metal 驱动程序为进程分配的 GPU 内存总量(以字节为单位)。 |


## MPS Profiler [¶](#mps-profiler "此标题的永久链接")


|  |  |
| --- | --- |
| 	[`profiler.start`](generated/torch.mps.profiler.start.html#torch.mps.profiler.start "torch.mps.profiler.start")	 | 	 Start OS Signpost tracing from MPS backend.	  |
| 	[`profiler.stop`](generated/torch.mps.profiler.stop.html#torch.mps.profiler.stop "torch.mps.profiler.stop")	 | 	 Stops generating OS Signpost tracing from MPS backend.	  |
| 	[`profiler.profile`](generated/torch.mps.profiler.profile.html#torch.mps.profiler.profile "torch.mps.profiler.profile")	 | 	 Context Manager to enabling generating OS Signpost tracing from MPS backend.	  |


## MPS 事件 [¶](#mps-event "此标题的固定链接")


|  |  |
| --- | --- |
| 	[`event.Event`](generated/torch.mps.event.Event.html#torch.mps.event.Event "torch.mps.event.Event")	 | 	 Wrapper around an MPS event.	  |