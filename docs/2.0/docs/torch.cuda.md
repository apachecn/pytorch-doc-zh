# TORCH.CUDA
> *斜体表示译者添加的解释或想法。*

这个包添加了对 CUDA 张量类型的支持，这些类型实现了与 CPU 张量相同的功能，但它们利用 GPU 进行计算。

初始化比较简单，可以随时导入，可以使用 [is_available()](https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html#torch.cuda.is_available) 来确定系统是否支持 CUDA。

[CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics) 提供了有关使用 CUDA 的更多详细信息。

|类或方法|功能描述|
| ---- | ---- |
| StreamContext | 上下文管理器，用于选择给定的数据流。|
| can_device_access_peer | 检查两个设备之间是否可以进行对等访问。 |
| current_blas_handle | 返回指向当前 cuBLAS 句柄的 cublasHandle_t 指针。*cuBLAS是CUDA基本线性代数子程序库，句柄(handle)可以把数据、函数等连接在一起* |
| current_device | 返回当前选择设备的索引，*即GPU号* |
| current_stream | 对于指定设备，返回当前选择的[数据流](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html#torch.cuda.Stream)|
|default_stream|返回指定设备上的默认[数据流](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html#torch.cuda.Stream)|
|device|可更改设备的上下文管理器|
|device_count|计算可用的GPU数|
|device_of|将当前设备更改为指定对象所在的设备|
|get_arch_list|返回这个包使用的CUDA架构列表|
|get_device_capability|获取设备的cuda功能|
|get_device_name|获取设备名称|
|get_device_properties|获取设备的属性：*设备标识、设备状态、设备位置、设备类型、设备连接方式等属性*|
|get_gencode_flags|返回编译这个库的时候使用的NVCC `gencode`，*gencode包含arch和code两部分，分别表示虚拟架构和生成代码*|
|get_sync_debug_mode|返回当前 cuda 同步操作调试模式的值|
|init|初始化 Pytorch 的 CUDA 状态|
|ipc_collect|在 CUDA IPC 释放显存后，强制回收显存|
|is_available|返回一个 bool 值，表示 CUDA 当前是否可用|
|is_initialized|返回 bool 值，表示 PyTorch 的 CUDA 状态是否已被初始化|
|memory_usage| 返回一个百分比，指的是采样周期内读写全局内存时间的占比|
|set_device|指定设备|
|set_stream|是一个设置指定数据流的API|
|set_sync_debug_mode|设置 cuda 同步操作的调试模式|
|stream|上下文管理器 StreamContext 的封装，用于选择指定的数据流|
|synchronize|等待 CUDA 设备上所有数据流中的所有 GPU 核处理完成|
|utilization|根据 nvidia-smi 提供的信息，返回在过去的采样周期内，一个或多个内核在 GPU 上执行的时间百分比|
|temperature|返回 GPU 传感器的平均温度，单位为摄氏度|
|power_draw|返回 GPU 传感器的平均耗电量，单位为 mW（毫瓦）|
|clock_rate|根据 nvidia-smi 提供的数据，以 Hz （赫兹）为单位返回过去采样周期内 GPU 的时钟速度|
|OutOfMemoryError|抛出 CUDA 显存不足异常|


### Random Number Generator *随机数生成器*
|类或方法|功能描述|
| ---- | ---- |
|get_rng_state|以 ByteTensor 的形式返回指定 GPU 的随机数生成器状态|
|get_rng_state_all|返回包含所有设备随机数状态的 ByteTensor 列表|
|set_rng_state|设置指定 GPU 的随机数生成器状态|
|set_rng_state_all|设置所有设备的随机数生成器状态|
|manual_seed|为当前 GPU 设置用于生成随机数的种子|
|manual_seed_all|设置用于在所有 GPU 上生成随机数的种子|
|seed|将生成随机数的种子设置为当前 GPU 的随机数|
|seed_all|在所有 GPU 上将生成随机数的种子设置为随机数|
|initial_seed|返回当前 GPU 的随机种子|


### Communication collectives *通信集合*
|类或方法|功能描述|
| ---- |----|
|comm.broadcast|将一个tensor向指定设备广播|
|comm.broadcast_coalesced|将一个tensor序列（tensors）向指定设备广播|
|comm.reduce_add|对来自多个 GPU 的张量求和|
|comm.scatter|将张量组播到多个 GPU 上|
|comm.gather|聚集来自多个GPU设备上的张量|


### Streams and events *数据流和事件*
|类或方法|功能描述|
|-|-|
|Stream|CUDA 数据流封装器|
|ExternalStream|外部 CUDA 数据流的封装器|
|Event|CUDA 事件封装器|


### Graphs (beta) *图*
|类或方法|功能描述|
|-|-|
|is_current_stream_capturing|返回 bool 值，表示当前 CUDA 流上是否处在图捕获状态。*这个函数可以用于控制流的捕获行为，例如在需要保存渲染结果或进行离线处理时启动捕获，而在不需要保存结果时停止捕获，以避免不必要的资源消耗*|
|graph_pool_handle|返回代表图形内存池 id 的不透明token，*利用该句柄提供了对图形资源池的操作和管理功能。使用它可以创建、销毁和管理图形资源池，以及在池中进行资源的分配和回收。这个句柄可以用于在GPU上执行图形任务，例如渲染、计算等*|
|CUDAGraph|CUDA 图封装器|
|graph|用于将 CUDA 工作捕获到 [torch.cuda.CUDAGraph](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph) 对象中的上下文管理器|
|make_graphed_callables|传入可调用的代码（函数或 [nn.Modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)）并返回图版本。|


### Memory management *内存管理*
|类或方法|功能描述|
|-|-|
|empty_cache|释放缓存分配器当前控制的所有未占用的缓存内存，以便这些内存可用于其他 GPU 应用程序，并在 nvidia-smi 中可见|
|list_gpu_processes|返回指定设备的运行进程及其 GPU 内存使用情况，并以人类可读方式输出|
|mem_get_info|使用 cudaMemGetInfo 返回指定设备的全局可用 GPU 内存和总内存|
|memory_stats|统计指定设备上的 CUDA 内存分配器数据，并以字典形式返回。*PyTorch 中，内存分配器（Memory Allocator）是指用于管理和分配 GPU 内存的模块或组件。内存分配器负责在 GPU 上分配和释放内存*|
|memory_summary|统计指定设备上的 CUDA 内存分配器信息，并以可读形式打印输出|
|memory_snapshot|返回所有设备的 CUDA 内存分配器状态快照|
|memory_allocated|以字节为单位返回给定设备当前占用的 GPU 内存|
|max_memory_allocated|以字节为单位返回指定设备的tensor占用 GPU 内存的最大值|
|reset_max_memory_allocated|重置GPU的最大内存使用量。*并不会实际释放显存。这个方法可以与`torch.cuda.empty_cache()`函数一起使用，以确保在需要时手动释放显存*|
|memory_reserved|以字节为单位返回当前内存分配器给指定设备分配的 GPU 内存|
|max_memory_reserved|以字节为单位返回当前内存分配器给指定设备分配的最大 GPU 内存|
|set_per_process_memory_fraction|设置一个进程在GPU上使用的显存比例，*限制上限，浮点数表示*|
|memory_cached|停用; 参照 [memory_reserved()](https://pytorch.org/docs/stable/generated/torch.cuda.memory_reserved.html#torch.cuda.memory_reserved)|
|max_memory_cached|停用; 参照 [max_memory_reserved()](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_reserved.html#torch.cuda.max_memory_reserved)|
|reset_max_memory_cached|重置指定设备缓存分配器管理的 GPU 内存最大值的起点。*训练循环等需要多次迭代的过程中，可以使用reset_max_memory_cached函数来测量每次迭代的峰值缓存内存量。结合torch.cuda.max_memory_cached函数，可以计算出每次迭代使用的缓存内存量，从而了解模型训练的内存需求*|
|reset_peak_memory_stats|重置 CUDA 内存分配器跟踪的 "峰值"。|
|caching_allocator_alloc|使用 CUDA 内存分配器分配 GPU 内存，*申请显存*|
|caching_allocator_delete|使用 CUDA 内存分配器释放 GPU 内存|
|get_allocator_backend|返回一个由 `PYTORCH_CUDA_ALLOC_CONF` 定义的激活的 GPU 内存分配器后台数据|
|CUDAPluggableAllocator|从 so 文件加载 CUDA 内存分配器|
|change_current_allocator|将当前使用的内存分配器更改为指定的内存分配器。*Pytorch默认使用缓存分配器(Caching Allocator)，该分配器通过缓存和重用内存模块来减少内存分配和释放的开销，某些情况下可能需要使用不同的内存分配器来优化内存使用或满足特定需求，在调试或提高性能时可用*|

### NVIDIA Tools Extension (NVTX) *NVIDIA工具扩展*
|类或方法|功能描述|
|----|----|
|nvtx.mark|描述某个时刻发生的瞬时事件，*用于在CUDA应用程序中添加一个标记（mark）。标记是一个命名的位置，表示在CUDA应用程序执行过程中的一个特定点。通过使用`nvtx.mark`函数添加标记，可以在CUDA应用程序中创建自定义的性能分析点，以便在分析和调试过程中更容易地识别和理解应用程序的不同阶段和执行路径*|
|nvtx.range_push|创建一个新的性能分析范围并入栈|
|nvtx.range_pop|从性能分析范围堆栈中弹出一个范围|

### Jiterator (beta)
*Jiterator是torch.utils.jit模块中的一个类，名为_JITIterator，它是PyTorch中的一个迭代器类，用于在JIT（Just-In-Time）编译模式下迭代数据。*
|类或方法|功能描述|
|----|----|
|jiterator._create_jit_fn|为元素操作创建一个由 jiterator 生成的 cuda 内核|
|jiterator._create_multi_output_jit_fn|为支持返回一个或多个输出的元素操作创建由 jiterator 生成的 cuda 内核，*作用是根据给定的输入参数和多个输出参数创建一个JIT编译函数*|

### Stream Sanitizer (prototype)

CUDA Sanitizer 是一个原型工具，用于检测 PyTorch 中流之间的同步错误。有关如何使用它的信息，请参阅[文档](https://pytorch.org/docs/stable/cuda._sanitizer.html)。