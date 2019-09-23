# torch.cuda

这个包增加了支持CUDA张类型，实现相同功能的CPU张量，但他们利用了计算的GPU。

据延迟初始化的，所以你可以随时导入它，然后用 `is_available（） `以确定您的系统支持CUDA。

[ CUDA语义 ](notes/cuda.html#cuda-semantics)有大约使用CUDA的更多细节。

`torch.cuda.``current_blas_handle`()[[source]](_modules/torch/cuda.html#current_blas_handle)

    

返回cublasHandle_t指向当前CUBLAS手柄

`torch.cuda.``current_device`()[[source]](_modules/torch/cuda.html#current_device)

    

返回当前选择的设备的索引。

`torch.cuda.``current_stream`( _device=None_
)[[source]](_modules/torch/cuda.html#current_stream)

    

返回当前选择的 `流 `对于给定的设备。

Parameters

    

**装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device
"torch.torch.device") _或_ [ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")
_，_ _可选_ ） - 选定的设备。返回当前选择的 `串流 `用于当前装置中，通过 `current_device给出（） `，如果 `装置 `
是`无 `（默认）。

`torch.cuda.``default_stream`( _device=None_
)[[source]](_modules/torch/cuda.html#default_stream)

    

返回默认 `流 `对于给定的设备。

Parameters

    

**装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device
"torch.torch.device") _或_ [ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")
_，_ _可选_ ） - 选定的设备。返回默认 `流 `用于当前装置，由下式给出 `current_device（） `下，如果 `装置 `是`
无 `（默认）。

_class_`torch.cuda.``device`( _device_
)[[source]](_modules/torch/cuda.html#device)

    

上下文经理改变所选的设备。

Parameters

    

**装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device
"torch.torch.device") _或_ [ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")）
- 设备索引来选择。它是一个无操作，如果该参数是负整数或`无 `。

`torch.cuda.``device_count`()[[source]](_modules/torch/cuda.html#device_count)

    

返回可用GPU的数量。

_class_`torch.cuda.``device_of`( _obj_
)[[source]](_modules/torch/cuda.html#device_of)

    

上下文管理器，改变当前设备到给定对象的。

您可以同时使用张量和储存作为参数。如果给定对象不是在GPU上分配的，这是一个空操作。

Parameters

    

**OBJ** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ _存放_ ） -
对象所选择的装置上分配。

`torch.cuda.``empty_cache`()[[source]](_modules/torch/cuda.html#empty_cache)

    

发布当前由高速缓存分配器保持，使得那些可以在其他的GPU应用中使用，并在可见的所有空闲的缓存内存的NVIDIA-SMI [HTG1。

注意

`empty_cache（） `不增加GPU存储器可用于PyTorch量。参见[ 内存管理 ](notes/cuda.html#cuda-memory-
management)关于GPU内存管理的更多细节。

`torch.cuda.``get_device_capability`( _device=None_
)[[source]](_modules/torch/cuda.html#get_device_capability)

    

获取设备的CUDA能力。

Parameters

    

**装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device
"torch.torch.device") _或_ [ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")
_，_ _可选_ ） - 装置，其用于归还该设备的能力。这个函数是一个无操作，如果这种说法是一个负整数。它使用当前装置中，通过 `
current_device给出（） `时，如果 `装置 `是`无 `（默认）。

Returns

    

该设备的主要和次要CUDA能力

Return type

    

[元组](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python
v3.7\)")（[ INT ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)")，[ INT ](https://docs.python.org/3/library/functions.html#int
"\(in Python v3.7\)")）

`torch.cuda.``get_device_name`( _device=None_
)[[source]](_modules/torch/cuda.html#get_device_name)

    

获取设备的名称。

Parameters

    

**装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device
"torch.torch.device") _或_ [ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")
_，_ _可选_ ） - 设备要返回的名称。这个函数是一个无操作，如果这种说法是一个负整数。它使用当前装置中，通过 `current_device给出（）
`时，如果 `装置 `是`无 `（默认）。

`torch.cuda.``init`()[[source]](_modules/torch/cuda.html#init)

    

初始化PyTorch的CUDA状态。您可能需要显式调用，如果你是PyTorch通过其C
API进行交互，为CUDA功能Python绑定要等到这个初始化发生。普通用户不应该需要这个，因为所有的PyTorch的CUDA方法自动初始化点播CUDA状态。

请问咱这CUDA状态已初始化。

`torch.cuda.``ipc_collect`()[[source]](_modules/torch/cuda.html#ipc_collect)

    

强制收集GPU内存已经通过CUDA IPC发布之后。

Note

检查是否有任何发送CUDA张量可以从内存中清除。力关闭用于参考计数，如果不存在激活的计数器共享内存文件。有用当生产者进程停止继续发张量和要释放未使用的内存。

`torch.cuda.``is_available`()[[source]](_modules/torch/cuda.html#is_available)

    

返回一个布尔值，指示是否CUDA是目前已经上市。

`torch.cuda.``max_memory_allocated`( _device=None_
)[[source]](_modules/torch/cuda.html#max_memory_allocated)

    

返回通过张量以字节为单位占用一个给定设备的最大GPU内存。

默认情况下，这将返回因为该程序的开始分配的内存峰值。`reset_max_memory_allocated（） `
可用于起点在跟踪该度量复位。例如，这两个功能可以测量在训练循环每次迭代的峰值分配内存使用情况。

Parameters

    

**装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device
"torch.torch.device") _或_ [ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")
_，_ _可选_ ） - 选定的设备。通过 `current_device给出返回当前设备统计量，（） `时，如果 `装置 `是`无 `（默认）。

Note

参见[ 内存管理 ](notes/cuda.html#cuda-memory-management)关于GPU内存管理的更多细节。

`torch.cuda.``max_memory_cached`( _device=None_
)[[source]](_modules/torch/cuda.html#max_memory_cached)

    

返回在对于给定的设备的字节高速缓存分配器管理的最大GPU存储器。

默认情况下，这将返回因为该节目的开头缓存内存中的峰值。`reset_max_memory_cached（） `
可用于起点在跟踪该度量复位。例如，这两个功能可以测量在训练循环每次迭代的峰值高速缓存的存储器的量。

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more
details about GPU memory management.

`torch.cuda.``memory_allocated`( _device=None_
)[[source]](_modules/torch/cuda.html#memory_allocated)

    

返回由张量以字节为单位所占用的指定设备的当前GPU内存。

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

这可能是比所示的量较少NVIDIA-SMI 因为一些未使用的存储器可以由高速缓存分配器被保持和一些上下文需要对GPU创建。参见[ 内存管理
](notes/cuda.html#cuda-memory-management)关于GPU内存管理的更多细节。

`torch.cuda.``memory_cached`( _device=None_
)[[source]](_modules/torch/cuda.html#memory_cached)

    

返回在对于给定的设备的字节高速缓存分配器管理的当前GPU存储器。

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more
details about GPU memory management.

`torch.cuda.``reset_max_memory_allocated`( _device=None_
)[[source]](_modules/torch/cuda.html#reset_max_memory_allocated)

    

复位在跟踪由张量对于给定的装置所占据最大GPU存储器的起点。

参见 `max_memory_allocated（） `的详细信息。

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more
details about GPU memory management.

`torch.cuda.``reset_max_memory_cached`( _device=None_
)[[source]](_modules/torch/cuda.html#reset_max_memory_cached)

    

在复位通过跟踪缓存分配器对于给定的设备所管理的最大GPU存储器的起点。

参见 `max_memory_cached（） `的详细信息。

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more
details about GPU memory management.

`torch.cuda.``set_device`( _device_
)[[source]](_modules/torch/cuda.html#set_device)

    

设置当前设备。

这个功能的用法有利于装置的 `气馁 `。在大多数情况下，最好使用`CUDA_VISIBLE_DEVICES`环境变量。

Parameters

    

**装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device
"torch.torch.device") _或_ [ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")）
- 选定的设备。这个函数是一个无操作，如果这个参数为负。

`torch.cuda.``stream`( _stream_ )[[source]](_modules/torch/cuda.html#stream)

    

上下文管理器，选择一个给定的流。

其范围内的所有排队CUDA内核将在选定的数据流进行排队。

Parameters

    

**串** （ _串流_ ） - 选择的流。这个经理是一个空操作，如果它是`无 [HTG9。`

Note

流是每设备。如果选定的流不是当前设备上时，该功能也将改变当前设备到流相匹配。

`torch.cuda.``synchronize`( _device=None_
)[[source]](_modules/torch/cuda.html#synchronize)

    

在CUDA设备上的所有数据流都内核等待完成。

Parameters

    

**装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device
"torch.torch.device") _或_ [ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")
_，_ _可选_ ） - 装置，其同步。它使用当前装置中，通过 `current_device给出（） `时，如果 `装置 `是`无 `（默认）。

## 随机数发生器

`torch.cuda.``get_rng_state`( _device='cuda'_
)[[source]](_modules/torch/cuda/random.html#get_rng_state)

    

返回指定GPU作为ByteTensor的随机数生成器的状态。

Parameters

    

**装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device
"torch.torch.device") _或_ [ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")
_，_ _可选_ ） - 该设备返回的RNG状态。默认值：`'CUDA' `（即`torch.device（ 'CUDA'） `，电流CUDA装置）。

警告

这个函数初始化热切CUDA。

`torch.cuda.``get_rng_state_all`()[[source]](_modules/torch/cuda/random.html#get_rng_state_all)

    

返回ByteTensor代表所有设备的随机数状态的元组。

`torch.cuda.``set_rng_state`( _new_state_ , _device='cuda'_
)[[source]](_modules/torch/cuda/random.html#set_rng_state)

    

设置指定GPU的随机数生成器的状态。

Parameters

    

  * **NEW_STATE** （ _torch.ByteTensor_ ） - 期望状态

  * **装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device "torch.torch.device") _或_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 设置RNG状态的装置。默认值：`'CUDA' `（即`torch.device（ 'CUDA'） `，电流CUDA装置）。

`torch.cuda.``set_rng_state_all`( _new_states_
)[[source]](_modules/torch/cuda/random.html#set_rng_state_all)

    

将所有设备的随机数生成器的状态。

Parameters

    

**NEW_STATE** （ _的torch.ByteTensor_ 元组） - 每个设备的所期望的状态

`torch.cuda.``manual_seed`( _seed_
)[[source]](_modules/torch/cuda/random.html#manual_seed)

    

设置为当前GPU产生随机数种子。它是安全的调用这个函数，如果CUDA不可用;在这种情况下，它被忽略。

Parameters

    

**种子** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)")） - 所需的种子。

Warning

如果你是一个多GPU模式工作时，这个功能是不足以获得确定性。种子所有的GPU中，用 `manual_seed_all（） `。

`torch.cuda.``manual_seed_all`( _seed_
)[[source]](_modules/torch/cuda/random.html#manual_seed_all)

    

设置上所有的GPU产生随机数种子。它是安全的调用这个函数，如果CUDA不可用;在这种情况下，它被忽略。

Parameters

    

**seed** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)")) – The desired seed.

`torch.cuda.``seed`()[[source]](_modules/torch/cuda/random.html#seed)

    

设置用于产生随机数的随机数为当前GPU种子。它是安全的调用这个函数，如果CUDA不可用;在这种情况下，它被忽略。

Warning

如果你是一个多GPU模式工作时，该功能只会初始化一个GPU的种子。初始化所有的GPU中，用 `seed_all（） `。

`torch.cuda.``seed_all`()[[source]](_modules/torch/cuda/random.html#seed_all)

    

设置生成随机数在所有GPU的随机数种子。它是安全的调用这个函数，如果CUDA不可用;在这种情况下，它被忽略。

`torch.cuda.``initial_seed`()[[source]](_modules/torch/cuda/random.html#initial_seed)

    

返回当前GPU的电流随机种子。

Warning

This function eagerly initializes CUDA.

## 通信集体

`torch.cuda.comm.``broadcast`( _tensor_ , _devices_
)[[source]](_modules/torch/cuda/comm.html#broadcast)

    

广播张到多个GPU的。

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量来广播。

  * **设备** （ _可迭代_ ） - 设备的一个可迭代其中广播。需要注意的是它应该像（SRC，DST1，DST2，...），所述第一元件，其是在源设备从广播。

Returns

    

含有的拷贝元组中的`张量 `，放置在从`设备 `对应于索引的装置。

`torch.cuda.comm.``broadcast_coalesced`( _tensors_ , _devices_ ,
_buffer_size=10485760_
)[[source]](_modules/torch/cuda/comm.html#broadcast_coalesced)

    

广播顺序张量到指定的GPU。小张量第一合并成一个缓冲区，以减少同步的数量。

Parameters

    

  * **张量** （ _序列_ ） - 张量来广播。

  * **devices** ( _Iterable_ ) – an iterable of devices among which to broadcast. Note that it should be like (src, dst1, dst2, …), the first element of which is the source device to broadcast from.

  * **BUFFER_SIZE** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 用于聚结的缓冲区的最大尺寸

Returns

    

A tuple containing copies of the `tensor`, placed on devices corresponding to
indices from `devices`.

`torch.cuda.comm.``reduce_add`( _inputs_ , _destination=None_
)[[source]](_modules/torch/cuda/comm.html#reduce_add)

    

从多个GPU的款项张量。

所有输入应该有匹配的形状。

Parameters

    

  * **输入** （ _可迭代_ _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 张量的一个可迭代添加。

  * **目的地** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 在其上输出将被置于一个设备（默认值：当前设备）。

Returns

    

含有的所有输入的元素单元的总和张量，放置在`目的地 `装置。

`torch.cuda.comm.``scatter`( _tensor_ , _devices_ , _chunk_sizes=None_ ,
_dim=0_ , _streams=None_ )[[source]](_modules/torch/cuda/comm.html#scatter)

    

跨散射多个GPU张量。

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量散射。

  * **设备** （ _可迭代_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") __ ） - 迭代的整数，其中指定哪些设备的张量应散射。

  * **chunk_sizes** （ _可迭代_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") __ _，_ _可选的_ ） - 组块的大小，以被放置在每个设备上。它应该匹配`设备 `在长度和总和为`tensor.size（DIM） `。如果没有指定，张量将被分成相等的块。

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 沿着以组块中的张量的尺寸。

Returns

    

含有跨越给定`设备 `中的`张量 `，传播块元组。

`torch.cuda.comm.``gather`( _tensors_ , _dim=0_ , _destination=None_
)[[source]](_modules/torch/cuda/comm.html#gather)

    

汇集了来自多个GPU张量。

张量大小在比`暗淡 `必须匹配不同所有维度。

Parameters

    

  * **张量** （ _可迭代_ _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 迭代张量的聚集。

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸沿其张量将是串联的。

  * **目的地** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 输出装置（-1表示CPU，默认：当前装置）

Returns

    

位于`目的地 `设备上的张量，即级联`张量的结果 `沿`暗淡 [ HTG11。`

## 流和事件

_class_`torch.cuda.``Stream`[[source]](_modules/torch/cuda/streams.html#Stream)

    

包裹一个CUDA流。

甲CUDA流是执行的线性序列属于特定设备，独立于其他流。参见[ CUDA语义 ](notes/cuda.html#cuda-semantics)了解详情。

Parameters

    

  * **装置** （[ _torch.device_ ](tensor_attributes.html#torch.torch.device "torch.torch.device") _或_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 在其上分配的流的装置。如果 `装置 `是`无 `（默认）或负整数，这将使用当前设备。

  * **优先权** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 流的优先级。数字越小，代表较高的优先级。

`query`()[[source]](_modules/torch/cuda/streams.html#Stream.query)

    

如果所有提交的工作已经完成检查。

Returns

    

布林值，表示如果在这个流中的所有内核都完成。

`record_event`( _event=None_
)[[source]](_modules/torch/cuda/streams.html#Stream.record_event)

    

记录的事件。

Parameters

    

**活动** （ _事件_ _，_ _可选_ ） - 事件记录。如果不给，一个新的将被分配。

Returns

    

记录的事件。

`synchronize`()[[source]](_modules/torch/cuda/streams.html#Stream.synchronize)

    

等待在这个流中的所有内核来完成。

Note

这是周围`cudaStreamSynchronize的包装（） `：参见 `CUDA documentation`_  以获得更多信息。

`wait_event`( _event_
)[[source]](_modules/torch/cuda/streams.html#Stream.wait_event)

    

使提交给流等待一个事件的所有未来的工作。

Parameters

    

**活动** （ _事件_ ） - 事件等待。

Note

这是周围`cudaStreamWaitEvent的包装（） `：参见 `CUDA documentation`_  以获得更多信息。

该函数返回，而无需等待`活动 `：只有未来的行动受到影响。

`wait_stream`( _stream_
)[[source]](_modules/torch/cuda/streams.html#Stream.wait_stream)

    

与其他流同步。

提交此流的所有未来的工作将等到调用完成时提交给定流的所有内核。

Parameters

    

**串** （ _串流_ ） - 一个流同步。

Note

该函数返回而不 `信息流 `等待当前排队的内核：只有未来的行动受到影响。

_class_`torch.cuda.``Event`[[source]](_modules/torch/cuda/streams.html#Event)

    

包裹一个CUDA事件。

CUDA事件是可以用于监视设备的进步，以精确地测量定时，并且向CUDA流进行同步的同步标记。

当第一次记录或导出到另一处理的情况下被延迟初始化底层CUDA事件。创建后，只流在同一设备上可以记录事件。然而，任何设备上的流可以等待该事件。

Parameters

    

  * **enable_timing** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 表示如果事件应该测量时间（默认值：`假 `）

  * **阻断** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `， `等待（） `将被阻断（默认值：`假 `）

  * **间** （ ） - 如果`真 `，则事件可以被处理（默认之间共享：`假 `）

`elapsed_time`( _end_event_
)[[source]](_modules/torch/cuda/streams.html#Event.elapsed_time)

    

返回以毫秒为单位经过的事件记录和end_event记录之前之后的时间。

_classmethod_`from_ipc_handle`( _device_ , _handle_
)[[source]](_modules/torch/cuda/streams.html#Event.from_ipc_handle)

    

从给定的设备上的手柄IPC重建的事件。

`ipc_handle`()[[source]](_modules/torch/cuda/streams.html#Event.ipc_handle)

    

返回此事件的IPC手柄。如果没有记录，该事件将使用当前的设备。

`query`()[[source]](_modules/torch/cuda/streams.html#Event.query)

    

检查当前事件捕获的所有工作已完成。

Returns

    

布林值，表示如果当前事件捕获的所有工作已完成。

`record`( _stream=None_
)[[source]](_modules/torch/cuda/streams.html#Event.record)

    

记录在给定流的情况下。

使用`torch.cuda.current_stream（）如果指定`没有流。流的设备必须在事件的设备匹配。

`synchronize`()[[source]](_modules/torch/cuda/streams.html#Event.synchronize)

    

该事件等待完成。

等待，直到所有工作的完成本次活动目前抓获。这可以防止CPU线程继续，直到事件结束。

> Note

>

> 这是周围`cudaEventSynchronize的包装（） `：参见 `CUDA documentation`_  以获得更多信息。

`wait`( _stream=None_
)[[source]](_modules/torch/cuda/streams.html#Event.wait)

    

使提交给定的流等待此事件的所有未来的工作。

使用`torch.cuda.current_stream（） `如果没有指定流。

## 存储器管理

`torch.cuda.``empty_cache`()[[source]](_modules/torch/cuda.html#empty_cache)

    

Releases all unoccupied cached memory currently held by the caching allocator
so that those can be used in other GPU application and visible in nvidia-smi.

Note

`empty_cache()`doesn’t increase the amount of GPU memory available for
PyTorch. See [Memory management](notes/cuda.html#cuda-memory-management) for
more details about GPU memory management.

`torch.cuda.``memory_allocated`( _device=None_
)[[source]](_modules/torch/cuda.html#memory_allocated)

    

Returns the current GPU memory occupied by tensors in bytes for a given
device.

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

This is likely less than the amount shown in nvidia-smi since some unused
memory can be held by the caching allocator and some context needs to be
created on GPU. See [Memory management](notes/cuda.html#cuda-memory-
management) for more details about GPU memory management.

`torch.cuda.``max_memory_allocated`( _device=None_
)[[source]](_modules/torch/cuda.html#max_memory_allocated)

    

Returns the maximum GPU memory occupied by tensors in bytes for a given
device.

By default, this returns the peak allocated memory since the beginning of this
program. `reset_max_memory_allocated()`can be used to reset the starting
point in tracking this metric. For example, these two functions can measure
the peak allocated memory usage of each iteration in a training loop.

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more
details about GPU memory management.

`torch.cuda.``reset_max_memory_allocated`( _device=None_
)[[source]](_modules/torch/cuda.html#reset_max_memory_allocated)

    

Resets the starting point in tracking maximum GPU memory occupied by tensors
for a given device.

See `max_memory_allocated()`for details.

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more
details about GPU memory management.

`torch.cuda.``memory_cached`( _device=None_
)[[source]](_modules/torch/cuda.html#memory_cached)

    

Returns the current GPU memory managed by the caching allocator in bytes for a
given device.

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more
details about GPU memory management.

`torch.cuda.``max_memory_cached`( _device=None_
)[[source]](_modules/torch/cuda.html#max_memory_cached)

    

Returns the maximum GPU memory managed by the caching allocator in bytes for a
given device.

By default, this returns the peak cached memory since the beginning of this
program. `reset_max_memory_cached()`can be used to reset the starting point
in tracking this metric. For example, these two functions can measure the peak
cached memory amount of each iteration in a training loop.

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more
details about GPU memory management.

`torch.cuda.``reset_max_memory_cached`( _device=None_
)[[source]](_modules/torch/cuda.html#reset_max_memory_cached)

    

Resets the starting point in tracking maximum GPU memory managed by the
caching allocator for a given device.

See `max_memory_cached()`for details.

Parameters

    

**device** ([ _torch.device_](tensor_attributes.html#torch.torch.device
"torch.torch.device") _or_[
_int_](https://docs.python.org/3/library/functions.html#int "\(in Python
v3.7\)") _,_ _optional_ ) – selected device. Returns statistic for the current
device, given by `current_device()`, if `device`is `None`(default).

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more
details about GPU memory management.

## NVIDIA工具扩展（NVTX）

`torch.cuda.nvtx.``mark`( _msg_
)[[source]](_modules/torch/cuda/nvtx.html#mark)

    

描述发生在某一时刻的瞬时事件。

Parameters

    

**MSG** （ _串_ ） - ASCII消息到与事件相关联。

`torch.cuda.nvtx.``range_push`( _msg_
)[[source]](_modules/torch/cuda/nvtx.html#range_push)

    

推的范围到嵌套范围跨度的堆叠。返回在启动该范围的零为基础的深度。

Parameters

    

**MSG** （ _串_ ） - ASCII消息发送到与相关联的范围

`torch.cuda.nvtx.``range_pop`()[[source]](_modules/torch/cuda/nvtx.html#range_pop)

    

弹出的范围内关断嵌套范围跨度的堆叠。返回结束范围的从零开始的深度。

[Next ![](_static/images/chevron-right-orange.svg)](storage.html
"torch.Storage") [![](_static/images/chevron-right-orange.svg)
Previous](sparse.html "torch.sparse")

* * *

©版权所有2019年，Torch 贡献者。