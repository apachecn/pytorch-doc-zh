

# torch.cuda

This package adds support for CUDA tensor types, that implement the same function as CPU tensors, but they utilize GPUs for computation.

It is lazily initialized, so you can always import it, and use [`is_available()`](#torch.cuda.is_available "torch.cuda.is_available") to determine if your system supports CUDA.

[CUDA semantics](notes/cuda.html#cuda-semantics) has more details about working with CUDA.

```py
torch.cuda.current_blas_handle()¶
```

Returns cublasHandle_t pointer to current cuBLAS handle

```py
torch.cuda.current_device()¶
```

Returns the index of a currently selected device.

```py
torch.cuda.current_stream()¶
```

Returns a currently selected [`Stream`](#torch.cuda.Stream "torch.cuda.Stream").

```py
class torch.cuda.device(device)¶
```

Context-manager that changes the selected device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – device index to select. It’s a no-op if this argument is a negative integer or `None`. |
| --- | --- |

```py
torch.cuda.device_count()¶
```

Returns the number of GPUs available.

```py
torch.cuda.device_ctx_manager¶
```

alias of [`torch.cuda.device`](#torch.cuda.device "torch.cuda.device")

```py
class torch.cuda.device_of(obj)¶
```

Context-manager that changes the current device to that of given object.

You can use both tensors and storages as arguments. If a given object is not allocated on a GPU, this is a no-op.

| Parameters: | **obj** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_ _Storage_) – object allocated on the selected device. |
| --- | --- |

```py
torch.cuda.empty_cache()¶
```

Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in &lt;cite&gt;nvidia-smi&lt;/cite&gt;.

Note

[`empty_cache()`](#torch.cuda.empty_cache "torch.cuda.empty_cache") doesn’t increase the amount of GPU memory available for PyTorch. See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

```py
torch.cuda.get_device_capability(device)¶
```

Gets the cuda capability of a device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – device for which to return the device capability. This function is a no-op if this argument is a negative integer. Uses the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |
| Returns: | the major and minor cuda capability of the device |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")([int](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) |
| --- | --- |

```py
torch.cuda.get_device_name(device)¶
```

Gets the name of a device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – device for which to return the name. This function is a no-op if this argument is a negative integer. Uses the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |

```py
torch.cuda.init()¶
```

Initialize PyTorch’s CUDA state. You may need to call this explicitly if you are interacting with PyTorch via its C API, as Python bindings for CUDA functionality will not be until this initialization takes place. Ordinary users should not need this, as all of PyTorch’s CUDA methods automatically initialize CUDA state on-demand.

Does nothing if the CUDA state is already initialized.

```py
torch.cuda.is_available()¶
```

Returns a bool indicating if CUDA is currently available.

```py
torch.cuda.max_memory_allocated(device=None)¶
```

Returns the maximum GPU memory usage by tensors in bytes for a given device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – selected device. Returns statistic for the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

```py
torch.cuda.max_memory_cached(device=None)¶
```

Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – selected device. Returns statistic for the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

```py
torch.cuda.memory_allocated(device=None)¶
```

Returns the current GPU memory usage by tensors in bytes for a given device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – selected device. Returns statistic for the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |

Note

This is likely less than the amount shown in &lt;cite&gt;nvidia-smi&lt;/cite&gt; since some unused memory can be held by the caching allocator and some context needs to be created on GPU. See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

```py
torch.cuda.memory_cached(device=None)¶
```

Returns the current GPU memory managed by the caching allocator in bytes for a given device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – selected device. Returns statistic for the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

```py
torch.cuda.set_device(device)¶
```

Sets the current device.

Usage of this function is discouraged in favor of [`device`](#torch.cuda.device "torch.cuda.device"). In most cases it’s better to use `CUDA_VISIBLE_DEVICES` environmental variable.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – selected device. This function is a no-op if this argument is negative. |
| --- | --- |

```py
torch.cuda.stream(stream)¶
```

Context-manager that selects a given stream.

All CUDA kernels queued within its context will be enqueued on a selected stream.

| Parameters: | **stream** ([_Stream_](#torch.cuda.Stream "torch.cuda.Stream")) – selected stream. This manager is a no-op if it’s `None`. |
| --- | --- |

Note

Streams are per-device, and this function changes the “current stream” only for the currently selected device. It is illegal to select a stream that belongs to a different device.

```py
torch.cuda.synchronize()¶
```

Waits for all kernels in all streams on current device to complete.

## Random Number Generator

```py
torch.cuda.get_rng_state(device=-1)¶
```

Returns the random number generator state of the current GPU as a ByteTensor.

| Parameters: | **device** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – The device to return the RNG state of. Default: -1 (i.e., use the current device). |
| --- | --- |

Warning

This function eagerly initializes CUDA.

```py
torch.cuda.set_rng_state(new_state, device=-1)¶
```

Sets the random number generator state of the current GPU.

| Parameters: | **new_state** ([_torch.ByteTensor_](tensors.html#torch.ByteTensor "torch.ByteTensor")) – The desired state |
| --- | --- |

```py
torch.cuda.manual_seed(seed)¶
```

Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

| Parameters: | **seed** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The desired seed. |
| --- | --- |

Warning

If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use [`manual_seed_all()`](#torch.cuda.manual_seed_all "torch.cuda.manual_seed_all").

```py
torch.cuda.manual_seed_all(seed)¶
```

Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

| Parameters: | **seed** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The desired seed. |
| --- | --- |

```py
torch.cuda.seed()¶
```

Sets the seed for generating random numbers to a random number for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

Warning

If you are working with a multi-GPU model, this function will only initialize the seed on one GPU. To initialize all GPUs, use [`seed_all()`](#torch.cuda.seed_all "torch.cuda.seed_all").

```py
torch.cuda.seed_all()¶
```

Sets the seed for generating random numbers to a random number on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

```py
torch.cuda.initial_seed()¶
```

Returns the current random seed of the current GPU.

Warning

This function eagerly initializes CUDA.

## Communication collectives

```py
torch.cuda.comm.broadcast(tensor, devices)¶
```

Broadcasts a tensor to a number of GPUs.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – tensor to broadcast.
*   **devices** (_Iterable_) – an iterable of devices among which to broadcast. Note that it should be like (src, dst1, dst2, …), the first element of which is the source device to broadcast from.

 |
| --- | --- |
| Returns: | A tuple containing copies of the `tensor`, placed on devices corresponding to indices from `devices`. |
| --- | --- |

```py
torch.cuda.comm.broadcast_coalesced(tensors, devices, buffer_size=10485760)¶
```

Broadcasts a sequence tensors to the specified GPUs. Small tensors are first coalesced into a buffer to reduce the number of synchronizations.

| Parameters: | 

*   **tensors** (_sequence_) – tensors to broadcast.
*   **devices** (_Iterable_) – an iterable of devices among which to broadcast. Note that it should be like (src, dst1, dst2, …), the first element of which is the source device to broadcast from.
*   **buffer_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – maximum size of the buffer used for coalescing

 |
| --- | --- |
| Returns: | A tuple containing copies of the `tensor`, placed on devices corresponding to indices from `devices`. |
| --- | --- |

```py
torch.cuda.comm.reduce_add(inputs, destination=None)¶
```

Sums tensors from multiple GPUs.

All inputs should have matching shapes.

| Parameters: | 

*   **inputs** (_Iterable__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – an iterable of tensors to add.
*   **destination** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – a device on which the output will be placed (default: current device).

 |
| --- | --- |
| Returns: | A tensor containing an elementwise sum of all inputs, placed on the `destination` device. |
| --- | --- |

```py
torch.cuda.comm.scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None)¶
```

Scatters tensor across multiple GPUs.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – tensor to scatter.
*   **devices** (_Iterable__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]_) – iterable of ints, specifying among which devices the tensor should be scattered.
*   **chunk_sizes** (_Iterable__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]__,_ _optional_) – sizes of chunks to be placed on each device. It should match `devices` in length and sum to `tensor.size(dim)`. If not specified, the tensor will be divided into equal chunks.
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – A dimension along which to chunk the tensor.

 |
| --- | --- |
| Returns: | A tuple containing chunks of the `tensor`, spread across given `devices`. |
| --- | --- |

```py
torch.cuda.comm.gather(tensors, dim=0, destination=None)¶
```

Gathers tensors from multiple GPUs.

Tensor sizes in all dimension different than `dim` have to match.

| Parameters: | 

*   **tensors** (_Iterable__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – iterable of tensors to gather.
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – a dimension along which the tensors will be concatenated.
*   **destination** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – output device (-1 means CPU, default: current device)

 |
| --- | --- |
| Returns: | A tensor located on `destination` device, that is a result of concatenating `tensors` along `dim`. |
| --- | --- |

## Streams and events

```py
class torch.cuda.Stream¶
```

Wrapper around a CUDA stream.

A CUDA stream is a linear sequence of execution that belongs to a specific device, independent from other streams. See [CUDA semantics](notes/cuda.html#cuda-semantics) for details.

| Parameters: | 

*   **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – a device on which to allocate the stream. If [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default) or a negative integer, this will use the current device.
*   **priority** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – priority of the stream. Lower numbers represent higher priorities.

 |
| --- | --- |

```py
query()¶
```

Checks if all the work submitted has been completed.

| Returns: | A boolean indicating if all kernels in this stream are completed. |
| --- | --- |

```py
record_event(event=None)¶
```

Records an event.

| Parameters: | **event** ([_Event_](#torch.cuda.Event "torch.cuda.Event")_,_ _optional_) – event to record. If not given, a new one will be allocated. |
| --- | --- |
| Returns: | Recorded event. |
| --- | --- |

```py
synchronize()¶
```

Wait for all the kernels in this stream to complete.

Note

This is a wrapper around `cudaStreamSynchronize()`: see [CUDA documentation](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) for more info.

```py
wait_event(event)¶
```

Makes all future work submitted to the stream wait for an event.

| Parameters: | **event** ([_Event_](#torch.cuda.Event "torch.cuda.Event")) – an event to wait for. |
| --- | --- |

Note

This is a wrapper around `cudaStreamWaitEvent()`: see [CUDA documentation](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) for more info.

This function returns without waiting for `event`: only future operations are affected.

```py
wait_stream(stream)¶
```

Synchronizes with another stream.

All future work submitted to this stream will wait until all kernels submitted to a given stream at the time of call complete.

| Parameters: | **stream** ([_Stream_](#torch.cuda.Stream "torch.cuda.Stream")) – a stream to synchronize. |
| --- | --- |

Note

This function returns without waiting for currently enqueued kernels in [`stream`](#torch.cuda.stream "torch.cuda.stream"): only future operations are affected.

```py
class torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False, _handle=None)¶
```

Wrapper around CUDA event.

| Parameters: | 

*   **enable_timing** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – indicates if the event should measure time (default: `False`)
*   **blocking** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – if `True`, [`wait()`](#torch.cuda.Event.wait "torch.cuda.Event.wait") will be blocking (default: `False`)
*   **interprocess** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – if `True`, the event can be shared between processes (default: `False`)

 |
| --- | --- |

```py
elapsed_time(end_event)¶
```

Returns the time elapsed before the event was recorded.

```py
ipc_handle()¶
```

Returns an IPC handle of this event.

```py
query()¶
```

Checks if the event has been recorded.

| Returns: | A boolean indicating if the event has been recorded. |
| --- | --- |

```py
record(stream=None)¶
```

Records the event in a given stream.

```py
synchronize()¶
```

Synchronizes with the event.

```py
wait(stream=None)¶
```

Makes a given stream wait for the event.

## Memory management

```py
torch.cuda.empty_cache()
```

Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in &lt;cite&gt;nvidia-smi&lt;/cite&gt;.

Note

[`empty_cache()`](#torch.cuda.empty_cache "torch.cuda.empty_cache") doesn’t increase the amount of GPU memory available for PyTorch. See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

```py
torch.cuda.memory_allocated(device=None)
```

Returns the current GPU memory usage by tensors in bytes for a given device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – selected device. Returns statistic for the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |

Note

This is likely less than the amount shown in &lt;cite&gt;nvidia-smi&lt;/cite&gt; since some unused memory can be held by the caching allocator and some context needs to be created on GPU. See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

```py
torch.cuda.max_memory_allocated(device=None)
```

Returns the maximum GPU memory usage by tensors in bytes for a given device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – selected device. Returns statistic for the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

```py
torch.cuda.memory_cached(device=None)
```

Returns the current GPU memory managed by the caching allocator in bytes for a given device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – selected device. Returns statistic for the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

```py
torch.cuda.max_memory_cached(device=None)
```

Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.

| Parameters: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – selected device. Returns statistic for the current device, given by [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device"), if [`device`](#torch.cuda.device "torch.cuda.device") is `None` (default). |
| --- | --- |

Note

See [Memory management](notes/cuda.html#cuda-memory-management) for more details about GPU memory management.

## NVIDIA Tools Extension (NVTX)

```py
torch.cuda.nvtx.mark(msg)¶
```

Describe an instantaneous event that occurred at some point.

| Parameters: | **msg** (_string_) – ASCII message to associate with the event. |
| --- | --- |

```py
torch.cuda.nvtx.range_push(msg)¶
```

Pushes a range onto a stack of nested range span. Returns zero-based depth of the range that is started.

| Parameters: | **msg** (_string_) – ASCII message to associate with range |
| --- | --- |

```py
torch.cuda.nvtx.range_pop()¶
```

Pops a range off of a stack of nested range spans. Returns the zero-based depth of the range that is ended.

