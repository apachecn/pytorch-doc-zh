# HIP (ROCm) 语义 [¶](#hip-rocm-semantics "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/hip>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/hip.html>


 ROCm™ 是 AMD 的开源软件平台，用于 GPU 加速的高性能计算和机器学习。 HIP 是 ROCm 的 C++ 方言，旨在轻松将 CUDA 应用程序转换为可移植的 C++ 代码。 HIP 用于将 PyTorch 等现有 CUDA 应用程序转换为可移植 C++ 以及需要 AMD 和 NVIDIA 之间可移植性的新项目。


## HIP 接口重用 CUDA 接口 [¶](#hip-interfaces-reuse-the-cuda-interfaces“此标题的永久链接”)


 PyTorch for HIP 有意重用现有的 [`torch.cuda`](../cuda.html#module-torch.cuda "torch.cuda") 接口。这有助于加速现有 PyTorch 代码和模型的移植，因为代码很少如果有的话，改变是必要的。


 [CUDA 语义](cuda.html#cuda-semantics) 中的示例对于 HIP 的工作方式完全相同：


```
cuda = torch.device('cuda')     # Default HIP device
cuda0 = torch.device('cuda:0')  # 'rocm' or 'hip' are not valid, use 'cuda'
cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

x = torch.tensor([1., 2.], device=cuda0)
# x.device is device(type='cuda', index=0)
y = torch.tensor([1., 2.]).cuda()
# y.device is device(type='cuda', index=0)

with torch.cuda.device(1):
    # allocates a tensor on GPU 1
    a = torch.tensor([1., 2.], device=cuda)

    # transfers a tensor from CPU to GPU 1
    b = torch.tensor([1., 2.]).cuda()
    # a.device and b.device are device(type='cuda', index=1)

    # You can also use ``Tensor.to`` to transfer a tensor:
    b2 = torch.tensor([1., 2.]).to(device=cuda)
    # b.device and b2.device are device(type='cuda', index=1)

    c = a + b
    # c.device is device(type='cuda', index=1)

    z = x + y
    # z.device is device(type='cuda', index=0)

    # even within a context, you can specify the device
    # (or give a GPU index to the .cuda call)
    d = torch.randn(2, device=cuda2)
    e = torch.randn(2).to(cuda2)
    f = torch.randn(2).cuda(cuda2)
    # d.device, e.device, and f.device are all device(type='cuda', index=2)

```


## 检查 HIP [¶](#checking-for-hip "此标题的永久链接")


 无论您使用 PyTorch 进行 CUDA 还是 HIP，调用 [`is_available()`](../generated/torch.cuda.is_available.html#torch.cuda.is_available "torch.cuda.is_available") 的结果会是一样的。如果您使用的是带有 GPU 支持的 PyTorch，它将返回 True 。如果您必须检查您正在使用的 PyTorch 版本，请参阅下面的示例：


```
if torch.cuda.is_available() and torch.version.hip:
    # do something specific for HIP
elif torch.cuda.is_available() and torch.version.cuda:
    # do something specific for CUDA

```


## ROCm 上的 TensorFloat-32(TF32) [¶](#tensorfloat-32-tf32-on-rocm“此标题的永久链接”)


 ROCm 不支持 TF32。


## 内存管理 [¶](#memory-management "此标题的永久链接")


 PyTorch 使用缓存内存分配器来加速内存分配。这允许快速内存释放而无需设备同步。但是，分配器管理的未使用内存仍将显示为在“rocm-smi”中使用。您可以使用 [`memory_allocated()`](../generated/torch.cuda.memory_allocated.html#torch.cuda.memory_alulated "torch.cuda.memory_alulated") 和 [`max_memory_allocated()`](../generated/torch.cuda.max_memory_alulated.html#torch.cuda.max_memory_alulated "torch.cuda.max_memory_alulated") 监视tensor占用的内存，并使用 [`memory_reserved()`](../generated/torch.cuda.memory_reserved.html#torch.cuda.memory_reserved "torch.cuda.memory_reserved") 和 [`max_memory_reserved()`](../generated/torch.cuda.max_memory_reserved.html#torch.cuda.max_memory_reserved "torch.cuda.max_memory_reserved") 来监视缓存分配器管理的内存总量。调用 [`empty_cache()`](../generated/torch.cuda.empty_cache.html#torch.cuda.empty_cache "torch.cuda.empty_cache") 释放 PyTorch 中所有**未使用的**缓存内存，以便这些可以被其他 GPU 应用程序使用。但是，tensor占用的 GPU 内存不会被释放，因此无法增加 PyTorch 可用的 GPU 内存量。


 对于更高级的用户，我们通过 [`memory_stats()`](../generated/torch.cuda.memory_stats.html#torch.cuda.memory_stats "torch.cuda.memory_stats") 提供更全面的内存基准测试。 我们还提供通过 [`memory_snapshot()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_snapshot.html#torch.cuda.memory_snapshot "torch.cuda.memory_snapshot") 捕获内存分配器状态的完整快照的功能，这可以帮助您了解代码生成的底层分配模式。


 要调试内存错误，请在您的环境中设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 以禁用缓存。


## hipFFT/rocFFT 计划缓存 [¶](#hipfft-rocft-plan-cache "此标题的永久链接") 


 不支持为 hipFFT/rocFFT 计划设置缓存大小。


## torch.distributed backends [¶](#torch-distributed-backends "此标题的永久链接")


 目前，ROCm 仅支持 torch.distributed 的“nccl”和“gloo”后端。


## C++ 中的 CUDA API 到 HIP API 映射 [¶](#cuda-api-to-hip-api-mappings-in-c "永久链接到此标题")


 请参考：<https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP_API_Guide.html>


 注意：CUDA_VERSION 宏、cudaRuntimeGetVersion 和 cudaDriverGetVersion API 在语义上不会映射到与 HIP_VERSION 宏、hipRuntimeGetVersion 和 hipDriverGetVersion API 相同的值。在进行版本检查时请不要互换使用它们。


 例如：而不是使用


`#if Defined(CUDA_VERSION) && CUDA_VERSION >= 11000` 隐式排除 ROCm/HIP，


 使用以下命令不采用 ROCm/HIP 的代码路径：


`#if 已定义(CUDA_VERSION) && CUDA_VERSION >= 11000 && !已定义(USE_ROCM)`


 或者，如果需要采用 ROCm/HIP 的代码路径：


`#if (已定义(CUDA_VERSION) && CUDA_VERSION >= 11000) ||定义(USE_ROCM)`


 或者，如果希望仅针对特定 HIP 版本采用 ROCm/HIP 的代码路径：


`#if (已定义(CUDA_VERSION) && CUDA_VERSION >= 11000) || (定义(USE_ROCM) && ROCM_VERSION >= 40300)`


## 请参阅 CUDA 语义文档 [¶](#refer-to-cuda-semantics-doc "此标题的永久链接")


 对于此处未列出的任何部分，请参阅 CUDA 语义文档：[CUDA 语义](cuda.html#cuda-semantics)


## 启用内核断言 [¶](#enabling-kernel-asserts "此标题的永久链接")


 ROCm 支持内核断言，但由于性能开销而被禁用。它可以通过从源代码重新编译 PyTorch 来启用。


 请将以下行添加为 cmake 命令参数的参数：


```
-DROCM_FORCE_ENABLE_GPU_ASSERTS:BOOL=ON

```