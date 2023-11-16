
（测试版）通过缩放点积注意力 (SDPA) 实现高性能 Transformer
 [¶](#beta-implementing-high-performance-transformers-with-scaled-dot-product-attention- sdpa“此标题的永久链接”）
==========================================================================================================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/scaled_dot_product_attention_tutorial#conclusion>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial#conclusion.html>




**作者：** 
[Driss Guessous](https://github.com/drisspg)





 摘要
 [¶](#summary "此标题的永久链接")
----------------------------------------------------------------



 在本教程中，我们想要重点介绍一个新的 
 `torch.nn.function`
 函数，它有助于实现 Transformer 架构。该函数名为
 `torch.nn.function.scaled_dot_product_attention`
 。
有关该函数的详细说明，请参阅
 [PyTorch 文档](https://pytorch.org/docs/master/generated/torch.nn.function.scaled_dot_product_attention.html#torch.nn.function.scaled_dot_product_attention) 
.
此函数已合并到
 `torch.nn.MultiheadAttention`
 和
 `torch.nn.TransformerEncoderLayer`
.






 概述
 [¶](#overview "此标题的永久链接")
------------------------------------------------------



 在较高层面上，此 PyTorch 函数根据
论文中的定义计算查询、键和值之间的
缩放点积注意力 (SDPA)
 [注意力就是您所需要的](https://arxiv.org/abs/1706.03762) 
 。虽然可以使用现有函数在 PyTorch 中编写此函数，但融合实现可以比原始实现提供更大的性能优势。






 融合实现
 [¶](#fused-implementations "永久链接到此标题")
--------------------------------------------------------------------------------



 对于 CUDA 张量输入，该函数将分派到以下实现之一
:



* [FlashAttention：具有 IO 感知的快速、内存高效的精确注意力](https://arxiv.org/abs/2205.14135)
* [内存高效的注意力](https://github.com/facebookresearch/xformers )
* 用 C++ 定义的 PyTorch 实现




 注意




 本教程需要 PyTorch 2.0.0 或更高版本。







```
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

# Example Usage:
query, key, value = torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device)
F.scaled_dot_product_attention(query, key, value)

```






```
tensor([[[-1.3321, -0.3489,  0.3015, -0.3912,  0.9867,  0.3137, -0.0691,
          -1.2593],
         [-1.0882,  0.2506,  0.6491,  0.1360,  0.5238, -0.2448, -0.0820,
          -0.6171],
         [-1.0012,  0.3990,  0.6441, -0.0277,  0.5325, -0.2564, -0.0607,
          -0.6404]],

        [[ 0.6091,  0.0708,  0.6188,  0.3252, -0.1598,  0.4197, -0.2335,
           0.0630],
         [ 0.5285,  0.3890, -0.2649,  0.3706, -0.3839,  0.1963, -0.6242,
           0.2312],
         [ 0.4048,  0.0762,  0.3777,  0.4689, -0.2978,  0.2754, -0.6429,
           0.1037]]], device='cuda:0')

```






 显式调度程序控制
 [¶](#explicit-dispatcher-control "永久链接到此标题")
------------------------------------------------------------------------------------------------



 虽然该函数将隐式分派到三个
实现之一，但用户还可以通过使用上下文管理器
显式控制分派。此上下文管理器允许用户
显式禁用某些实现。如果用户想要确保
该函数确实对其特定输入使用
最快的实现，
可以使用上下文管理器来扫描
测量性能。






```
# Lets define a helpful benchmarking function:
import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# Lets define the hyper-parameters of our input
batch_size = 32
max_sequence_len = 1024
num_heads = 32
embed_dimension = 32

dtype = torch.float16

query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# Lets explore the speed of each of the 3 implementations
from torch.backends.cuda import sdp_kernel, SDPBackend

# Helpful arguments mapper
backend_map = {
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}

with sdp_kernel(**backend_map[SDPBackend.MATH]):
    print(f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")


with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
    try:
        print(f"The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
    try:
        print(f"The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")

```






```
The default implementation runs in 4741.745 microseconds
The math implementation runs in 19249.446 microseconds
The flash attention implementation runs in 4741.583 microseconds
The memory efficient implementation runs in 4193.383 microseconds

```






 硬件依赖性
 [¶](#hardware-dependence "永久链接到此标题")
------------------------------------------------------------------------------



 根据您运行上述单元的机器以及可用的硬件，您的结果可能会有所不同。
- 如果您没有’ 没有 GPU 并且在 CPU 上运行，则上下文管理器\ n 将没有任何效果，并且所有三个运行都应返回相似的计时。
- 取决于您的显卡支持的计算能力
闪存关注或内存效率可能会失败。






 因果自注意力
 [¶](#causal-self-attention "永久链接到此标题")
------------------------------------------------------------------------------------------------



 下面是一个多头因果自我注意力块的示例实现，灵感来自于
 [Andrej Karpathy NanoGPT](https://github.com/karpathy/nanoGPT)
 存储库。






```
class CausalSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y


num_heads = 8
heads_per_dim = 64
embed_dimension = num_heads * heads_per_dim
dtype = torch.float16
model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to("cuda").to(dtype).eval()
print(model)

```






```
CausalSelfAttention(
  (c_attn): Linear(in_features=512, out_features=1536, bias=False)
  (c_proj): Linear(in_features=512, out_features=512, bias=False)
  (resid_dropout): Dropout(p=0.1, inplace=False)
)

```




### 
`NestedTensor`
 和密集张量支持
 [¶](#nestedtensor-and-dense-tensor-support "永久链接到此标题")



 SDPA 支持
 `NestedTensor`
 和密集张量输入。
 `NestedTensor`
 处理输入是一批可变长度序列的情况
无需将每个序列填充到最大长度批。有关 
 `NestedTensors` 的更多信息，请参阅
 [torch.nested](https://pytorch.org/docs/stable/nested.html) 
 和
 [NestedTensors 教程](https://pytorch.org/tutorials/prototype/nestedtensor.html) 
.






```
import random
def generate_rand_batch(
    batch_size,
    max_sequence_len,
    embed_dimension,
    pad_percentage=None,
    dtype=torch.float16,
    device="cuda",
):
    if not pad_percentage:
        return (
            torch.randn(
                batch_size,
                max_sequence_len,
                embed_dimension,
                dtype=dtype,
                device=device,
            ),
            None,
        )
    # Random sequence lengths
    seq_len_list = [
        int(max_sequence_len * (1 - random.gauss(pad_percentage, 0.01)))
        for _ in range(batch_size)
    ]
    # Make random entry in the batch have max sequence length
    seq_len_list[random.randint(0, batch_size - 1)] = max_sequence_len
    return (
        torch.nested.nested_tensor(
            
                [torch.randn(seq_len, embed_dimension,
                            dtype=dtype, device=device)
                for seq_len in seq_len_list
            ]
        ),
        seq_len_list,
    )

random_nt, _ = generate_rand_batch(32, 512, embed_dimension, pad_percentage=0.5, dtype=dtype, device=device)
random_dense, _ = generate_rand_batch(32, 512, embed_dimension, pad_percentage=None, dtype=dtype, device=device)

# Currently the fused implementations don't support ``NestedTensor`` for training
model.eval()

with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
    try:
        print(f"Random NT runs in {benchmark_torch_function_in_microseconds(model, random_nt):.3f} microseconds")
        print(f"Random Dense runs in {benchmark_torch_function_in_microseconds(model, random_dense):.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

```






```
/var/lib/jenkins/workspace/intermediate_source/scaled_dot_product_attention_tutorial.py:226: UserWarning:

The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)

Random NT runs in 679.281 microseconds
Random Dense runs in 1183.933 microseconds

```








 使用 SDPA 与
 `torch.compile`
[¶](#using-sdpa-with-torch-compile "永久链接到此标题")
================================================================================================


 随着 PyTorch 2.0 的发布，引入了一项名为
 `torch.compile()`
 的新功能，与 eager 模式相比
它可以提供
显着的性能改进。
缩放点积注意力完全可以与
组合`torch.compile()`
 。
为了演示这一点，让’s 使用
 `CausalSelfAttention`
 模块编译
 `torch.compile()`
 并观察由此产生的性能改进.






```
batch_size = 32
max_sequence_len = 256
x = torch.rand(batch_size, max_sequence_len,
               embed_dimension, device=device, dtype=dtype)
print(
    f"The non compiled module runs in {benchmark_torch_function_in_microseconds(model, x):.3f} microseconds")


compiled_model = torch.compile(model)
# Let's compile it
compiled_model(x)
print(
    f"The compiled module runs in {benchmark_torch_function_in_microseconds(compiled_model, x):.3f} microseconds")

```






```
The non compiled module runs in  416.696 microseconds
The compiled module runs in  453.513 microseconds

```




 确切的执行时间取决于机器，但是我的结果：
未编译的模块在 166.616 微秒内运行
编译的模块在 166.726 微秒内运行
这不是我们所期望的。让’s 更深入地挖掘一下。
PyTorch 附带了一个令人惊叹的内置分析器，您可以使用它
检查代码的性能特征。






```
from torch.profiler import profile, record_function, ProfilerActivity
activities = [ProfilerActivity.CPU]
if device == 'cuda':
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=False) as prof:
    with record_function(" Non-Compilied Causal Attention"):
        for _ in range(25):
            model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


with profile(activities=activities, record_shapes=False) as prof:
    with record_function("Compiled Causal Attention"):
        for _ in range(25):
            compiled_model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# For even more insights, you can export the trace and use ``chrome://tracing`` to view the results
# ::
#
# prof.export_chrome_trace("compiled_causal_attention_trace.json").

```






```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                         Non-Compilied Causal Attention        16.91%       1.981ms        70.42%       8.250ms       8.250ms       0.000us         0.00%      11.013ms      11.013ms             1
                                           aten::matmul         2.48%     291.000us        26.92%       3.154ms      63.080us       0.000us         0.00%       8.378ms     167.560us            50
                                               aten::mm        18.89%       2.213ms        22.68%       2.657ms      53.140us       7.743ms        74.61%       8.378ms     167.560us            50
                                           aten::linear         2.50%     293.000us        30.21%       3.539ms      70.780us       0.000us         0.00%       7.893ms     157.860us            50
         ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn         0.00%       0.000us         0.00%       0.000us       0.000us       5.550ms        53.48%       5.550ms     222.000us            25
                     aten::scaled_dot_product_attention         1.85%     217.000us        14.66%       1.718ms      68.720us       0.000us         0.00%       2.635ms     105.400us            25
          aten::_scaled_dot_product_efficient_attention         3.61%     423.000us        12.81%       1.501ms      60.040us       0.000us         0.00%       2.635ms     105.400us            25
                     aten::_efficient_attention_forward         3.36%     394.000us         8.33%     976.000us      39.040us       2.635ms        25.39%       2.635ms     105.400us            25
fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us       2.635ms        25.39%       2.635ms     105.400us            25
ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us       2.193ms        21.13%       2.193ms      87.720us            25
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 11.715ms
Self CUDA time total: 10.378ms

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                              Compiled Causal Attention        14.58%       1.889ms        90.02%      11.660ms      11.660ms       0.000us         0.00%      12.187ms      12.187ms             1
                                       CompiledFunction        37.96%       4.916ms        66.21%       8.575ms     343.000us       0.000us         0.00%      12.187ms     487.480us            25
                                               aten::mm         6.82%     883.000us        10.76%       1.393ms      27.860us       7.767ms        68.85%       8.306ms     166.120us            50
         ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn         0.00%       0.000us         0.00%       0.000us       0.000us       5.572ms        49.39%       5.572ms     222.880us            25
          aten::_scaled_dot_product_efficient_attention         2.01%     260.000us        10.57%       1.369ms      54.760us       0.000us         0.00%       2.867ms     114.680us            25
                     aten::_efficient_attention_forward         3.08%     399.000us         7.42%     961.000us      38.440us       2.639ms        23.39%       2.867ms     114.680us            25
fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMemEf...         0.00%       0.000us         0.00%       0.000us       0.000us       2.639ms        23.39%       2.639ms     105.560us            25
ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us       2.195ms        19.46%       2.195ms      87.800us            25
                               triton_poi_fused_clone_0         2.84%     368.000us         3.92%     508.000us      20.320us     875.000us         7.76%       1.014ms      40.560us            25
                                          triton__0d1de         0.00%       0.000us         0.00%       0.000us       0.000us     875.000us         7.76%     875.000us      35.000us            25
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 12.952ms
Self CUDA time total: 11.281ms

```




 前面的代码片段生成了编译模块和非编译模块中消耗最多 GPU 执行时间的前 10 个 PyTorch 函数的报告。
分析表明，花费在 GPU 上的大部分时间是两个模块集中
相同的函数集。
原因是
 `torch.compile`
非常擅长消除
与 PyTorch 相关的框架开销。如果您的模型正在启动大型、高效的 CUDA 内核（在本例中就是“CausalSelfAttention”），则可以隐藏 PyTorch 的开销。




 实际上，您的模块通常不包含单个
 `CausalSelfAttention`
 块。在使用 [Andrej Karpathy NanoGPT](https://github.com/karpathy/nanoGPT) 存储库进行实验时，编译
模块每个训练步骤的时间从：
 `6090.49ms`
 到
 `3273.17ms`
！这是在 Shakespeare 数据集上的 NanoGPT 训练提交时完成的：
 `ae3a8d5`
。






 结论
 [¶](#conclusion "永久链接到此标题")
=======================================================


 在本教程中，我们演示了 
 `torch.nn.function.scaled_dot_product_attention`
 的基本用法。我们已经展示了如何使用
`sdp_kernel`
 上下文管理器来断言在 GPU 上使用了某个
实现。此外，我们还构建了一个简单的“CausalSelfAttention”模块，该模块可与“NestedTensor”配合使用，并且可进行 torch 编译。在此过程中，我们展示了如何使用分析工具
来探索用户定义
模块的性能特征。




**脚本的总运行时间:** 
 ( 0 分 8.239 秒)
