# （测试版）使用缩放点积注意力（SDPA）实现高性能Transformers [¶](#beta-implementing-high-performance-transformers-with-scaled-dot-product-attention- sdpa"此标题的永久链接")

> 译者：[liuenci](https://github.com/liuenci)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/scaled_dot_product_attention_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>

**作者**: [Driss Guessous](https://github.com/drisspg)

## 摘要 [¶](#summary "此标题的永久链接")
在本教程中，我们将介绍一个新的torch.nn.functional函数，它对于实现 Transformers 架构非常有帮助。这个函数名为torch.nn.functional.scaled_dot_product_attention。有关该函数的详细描述，请参阅[PyTorch 文档](https://pytorch.org/docs/master/generated/torch.nn.function.scaled_dot_product_attention.html#torch.nn.function.scaled_dot_product_attention) 。此函数已经被整合到torch.nn.MultiheadAttention和torch.nn.TransformerEncoderLayer中。

## 概述 [¶](#overview "此标题的永久链接")
从深层次来看，这个PyTorch函数根据论文《Attention is all you need》中的定义，计算查询（query）、键（key）和值（value）之间的缩放点积注意力（SDPA）。虽然这个函数可以使用现有的PyTorch函数编写，但一个融合实现（fused implementation）可以比朴素实现提供更大的性能优势。

## 融合实现 [¶](#fused-implementations "永久链接到此标题")

对于CUDA张量输入，该函数将分派到以下实现之一：

1. **FlashAttention**：这是一种快速且内存高效的精确注意力机制，具有IO感知能力。这种实现优化了计算速度，并考虑到输入/输出操作对性能的影响。
2. **内存高效注意力**：这种实现旨在减少在执行缩放点积注意力时所需的内存占用，这对于处理大型模型或长序列尤为重要。
3. **C++中定义的PyTorch实现**：这指的是在C++中编写的PyTorch函数实现，通常用于提高性能，因为C++编写的代码可以直接与底层硬件进行交互，从而优化计算效率。

本教程需要PyTorch 2.0.0或更高版本。

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

# Example Usage:
query, key, value = torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device)
F.scaled_dot_product_attention(query, key, value)
```

```py
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

## 显式调度器控制 [¶](#explicit-dispatcher-control "永久链接到此标题")

虽然该函数会隐式地分派到三种实现之一，但用户也可以通过使用上下文管理器（context manager）来显式控制分派。这个上下文管理器允许用户显式禁用某些实现。如果用户想确保函数确实针对他们的特定输入使用最快的实现，可以使用上下文管理器来遍历并测量性能。

```py
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
from torch.nn.attention import SDPBackend, sdpa_kernel


with sdpa_kernel(SDPBackend.MATH):
    math_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
    print(f"The math implementation runs in {math_time:.3f} microseconds")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        flash_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The flash attention implementation runs in {flash_time:.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        efficient_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")
```


```py
The default implementation runs in 2304.977 microseconds
The math implementation runs in 19249.369 microseconds
The flash attention implementation runs in 2304.600 microseconds
The memory efficient implementation runs in 4197.082 microseconds
```

## 硬件依赖性 [¶](#hardware-dependence "永久链接到此标题")

根据您在上面代码单元运行的机器以及可用的硬件，您得到的结果可能会有所不同：

- 如果您没有GPU并且是在CPU上运行，那么上下文管理器将不起作用，三次运行应该返回相似的时间。
- 根据您的显卡支持的计算能力，FlashAttention或内存高效注意力可能会失败。

## 因果自注意力[¶](#causal-self-attention "永久链接到此标题")
下面是一个因果自注意力（multi-headed causal self attention）块的示例实现，灵感来源于Andrej Karpathy的NanoGPT仓库。

```py
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


```py
CausalSelfAttention(
  (c_attn): Linear(in_features=512, out_features=1536, bias=False)
  (c_proj): Linear(in_features=512, out_features=512, bias=False)
  (resid_dropout): Dropout(p=0.1, inplace=False)
)
```

## NestedTensor 和 Dense 张量支持

SDPA支持NestedTensor和Dense张量输入。NestedTensors处理的情况是输入是一个不等长序列的批次，而无需将每个序列填充到批次中的最大长度。有关NestedTensors的更多信息，请参阅torch.nested和NestedTensors教程。

```py
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
            [
                torch.randn(seq_len, embed_dimension,
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

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        print(f"Random NT runs in {benchmark_torch_function_in_microseconds(model, random_nt):.3f} microseconds")
        print(f"Random Dense runs in {benchmark_torch_function_in_microseconds(model, random_dense):.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")
```


```py
/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nested/__init__.py:166: UserWarning:

The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)

Random NT runs in 558.517 microseconds
Random Dense runs in 936.630 microseconds
```

## 使用 torch.compile 与 SDPA [¶](#using-sdpa-with-torch-compile "永久链接到此标题")

随着PyTorch 2.0的发布，引入了一个名为torch.compile()的新特性，它可以在急切模式（eager mode）上提供显著性能提升。缩放点积注意力（SDPA）与torch.compile()完全兼容。为了演示这一点，我们将使用torch.compile()编译CausalSelfAttention模块，并观察由此带来的性能提升。

```py
batch_size = 32
max_sequence_len = 256
x = torch.rand(batch_size, max_sequence_len,
               embed_dimension, device=device, dtype=dtype)
print(
    f"The non compiled module runs in  {benchmark_torch_function_in_microseconds(model, x):.3f} microseconds")


compiled_model = torch.compile(model)
# Let's compile it
compiled_model(x)
print(
    f"The compiled module runs in  {benchmark_torch_function_in_microseconds(compiled_model, x):.3f} microseconds")
```


```py
The non compiled module runs in  408.207 microseconds
The compiled module runs in  516.612 microseconds
```

具体的执行时间取决于机器，但我的结果是：未编译的模块运行时间为166.616微秒，编译后的模块运行时间为166.726微秒。这并不是我们期望的结果。让我们深入探究一下。PyTorch内置了一个惊人的性能分析器（profiler），您可以使用它来检查代码的性能特征。

```py
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
#
# .. code-block:: python
#
#    prof.export_chrome_trace("compiled_causal_attention_trace.json").
```


```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                         Non-Compilied Causal Attention        20.01%       2.285ms        77.24%       8.821ms       8.821ms       0.000us         0.00%      11.098ms      11.098ms             1
                         Non-Compilied Causal Attention         0.00%       0.000us         0.00%       0.000us       0.000us      10.328ms        50.41%      10.328ms      10.328ms             1
                                           aten::matmul         2.36%     269.000us        27.28%       3.115ms      62.300us       0.000us         0.00%       8.156ms     163.120us            50
                                               aten::mm        18.72%       2.138ms        22.97%       2.623ms      52.460us       7.750ms        37.83%       8.156ms     163.120us            50
                                           aten::linear         1.62%     185.000us        30.99%       3.539ms      70.780us       0.000us         0.00%       8.068ms     161.360us            50
         ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn         0.00%       0.000us         0.00%       0.000us       0.000us       5.552ms        27.10%       5.552ms     222.080us            25
                     aten::scaled_dot_product_attention         1.97%     225.000us        17.75%       2.027ms      81.080us       0.000us         0.00%       2.942ms     117.680us            25
              aten::_scaled_dot_product_flash_attention         3.38%     386.000us        15.78%       1.802ms      72.080us       0.000us         0.00%       2.942ms     117.680us            25
                         aten::_flash_attention_forward         4.45%     508.000us        11.48%       1.311ms      52.440us       2.411ms        11.77%       2.942ms     117.680us            25
void pytorch_flash::flash_fwd_kernel<pytorch_flash::...         0.00%       0.000us         0.00%       0.000us       0.000us       2.411ms        11.77%       2.411ms      96.440us            25
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 11.420ms
Self CUDA time total: 20.489ms

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                              Compiled Causal Attention         6.44%     748.000us        90.99%      10.575ms      10.575ms       0.000us         0.00%      10.978ms      10.978ms             1
                                  Torch-Compiled Region        10.49%       1.219ms        83.33%       9.685ms     387.400us       0.000us         0.00%      10.978ms     439.120us            25
                                       CompiledFunction        43.24%       5.025ms        71.65%       8.327ms     333.080us       0.000us         0.00%      10.978ms     439.120us            25
                              Compiled Causal Attention         0.00%       0.000us         0.00%       0.000us       0.000us      10.359ms        50.50%      10.359ms      10.359ms             1
                                               aten::mm         8.22%     955.000us        12.70%       1.476ms      29.520us       7.751ms        37.78%       8.159ms     163.180us            50
         ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn         0.00%       0.000us         0.00%       0.000us       0.000us       5.553ms        27.07%       5.553ms     222.120us            25
              aten::_scaled_dot_product_flash_attention         2.41%     280.000us        14.79%       1.719ms      68.760us       0.000us         0.00%       2.819ms     112.760us            25
                         aten::_flash_attention_forward         4.48%     521.000us        11.07%       1.287ms      51.480us       2.404ms        11.72%       2.819ms     112.760us            25
void pytorch_flash::flash_fwd_kernel<pytorch_flash::...         0.00%       0.000us         0.00%       0.000us       0.000us       2.404ms        11.72%       2.404ms      96.160us            25
ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_3...         0.00%       0.000us         0.00%       0.000us       0.000us       2.198ms        10.71%       2.198ms      87.920us            25
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 11.622ms
Self CUDA time total: 20.514ms
```

之前的代码片段生成了编译和未编译模块在GPU执行时间上消耗最多的前10个PyTorch函数的报告。分析显示，对于两个模块，在GPU上花费的大部分时间集中在同一组函数上。这里的原因是torch.compile非常擅长消除与PyTorch相关的高级框架开销。如果您的模型正在启动大型、高效的CUDA内核，正如本例中的CausalSelfAttention，那么PyTorch的开销可以被隐藏。
实际上，您的模块通常不仅仅包含一个CausalSelfAttention块。在尝试Andrej Karpathy的NanoGPT仓库时，编译模块将每步训练时间从6090.49毫秒减少到3273.17毫秒！这是在NanoGPT的Shakespeare数据集训练的ae3a8d5提交上完成的。

## 使用 SDPA 与 attn_bias 子类 [¶](#using-sdpa-with-attn-bias-subclass "永久链接到此标题")

截至PyTorch 2.3版本，我们增加了一个新的子模块，其中包含了张量的子类。这些子类被设计用于与torch.nn.functional.scaled_dot_product_attention一起使用。该模块名为torch.nn.attention.bias，并包含以下两个用于生成因果注意力变体的工具：

* torch.nn.attention.bias.causal_upper_left
* torch.nn.attention.bias.causal_lower_right


```py
The current argument is_causal in torch.nn.functional.scaled_dot_product_attention is the same as using torch.nn.attention.bias.causal_upper_left.
```

```py
from torch.nn.attention.bias import causal_lower_right, causal_upper_left

batch_size = 32
sequence_length_q = 2
sequence_length_kv = 10
num_heads = 16
embed_dimension = 32

dtype = torch.float16

query = torch.rand(batch_size, num_heads, sequence_length_q, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, sequence_length_kv, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, sequence_length_kv, embed_dimension, device=device, dtype=dtype)

upper_left_bias = causal_upper_left(sequence_length_q, sequence_length_kv)
lower_right_bias = causal_lower_right(sequence_length_q, sequence_length_kv)

print(type(upper_left_bias))
print(type(lower_right_bias))

assert type(upper_left_bias) == type(lower_right_bias)
assert issubclass(type(upper_left_bias), torch.Tensor)

# As you can see from the previous output, are the same type ``torch.nn.attention.bias.CausalBias``
# and subclass ``torch.Tensor``

# Lets see what these tensors look like
print(upper_left_bias)
print(lower_right_bias)

# Upper Left Bias aligns the causal attention mask to the upper left corner of the attention scores matrix.
# This only has an impact when the attention scores matrix is not square, which is common for decoding use cases.
# Another way of thinking about this concept is that when you use upper left bias,
# the 0th token in the query is aligned to the 0th token in the key, while for lower right bias,
# Assuming the attention score matrix is two dimensional, ``attn_score[0][0]`` is the attention score
# between the 0th token in the query and the 0th token in the key.
# For lower right bias, the sequence of q is aligned so that the last token in q is aligned to the last token in k
# (for example, ``attn_score[-1][-1])`` is all True since the last token in q is at the same position as the last token in k
# even if the sequence length of q and k are different.

# These objects are intended to be used with sdpa
out_upper_left = F.scaled_dot_product_attention(query, key, value, upper_left_bias)
out_lower_right = F.scaled_dot_product_attention(query, key, value, lower_right_bias)
out_is_causal = F.scaled_dot_product_attention(query, key, value, is_causal=True)

assert torch.allclose(out_upper_left, out_is_causal)
assert not torch.allclose(out_upper_left, out_lower_right)

# These attention biases should also be compatible with torch.compile
compiled_sdpa = torch.compile(F.scaled_dot_product_attention, fullgraph=True)
out_upper_left = compiled_sdpa(query, key, value, upper_left_bias)
```


```py
<class 'torch.nn.attention.bias.CausalBias'>
<class 'torch.nn.attention.bias.CausalBias'>
tensor([[ True, False, False, False, False, False, False, False, False, False],
        [ True,  True, False, False, False, False, False, False, False, False]])
tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]])
```

## 结论

在本教程中，我们演示了torch.nn.functional.scaled_dot_product_attention的基本用法。我们展示了如何使用sdpa_kernel上下文管理器来确保在GPU上使用特定的实现。此外，我们还构建了一个简单的CausalSelfAttention模块，该模块与NestedTensor兼容，并且可以被torch编译。在这个过程中，我们还展示了如何使用性能分析工具来探索用户定义模块的性能特征。

脚本总运行时间：（0分钟7.894秒）