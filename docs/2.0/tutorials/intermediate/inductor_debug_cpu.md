# 电感器 CPU 后端调试和分析 [¶](#inductor-cpu-backend-debugging-and-profiling "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/inductor_debug_cpu>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/inductor_debug_cpu.html>




**Authors** 
 :
 [Xuan Liao](https://github.com/Valentine233) 
 ,
 [Haozhe Zhu](https://github.com/zhuhaozhe) 
 ,
 [Jiong Gong](https://github.com/jgong5) 
 ,
 [Weihan Wang](https://github.com/EikanWang)





## 概述 [¶](#overview "此标题的永久链接")




 PyTorch 2.0 引入了名为
 `torch.compile`
 的编译 API。
这一新功能通过由默认 Inductor 后端支持的图形级优化，显着提升了急切模式的执行速度。




 本教程旨在通过深入研究 
 `torch.compile` 的复杂性，深入介绍 Inductor CPU 后端的调试
和性能分析。
 。




 同时，你还可以在[基本用法](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)周围找到
 `torch.compile`的相关教程

 、
综合\ n [故障排除](https://pytorch.org/docs/stable/dynamo/troubleshooting.html) 
 和 GPU 特定知识，例如
 [GPU 性能分析](https://github.com/pytorch/pytorch /blob/main/docs/source/compile/profiling_torch_compile.rst) 
.




 我们将从一个触发编译问题和准确性问题的示例开始调试，
通过演示调试过程来查明问题。




 通过启用日志记录并探索底层生成的代码，
您可以了解如何逐步缩小失败范围并最终找出路由原因。




 接下来，我们将继续讨论如何分析编译后的代码，
通过与 eager 模式的性能比较，
详细说明为什么
 `torch.compile`
 与它的热切对手。





## 调试 [¶](#debugging "此标题的永久链接")




 这是一个使用 Inductor 运行
 `torch.compile`
 并将其结果与 eager 模式进行比较的简单示例：






```
import torch

def foo1(x1, x2):
    a = torch.neg(x1)
    b = torch.maximum(x2, a)
    y = torch.cat([b], dim=0)
    return y

x1 = torch.randint(256, (1, 8), dtype=torch.uint8)
x2 = torch.randint(256, (8390, 8), dtype=torch.uint8)

compiled_foo1 = torch.compile(foo1)
result = compiled_foo1(x1, x2)

```




 `cpp`
 codegen 中 
 `neg`
 的正确实现如下：






```
def neg1(x):
    return f"decltype({x})(-{x})"

```




 为了演示调试，稍后我们将函数修改为错误的。




### 获取更多日志记录信息 [¶](#get-more-logging-information "永久链接到此标题")



 如果默认运行这个简单的示例，则不会提供调试信息。为了获得更有用的调试和日志信息，我们通常添加一个
 `TORCH_COMPILE_DEBUG`
 环境变量，如下所示：






```
TORCH_COMPILE_DEBUG=1 python xx.py

```




 这将在输出日志中打印更多调试信息，并转储在代码生成过程中生成的中间 IR。您可以在日志中找到转储的文件路径，如下所示：






```
torch._inductor.debug: [WARNING] model___20 debug trace: /tmp/torchinductor_root/rx/crxfi2ybd7yp5sbj2pnhw33wfhtdw7wumvrobyp5sjvdui5ktjc2.debug

```




 在此目录中，保存以下文件用于调试目的:








| 
 文件
 | 
 描述
 |
| --- | --- |
| 
`fx_graph_runnable.py`
 | 
 可执行 FX 图表，分解后、模式匹配前
 |
| 
`fx_graph_transformed.py`
 | 
 模式匹配后转换后的 FX 图表
 |
| 
`ir_post_fusion.txt`
 | 
 融合前的电感器 IR
 |
| 
`ir_pre_fusion.txt`
 | 
 融合后的电感 IR
 |
| 
`输出_code.py`
 | 
 使用 C++/Triton 内核生成图形的 Python 代码
 |



 请注意，
 `fx_graph_runnable.py`
 和
 `output_code.py`
 都是可运行和可编辑的，以便于调试。
以下是主要部分从文件中提取的代码行，我们将 C++ 生成的行与 FX 代码行相关联。




`fx_graph_runnable`
 :






```
def forward1(self, arg0_1, arg1_1):
    neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
    maximum = torch.ops.aten.maximum.default(arg1_1, neg);  arg1_1 = neg = None
    clone = torch.ops.aten.clone.default(maximum);  maximum = None
    return (clone,)

```




 `output_code` 中的 C++ 内核
 :






```
from torch._inductor.codecache import AsyncCompile
async_compile = AsyncCompile()

cpp_fused_cat_maximum_neg_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
extern "C" void kernel(const unsigned char* in_ptr0,
 const unsigned char* in_ptr1,
 unsigned char* out_ptr0)
{
 {
 #pragma GCC ivdep
 for(long i0=static_cast<long>(0L); i0<static_cast<long>(8390L); i0+=static_cast<long>(1L))
 {
 #pragma GCC ivdep
 for(long i1=static_cast<long>(0L); i1<static_cast<long>(8L); i1+=static_cast<long>(1L))
 {
 auto tmp0 = in_ptr0[static_cast<long>(i1 + (8L*i0))];
 auto tmp1 = in_ptr1[static_cast<long>(i1)];
 // Corresponding FX code line: neg = torch.ops.aten.neg.default(arg0_1); arg0_1 = None
 auto tmp2 = decltype(tmp1)(-tmp1);
 // Corresponding FX code line: maximum = torch.ops.aten.maximum.default(arg1_1, neg); arg1_1 = neg = None
 auto tmp3 = max_propagate_nan(tmp0, tmp2);
 // Corresponding FX code line: clone = torch.ops.aten.clone.default(maximum); maximum = None
 out_ptr0[static_cast<long>(i1 + (8L*i0))] = tmp3;
 }
 }
 }
}''')

```





### 确定错误的组成部分 [¶](#define-component-of-error "永久链接到此标题")



 当遇到错误或准确性问题时，找到错误的一个直接解决方案是缩小问题范围。首先要做的是确定发生错误的组件。幸运的是，它可以通过更改
 `torch.compile`
 的后端来简单地实现。








| 
 代码
 | 
 描述
 |
| --- | --- |
| 
`torch.compile(fn,
 

 backend="eager")`
 | 
 启用 Dynamo
 |
| 
`torch.compile(fn,
 

 backend="aot_eager")`
 | 
 启用 Dynamo + AOT Autograd
 |
| 
`torch.compile(fn,
 

 backend="电感器")`
 | 
 启用 Dynamo + AOT Autograd + 电感器
 |



 如果后端设置为
 `eager`
 或
 `aot_eager`
 时模型可以成功运行，而模型失败时
 `inductor`
 ，我们可以缩小故障范围到电感器。





### 编译错误 [¶](#compilation-error "永久链接到此标题")



 众所周知，图级优化的演化链是这样的：






```
torch.neg (Python) -> torch.ops.aten.neg.default (within FX graph) -> ops.neg (within IR node) -> tmp2 = -tmp1 (within C++ kernel)

```




 如果遇到编译错误，则说明输出代码中编译 C++ 内核时出现问题。
此类错误表明在降低 IR 节点输出代码时引入了 bug。
编译错误的根本原因是通常显示在回溯日志中。




 例如，
 `neg`
 函数修改如下：






```
def neg2(x):
    return f"-{x}"

```




 日志记录给出了以下编译错误，并且原因相当明确。






```
 torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
 CppCompileError: C++ compile error
 /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp: In function ‘void kernel(const unsigned char*, const unsigned char*, unsigned char*)’:
 /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp:17:57: error: no matching function for call to ‘max_propagate_nan(unsigned char&, int&)’
   17 |                 auto tmp3 = max_propagate_nan(tmp0, tmp2);
        |                                                         ^
 In file included from /tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp:2:
 /tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note: candidate: ‘template<class scalar_t> scalar_t max_propagate_nan(scalar_t, scalar_t)’
 27 | inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
      |                 ^~~~~~~~~~~~~~~~~
 /tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h:27:17: note:   template argument deduction/substitution failed:
/tmp/torchinductor_root/xg/cxga5tk3b4lkwoxyigrtocjp5s7vc5cg2ikuscf6bk6pjqip2bhx.cpp:17:57: note:   deduced conflicting types for parameter ‘scalar_t’ (‘unsigned char’ and ‘int’)
 17 |                 auto tmp3 = max_propagate_nan(tmp0, tmp2);
      |                                                         ^

```




 让我们也看看输出代码和 IR 节点中相应的 C++ 内核。




 C++ 内核:






```
include "/tmp/torchinductor_root/gv/cgv6n5aotqjo5w4vknjibhengeycuattfto532hkxpozszcgxr3x.h"
extern "C" void kernel(const unsigned char* in_ptr0,
 const unsigned char* in_ptr1,
 unsigned char* out_ptr0)
{
 {
 #pragma GCC ivdep
 for(long i0=static_cast<long>(0L); i0<static_cast<long>(8390L); i0+=static_cast<long>(1L))
 {
 #pragma GCC ivdep
 for(long i1=static_cast<long>(0L); i1<static_cast<long>(8L); i1+=static_cast<long>(1L))
 {
 auto tmp0 = in_ptr0[static_cast<long>(i1 + (8L*i0))];
 auto tmp1 = in_ptr1[static_cast<long>(i1)];
 auto tmp2 = -tmp1;
 auto tmp3 = max_propagate_nan(tmp0, tmp2);
 out_ptr0[static_cast<long>(i1 + (8L*i0))] = tmp3;
 }
 }
 }
}

```




 IR 节点:






```
buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep('buf0', c0, {c0: 67120})]
buf0.unmet_dependencies = []
buf0.met_dependencies =
    [   MemoryDep('arg0_1', c1, {c0: 8390, c1: 8}),
        MemoryDep('arg1_1', c0, {c0: 67120})]
buf0.users = [NodeUser(node=OUTPUT, can_inplace=False)]
buf0.group.device = cpu
buf0.group.iteration = ((8390, 8), ())
buf0.sizes = ([8390, 8], [])
class buf0_loop_body:
    var_ranges = {z0: 8390, z1: 8}
    index0 = 8*z0 + z1
    index1 = z1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg1_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg0_1', get_index_1)
        neg = ops.neg(load_1)
        maximum = ops.maximum(load, neg)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, maximum, None)
        return store

```




 根据回溯日志，编译错误是由于 
 `max_propagate_nan`
 ’s 输入的数据类型不一致引起的。
通过检查 C++ 内核，我们发现知道
 `tmp2`
 不再
 `long`
 执行完
 `-`
 后
 `tmp0`
 是
 `long`
 。
我们可以轻松匹配C++ 内核中的 
 `-`
 和
 `max_propagate_nan`
 分别在 IR 节点中
 `ops.neg`
 和
 `ops.maximum`
。\ n



 现在我们成功发现根本原因是
 `cpp`
 codegen中
 `ops.neg`的实现，在执行
 `neg`
的时候默默的改变了数据类型。\ n




### 精度调试 [¶](#accuracy-debugging "永久链接到此标题")



 否则，如果模型运行时出现其他错误或精度问题，您可以使用名为
 [Minifier](https://pytorch.org/functorch/stable/notebooks/minifier.html) 的 PyTorch 调试工具 
 。 




 `Minifier` 的核心思想是不断删除图的节点和输入，直到找到有问题的最小图。
它通过 4 种策略帮助自动生成缩小的有问题的图：截断后缀、增量调试，消除死代码并删除未使用的输入。




 下面我们将通过
 `Minifer`
 来展示准确度问题的调试过程。
准确度问题是指后端 eager 和 detector 的输出不同的情况。




 例如，我们将示例修改为这样：






```
from torch._dynamo.utils import same

def foo2(x1, x2):
    a = torch.neg(x1)
    b = torch.maximum(x2, a)
    y = torch.cat([b], dim=0)
    return y

x1 = torch.randn((1, 8), dtype=torch.float32)
x2 = torch.randn((8390, 8), dtype=torch.float32)

expected_result = foo2(x1, x2)

compiled_foo2 = torch.compile(foo2)
actual_result = compiled_foo2(x1, x2)

assert same(expected_result, actual_result) == True

```




 并且还要修改
 `neg`
 函数:






```
def neg3(x):
    return f"decltype({x})(2 * {x})"

```




 将出现如下精度问题：






```
torch._dynamo.utils: [ERROR] Accuracy failed: allclose not within tol=0.0001
Traceback (most recent call last):
  File "test_script.py", line 18, in <module>
    assert same(expected_result, actual_result) == True
AssertionError

```




 要使用 Minifier 调试准确性问题，需要两个环境变量:






```
TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4 python xx.py

```




 这为我们提供了演示缩小步骤的日志信息：






```
Started off with 6 nodes

Trying granularity 2
Strategy: Truncate suffix (G: 2) (6 nodes, 2 inputs)
SUCCESS: Went from 6 to 4 nodes

Trying granularity 4
Strategy: Remove unused inputs (G: 4) (4 nodes, 2 inputs)
SUCCESS: Went from 4 to 3 nodes

```




 运行后，我们得到了目标节点的最终缩小图
 `neg`
 :






```
def forward2(self, arg0_1):
    neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
    return (neg,)

```




 有关 Minifier 的更多使用详情，请参阅
 [问题排查](https://pytorch.org/docs/stable/dynamo/troubleshooting.html) 
.





## 性能分析 [¶](#performance-profiling "永久链接到此标题")




 在本节中，我们将演示对使用 Inductor CPU 后端编译的模型进行性能分析的过程。
在下面的示例中，我们对 Hugging Face Transformer 模型进行基准测试
 `MobileBertForQuestionAnswering`
 Eager 模式和 Inductor 图形模式。
基准测试后打印 Inductor 的执行时间和加速比。
我们使用 Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz 并在第一个插槽上运行基准测试演示本部分中的优化。
我们设置以下环境变量作为在 Intel(R) CPU 上进行基准测试的最佳实践。






```
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
numactl -C 0-31 -m 0 python bench.py

```






```
# bench.py
from transformers import MobileBertForQuestionAnswering
# Initialize an eager model
model = MobileBertForQuestionAnswering.from_pretrained("csarron/mobilebert-uncased-squad-v2")
seq_length = 128
bs = 128
vocab_size = model.config.vocab_size
input = torch.randint(0, vocab_size, (bs, seq_length), dtype=torch.int64)
input_dict = {"input_ids": input}

# Initialize the inductor model
compiled_model = torch.compile(model)
with torch.no_grad():
    compiled_model(**input_dict)

NUM_ITERS=50
import timeit
with torch.no_grad():
    # warmup
    for _ in range(10):
        model(**input_dict)
    eager_t = timeit.timeit("model(**input_dict)", number=NUM_ITERS, globals=globals())

with torch.no_grad():
    # warmup
    for _ in range(10):
        compiled_model(**input_dict)
    inductor_t = timeit.timeit("compiled_model(**input_dict)", number=NUM_ITERS, globals=globals())
# print(f"eager use: {eager_t * 1000 / NUM_ITERS} ms/iter")
# print(f"inductor use: {inductor_t * 1000 / NUM_ITERS} ms/iter")
# print(f"speed up ratio: {eager_t / inductor_t}")

```






```
Downloading (…)lve/main/config.json:   0%|          | 0.00/765 [00:00<?, ?B/s]
Downloading (…)lve/main/config.json: 100%|##########| 765/765 [00:00<00:00, 4.17MB/s]

Downloading model.safetensors:   0%|          | 0.00/98.5M [00:00<?, ?B/s]
Downloading model.safetensors:  21%|##1       | 21.0M/98.5M [00:00<00:01, 55.7MB/s]
Downloading model.safetensors:  32%|###1      | 31.5M/98.5M [00:00<00:01, 54.5MB/s]
Downloading model.safetensors:  43%|####2     | 41.9M/98.5M [00:00<00:01, 55.8MB/s]
Downloading model.safetensors:  53%|#####3    | 52.4M/98.5M [00:01<00:00, 49.3MB/s]
Downloading model.safetensors:  75%|#######4  | 73.4M/98.5M [00:01<00:00, 57.7MB/s]
Downloading model.safetensors:  85%|########5 | 83.9M/98.5M [00:01<00:00, 36.2MB/s]
Downloading model.safetensors: 100%|##########| 98.5M/98.5M [00:02<00:00, 35.8MB/s]
Downloading model.safetensors: 100%|##########| 98.5M/98.5M [00:02<00:00, 42.3MB/s]

```




 输出：






```
eager use: 802.1023553796113 ms/iter
inductor use: 339.95180135127157 ms/iter
speed up ratio: 2.359459053287382

```




 在我们自己的测试中，我们发现 Inductor CPU 后端将模型速度提高了约 2.355 倍。




 接下来，让’s 深入了解操作层面的性能，了解加速从何而来。
 [Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes /profiler_recipe.html) 
 是一个帮助我们的好工具。
感应器 CPU 后端支持使用
 `enable_kernel_profile`
 配置选项将融合内核的时间报告给分析器:






```
from torch._inductor import config
config.cpp.enable_kernel_profile = True

```




 按照 [Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 中的步骤操作
 我们能够获取分析表和跟踪文件。






```
# bench.py
from torch.profiler import profile, schedule, ProfilerActivity
RESULT_DIR = "./prof_trace"
my_schedule = schedule(
    skip_first=10,
    wait=5,
    warmup=5,
    active=1,
    repeat=5)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
    # print(output)
    p.export_chrome_trace(f"{RESULT_DIR}/{p.step_num}.json")

for _ in range(10):
    model(**input_dict)  # compiled_model(**input_dict) to get inductor model profiling

total = 0
with profile(
    activities=[ProfilerActivity.CPU],
    schedule=my_schedule,
    on_trace_ready=trace_handler
) as p:
    for _ in range(50):
        model(**input_dict)  # compiled_model(**input_dict) to get inductor model profiling
        p.step()

```




 我们得到以下 Eager-Mode 模型的性能分析表(省略一些列)：






```
------------------------- ------------ ------------ ------------
 Name CPU total % CPU total # of Calls
------------------------- ------------ ------------ ------------
 aten::addmm 45.73% 370.814ms 362
 aten::add 19.89% 161.276ms 363
 aten::copy_ 14.97% 121.416ms 488
 aten::mul 9.02% 73.154ms 194
 aten::clamp_min 8.81% 71.444ms 96
 aten::bmm 5.46% 44.258ms 48
 ProfilerStep* 100.00% 810.920ms 1
 aten::div 2.89% 23.447ms 24
 aten::_softmax 1.00% 8.087ms 24
 aten::linear 46.48% 376.888ms 362
 aten::clone 2.77% 22.430ms 98
 aten::t 0.31% 2.502ms 362
 aten::view 0.14% 1.161ms 850
 aten::transpose 0.17% 1.377ms 386
 aten::index_select 0.12% 952.000us 3
 aten::expand 0.12% 986.000us 458
 aten::matmul 8.31% 67.420ms 48
 aten::cat 0.09% 703.000us 1
 aten::as_strided 0.08% 656.000us 963
 aten::relu 8.86% 71.864ms 96
------------------------- ------------ ------------ ------------
Self CPU time total: 810.920ms

```




 同样，我们还得到了带有 Inductor 的编译模型的表格(省略了一些列)：






```
----------------------------------------------- ------------ ------------ ------------
 Name CPU total % CPU total # of Calls
----------------------------------------------- ------------ ------------ ------------
 mkl::_mkl_linear 68.79% 231.573ms 362
 aten::bmm 8.02% 26.992ms 48
 ProfilerStep* 100.00% 336.642ms 1
 graph_0_cpp_fused_constant_pad_nd_embedding_0 0.27% 915.000us 1
 aten::empty 0.27% 911.000us 362
 graph_0_cpp_fused__mkl_linear_add_mul_relu_151 0.27% 901.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_226 0.27% 899.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_361 0.27% 898.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_121 0.27% 895.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_31 0.27% 893.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_76 0.26% 892.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_256 0.26% 892.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_346 0.26% 892.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_241 0.26% 891.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_316 0.26% 891.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_91 0.26% 890.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_106 0.26% 890.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_211 0.26% 890.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_61 0.26% 889.000us 1
 graph_0_cpp_fused__mkl_linear_add_mul_relu_286 0.26% 889.000us 1
----------------------------------------------- ------------ ------------ ------------
Self CPU time total: 336.642ms

```




 从 eager 模型的分析表中，我们可以看到最耗时的操作是 [
 `aten::addmm`
 ,
 `aten::add`
 ,
 `aten: :copy_`
 ,
 `aten::mul`
 ,
 `aten::clamp_min`
 ,
 `aten::bmm`
 ].
与在电感器模型分析表中，我们注意到
 `mkl::_mkl_linear`
 条目和多个融合内核，其形式
 `graph_0_cpp_fused_* `
 。它们是电感器模型正在进行的主要优化。让我们分别讨论它们。




 (1) 关于
 `mkl::_mkl_linear`
 : 你可能会注意到这个内核的调用次数是 362，与
 `aten::linear` 完全一样
 n 在 eager 模型分析表中。
 `aten::linear`
 的 CPU 总计为 376.888ms，而 
 `mkl::_mkl_linear`
 的 CPU 总计为 231.573ms。这表明 “linear” 部分约为 1.63 倍。
加速主要来自
 [将权重tensor打包为块内存格式](https://www.intel.com /content/www/us/en/docs/onemkl/developer-reference-c/2023-1/cblas-gemm-pack-002.html) 
 并调用
 [cblas_sgemm_compute](https ://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/cblas-gemm-compute-002.html)
 在电感器 CPU 后端
在 GEMM 计算期间具有更好的缓存行为。




 (2) 关于其他内存密集型操作：在我们的测试中，eager/inductor 模型的端到端延迟为 802/339ms。因此我们可以粗略地推断其他内存密集型操作的速度约为 3.94 倍。
让’s 阅读生成的代码以了解电感器如何实现这一令人印象深刻的优化。您可以通过在
 `output_code.py 中搜索
 `cpp_fused__mkl_linear_add_mul_relu_151`
 找到生成的代码`






```
cpp_fused__mkl_linear_add_mul_relu_151 = async_compile.cpp('''
#include <ATen/record_function.h>
#include "/tmp/torchinductor_root/lr/clrlgu27q4ggd472umdzwsu6qcpqxcuusjxqvx2hwitjbujiiz7z.h"
extern "C" void kernel(float* in_out_ptr0,
 const float* in_ptr0,
 const float* in_ptr1,
 const float* in_ptr2,
 const float* in_ptr3)
{
 RECORD_FUNCTION("graph_0_cpp_fused__mkl_linear_add_mul_relu_151", c10::ArrayRef<c10::IValue>({}));
 #pragma omp parallel num_threads(32)
 {
 {
 #pragma omp for
 for(long i0=static_cast<long>(0L); i0<static_cast<long>(16384L); i0+=static_cast<long>(1L))
 {
 for(long i1=static_cast<long>(0L); i1<static_cast<long>(512L); i1+=static_cast<long>(8L))
 {
 auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (512L*i0)));
 auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i1));
 auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(i1 + (512L*i0)));
 auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(i1));
 auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(i1));
 auto tmp2 = tmp0 + tmp1;
 auto tmp4 = tmp2 + tmp3;
 auto tmp6 = tmp4 * tmp5;
 auto tmp8 = tmp6 + tmp7;
 tmp8.store(in_out_ptr0 + static_cast<long>(i1 + (512L*i0)));
 }
 }
 }
 }
}''')

```




 从上面生成的代码中，我们可以看到这个内核做了一个典型的
 [Loop Fusion](https://en.wikipedia.org/wiki/Loop_fission_and_fusion) 
 on
 `[add,\ n 

 add,
 

 mul,
 

 add]`
 。
这是一个阻碍良好性能的内存限制瓶颈。为了更直观地感受这种优化，
我们可以推断输入的大小和步幅，并进一步对此进行基准测试
 `[add,
 

 add,
 

 mul,
 \ n
 添加]`
 模式。






```
# bench.py
def func(arg_0, arg_1, arg_2, arg_3, arg_4):
    add_0 = arg_0 + arg_1
    add_1 = add_0 + arg_2
    mul_1 = add_1 * arg_3
    add_2 = mul_1 + arg_4
    arg_2 = add_2
    return arg_2

arg_0 = torch.rand(16384, 512)
arg_1 = torch.rand(1, 512)
arg_2 = torch.zeros(16384, 512)
arg_3 = torch.rand(1, 512)
arg_4 = torch.rand(1, 512)

input = (arg_0, arg_1, arg_2, arg_3, arg_4)
inductor_func = torch.compile(func)
with torch.no_grad():
    inductor_func(*input)

import timeit
NUM_ITERS=100
with torch.no_grad():
    # warmup
    for _ in range(10):
        func(*input)
    eager_t = timeit.timeit("func(*input)", number=NUM_ITERS, globals=globals())

with torch.no_grad():
    # warmup
    for _ in range(10):
        inductor_func(*input)
    inductor_t = timeit.timeit("inductor_func(*input)", number=NUM_ITERS, globals=globals())
# print(f"eager use: {eager_t * 1000 / NUM_ITERS} ms/iter")
# print(f"inductor use: {inductor_t * 1000 / NUM_ITERS} ms/iter")
# print(f"speed up ratio: {eager_t / inductor_t}")

```




 输出：






```
eager use: 5.780875144992024 ms/iter
inductor use: 0.9588955780491233 ms/iter
speed up ratio: 6.0286805751604735

```




 这只是一个例子。分析表显示，在此模型中，所有按元素运算均自动融合在电感器内。您可以在

output_code.py
中阅读更多内核





## 结论 [¶](#conclusion "此标题的永久链接")




 该文档提供了 Inductor CPU 后端的深入教程。




 通过激励性示例，我们逐步完成调试和分析的过程。
主要思想是缩小问题范围。




 我们一步步演示如何借助调试日志记录和工具 Minifier 来深入研究问题并找到故障的根本原因。
首先确定故障发生在哪个组件，然后尝试生成最小的代码片段可以重现该故障的代码。




 当 Inductor 的性能优于 eager 模式时，我们提供了可靠的性能分析方法。
我们展示了如何使用 PyTorch Profiler 找到耗时热点，并找出操作员级别或内核级别解释该现象的原因。




**脚本总运行时间:** 
 ( 10 分钟 49.837 秒)
