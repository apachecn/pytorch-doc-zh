# 数值精度 [¶](#numerical-accuracy "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/numerical_accuracy>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/numerical_accuracy.html>


 在现代计算机中，浮点数使用 IEEE 754 标准来表示。有关浮点算术和 IEEE 754 标准的更多详细信息，请参阅[浮点算术](https://en.wikipedia.org/wiki/Floating-point_arithmetic)特别要注意的是，浮点提供的精度有限(单精度浮点数大约为 7 位小数，双精度浮点数大约为 16 位小数)，并且浮点加法和乘法不具有关联性，因此运算顺序会影响结果因此，PyTorch 不能保证为数学上相同的浮点计算产生按位相同的结果。同样，不保证跨 PyTorch 版本、单独提交或不同平台的按位相同结果。特别是，即使对于按位相同的输入，甚至在控制随机源之后，CPU 和 GPU 结果也可能不同。


## 批量计算或切片计算 [¶](#batched-computations-or-slice-computations "永久链接到此标题")


 PyTorch 中的许多操作支持批量计算，其中对批量输入的元素执行相同的操作。一个例子是 [`torch.mm()`](../generated/torch.mm.html#torch.mm "torch.mm") 和 [`torch.bmm()`](../generated/torch.bmm.html#torch.bmm "torch.bmm") 。可以将批处理计算实现为批处理元素上的循环，并对各个批处理元素应用必要的数学运算，出于效率原因，我们不这样做，并且通常对整个批处理执行计算。在这种情况下，与非批处理计算相比，我们调用的数学库和 PyTorch 内部运算实现可能会产生略有不同的结果。特别是，让“A”和“B”是尺寸适合批量矩阵乘法的3D张量。然后“(A@B)[0]”(批量结果的第一个元素)不能保证按位相同`A[0]@B[0]` (输入批次的第一个元素的矩阵乘积)即使在数学上它是相同的计算。


 类似地，应用于张量切片的操作不能保证产生与应用于完整张量的相同操作的结果切片相同的结果。例如。令“A”为二维张量。 `A.sum(-1)[0]` 不保证按位等于 `A[:,0].sum()` 。


## 极值 [¶](#extremal-values "此标题的固定链接")


 当输入包含较大值时，中间结果可能会溢出所使用数据类型的范围，最终结果也可能会溢出，即使它可以用原始数据类型表示。例如。：


```
import torch
a=torch.tensor([1e20, 1e20]) # fp32 type by default
a.norm() # produces tensor(inf)
a.double().norm() # produces tensor(1.4142e+20, dtype=torch.float64), representable in fp32

```


## 线性代数 (`torch.linalg`) [¶](#linear-algebra-torch-linalg "永久链接到此标题") 


### 非有限值 [¶](#non-finite-values "永久链接到此标题")


 当输入具有诸如“inf”或“NaN”之类的非有限值时，“torch.linalg”使用的外部库(后端)无法保证其行为。因此，PyTorch 也不会。操作可能会返回具有非有限值的张量，或者引发异常，甚至出现段错误。


 在调用这些函数来检测这种情况之前，请考虑使用 [`torch.isfinite()`](../generated/torch.isfinite.html#torch.isfinite "torch.isfinite")。


### linalg 中的极值 [¶](#extremal-values-in-linalg "永久链接到此标题")


 `torch.linalg` 中的函数比其他 PyTorch 函数具有更多的[极值](#extremal-values)。


[求解器](../linalg.html#linalg-solvers) 和[逆](../linalg.html#linalg-inverses) 假设输入矩阵“A”是可逆的。如果它接近不可逆(例如，如果它具有非常小的奇异值)，那么这些算法可能会默默地返回不正确的结果。这些矩阵被认为是[病态](https://nhigham.com/2020/03/19/what-is-a-condition-number/)。如果提供病态输入，这些矩阵的结果当在不同设备上使用相同输入或通过关键字“driver”使用不同后端时，它们的功能可能会有所不同。


 当“svd”、“eig”和“eigh”等频谱运算的输入具有彼此接近的奇异值时，它们也可能返回不正确的结果(并且它们的梯度可能是无限的)。这是因为用于计算这些分解的算法很难针对这些输入收敛。


 在 `float64` 中运行计算(NumPy 默认情况下这样做)通常会有所帮助，但它并不能在所有情况下解决这些问题。通过 [`torch.linalg.svdvals()`](../generated/torch.linalg.svdvals.html#torch.linalg.svdvals "torch.linalg.svdvals") 或其条件编号通过 [`torch.linalg.cond()`](../generated/torch.linalg.cond.html#torch.linalg.cond“torch.linalg.cond”)可能有助于检测这些问题。


## Nvidia Ampere 设备上的 TensorFloat-32(TF32) [¶](#tensorfloat-32-tf32-on-nvidia-ampere-devices“此标题的永久链接”)


 在 Ampere Nvidia GPU 上，PyTorch 可以使用 TensorFloat32 (TF32) 来加速数学密集型运算，特别是矩阵乘法和卷积。当使用 TF32 张量核心执行运算时，仅读取输入尾数的前 10 位。这可能降低准确性并产生令人惊讶的结果(例如，将矩阵乘以单位矩阵可能会产生与输入不同的结果)。默认情况下，TF32 张量核心禁用矩阵乘法并启用卷积，尽管大多数神经网络工作负载都具有使用 TF32 时的收敛行为与使用 fp32 时的收敛行为相同。如果您的网络不需要完整的 float32 精度，我们建议使用`torch.backends.cuda.matmul.allow_tf32 = True`启用 TF32 张量核心进行矩阵乘法。如果您的网络矩阵乘法和卷积都需要完整的 float32 精度，那么也可以通过 `torch.backends.cudnn.allow_tf32 = False` 禁用 TF32 张量核心进行卷积。


 有关更多信息，请参阅 [TensorFloat32](cuda.html#tf32-on-ampere) 。


## 降低 FP16 和 BF16 GEMM 的精度


 半精度 GEMM 运算通常通过单精度的中间累加(归约)来完成，以提高数值精度并提高溢出弹性。为了提高性能，某些 GPU 架构(尤其是较新的 GPU 架构)允许对中间累加结果进行一些截断，以降低精度(例如，半精度)。从模型收敛的角度来看，这种变化通常是良性的，尽管它可能会导致意外的结果(例如，当最终结果应该以半精度表示时，会出现“inf”值)。如果降低精度的降低有问题，则可以通过 `torch.backends.cuda.matmul.allow_fp16_reduced_ precision_reduction = False` 关闭


 BF16 GEMM 操作存在类似的标志，并且默认情况下处于打开状态。如果 BF16 降低精度减少有问题，可以使用“torch.backends.cuda.matmul.allow_bf16_reduced_ precision_reduction = False”关闭它们


 有关详细信息，请参阅 [allow_fp16_reduced_precision_reduction](cuda.html#fp16reducedprecision) 和 [allow_bf16_reduced_ precision_reduction](cuda.html#bf16reducedprecision)


## AMD Instinct MI200 设备上的降低精度 FP16 和 BF16 GEMM 和卷积 [¶](#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices“此标题的永久链接”) 


 在 AMD Instinct MI200 GPU 上，FP16 和 BF16 V_DOT2 以及 MFMA 矩阵指令将输入和输出非正规值刷新为零。 FP32 和 FP64 MFMA 矩阵指令不会将输入和输出非正规值刷新为零。受影响的指令仅由 rocBLAS (GEMM) 和 MIOpen (卷积) 内核使用；所有其他 PyTorch 操作都不会遇到此行为。所有其他受支持的 AMD GPU 都不会遇到此行为。


 rocBLAS 和 MIOpen 为受影响的 FP16 操作提供替代实现。未提供 BF16 操作的替代实现； BF16 数字比 FP16 数字具有更大的动态范围，并且不太可能遇到非正规值。对于 FP16 替代实现，FP16 输入值被转换为中间 BF16 值，然后在累加 FP32 操作后转换回 FP16 输出。这样，输入输出类型就不变了。


 使用 FP16 精度进行训练时，某些模型可能无法在 FP16 分母刷新为零的情况下收敛。非正规值更频繁地出现在梯度计算期间训练的向后传递中。默认情况下，PyTorch 将在向后传递过程中使用 rocBLAS 和 MIOpen 替代实现。可以使用环境变量 ROCBLAS_INTERNAL_FP16_ALT_IMPL 和 MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL 覆盖默认行为。这些环境变量的行为如下：


|  | 	 forward	  | 	 backward	  |
| --- | --- | --- |
| 	 Env unset	  | 	 original	  | 	 alternate	  |
| 	 Env set to 1	  | 	 alternate	  | 	 alternate	  |
| 	 Env set to 0	  | 	 original	  | 	 original	  |


 以下是可以使用 rocBLAS 的操作列表：



* torch.addbmm
* torch.addmm
* torch.baddbmm
* torch.bmm
* torch.mm
* torch.nn.GRUCell
* torch.nn.LSTMCell
* torch.nn.Linear
* torch.sparse.addmm
* 以下 torch._C._ConvBackend 实现：
    + SlowNd 
    + SlowNd_transpose 
    + SlowNd_dilated 
    + SlowNd_dilated_transpose


 以下是可以使用 MIOpen 的操作列表：



* torch.nn.Conv[Transpose]Nd
* 以下 torch._C._ConvBackend 实现：
    + ConvBackend::Miopen 
    + ConvBackend::MiopenDepthwise 
    + ConvBackend::MiopenTranspose