# (beta) PyTorch 中的通道最后内存格式 [¶](#beta-channels-last-memory-format-in-pytorch "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/memory_format_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html>




**作者** 
 :
 [Vitaly Fedyunin](https://github.com/VitalyFedyunin)





## 什么是 Channels Last [¶](#what-is-channels-last "此标题的永久链接")




 通道最后内存格式是在内存保留维度排序中对 NCHW tensor进行排序的另一种方法。通道最后的tensor以通道成为最密集维度的方式排序(也称为逐像素存储图像)。




 例如，NCHW tensor的经典(连续)存储(在我们的例子中是两个具有 3 个颜色通道的 4x4 图像)如下所示：




![classic_memory_format](https://pytorch.org/tutorials/_images/classic_memory_format.png)


 通道最后的内存格式对数据的排序不同:




![channels_last_memory_format](https://pytorch.org/tutorials/_images/channels_last_memory_format.png)


 Pytorch 通过利用现有的步幅结构来支持内存格式(并提供与现有模型的向后兼容性，包括 eager、JIT 和 TorchScript)。
例如，Channels Last 格式中的 10x3x16x16 批次的步幅等于 (768, 1, 48, 3).




 通道最后的内存格式仅针对 4D NCHW tensor实现。





## 内存格式 API [¶](#memory-format-api "此标题的永久链接")




 以下是如何在连续和通道之间转换tensor
最后的内存格式。




 经典 PyTorch 连续tensor






```
import torch

N, C, H, W = 10, 3, 32, 32
x = torch.empty(N, C, H, W)
print(x.stride())  # Outputs: (3072, 1024, 32, 1)

```






```
(3072, 1024, 32, 1)

```




 转换运算符






```
x = x.to(memory_format=torch.channels_last)
print(x.shape)  # Outputs: (10, 3, 32, 32) as dimensions order preserved
print(x.stride())  # Outputs: (3072, 1, 96, 3)

```






```
torch.Size([10, 3, 32, 32])
(3072, 1, 96, 3)

```




 返回连续






```
x = x.to(memory_format=torch.contiguous_format)
print(x.stride())  # Outputs: (3072, 1024, 32, 1)

```






```
(3072, 1024, 32, 1)

```




 替代选项






```
x = x.contiguous(memory_format=torch.channels_last)
print(x.stride())  # Outputs: (3072, 1, 96, 3)

```






```
(3072, 1, 96, 3)

```




 格式检查






```
print(x.is_contiguous(memory_format=torch.channels_last))  # Outputs: True

```






```
True

```




 这两个 API 之间存在细微差别
 `to`
 和
 `contigious`
 。我们建议在显式转换tensor的内存格式时坚持使用“to”。




 对于一般情况，两个 API 的行为相同。然而，在特殊
情况下，对于大小为
 `NCHW`
 的 4D tensor，当：
 `C==1`
 或
 `H==1
 

 &&
 
 
 W==1`
 ，只有
 `to`
 才会生成适当的步幅来
表示通道最后的内存格式。




 这是因为在上述两种情况中，tensor的内存格式
是不明确的，即大小为
 `N1HW`
 的连续tensor
 既是
 `连续`
 又是内存中的最后一个通道存储。
因此，对于给定的内存格式，它们已被视为
 `is_contigious`
，因此
 `contigulous`
 调用变成
无操作，并且不会更新步幅。相反，
 `to`
 会在大小为 1 的维度上以有意义的步幅重新调整tensor，
以便正确表示预期的内存
格式






```
special_x = torch.empty(4, 1, 4, 4)
print(special_x.is_contiguous(memory_format=torch.channels_last))  # Outputs: True
print(special_x.is_contiguous(memory_format=torch.contiguous_format))  # Outputs: True

```






```
True
True

```




 同样的情况也适用于显式排列 API
 `permute`
 。在可能出现歧义的特殊情况下，“permute”不能保证产生正确携带预期内存格式的步幅。我们建议使用
 `to`
 和显式内存格式
以避免意外行为。




 顺便指出，在极端情况下，三个非批量
维度都等于
 `1`
 (
 `C==1
 

 &&
 
\ n H==1
 

 &&
 

 W==1`
 )，
当前实现无法将tensor标记为通道最后内存
格式。




 最后创建为频道






```
x = torch.empty(N, C, H, W, memory_format=torch.channels_last)
print(x.stride())  # Outputs: (3072, 1, 96, 3)

```






```
(3072, 1, 96, 3)

```




`克隆`
 保留内存格式






```
y = x.clone()
print(y.stride())  # Outputs: (3072, 1, 96, 3)

```






```
(3072, 1, 96, 3)

```




`to`
 、
 `cuda`
 、
 `float`
 … 保留内存格式






```
if torch.cuda.is_available():
    y = x.cuda()
    print(y.stride())  # Outputs: (3072, 1, 96, 3)

```






```
(3072, 1, 96, 3)

```




`empty_like`
 ,
 `*_like`
 运算符保留内存格式






```
y = torch.empty_like(x)
print(y.stride())  # Outputs: (3072, 1, 96, 3)

```






```
(3072, 1, 96, 3)

```




 逐点运算符保留内存格式






```
z = x + y
print(z.stride())  # Outputs: (3072, 1, 96, 3)

```






```
(3072, 1, 96, 3)

```




`Conv`
 、
 `Batchnorm`
 模块使用
 `cudnn`
 后端最后支持通道
(仅适用于 cuDNN >= 7.6)。与二进制
p-wise 运算符不同，卷积模块将通道最后作为主导内存格式。
如果所有输入均为连续内存格式，则运算符
以连续内存格式生成输出。否则，
输出将采用通道最后的内存格式。






```
if torch.backends.cudnn.is_available() and torch.backends.cudnn.version() >= 7603:
    model = torch.nn.Conv2d(8, 4, 3).cuda().half()
    model = model.to(memory_format=torch.channels_last)  # Module parameters need to be channels last

    input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, requires_grad=True)
    input = input.to(device="cuda", memory_format=torch.channels_last, dtype=torch.float16)

    out = model(input)
    print(out.is_contiguous(memory_format=torch.channels_last))  # Outputs: True

```






```
True

```




 当输入tensor到达没有通道最后支持的运算符时，
 排列应自动在内核中应用以恢复
输入tensor上的连续性。这会引入开销并停止通道最后的内存格式传播。尽管如此，它还是
保证了正确的输出。





## 性能增益 [¶](#performance-gains "此标题的固定链接")




 通道最后内存格式优化在 GPU 和 CPU 上均可用。
在 GPU 上，在 NVIDIA’s
支持在降低精度下运行的 Tensor Core 硬件上观察到最显着的性能提升
(
 ` torch.float16`
 )。
与连续格式相比，最后的通道能够实现超过 22% 的性能提升，同时利用
‘AMP(自动混合精度)’训练脚本。
我们的脚本使用 NVIDIA 提供的 AMP
 <https://github.com/NVIDIA/apex>
 。




`python
 

 main_amp.py
 

 -a
 

 resnet50
 

 --b
 

 200
 \ n
 --workers
 

 16
 

 --opt-level
 

 O2
 

./data`






```
# opt_level = O2
# keep_batchnorm_fp32 = None <class 'NoneType'>
# loss_scale = None <class 'NoneType'>
# CUDNN VERSION: 7603
# => creating model 'resnet50'
# Selected optimization level O2: FP16 training with FP32 batchnorm and FP32 master weights.
# Defaults for this optimization level are:
# enabled : True
# opt_level : O2
# cast_model_type : torch.float16
# patch_torch_functions : False
# keep_batchnorm_fp32 : True
# master_weights : True
# loss_scale : dynamic
# Processing user overrides (additional kwargs that are not None)...
# After processing overrides, optimization options are:
# enabled : True
# opt_level : O2
# cast_model_type : torch.float16
# patch_torch_functions : False
# keep_batchnorm_fp32 : True
# master_weights : True
# loss_scale : dynamic
# Epoch: [0][10/125] Time 0.866 (0.866) Speed 230.949 (230.949) Loss 0.6735125184 (0.6735) Prec@1 61.000 (61.000) Prec@5 100.000 (100.000)
# Epoch: [0][20/125] Time 0.259 (0.562) Speed 773.481 (355.693) Loss 0.6968704462 (0.6852) Prec@1 55.000 (58.000) Prec@5 100.000 (100.000)
# Epoch: [0][30/125] Time 0.258 (0.461) Speed 775.089 (433.965) Loss 0.7877287269 (0.7194) Prec@1 51.500 (55.833) Prec@5 100.000 (100.000)
# Epoch: [0][40/125] Time 0.259 (0.410) Speed 771.710 (487.281) Loss 0.8285319805 (0.7467) Prec@1 48.500 (54.000) Prec@5 100.000 (100.000)
# Epoch: [0][50/125] Time 0.260 (0.380) Speed 770.090 (525.908) Loss 0.7370464802 (0.7447) Prec@1 56.500 (54.500) Prec@5 100.000 (100.000)
# Epoch: [0][60/125] Time 0.258 (0.360) Speed 775.623 (555.728) Loss 0.7592862844 (0.7472) Prec@1 51.000 (53.917) Prec@5 100.000 (100.000)
# Epoch: [0][70/125] Time 0.258 (0.345) Speed 774.746 (579.115) Loss 1.9698858261 (0.9218) Prec@1 49.500 (53.286) Prec@5 100.000 (100.000)
# Epoch: [0][80/125] Time 0.260 (0.335) Speed 770.324 (597.659) Loss 2.2505953312 (1.0879) Prec@1 50.500 (52.938) Prec@5 100.000 (100.000)

```




 传递
 `--channels-last
 

 true`
 允许以 Channels Last 格式运行模型，并观察到 ​​22% 的性能增益。




`python
 

 main_amp.py
 

 -a
 

 resnet50
 

 --b
 

 200
 \ n
 --workers
 

 16
 

 --opt-level
 

 O2
 

 --channels-last
 

 true\ n 

./数据`






```
# opt_level = O2
# keep_batchnorm_fp32 = None <class 'NoneType'>
# loss_scale = None <class 'NoneType'>
#
# CUDNN VERSION: 7603
#
# => creating model 'resnet50'
# Selected optimization level O2: FP16 training with FP32 batchnorm and FP32 master weights.
#
# Defaults for this optimization level are:
# enabled : True
# opt_level : O2
# cast_model_type : torch.float16
# patch_torch_functions : False
# keep_batchnorm_fp32 : True
# master_weights : True
# loss_scale : dynamic
# Processing user overrides (additional kwargs that are not None)...
# After processing overrides, optimization options are:
# enabled : True
# opt_level : O2
# cast_model_type : torch.float16
# patch_torch_functions : False
# keep_batchnorm_fp32 : True
# master_weights : True
# loss_scale : dynamic
#
# Epoch: [0][10/125] Time 0.767 (0.767) Speed 260.785 (260.785) Loss 0.7579724789 (0.7580) Prec@1 53.500 (53.500) Prec@5 100.000 (100.000)
# Epoch: [0][20/125] Time 0.198 (0.482) Speed 1012.135 (414.716) Loss 0.7007197738 (0.7293) Prec@1 49.000 (51.250) Prec@5 100.000 (100.000)
# Epoch: [0][30/125] Time 0.198 (0.387) Speed 1010.977 (516.198) Loss 0.7113101482 (0.7233) Prec@1 55.500 (52.667) Prec@5 100.000 (100.000)
# Epoch: [0][40/125] Time 0.197 (0.340) Speed 1013.023 (588.333) Loss 0.8943189979 (0.7661) Prec@1 54.000 (53.000) Prec@5 100.000 (100.000)
# Epoch: [0][50/125] Time 0.198 (0.312) Speed 1010.541 (641.977) Loss 1.7113249302 (0.9551) Prec@1 51.000 (52.600) Prec@5 100.000 (100.000)
# Epoch: [0][60/125] Time 0.198 (0.293) Speed 1011.163 (683.574) Loss 5.8537774086 (1.7716) Prec@1 50.500 (52.250) Prec@5 100.000 (100.000)
# Epoch: [0][70/125] Time 0.198 (0.279) Speed 1011.453 (716.767) Loss 5.7595844269 (2.3413) Prec@1 46.500 (51.429) Prec@5 100.000 (100.000)
# Epoch: [0][80/125] Time 0.198 (0.269) Speed 1011.827 (743.883) Loss 2.8196096420 (2.4011) Prec@1 47.500 (50.938) Prec@5 100.000 (100.000)

```




 以下模型列表完全支持 Channels Last，并在 Volta 设备上显示 8%-35% 的性能提升:
 `alexnet`
 ,
 `mnasnet0_5`
 ,
 `mnasnet0_75`
 、
 `mnasnet1_0`
 、
 `mnasnet1_3`
 、
 `mobilenet_v2`
 、
 `resnet101`
 、 
 `resnet152`
 、
 `resnet18`
 、
 `resnet34`
 、
 `resnet50`
 、
 `resnext50_32x4d`
 、
 `shufflenet_v2 _x0_5`
 ,
 `shufflenet_v2_x1_0`
 ,
 `shufflenet_v2_x1_5`
 ,
 `shufflenet\ _v2_x2_0`
 ,
 `squeezenet1_0`
 ,
 `squeezenet1_1`
 ,
 `vgg11`
 ,
 `vgg11_bn`\ n ,
 `vgg13`
 ,
 `vgg13_bn`
 ,
 `vgg16`
 ,
 `vgg16_bn`
 ,
 `vgg19`
 ,
 `vgg19_bn`
 ,
 `wide_resnet101_2`
 ,
 `wide_resnet50_2`




 以下型号列表完全支持最后通道，并在 Intel(R) Xeon(R) Ice Lake(或更新的)CPU 上显示出 26%-76% 的性能提升：
 `alexnet`
 , 
 `densenet121`
 、
 `densenet161`
 、
 `densenet169`
 、
 `googlenet`
 、
 `inception_v3`
 、
 `mnasnet0_5 `
 、
 `mnasnet1_0`
 、
 `resnet101`
 、
 `resnet152`
 、
 `resnet18`
 、
 `resnet34`
 、
 ` resnet50`
 、
 `resnext101_32x8d`
 、
 `resnext50_32x4d`
 、
 `shufflenet_v2_x0_5`
 、
 `shufflenet\ _v2_x1_0`
 ,
 `squeezenet1_0`
 ,
 `squeezenet1_1`
 ,
 `vgg11`
 ,
 `vgg11_bn`\ n ,
 `vgg13`
 ,
 `vgg13_bn`
 ,
 `vgg16`
 ,
 `vgg16_bn`
 ,
 `vgg19`
 ,
 `vgg19_bn`
 ,
 `wide_resnet101_2`
 ,
 `wide_resnet50_2`





## 转换现有模型 [¶](#converting-existing-models "永久链接到此标题")




 通道最后支持不受现有模型的限制，因为只要输入(或特定权重)格式正确，
任何模型都可以转换为通道最后并通过图表
传播格式。






```
# Need to be done once, after model initialization (or load)
model = model.to(memory_format=torch.channels_last)  # Replace with your model

# Need to be done for every input
input = input.to(memory_format=torch.channels_last)  # Replace with your input
output = model(input)

```




 然而，并非所有运算符最后都完全转换为支持通道(通常返回连续的输出)。在上面发布的示例中，最后不支持通道的层将停止内存格式传播。尽管如此，由于我们已将模型转换为通道最后的格式，这意味着每个卷积层(其在通道最后的内存格式中具有 4 维权重)将恢复通道最后的内存格式并受益于更快的内核。 




 但最后不支持通道的运算符确实会通过排列引入
开销。或者，如果您想要提高转换模型的性能，
您可以调查并识别模型中
最后不支持通道的运算符。




 这意味着您需要根据支持的运算符列表验证已使用的运算符列表
 <https://github.com/pytorch/pytorch/wiki/Operators-with-Channels-Last-support>
 ，
将内存格式检查引入急切执行模式并运行您的模型。




 运行下面的代码后，如果运算符的输出’ 与输入的内存格式不匹配，运算符将引发异常。






```
def contains_cl(args):
    for t in args:
        if isinstance(t, torch.Tensor):
            if t.is_contiguous(memory_format=torch.channels_last) and not t.is_contiguous():
                return True
        elif isinstance(t, list) or isinstance(t, tuple):
            if contains_cl(list(t)):
                return True
    return False


def print_inputs(args, indent=""):
 for t in args:
 if isinstance(t, [torch.Tensor](https://pytorch.org/docs/stable/tensors.html# torch.Tensor "torch.Tensor")):
 print(indent, t.stride(), t.shape, t.device, t.dtype)
 elif isinstance(t, list) 或 isinstance(t, tuple ):
 print(indent, type(t))
 print_inputs(list(t), indent=indent + " ")
 else:
 print(indent, t)


def check_wrapper(fn):
 name = fn.__name__

 def check_cl(*args, **kwargs) :
 was_cl = contains_cl(args)
 try:
 result = fn(*args, **kwargs)
 except Exception as e:
 print(" `{}` 输入为：".format(name))
 print_inputs(args)
 print("--------------------")\ n raise e
 failed = False
 if was_cl:
 if isinstance(结果, [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")):
 如果 result.dim() == 4 并且不是 result.is_contigious(memory_format=[torch.channels_last](https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format "torch.memory_format")):
 print(
 "`{}` 获得了channels_last 输入，但输出不是channels_last:".format(name) ,
 result.shape,
 result.stride(),
 result.device,
 result.dtype,
 )
 failed = True
 如果失败且 True:
 print("`{} ` 输入为：".format(name))
 print_inputs(args)
 raise Exception("Operator `{}` 丢失了channel_last 属性".format(name))
 返回结果
 
 返回检查_cl


旧_attrs = dict()


def 属性(m):
 old_attrs[m] = dict()
 for i in dir(m):
 e = getattr(m, i)
 except_functions = ["is\ _cuda", "has_names", "numel", "stride", "Tensor", "is_contigious", "__class__"]
 如果我不在排除 dir(e) 中的 _functions 而不是 i.startswith("_") 和 "__call__":
 尝试:
 old_attrs[m] [i] = e
 setattr(m, i, check_wrapper(e))
 except Exception as e:
 print(i)
 print(e)


attribute(torch.Tensor)
attribute(torch.nn.functional)
attribute(torch)

```




 如果您发现一个不’t 支持通道最后tensor的运算符
并且您想做出贡献，请随意使用以下开发人员
指南
 <https://github.com/pytorch/pytorch /wiki/Writing-memory-format-aware-operators>
.




 下面的代码是恢复torch的属性。






```
for (m, attrs) in old_attrs.items():
    for (k, v) in attrs.items():
        setattr(m, k, v)

```





## 待办事项 [¶](#work-to-do "此标题的永久链接")




 还有很多事情要做，比如：



* 解决
 `N1HW`
 和
 `NC11`
 tensor的歧义；
* 测试分布式训练支持；
* 提高算子覆盖范围。



 如果您有反馈和/或改进建议，请通过创建
[问题](https://github.com/pytorch/pytorch/issues) 告诉我们

 。




**脚本总运行时间:** 
 ( 0 分 0.158 秒)
