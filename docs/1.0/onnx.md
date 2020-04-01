# torch.onnx

> 译者：[guobaoyo](https://github.com/guobaoyo)

## 示例:从Pytorch到Caffe2的端对端AlexNet模型
这里是一个简单的脚本程序,它将一个在 torchvision 中已经定义的预训练 AlexNet 模型导出到 ONNX 格式. 它会运行一次,然后把模型保存至 `alexnet.onnx`:
```py
import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

# 可以根据模块图形的数值设置输入输出的显示名称。这些设置不会改变此图形的语义。只是会变得更加可读了。
#该网络的输入包含了输入的扁平表(flat list)。也就是说传入forward()里面的值，其后是扁平表的参数。你可以指定一部分名字，例如指定一个比该模块输入数量更少的表，随后我们会从一开始就设定名字。
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
```
得到的 `alexnet.onnx` 是一个 protobuf 二值文件, 它包含所导出模型 ( 这里是 AlexNet )中网络架构和网络参数. 关键参数 `verbose=True` 会使导出过程中打印出的网络更可读:

```py
#这些是网络的输入和参数，包含了我们之前设定的名称。
graph(%actual_input_1 : Float(10, 3, 224, 224)
      %learned_0 : Float(64, 3, 11, 11)
      %learned_1 : Float(64)
      %learned_2 : Float(192, 64, 5, 5)
      %learned_3 : Float(192)
      # ---- 为了简介可以省略 ----
      %learned_14 : Float(1000, 4096)
      %learned_15 : Float(1000)) {
  # 每个声明都包含了一些输出张量以及他们的类型，以及即将运行的操作符(并且包含它的属性，例如核部分，步长等等）它的输入张量(%actual_input_1, %learned_0, %learned_1）
  %17 : Float(10, 64, 55, 55) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[11, 11], pads=[2, 2, 2, 2], strides=[4, 4]](%actual_input_1, %learned_0, %learned_1), scope: AlexNet/Sequential[features]/Conv2d[0]
  %18 : Float(10, 64, 55, 55) = onnx::Relu(%17), scope: AlexNet/Sequential[features]/ReLU[1]
  %19 : Float(10, 64, 27, 27) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%18), scope: AlexNet/Sequential[features]/MaxPool2d[2]
  # ---- 为了简洁可以省略 ----
  %29 : Float(10, 256, 6, 6) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%28), scope: AlexNet/Sequential[features]/MaxPool2d[12]
  #动态意味着它的形状是未知的。这可能是因为我们的执行操作或者其形状大小是否确实为动态的而受到了限制。(这一点我们想在将来的版本中修复）
  %30 : Dynamic = onnx::Shape(%29), scope: AlexNet
  %31 : Dynamic = onnx::Slice[axes=[0], ends=[1], starts=[0]](%30), scope: AlexNet
  %32 : Long() = onnx::Squeeze[axes=[0]](%31), scope: AlexNet
  %33 : Long() = onnx::Constant[value={9216}](), scope: AlexNet
  # ---- 为了简洁可以省略 ----
  %output1 : Float(10, 1000) = onnx::Gemm[alpha=1, beta=1, broadcast=1, transB=1](%45, %learned_14, %learned_15), scope: AlexNet/Sequential[classifier]/Linear[6]
  return (%output1);
}
```
你可以使用 [onnx](https://github.com/onnx/onnx/) 库验证 protobuf, 并且用 conda 安装 `onnx`
```py
conda install -c conda-forge onnx

```

然后运行:

```py
import onnx

# 载入onnx模块
model = onnx.load("alexnet.onnx")

#检查IR是否良好
onnx.checker.check_model(model)

#输出一个图形的可读表示方式
onnx.helper.printable_graph(model.graph)

```

为了能够使用 [caffe2](https://caffe2.ai/) 运行脚本，你需要安装 Caffe2\. 如果你之前没有安装,请参照 [安装指南](https://caffe2.ai/docs/getting-started.html)。
一旦这些安装完成,你就可以在后台使用 Caffe2 :

```py
# ...接着上面的继续
import onnx_caffe2.backend as backend
import numpy as np

rep = backend.prepare(model, device="CUDA:0") #或者 "CPU"
#后台运行Caffe2：
#rep.predict_net是该网络的Caffe2 protobuf
#rep.workspace是该网络的Caffe2 workspace
#(详见类“onnx_caffe2.backend.Workspace”）
outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
#为了多输入地运行该网络，应该传递元组而不是一个单元格。
print(outputs[0])

```

之后,我们还会提供其它框架的后端支持.

## 局限

*   ONNX 导出器是一个基于轨迹的导出器，这意味着它执行时需要运行一次模型，然后导出实际参与运算的运算符。这也意味着，如果你的模型是动态的，例如，改变一些依赖于输入数据的操作，这时的导出结果是不准确的。同样，一个轨迹可能只对一个具体的输入尺寸有效 (这就是我们在轨迹中需要有明确的输入的原因之一。) 我们建议检查模型的轨迹，确保被追踪的运算符是合理的。
*   Pytorch和Caffe2中的一些运算符经常有着数值上的差异.根据模型的结构,这些差异可能是微小的,但它们会在表现上产生很大的差别 (尤其是对于未训练的模型。)之后，为了帮助你在准确度要求很高的情况中，能够轻松地避免这些差异带来的影响，我们计划让Caffe2能够直接调用Torch的运算符.

## 支持的运算符

以下是已经被支持的运算符:

*   add (不支持非零α)
*   sub (不支持非零α)
*   mul
*   div
*   cat
*   mm
*   addmm
*   neg
*   sqrt
*   tanh
*   sigmoid
*   mean
*   sum
*   prod
*   t
*   expand (只有在扩展onnx操作符之前可以使用，例如add)
*   transpose
*   view
*   split
*   squeeze
*   prelu (不支持输入通道之间的单重共享)
*   threshold (不支持非零值阈值/非零值)
*   leaky_relu
*   glu
*   softmax (只支持dim=-1)
*   avg_pool2d (不支持ceil_mode)
*   log_softmax
*   unfold (为ATen-Caffe2集成作实验支撑)
*   elu
*   concat
*   abs
*   index_select
*   pow
*   clamp
*   max
*   min
*   eq
*   gt
*   lt
*   ge
*   le
*   exp
*   sin
*   cos
*   tan
*   asin
*   acos
*   atan
*   permute
*   Conv
*   BatchNorm
*   MaxPool1d (不支持ceil_mode)
*   MaxPool2d (不支持ceil_mode)
*   MaxPool3d (不支持ceil_mode)
*   Embedding (不支持可选参数)
*   RNN
*   ConstantPadNd
*   Dropout
*   FeatureDropout (不支持训练模式)
*   Index (支持常量整数和元组索引)

上面的运算符足够导出下面的模型:

*   AlexNet
*   DCGAN
*   DenseNet
*   Inception (注意:该模型对操作符十分敏感)
*   ResNet
*   SuperResolution
*   VGG
* [word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model)

为操作符增加导出支持是一种 _提前的用法_。为了实现这一点，开发者需要掌握PyTorch的源代码。请按照这个[网址链接](https://github.com/pytorch/pytorch#from-source) 去下载PyTorch。如果您想要的运算符已经在ONNX标准化了，那么支持对导出此类运算符的操作(为运算符添加符号函数）就很容易了。为了确认运算符是否已经被标准化，请检查[ONNX 操作符列表](https://github.com/onnx/onnx/blob/master/docs/Operators.md).如果这个操作符是ATen操作符，这就意味着你可以在 `torch/csrc/autograd/generated/VariableType.h`找到它的定义。(在PyTorch安装文件列表的合成码中可见)，你应该在 `torch/onnx/symbolic.py`里面加上符号并且遵循下面的指令：
*   在 [torch/onnx/symbolic.py](https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic.py)里面定义符号。确保该功能与在ATen操作符在`VariableType.h`的功能相同。
*   第一个参数总是ONNX图形参数，参数的名字必须与 `VariableType.h`里的匹配，因为调度是依赖于关键字参数完成的。
*   参数排序不需要严格与`VariableType.h`匹配，首先的张量一定是输入的张量，然后是非张量参数。
*   在符号功能里，如果操作符已经在ONNX标准化了，我们只需要创建一个代码去表示在图形里面的ONNX操作符。
*   如果输入参数是一个张量，但是ONNX需要的是一个标量形式的输入，我们需要做个转化。`_scalar`可以帮助我们将一个张量转化为一个python标量，并且`_if_scalar_type_as`函数可以将python标量转化为PyTorch张量。

如果操作符是一个非ATen操作符，那么符号功能需要加在相应的PyTorch函数类中。请阅读下面的指示：

*   在相应的函数类中创建一个符号函数命名为`symbolic`。
*   第一个参数总是导出ONNX图形参数。
*   参数的名字除了第一个必须与`前面的形式`严格匹配。
*   输出元组大小必须与`前面的形式`严格匹配。
*   在符号功能中，如果操作符已经在ONNX标准化了，我们只需要创建一个代码去表示在图形里面的ONNX操作符。

符号功能应该在Python里面配置好。所有的这些与Python方法相关的功能都通过C++-Python绑定配置好，且上者提供的界面直观地显示如下：

```py
def operator/symbolic(g, *inputs):
  """
 修改图像(例如使用 "op")，加上代表这个PyTorch功能的ONNX操作符，并且返回一个指定的ONNX输出值或者元组值，这些值与最开始PyTorch返回的自动求导功能相关(或者如果ONNX不支持输出，则返回none。 ).

参数：
 g (图形)：写入图形的ONNX表示方法。
 inputs (值...)：该值的列表表示包含这个功能的输入的可变因素。
 """

class Value(object):
  """代表一个在ONNX里计算的中间张量。"""
  def type(self):
    """返回值的类型"""

class Type(object):
  def sizes(self):
    """返回代表这个张量大小形状的整数元组"""

class Graph(object):
  def op(self, opname, *inputs, **attrs):
    """
 闯将一个ONNX操作符'opname'，将'args'作为输入和属性'kwargs'并且将它作为当前图形的节点，返回代表这个操作符的单一输出值(详见`outputs`多关键参数返回节点)。

 操作符的设置和他们输入属性详情请见 https://github.com/onnx/onnx/blob/master/docs/Operators.md

 参数：
 opname (字符串)：ONNX操作符的名字，例如`Abs`或者`Add`。
 args (值...)：该操作符的输入经常被作为`symbolic`定义参数输入。
 kwargs：该ONNX操作符的属性键名根据以下约定：`alpha_f` 代表着`alpha`具有`f`的属性。有效的类型说明符是
  `f`(float），`i`(int），`s`(string）或`t`(Tensor）。使用float类型指定的属性接受单个float或float列表(例如，对于带有整数列表的`dims`属性，你可以称其为'dims_i`）。
 outputs (证书，可选)：这个运算符返回的输出参数的数量，默认情况下，假定运算符返回单个输出。
 如果`输出`不止一个，这个功能将会返回一个输出值的元组，代表着每个ONNX操作符的输出的位置。
 """

```

ONNX的图形C++定义详情请见`torch/csrc/jit/ir.h`。

这是一个处理`elu`操作符缺少符号函数的例子。我们尝试导出模型并查看错误消息，如下所示：

```py
UserWarning: ONNX export failed on elu because torch.onnx.symbolic.elu does not exist
RuntimeError: ONNX export failed: Couldn't export operator elu

```

导出失败，因为PyTorch不支持导出`elu`操作符。 我们发现`virtual Tensor elu(const Tensor＆input，Scalar alpha，bool inplace）const override;```VariableType.h`。 这意味着`elu`是一个ATen操作符。
我们可以参考[ONNX操作运算符列表](http://https://github.com/onnx/onnx/blob/master/docs/Operators.md)，并且确认 `Elu` 在ONNX中已经被标准化。我们将以下行添加到`symbolic.py`：

```py
def elu(g, input, alpha, inplace=False):
    return g.op("Elu", input, alpha_f=_scalar(alpha))

```

现在PyTorch能够导出`elu`操作符：

在下面的链接中有更多的例子： [symbolic.py](https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic.py), [tensor.py](https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/autograd/_functions/tensor.py#L24), [padding.py](https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/nn/_functions/padding.py#L8).

用于指定运算符定义的接口是实验性的; 喜欢尝试的用户应该注意，API可能会在未来的界面中发生变化。

## 功能函数

```py
torch.onnx.export(*args, **kwargs)
```
