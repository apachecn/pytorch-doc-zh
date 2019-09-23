# torch.onnx

  * [HTG0例：端至端AlexNet从PyTorch到ONNX 

  * 跟踪VS脚本

  * 局限性

  * 支持的运营商

  * 为运营商添加支持

    * 阿坦运营

    * 非宏正操作符

    * 自定义操作符

  * 常见问题

  * 功能

## [HTG0例：端至端从PyTorch AlexNet到ONNX

这里是一个出口预训练AlexNet在torchvision定义成ONNX一个简单的脚本。它运行一个单轮推理的，然后保存该所得的跟踪模型`
alexnet.onnx`：

    
    
    import torch
    import torchvision
    
    dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
    model = torchvision.models.alexnet(pretrained=True).cuda()
    
    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]
    
    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
    

将得到的`alexnet.onnx`是包含两者的网络结构和导出的模型（在此情况下，AlexNet）的参数的二进制protobuf的文件。关键字参数`
冗长=真 `使出口打印出所述网络的人类可读表示：

    
    
    # These are the inputs and parameters to the network, which have taken on
    # the names we specified earlier.
    graph(%actual_input_1 : Float(10, 3, 224, 224)
          %learned_0 : Float(64, 3, 11, 11)
          %learned_1 : Float(64)
          %learned_2 : Float(192, 64, 5, 5)
          %learned_3 : Float(192)
          # ---- omitted for brevity ----
          %learned_14 : Float(1000, 4096)
          %learned_15 : Float(1000)) {
      # Every statement consists of some output tensors (and their types),
      # the operator to be run (with its attributes, e.g., kernels, strides,
      # etc.), its input tensors (%actual_input_1, %learned_0, %learned_1)
      %17 : Float(10, 64, 55, 55) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[11, 11], pads=[2, 2, 2, 2], strides=[4, 4]](%actual_input_1, %learned_0, %learned_1), scope: AlexNet/Sequential[features]/Conv2d[0]
      %18 : Float(10, 64, 55, 55) = onnx::Relu(%17), scope: AlexNet/Sequential[features]/ReLU[1]
      %19 : Float(10, 64, 27, 27) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%18), scope: AlexNet/Sequential[features]/MaxPool2d[2]
      # ---- omitted for brevity ----
      %29 : Float(10, 256, 6, 6) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%28), scope: AlexNet/Sequential[features]/MaxPool2d[12]
      # Dynamic means that the shape is not known. This may be because of a
      # limitation of our implementation (which we would like to fix in a
      # future release) or shapes which are truly dynamic.
      %30 : Dynamic = onnx::Shape(%29), scope: AlexNet
      %31 : Dynamic = onnx::Slice[axes=[0], ends=[1], starts=[0]](%30), scope: AlexNet
      %32 : Long() = onnx::Squeeze[axes=[0]](%31), scope: AlexNet
      %33 : Long() = onnx::Constant[value={9216}](), scope: AlexNet
      # ---- omitted for brevity ----
      %output1 : Float(10, 1000) = onnx::Gemm[alpha=1, beta=1, broadcast=1, transB=1](%45, %learned_14, %learned_15), scope: AlexNet/Sequential[classifier]/Linear[6]
      return (%output1);
    }
    

您也可以使用[ onnx ](https://github.com/onnx/onnx/)库验证的protobuf。您可以安装`onnx`与畅达：

    
    
    conda install -c conda-forge onnx
    

然后，你可以运行：

    
    
    import onnx
    
    # Load the ONNX model
    model = onnx.load("alexnet.onnx")
    
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)
    

要运行用[导出脚本caffe2 ](https://caffe2.ai/)，你需要安装 caffe2
：如果你没有一个已经，请[按照安装说明[HTG5。](https://caffe2.ai/docs/getting-started.html)

一旦安装了这些，您可以使用Caffe2后端：

    
    
    # ...continuing from above
    import caffe2.python.onnx.backend as backend
    import numpy as np
    
    rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    # For the Caffe2 backend:
    #     rep.predict_net is the Caffe2 protobuf for the network
    #     rep.workspace is the Caffe2 workspace for the network
    #       (see the class caffe2.python.onnx.backend.Workspace)
    outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
    # To run networks with more than one input, pass a tuple
    # rather than a single numpy ndarray.
    print(outputs[0])
    

您也可以运行[导出模型ONNXRuntime ](https://github.com/microsoft/onnxruntime)，你需要安装
ONNXRuntime
：请[按照这些指示[HTG5。](https://github.com/microsoft/onnxruntime#installation)

一旦安装了这些，您可以使用ONNXRuntime后端：

    
    
    # ...continuing from above
    import onnxruntime as ort
    
    ort_session = ort.InferenceSession('alexnet.onnx')
    
    outputs = ort_session.run(None, {'actual_input_1': np.random.randn(10, 3, 224, 224).astype(np.float32)})
    
    print(outputs[0])
    

这里是出口超高分辨率模型ONNX的另一个[教程。
](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)。

在未来，还会有其他框架后端为好。

## 跟踪VS脚本

该ONNX出口既可以是 _追踪基_ 和 __ 基于脚本的出口国。

  * _跟踪为主_ 意味着它通过执行模型一次，导出此运行期间实际运行操作人员进行操作。这意味着，如果你的模型是动态的，例如，改变依赖于输入数据的行为，出口将是不准确的。同样，跟踪可能只针对特定的输入大小是有效的（这就是为什么我们需要在跟踪明确的输入。）我们建议检查模型跟踪，并确保跟踪的运营商希望合理。如果模型包含像for循环，如果条件控制流， _跟踪为主_ 出口将展开的循环和if条件，导出一个静态的图形是完全一样的，因为这跑。如果你要导出动态控制流模型，您将需要使用 _基于脚本的_ 出口国。

  * _基于脚本的_ 意味着你正在尝试导出模型是[ ScriptModule [HTG3。  ScriptModule 是在 TorchScript 核心数据结构，和 TorchScript 是Python语言，从PyTorch代码创建和序列优化的模型的子集。](../jit.html)

我们允许混合跟踪和脚本。您可以撰写跟踪和脚本，以适应模型的一部分的特殊要求。结帐这个例子：

    
    
    import torch
    
    # Trace-based only
    
    class LoopModel(torch.nn.Module):
        def forward(self, x, y):
            for i in range(y):
                x = x + i
            return x
    
    model = LoopModel()
    dummy_input = torch.ones(2, 3, dtype=torch.long)
    loop_count = torch.tensor(5, dtype=torch.long)
    
    torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True)
    

随着 _跟踪为主_ 出口商，我们得到的结果ONNX图其器展开for循环：

    
    
    graph(%0 : Long(2, 3),
          %1 : Long()):
      %2 : Tensor = onnx::Constant[value={1}]()
      %3 : Tensor = onnx::Add(%0, %2)
      %4 : Tensor = onnx::Constant[value={2}]()
      %5 : Tensor = onnx::Add(%3, %4)
      %6 : Tensor = onnx::Constant[value={3}]()
      %7 : Tensor = onnx::Add(%5, %6)
      %8 : Tensor = onnx::Constant[value={4}]()
      %9 : Tensor = onnx::Add(%7, %8)
      return (%9)
    

为了利用 _基于脚本的_ 出口商捕捉动态环路，我们可以编写脚本的循环，并从正规nn.Module调用它：

    
    
    # Mixing tracing and scripting
    
    @torch.jit.script
    def loop(x, y):
        for i in range(int(y)):
            x = x + i
        return x
    
    class LoopModel2(torch.nn.Module):
        def forward(self, x, y):
            return loop(x, y)
    
    model = LoopModel2()
    dummy_input = torch.ones(2, 3, dtype=torch.long)
    loop_count = torch.tensor(5, dtype=torch.long)
    torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True,
                      input_names=['input_data', 'loop_range'])
    

现在出口ONNX图变为：

    
    
    graph(%input_data : Long(2, 3),
          %loop_range : Long()):
      %2 : Long() = onnx::Constant[value={1}](), scope: LoopModel2/loop
      %3 : Tensor = onnx::Cast[to=9](%2)
      %4 : Long(2, 3) = onnx::Loop(%loop_range, %3, %input_data), scope: LoopModel2/loop # custom_loop.py:240:5
        block0(%i.1 : Long(), %cond : bool, %x.6 : Long(2, 3)):
          %8 : Long(2, 3) = onnx::Add(%x.6, %i.1), scope: LoopModel2/loop # custom_loop.py:241:13
          %9 : Tensor = onnx::Cast[to=9](%2)
          -> (%9, %8)
      return (%4)
    

动态控制流得到正确捕获。我们可以在不同的循环范围的后端验证。

    
    
    import caffe2.python.onnx.backend as backend
    import numpy as np
    import onnx
    model = onnx.load('loop.onnx')
    
    rep = backend.prepare(model)
    outputs = rep.run((dummy_input.numpy(), np.array(9).astype(np.int64)))
    print(outputs[0])
    #[[37 37 37]
    # [37 37 37]]
    
    
    import onnxruntime as ort
    ort_sess = ort.InferenceSession('loop.onnx')
    outputs = ort_sess.run(None, {'input_data': dummy_input.numpy(),
                                  'loop_range': np.array(9).astype(np.int64)})
    print(outputs)
    #[array([[37, 37, 37],
    #       [37, 37, 37]], dtype=int64)]
    

## 局限性

  * 张量就地如索引分配数据[索引] = NEW_DATA 目前不出口支撑。解决这种问题的一种方法是使用操作符散射，明确地更新原来的张量。
    
        data = torch.zeros(3, 4)
    index = torch.tensor(1)
    new_data = torch.arange(4).to(torch.float32)
    
    # Assigning to left hand side indexing is not supported in exporting.
    # class InPlaceIndexedAssignment(torch.nn.Module):
    # def forward(self, data, index, new_data):
    #     data[index] = new_data
    #     return data
    
    class InPlaceIndexedAssignmentONNX(torch.nn.Module):
        def forward(self, data, index, new_data):
            new_data = new_data.unsqueeze(0)
            index = index.expand(1, new_data.size(1))
            data.scatter_(0, index, new_data)
            return data
    
    out = InPlaceIndexedAssignmentONNX()(data, index, new_data)
    
    torch.onnx.export(InPlaceIndexedAssignmentONNX(), (data, index, new_data), 'inplace_assign.onnx')
    
    # caffe2
    import caffe2.python.onnx.backend as backend
    import onnx
    
    onnx_model = onnx.load('inplace_assign.onnx')
    rep = backend.prepare(onnx_model)
    out_caffe2 = rep.run((torch.zeros(3, 4).numpy(), index.numpy(), new_data.numpy()))
    
    assert torch.all(torch.eq(out, torch.tensor(out_caffe2)))
    
    # onnxruntime
    import onnxruntime
    sess = onnxruntime.InferenceSession('inplace_assign.onnx')
    out_ort = sess.run(None, {
        sess.get_inputs()[0].name: torch.zeros(3, 4).numpy(),
        sess.get_inputs()[1].name: index.numpy(),
        sess.get_inputs()[2].name: new_data.numpy(),
    })
    
    assert torch.all(torch.eq(out, torch.tensor(out_ort)))
    

  * 有没有在ONNX张量清单的概念。如果没有这个概念，它是非常困难的出口消耗或产生张量清单运营商，尤其是当张列表的长度在出口时并不知道。
    
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    
    # This is not exportable
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.unbind(0)
    
    # This is exportable.
    # Note that in this example we know the split operator will always produce exactly three outputs,
    # Thus we can export to ONNX without using tensor list.
    class AnotherModel(torch.nn.Module):
        def forward(self, x):
            return [torch.squeeze(out, 0) for out in torch.split(x, [1,1,1], dim=0)]
    

  * PyTorch和ONNX后端（Caffe2，ONNXRuntime等）经常与某些数字差异运营商的实现。根据模型结构，这些差异可能是微不足道的，但他们也可能会导致行为的主要分歧（特别是未经训练的模型。）我们允许Caffe2直接调用运营商的火炬实现，帮你抚平这些差异时的精度是非常重要的，并且还记录这些差异。

## 支持的运营商

下面的运营商的支持：

  * BatchNorm

  * ConstantPadNd

  * CONV

  * 退出

  * 嵌入（不支持任何可选参数）

  * FeatureDropout（不支持培训模式）

  * 指数

  * MaxPool1d

  * MaxPool2d

  * MaxPool3d

  * RNN

  * ABS

  * ACOS

  * adaptive_avg_pool1d

  * adaptive_avg_pool2d

  * adaptive_avg_pool3d

  * adaptive_max_pool1d

  * 自适应max_pool2d

  * adaptive_max_pool3d

  * 添加（不支持非零的α）

  * addmm

  * 和

  * 人气指数

  * argmax

  * argmin

  * ASIN

  * 晒黑

  * avg_pool1d

  * avg_pool2d

  * avg_pool2d

  * avg_pool3d

  * 猫

  * 小区

  * 钳

  * clamp_max

  * clamp_min

  * CONCAT

  * COS

  * dim_arange

  * DIV

  * 退出

  * 埃卢

  * EQ

  * ERF

  * EXP

  * 扩大

  * expand_as

  * 弄平

  * 地板

  * 充分

  * full_like

  * 收集

  * 通用电器

  * 谷氨酸

  * GT

  * hardtanh

  * index_copy

  * index_fill

  * index_select

  * instance_norm

  * isnan

  * layer_norm

  * 乐

  * leaky_relu

  * 日志

  * LOG2

  * log_sigmoid

  * log_softmax

  * logsumexp

  * LT

  * masked_fill

  * 最大

  * 意思

  * 分

  * 毫米

  * MUL

  * 狭窄

  * NE

  * NEG

  * 非零

  * 规范

  * 那些

  * ones_like

  * 要么

  * 置换

  * pixel_shuffle

  * POW

  * prelu（输入通道之间共享单个重量不支持）

  * 刺

  * 兰特

  * randn

  * randn_like

  * 倒数

  * reflection_pad

  * RELU

  * 重复

  * replication_pad

  * 重塑

  * reshape_as

  * rrelu

  * RSUB

  * 分散

  * scatter_add

  * 选择

  * 九色鹿

  * 乙状结肠

  * 标志

  * 罪

  * 尺寸

  * 切片

  * SOFTMAX（仅暗淡= -1支持）

  * softplus

  * 分裂

  * 开方

  * 挤

  * 堆

  * 子（不支持非零的α）

  * 和

  * Ť

  * 黄褐色

  * 正切

  * （不支持非零阈值/非零值）的阈值

  * 至

  * TOPK

  * 颠倒

  * type_as

  * 展开（有了ATEN-Caffe2集成的实验性支持）

  * unsqueeze

  * upsample_nearest1d

  * upsample_nearest2d

  * upsample_nearest3d

  * 视图

  * 哪里

  * 零

  * zeros_like

上述设置操作员足以导出以下机型：

  * AlexNet

  * DCGAN

  * DenseNet

  * 盗梦空间（警告：这种模式是高度敏感的操作方式有变动）

  * RESNET

  * 超高分辨率

  * VGG

  * [ word_language_model ](https://github.com/pytorch/examples/tree/master/word_language_model)

## 为运营商添加支持

增加对运营商出口的支持力度是
_提前使用[HTG1。为了实现这一目标，开发人员需要触摸PyTorch的源代码。请遵循从源代码安装PyTorch的[说明书[HTG3。如果想要的操作是ONNX规范，它应该很容易增加对出口此类操作（添加操作员的符号功能）的支持。要确认操作是否规范与否，请检查](https://github.com/pytorch/pytorch#from-
source)[
ONNX操作员列表[HTG5。](https://github.com/onnx/onnx/blob/master/docs/Operators.md)_

### 阿坦运营

如果运营商是ATEN运营商，这意味着你可以找到函数的`火炬/中国证监会申报/ autograd /生成/ VariableType.h
`（在PyTorch生成的代码可安装DIR），应添加符号函数在`炬/ onnx / symbolic_opset & LT ;版本& GT ;。PY
`，并按照列为以下说明：

  * 定义`炬/ onnx / symbolic_opset & LT符号函数;版本& GT ;。PY`，例如[炬/onnx/symbolic_opset9.py ](https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py)。确保函数具有相同的名称`VariableType.h`定义的ATEN操作/功能。

  * 第一个参数始终是出口ONNX图。参数名称必须完全匹配`VariableType.h`的名字，因为调度与关键字参数来完成。

  * 参数顺序不一定匹配是什么`VariableType.h`，张量（输入）总是第一个，那么非张量参数。

  * 在象征性的功能，如果操作员在ONNX已经标准化，我们只需要创建一个节点来表示ONNX操作者在图中。

  * 如果输入参数是一个张量，但ONNX要求一个标量，我们必须明确地做转换。辅助函数`_scalar`可以标量张量转换成一个Python标量，并且`_if_scalar_type_as`可以将一个Python标量成PyTorch张量。

### 非宏正操作符

如果操作员是一个非ATEN运算符，符号函数具有在对应PyTorch Function类被添加。请阅读以下说明：

  * 创建名为`的符号函数的符号 `在相应的功能类。

  * 第一个参数始终是出口ONNX图。

  * 除了首先必须参数名完全匹配`转发 `的名字。

  * 输出元组大小必须的`向前 `的输出相匹配。

  * 在象征性的功能，如果操作员在ONNX已经标准化，我们只需要创建一个节点来表示ONNX操作者在图中。

象征性的功能应该在Python中实现。所有这些功能与Python的方法这是通过C ++实现互动 - Python绑定，但直观的界面，他们提供这个样子的：

    
    
    def operator/symbolic(g, *inputs):
      """
      Modifies Graph (e.g., using "op"), adding the ONNX operations representing
      this PyTorch function, and returning a Value or tuple of Values specifying the
      ONNX outputs whose values correspond to the original PyTorch return values
      of the autograd Function (or None if an output is not supported by ONNX).
    
      Arguments:
        g (Graph): graph to write the ONNX representation into
        inputs (Value...): list of values representing the variables which contain
            the inputs for this function
      """
    
    class Value(object):
      """Represents an intermediate tensor value computed in ONNX."""
      def type(self):
        """Returns the Type of the value."""
    
    class Type(object):
      def sizes(self):
        """Returns a tuple of ints representing the shape of a tensor this describes."""
    
    class Graph(object):
      def op(self, opname, *inputs, **attrs):
        """
        Create an ONNX operator 'opname', taking 'args' as inputs
        and attributes 'kwargs' and add it as a node to the current graph,
        returning the value representing the single output of this
        operator (see the `outputs`keyword argument for multi-return
        nodes).
    
        The set of operators and the inputs/attributes they take
        is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md
    
        Arguments:
            opname (string): The ONNX operator name, e.g., `Abs`or `Add`.
            args (Value...): The inputs to the operator; usually provided
                as arguments to the `symbolic`definition.
            kwargs: The attributes of the ONNX operator, with keys named
                according to the following convention: `alpha_f`indicates
                the `alpha`attribute with type `f`.  The valid type specifiers are
              `f`(float), `i`(int), `s`(string) or `t`(Tensor).  An attribute
                specified with type float accepts either a single float, or a
                list of floats (e.g., you would say `dims_i`for a `dims`attribute
                that takes a list of integers).
            outputs (int, optional):  The number of outputs this operator returns;
                by default an operator is assumed to return a single output.
                If `outputs`is greater than one, this functions returns a tuple
                of output `Value`, representing each output of the ONNX operator
                in positional.
        """
    

所述ONNX曲线C ++定义在`炬/ CSRC / JIT / ir.h`。

下面是处理缺失符号函数为`ELU`操作者的例子。我们尝试导出模型，并看到错误消息如下：

    
    
    UserWarning: ONNX export failed on elu because torch.onnx.symbolic_opset9.elu does not exist
    RuntimeError: ONNX export failed: Couldn't export operator elu
    

因为PyTorch不支持导出`埃卢 `操作导出失败。我们发现`虚拟 张量 埃卢（常量 张量 [ - ]放;  输入， [HTG17标量 α， 布尔
就地） 常量 覆盖[ ]`在`VariableType.h`。这意味着`ELU`是一个宏正操作符。我们检查[
ONNX操作员列表](https://github.com/onnx/onnx/blob/master/docs/Operators.md)，并确认`恶露
`在ONNX标准化。我们将下列行添加到`symbolic_opset9.py`：

    
    
    def elu(g, input, alpha, inplace=False):
        return g.op("Elu", input, alpha_f=_scalar(alpha))
    

现在PyTorch能够输出`埃卢 `运营商。

有在[ symbolic_opset9.py
](https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py)以上实例中，[
symbolic_opset10.py
](https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset10.py)。

指定运营商定义的接口是实验;冒险的用户应注意，这些API将在未来的接口可能会改变。

### 自定义操作符

在此之后教程[扩展TorchScript用自定义的C
++运算符](/advanced/torch_script_custom_ops.html)，您可以创建并注册在PyTorch自己的自定义OPS的实现。以下是如何这样的模型导出到ONNX：

    
    
    # Create custom symbolic function
    from torch.onnx.symbolic_helper import parse_args
    @parse_args('v', 'v', 'f', 'i')
    def symbolic_foo_forward(g, input1, input2, attr1, attr2):
        return g.op("Foo", input1, input2, attr1_f=attr1, attr2_i=attr2)
    
    # Register custom symbolic function
    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('custom_ops::foo_forward', symbolic_foo_forward, 9)
    
    class FooModel(torch.nn.Module):
        def __init__(self, attr1, attr2):
            super(FooModule, self).__init__()
            self.attr1 = attr1
            self.attr2 = attr2
    
        def forward(self, input1, input2):
            # Calling custom op
            return torch.ops.custom_ops.foo_forward(input1, input2, self.attr1, self.attr2)
    
    model = FooModel(attr1, attr2)
    torch.onnx.export(model, (dummy_input1, dummy_input2), 'model.onnx')
    

根据自定义操作，可以将其导出为一个或现有ONNX
OPS的组合。您也可以将其导出为自运在ONNX为好。在这种情况下，你将需要相匹配的定制OPS实现，例如延长您所选择的后端[ Caffe2定制OPS
](https://caffe2.ai/docs/custom-operators.html)，[ ONNXRuntime定制OPS
](https://github.com/microsoft/onnxruntime/blob/master/docs/AddingCustomOp.md)。

## 常见问题

问：我已出口我LSTM模式，但其输入的大小似乎是固定的？

> 示踪剂记录在图中的示例输入形状。在情况下，模型应该接受动态形状的输入，你可以利用出口API参数 dynamic_axes [HTG1。

>  
>  
>     layer_count = 4

>  
>     model = nn.LSTM(10, 20, num_layers=layer_count, bidirectional=True)

>     model.eval()

>  
>     with torch.no_grad():

>         input = torch.randn(5, 3, 10)

>         h0 = torch.randn(layer_count * 2, 3, 20)

>         c0 = torch.randn(layer_count * 2, 3, 20)

>         output, (hn, cn) = model(input, (h0, c0))

>  
>         # default export

>         torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx')

>         onnx_model = onnx.load('lstm.onnx')

>         # input shape [5, 3, 10]

>         print(onnx_model.graph.input[0])

>  
>         # export with `dynamic_axes`

>         torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx',

>                         input_names=['input', 'h0', 'c0'],

>                         output_names=['output', 'hn', 'cn'],

>                         dynamic_axes={'input': {0: 'sequence'}, 'output':
{0: 'sequence'}})

>         onnx_model = onnx.load('lstm.onnx')

>         # input shape ['sequence', 3, 10]

>         print(onnx_model.graph.input[0])

>  

问：如何与它的循环导出模型？

> 请结算跟踪VS脚本[HTG1。

问：ONNX支持隐式数据类型标铸造？

>
没有，但出口商会尽量处理的那部分。标量被转换为在ONNX恒定张量。出口商会揣摩标量正确的数据类型。但是，对于它没有这样做的情况下，您将需要手动提供的数据类型信息。我们正在努力改善，使得手动更改不会在将来要求出口数据类型的传播。

>  
>  
>     class ImplicitCastType(torch.jit.ScriptModule):

>         @torch.jit.script_method

>         def forward(self, x):

>             # Exporter knows x is float32, will export '2' as float32 as
well.

>             y = x + 2

>             # Without type propagation, exporter doesn't know the datatype
of y.

>             # Thus '3' is exported as int64 by default.

>             return y + 3

>             # The following will export correctly.

>             # return y + torch.tensor([3], dtype=torch.float32)

>  
>     x = torch.tensor([1.0], dtype=torch.float32)

>     torch.onnx.export(ImplicitCastType(), x, 'models/implicit_cast.onnx',

>                       example_outputs=ImplicitCastType()(x))

>  

## 功能

`torch.onnx.``export`( _model_ , _args_ , _f_ , _export_params=True_ ,
_verbose=False_ , _training=False_ , _input_names=None_ , _output_names=None_
, _aten=False_ , _export_raw_ir=False_ , _operator_export_type=None_ ,
_opset_version=None_ , __retain_param_name=True_ , _do_constant_folding=False_
, _example_outputs=None_ , _strip_doc_string=True_ , _dynamic_axes=None_
)[[source]](_modules/torch/onnx.html#export)

    

导出模型到ONNX格式。该出口国，一旦运行模型，以获得其执行一丝要导出;目前，它支持一组有限的动态模型（例如，RNNs）另见：
onnx出口：PARAM模型：要导出的模型。 ：类型的模型：torch.nn.Module：PARAM ARGS：输入以

> 该模型，例如，使得`模型（*参数）
`是该模型的有效调用。任何非张量参数将被硬编码到导出的模型;任何张量参数将成为导出的模型的输入，在它们出现在args的顺序。如果ARGS是张量，这等同于具有与该张量的1元元组称为它。
（注：传递关键字参数到模型目前不支持给我们留言，如果你需要它。）

Parameters

    

  * **F** \- 一个类文件对象（必须实现的fileno返回文件描述符）或包含文件名的字符串。二进制的Protobuf将被写入该文件。

  * **export_params** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _默认真_ ） - 如果指定，所有的参数将被导出。如果你要导出未经训练的模式设置为False。在这种情况下，如由`model.state_dict（）中指定的输出模式将首先采取它的所有参数作为参数，排序。值（） `

  * **详细** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _默认为false_ ） - 如果指定，我们将打印出的调试说明跟踪被导出。

  * **训练** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _默认为false_ ） - 导出模型训练模式。目前，ONNX是面向仅供推理模型导出，所以你一般不会需要将其设置为True。

  * **input_names** （ _字符串列表_ _，_ _默认空列表_ ） - 名称分配给图的输入节点，为了

  * **output_names** （ _字符串列表_ _，_ _默认空列表_ ） - 名称分配给图的输出节点，为了

  * **ATEN** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _默认假_ ） - [已过时。使用operator_export_type]的模型导出ATEN模式。 ;版本& GT ;。PY导出为阿坦OPS如果使用宏正模式时，所有的OPS原稿symbolic_opset & LT出口通过的功能。

  * **export_raw_ir** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _默认假_ ） - [已过时。使用operator_export_type]导出内部IR，而不是直接将其转换为ONNX OPS的。

  * **operator_export_type** （ _枚举_ _，_ _默认OperatorExportTypes.ONNX_ ） - 

OperatorExportTypes.ONNX：所有OPS导出为普通ONNX欢声笑语。
OperatorExportTypes.ONNX_ATEN：所有OPS导出为阿滕欢声笑语。
OperatorExportTypes.ONNX_ATEN_FALLBACK：如果象征性的缺失，

> 依傍阿滕运算。

OperatorExportTypes.RAW：出口原料IR。

  * **opset_version** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _默认为9 HTG9]） - 默认情况下，我们的模型导出到的opset所版本该onnx子模块。由于ONNX最新opset所下一个稳定版本之前可能演变，在默认情况下我们出口到一个稳定opset所版本。眼下，支持稳定opset所版本9. opset_version必须_onnx_master_opset或者其中的火炬/ onnx / symbolic_helper.py定义_onnx_stable_opsets_

  * **do_constant_folding** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _默认假_ ） - 如果为True，恒定折叠优化施加到出口过程中的模型。恒定折叠优化将取代一些具有所有常量输入，与预先计算的常数的节点OPS的。

  * **example_outputs** （ _张量_ _的元组，_ _默认无_ ） - example_outputs必须导出ScriptModule或TorchScript功能时提供。

  * **strip_doc_string** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _默认真_ ） - 如果为True，从导出的剥离字段“doc_string”模型，对堆栈跟踪哪些信息。

  * **example_outputs** \- 正被导出的模型的示例输出。

  * **dynamic_axes** （ _DICT & LT ;串_ _，_ _字典 & LT ;蟒：INT _ _，_ _串 & GT ; & GT ; _ _或_ _字典 & LT ;串_ _，_ [ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)") _（_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _）_ _ & GT ; _ _，_ _默认空字典_ ） - 

一个字典，以指定的输入/输出的动态轴，使得： - KEY：输入和/或输出的名称 - 值：对于给定的密钥动态轴的指数的和潜在的名称将被用于导出动态轴。
（1）：在一般的值根据以下方式或二者的组合中的一个来定义。整数specifiying提供的输入的动态轴的列表。在这种情况下自动名称将被产生和导出过程中施加到所提供的输入/输出的动态轴。
OR（2）。内的字典，它指定在相应的输入/输出到期望出口期间这样的输入/输出的这种轴所施加的名字从动态轴的索引的映射。例。如果我们有用于输入和输出如下形状：

> 形状（INPUT_1）=（“B”，3，“W”，“H”）和形状（INPUT_2）=（“B”，4）和形状（输出）=（“B”，“d”，5）

Then dynamic axes can be defined either as:

    

(a). ONLY INDICES:

    

dynamic_axes = {“INPUT_1”：[0，2，3]，“INPUT_2”：[0]，“输出”：[0，1]}

其中自动名称将用于导出动态轴来生成

(b). INDICES WITH CORRESPONDING NAMES:

    

dynamic_axes = { 'INPUT_1'：{0： '批'，1： '宽度'，2： '高度'}， 'INPUT_2'：{0： '批'}，
'输出'：{0： '批'， 1： '检测'}

其中提供的名称将被应用到导出动态轴

(c). MIXED MODE OF (a) and (b)

    

dynamic_axes = {“INPUT_1”：[0，2，3]，“INPUT_2”：{0：”批”}，‘输出’：[0,1]}

`torch.onnx.``register_custom_op_symbolic`( _symbolic_name_ , _symbolic_fn_ ,
_opset_version_
)[[source]](_modules/torch/onnx.html#register_custom_op_symbolic)

    

`torch.onnx.operators.``shape_as_tensor`( _x_
)[[source]](_modules/torch/onnx/operators.html#shape_as_tensor)

    

`torch.onnx.``set_training`( _model_ , _mode_
)[[source]](_modules/torch/onnx.html#set_training)

    

上下文管理者临时设置“模式”到“模式”的训练模式，重置它，当我们退出与块。甲如果无操作模式是无。

`torch.onnx.``is_in_onnx_export`()[[source]](_modules/torch/onnx.html#is_in_onnx_export)

    

检查它是否是在ONNX出口的中间。此功能在torch.onnx.export中旬返回True（）。 torch.onnx.export应与单个线程执行。

[Next ![](_static/images/chevron-right-orange.svg)](__config__.html
"torch.__config__") [![](_static/images/chevron-right-orange.svg)
Previous](tensorboard.html "torch.utils.tensorboard")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * torch.onnx 
    * [HTG0例：端至端AlexNet从PyTorch到ONNX 
    * 跟踪VS脚本
    * 局限性
    * 支持的运营商
    * 为运营商添加支持
      * 阿坦运营
      * 非宏正操作符
      * 自定义操作符
    * 常见问题
    * 功能

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

