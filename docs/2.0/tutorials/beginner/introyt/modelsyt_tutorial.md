# 使用 PyTorch 构建模型

> 译者：[Fadegentle](https://github.com/Fadegentle)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/introyt/modelsyt_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html>

请跟随下面的视频或在 [youtube](https://www.youtube.com/watch?v=OSqIP-mOWOI) 上观看。

<iframe width="560" height="315" src="https://www.youtube.com/embed/OSqIP-mOWOI" title="Building Models with PyTorch" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## `torch.nn.Module` 和 `torch.nn.Parameter`

在本视频中，我们将讨论 PyTorch 为构建深度学习网络提供的一些工具。

除了 `Parameter` 之外，我们在本视频中讨论的类都是 `torch.nn.Module` 的子类。这是 PyTorch 的基类，用于封装 PyTorch 模型及其组件的特定行为。

`torch.nn.Module` 的一个重要行为是注册参数。如果一个特定的 `Module` 子类有学习权重，这些权重将以 `torch.nn.Parameter` 的实例来表示。`Parameter` 类是 `torch.Tensor` 的子类，其特殊行为是当它们被指定为 `Module` 的属性时，会被添加到该模块的参数列表中。这些参数可以通过 `Module` 类的 `parameters()` 方法访问。

举个简单的例子，这里有一个非常简单的模型，包含两个线性层和一个激活函数。我们将创建一个实例，并要求它报告其参数：

```python
import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)
```

输出：
```shell
The model:
TinyModel(
  (linear1): Linear(in_features=100, out_features=200, bias=True)
  (activation): ReLU()
  (linear2): Linear(in_features=200, out_features=10, bias=True)
  (softmax): Softmax(dim=None)
)


Just one layer:
Linear(in_features=200, out_features=10, bias=True)


Model params:
Parameter containing:
tensor([[ 0.0765,  0.0830, -0.0234,  ..., -0.0337, -0.0355, -0.0968],
        [-0.0573,  0.0250, -0.0132,  ..., -0.0060,  0.0240,  0.0280],
        [-0.0908, -0.0369,  0.0842,  ..., -0.0078, -0.0333, -0.0324],
        ...,
        [-0.0273, -0.0162, -0.0878,  ...,  0.0451,  0.0297, -0.0722],
        [ 0.0833, -0.0874, -0.0020,  ..., -0.0215,  0.0356,  0.0405],
        [-0.0637,  0.0190, -0.0571,  ..., -0.0874,  0.0176,  0.0712]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0304, -0.0758, -0.0549, -0.0893, -0.0809, -0.0804, -0.0079, -0.0413,
        -0.0968,  0.0888,  0.0239, -0.0659, -0.0560, -0.0060,  0.0660, -0.0319,
        -0.0370,  0.0633, -0.0143, -0.0360,  0.0670, -0.0804,  0.0265, -0.0870,
         0.0039, -0.0174, -0.0680, -0.0531,  0.0643,  0.0794,  0.0209,  0.0419,
         0.0562, -0.0173, -0.0055,  0.0813,  0.0613, -0.0379,  0.0228,  0.0304,
        -0.0354,  0.0609, -0.0398,  0.0410,  0.0564, -0.0101, -0.0790, -0.0824,
        -0.0126,  0.0557,  0.0900,  0.0597,  0.0062, -0.0108,  0.0112, -0.0358,
        -0.0203,  0.0566, -0.0816, -0.0633, -0.0266, -0.0624, -0.0746,  0.0492,
         0.0450,  0.0530, -0.0706,  0.0308,  0.0533,  0.0202, -0.0469, -0.0448,
         0.0548,  0.0331,  0.0257, -0.0764, -0.0892,  0.0783,  0.0062,  0.0844,
        -0.0959, -0.0468, -0.0926,  0.0925,  0.0147,  0.0391,  0.0765,  0.0059,
         0.0216, -0.0724,  0.0108,  0.0701, -0.0147, -0.0693, -0.0517,  0.0029,
         0.0661,  0.0086, -0.0574,  0.0084, -0.0324,  0.0056,  0.0626, -0.0833,
        -0.0271, -0.0526,  0.0842, -0.0840, -0.0234, -0.0898, -0.0710, -0.0399,
         0.0183, -0.0883, -0.0102, -0.0545,  0.0706, -0.0646, -0.0841, -0.0095,
        -0.0823, -0.0385,  0.0327, -0.0810, -0.0404,  0.0570,  0.0740,  0.0829,
         0.0845,  0.0817, -0.0239, -0.0444, -0.0221,  0.0216,  0.0103, -0.0631,
         0.0831, -0.0273,  0.0756,  0.0022,  0.0407,  0.0072,  0.0374, -0.0608,
         0.0424, -0.0585,  0.0505, -0.0455,  0.0268, -0.0950, -0.0642,  0.0843,
         0.0760, -0.0889, -0.0617, -0.0916,  0.0102, -0.0269, -0.0011,  0.0318,
         0.0278, -0.0160,  0.0159, -0.0817,  0.0768, -0.0876, -0.0524, -0.0332,
        -0.0583,  0.0053,  0.0503, -0.0342, -0.0319, -0.0562,  0.0376, -0.0696,
         0.0735,  0.0222, -0.0775, -0.0072,  0.0294,  0.0994, -0.0355, -0.0809,
        -0.0539,  0.0245,  0.0670,  0.0032,  0.0891, -0.0694, -0.0994,  0.0126,
         0.0629,  0.0936,  0.0058, -0.0073,  0.0498,  0.0616, -0.0912, -0.0490],
       requires_grad=True)
Parameter containing:
tensor([[ 0.0504, -0.0203, -0.0573,  ...,  0.0253,  0.0642, -0.0088],
        [-0.0078, -0.0608, -0.0626,  ..., -0.0350, -0.0028, -0.0634],
        [-0.0317, -0.0202, -0.0593,  ..., -0.0280,  0.0571, -0.0114],
        ...,
        [ 0.0582, -0.0471, -0.0236,  ...,  0.0273,  0.0673,  0.0555],
        [ 0.0258, -0.0706,  0.0315,  ..., -0.0663, -0.0133,  0.0078],
        [-0.0062,  0.0544, -0.0280,  ..., -0.0303, -0.0326, -0.0462]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0385, -0.0116,  0.0703,  0.0407, -0.0346, -0.0178,  0.0308, -0.0502,
         0.0616,  0.0114], requires_grad=True)


Layer params:
Parameter containing:
tensor([[ 0.0504, -0.0203, -0.0573,  ...,  0.0253,  0.0642, -0.0088],
        [-0.0078, -0.0608, -0.0626,  ..., -0.0350, -0.0028, -0.0634],
        [-0.0317, -0.0202, -0.0593,  ..., -0.0280,  0.0571, -0.0114],
        ...,
        [ 0.0582, -0.0471, -0.0236,  ...,  0.0273,  0.0673,  0.0555],
        [ 0.0258, -0.0706,  0.0315,  ..., -0.0663, -0.0133,  0.0078],
        [-0.0062,  0.0544, -0.0280,  ..., -0.0303, -0.0326, -0.0462]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0385, -0.0116,  0.0703,  0.0407, -0.0346, -0.0178,  0.0308, -0.0502,
         0.0616,  0.0114], requires_grad=True)
```

这显示了 PyTorch 模型的基本结构： `__init__()` 方法定义模型的层和其他组件，而 `forward()` 方法完成计算。请注意，我们可以打印模型或它的任何子模块，以了解它的结构。

## 常见网络层类型

### 线性层

神经网络层的最基本类型是 _线性层或全连接层_。在这种层中，每个输入都会影响每个输出，影响程度由层的权重决定。如果一个模型有 `m` 个输入和 `n` 个输出，那么权重就是一个 _m_ x _n_ 矩阵。例如

```python
lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)
```

输出：
```shell
Input:
tensor([[0.8790, 0.9774, 0.2547]])


Weight and Bias parameters:
Parameter containing:
tensor([[ 0.1656,  0.4969, -0.4972],
        [-0.2035, -0.2579, -0.3780]], requires_grad=True)
Parameter containing:
tensor([0.3768, 0.3781], requires_grad=True)


Output:
tensor([[ 0.8814, -0.1492]], grad_fn=<AddmmBackward0>)
```

如果将 `x` 与线性层的权重进行矩阵乘法运算，再加上偏置，就能得到输出向量 `y`。

还有一个重要特征值得注意：当我们使用 `lin.weight` 检查层的权重时，它将自己报告为 `Parameter`(`Tensor` 的子类)，并让我们知道它正在使用自动微分跟踪梯度。与 `Tensor` 不同，这是 `Parameter` 的默认行为。

线性层广泛应用于深度学习模型中。在分类器模型中最常见到的就是线性层，分类器模型的最后通常会有一个或多个线性层，最后一层会有 `n` 个输出，其中 `n` 是分类器处理的类的数量。

### 卷积层

_卷积_ 层用于处理具有高度空间相关性的数据。卷积层常用于计算机视觉领域，它们能检测到特征的紧密组合，并将其组成更高级别的特征。卷积层也会出现在其他场合——例如，在 NLP 应用中，一个单词的上下文(即序列中附近的其他单词)会影响句子的意思。

我们在之前的视频中看到了卷积层在 LeNet5 中的应用：

```python
import torch.functional as F


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

让我们来分析一下这个模型的卷积层中发生了什么。从 `conv1` 开始：

- LeNet5 接收的是一幅 1x32x32 的黑白图像。**卷积层构造函数的第一个参数是输入通道数**，这里是 1。如果我们建立这个模型是为了查看 3 色通道，那么参数就是 3。
- 卷积层就像一个窗口，它会扫描图像，寻找它能识别的模式。这些模式被称为 _特征_，卷积层的参数之一就是我们希望它学习的特征数量。**这就是构造函数的第二个参数，输出特征的数量**。在这里，我们要求卷积层学习 6 个特征。
- 在上文，我把卷积层比作一个窗口，但窗口到底有多大？**第三个参数是窗口或内核大小**。这里的 "5" 表示我们选择了一个 5x5 的内核。(如果您想要一个宽高不同的内核，可以为这个参数指定一个元组，例如，(`3`, `5`)可以得到一个 3x5 的卷积内核)。

卷积层的输出是 _激活图_，即输入tensor中存在特征的空间表示。`conv1` 将为我们提供一个 6x28x28 的输出tensor，6 是特征的数量，28 是图的宽高。(28 是指在 32 像素行上扫描 5 像素窗口时，只有 28 个有效位置)。

然后，我们将卷积输出通过 ReLU 激活函数(激活函数稍后详述)，再通过最大池化层。最大池化层将激活图中相互靠近的特征集中在一起。具体做法是缩小tensor，将输出中的每组 2x2 单元合并为一个单元，并为该单元分配 4 个单元中的最大值。这样，我们就得到了激活图的低分辨率版本，尺寸为 6x14x14。

下一个卷积层 `conv2` 需要 6 个输入通道(与第一层的 6 个特征相对应)，16 个输出通道和一个 3x3 内核。它输出一个 16x12x12 的激活图，通过最大池化层再次将激活图缩小为 16x6x6。在将输出传递给线性层之前，它被重塑为一个 16 * 6 * 6 = 576 元素的向量，供下一层使用。

卷积层可用于处理一维、二维和三维tensor。卷积层构造函数还有更多可选参数，包括输入中的跨距长度(例如，每次只扫描第二或第三个位置)、填充(以便扫描到输入的边缘)等。更多信息请参阅[文档](https://pytorch.org/docs/stable/nn.html#convolution-layers)。

### 循环层

_RNN(Recurrent neural network，递归神经网络)_ 用于处理序列数据——从科学仪器的时间序列测量到自然语言句子再到 DNA 核苷酸，无所不包。RNN 通过保持一种 _隐藏状态_ 来实现这一功能，这种隐藏状态就像一种记忆，记录了它迄今为止在序列中看到的内容。

RNN 层或其变体 LSTM(long short-term memory，长短期记忆)和 GRU(gated recurrent unit，门控循环单元)的内部结构非常复杂，超出了本视频的讨论范围，但我们将向您展示基于 LSTM 的语音部分标记(一种分类器，能告诉您一个词是名词还是动词等)的实际效果：

```python
class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
```

这个构造函数有四个参数：

- `vocab_size` 是输入词汇的字数。每个单词都是 `vocab_size` 维空间中的独热(one-hot)向量(或单位向量)。
- `tagset_size` 是输出集合中标签的数量。
- `embedding_dim` 是词汇 _嵌入(embedding)_ 空间的大小。嵌入空间将词汇映射到一个低维空间中，在这个空间中，词义相近的词会靠得很近。
- `hidden_dim` 是 LSTM 内存的大小。

输入将是一个句子，其中的单词表示为独热(one-hot)向量的索引。然后，嵌入层会将这些词映射到一个 `embedding_dim` 维空间。LSTM 获取嵌入序列并对其进行迭代，得到长度为 `hidden_dim` 的输出向量。最后的线性层充当分类器，对最后一层的输出应用 `log_softmax()`，可将输出转换为一组归一化的估计概率，表示指定单词映射到指定标签的概率。

如果您想看看这个网络的实际应用，请查看 pytorch.org 上的[序列模型和 LSTM 网络教程](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)。

### 变换器

_变换器(Transformer)_ 是一种多用途网络，它与 BERT 等模型一起，称霸了 NLP 技术领域。讨论变换器架构超出了本视频的范围，但 PyTorch 有一个 `Transformer` 类，可以定义变换器模型的整体参数——注意力头的数量、编码器和解码器层的数量、随机失活(dropout)和激活函数等(甚至可以通过正确的参数用这个类来构建 BERT 模型！)。`torch.nn.Transformer` 类还封装了各个组件(`TransformerEncoder`、`TransformerDecoder`)和子组件(`TransformerEncoderLayer`、`TransformerDecoderLayer`)的类。有关详细信息，请查阅有关变换器类的[文档](https://pytorch.org/docs/stable/nn.html#transformer-layers)以及 pytorch.org 上的相关[教程](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)。

## 其他层和函数

### 数据操作层

还有其他一些类型的层在模型中起重要作用，但它们本身并不参与学习过程。

**最大池化**(和它兄弟，最小池化)通过合并单元来减少tensor，并将输入单元的最大值分配给输出单元(我们看到过这个)。例如

```python
my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))
```

输出：
```shell
tensor([[[0.5036, 0.6285, 0.3460, 0.7817, 0.9876, 0.0074],
         [0.3969, 0.7950, 0.1449, 0.4110, 0.8216, 0.6235],
         [0.2347, 0.3741, 0.4997, 0.9737, 0.1741, 0.4616],
         [0.3962, 0.9970, 0.8778, 0.4292, 0.2772, 0.9926],
         [0.4406, 0.3624, 0.8960, 0.6484, 0.5544, 0.9501],
         [0.2489, 0.8971, 0.7499, 0.1803, 0.9571, 0.6733]]])
tensor([[[0.7950, 0.9876],
         [0.9970, 0.9926]]])
```

如果仔细观察上面的值，就会发现已最大池化的输出中每个值都是 6x6 输入的各象限最大值。

**归一化层**会将一层的输出重新居中并归一化，然后再输入到另一层。将中间tensor居中和缩放有很多好处，比如可以让您使用更高的学习率，而不会导致梯度爆炸/消失。

```python
my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())
```

输出：
```shell
tensor([[[ 7.7375, 23.5649,  6.8452, 16.3517],
         [19.5792, 20.3254,  6.1930, 23.7576],
         [23.7554, 20.8565, 18.4241,  8.5742],
         [22.5100, 15.6154, 13.5698, 11.8411]]])
tensor(16.2188)
tensor([[[-0.8614,  1.4543, -0.9919,  0.3990],
         [ 0.3160,  0.4274, -1.6834,  0.9400],
         [ 1.0256,  0.5176,  0.0914, -1.6346],
         [ 1.6352, -0.0663, -0.5711, -0.9978]]],
       grad_fn=<NativeBatchNormBackward0>)
tensor(3.3528e-08, grad_fn=<MeanBackward0>)
```

运行上面的单元格时，我们为输入tensor添加了一个较大的缩放因子和偏移量，您应该看到输入tensor的 `mean()` 在 15 左右。在通过归一化层运行后，您可以看到数值变小了，并聚集在零附近——事实上，平均值应该非常小(> 1e-8)。

这样做是有好处的，因为许多激活函数(下面会讨论)在 0 附近的梯度最大，但有时把输入推远离 0 会出现梯度消失或爆炸。将数据保持在梯度最陡峭区域的中心，往往意味着更快更好的学习和更高的可行学习率。

**随机失活(Dropout)层**是模型中一种鼓励 _稀疏表示_ 的工具，即让模型用更少的数据进行推理。

随机失活层的工作原理是在 _训练过程中_ 随机设置输入tensor的一部分——随机失活层在推理时始终处于关闭状态。这就迫使模型根据这种被屏蔽或减少的数据集进行学习。例如：

```python
my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))
```

输出：
```shell
tensor([[[0.8869, 0.6595, 0.2098, 0.0000],
         [0.5379, 0.0000, 0.0000, 0.0000],
         [0.1950, 0.2424, 1.3319, 0.5738],
         [0.5676, 0.8335, 0.0000, 0.2928]]])
tensor([[[0.8869, 0.6595, 0.2098, 0.2878],
         [0.5379, 0.0000, 0.4029, 0.0000],
         [0.0000, 0.2424, 1.3319, 0.5738],
         [0.0000, 0.8335, 0.9647, 0.0000]]])
```

从上面，您可以看到样本tensor应用随机失活的效果。您可以使用可选的 `p` 参数设置单个权重失活的概率，如果不使用，默认值为 0.5。

### 激活函数

激活函数使得深度学习成为可能。神经网络实际上是一个带有许多参数的程序，用于 _模拟数学函数_。如果我们只是反复将tensor乘以层权重，那么我们只能模拟 _线性函数_，另外没有必要使用多层，因为整个网络可以简化为一次矩阵乘法。在层之间插入非线性激活函数允许深度学习模型模拟任何函数，而不仅仅是线性函数。

`torch.nn.Module` 中包含封装了所有主要激活函数的对象，包括 ReLU 及其许多变种、Tanh、Hardtanh、sigmoid 等。它还包括其他在模型输出阶段最有用的函数，例如 Softmax。

### 损失函数

损失函数告诉我们一个模型的预测离正确答案有多远。PyTorch 包含多种损失函数，包括常见的 MSE(均方误差 = L2 范数)、交叉熵损失和负对数似然损失(适用于分类器)等。