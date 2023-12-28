
# 使用 Better Transformer 进行快速变压器推理 [¶](#fast-transformer-inference-with-better-transformer "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)

>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/bettertransformer_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/bettertransformer_tutorial.html>

**作者**: [Michael Gschwind](https://github.com/mikekgfb)

 本教程介绍了 Better Transformer (BT)，作为 PyTorch 1.12 版本的一部分。
在本教程中，我们将展示如何使用 Better Transformer 与 torchtext 进行生产\推理。 Better Transformer 是一个生产就绪的快速路径，
可在 CPU 和 GPU 上加速 Transformer 模型的部署，并具有高性能。
快速路径功能对于直接基于 PyTorch 核心`nn.module`或使用 torchtext 的模型透明地工作。

 可以通过 Better Transformer 快速路径执行加速的模型是那些使用以下 PyTorch 核心的`torch.nn.module`类 `TransformerEncoder`、`TransformerEncoderLayer` 和 `MultiHeadAttention`. 此外，torchtext 已更新为使用核心库模块，从而受益于快速路径加速。(将来可能会通过快速路径执行启用其他模块。)

 Better Transformer 提供两种类型的加速：

* 针对 CPU 和 GPU 的本机多头注意力 (MHA) 实现，以提高整体执行效率。
* 利用 NLP 推理中的稀疏性。由于输入长度可变，输入令牌可能包含大量填充令牌，可能会跳过这些填充令牌，从而显着提高速度。

 快速路径执行需要遵守一些标准。最重要的是，模型必须在推理模式下执行，并在不收集梯度带信息的输入tensor上运行(例如，使用 torch.no_grad 运行)。

 要在 Google Colab 中遵循此示例，[单击此处](https://colab.research.google.com/drive/1KZnMJYhYkOMYtNIX5S3AGIYnjyG0AojN?usp=sharing).

## 本教程中更好的 Transformer 功能 [¶](#better-transformer-features-in-this-tutorial "永久链接到此标题")

* 加载预训练模型(在 PyTorch 版本 1.12 之前创建，没有 Better Transformer)
* 在具有或不具有 BT 快速路径的 CPU 上运行推理并进行基准测试(仅限本机 MHA)
* 在具有或不具有 BT 快速路径的(可配置)设备上运行推理并进行基准测试(仅限本机 MHA)
* 启用稀疏性支持
* 在具有和不具有 BT 快速路径的(可配置)设备上运行和基准测试推理(本机 MHA + 稀疏性)

## 附加信息 [¶](#additional-information "此标题的永久链接")

 有关 Better Transformer 的更多信息，请参阅 PyTorch.Org 博客 [用于快速 Transformer 推理的 Better Transformer](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-编码器推断//).

1. 设置

 1.1 加载预训练模型

 我们按照 [torchtext.models](https://pytorch.org/text/main/models.html) 中的说明从预定义的 torchtext 模型中下载 XLM-R 模型。我们还将设备设置为执行\非加速器测试。 (根据需要为您的环境启用 GPU 执行。)

```python
import torch
import torch.nn as nn

print(f"torch version: {torch.__version__}")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"torch cuda available: {torch.cuda.is_available()}")

import torch, torchtext
from torchtext.models import RobertaClassificationHead
from torchtext.functional import to_tensor
xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
classifier_head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim = 1024)
model = xlmr_large.get_model(head=classifier_head)
transform = xlmr_large.transform()

```

 1.2 数据集设置

 我们设置了两种类型的输入：小型输入批次和稀疏的大型输入批次。

```python
small_input_batch = [
               "Hello world",
               "How are you!"
]
big_input_batch = [
               "Hello world",
               "How are you!",
 """`Well, Prince, so Genoa and Lucca are now just family estates of the
Buonapartes. But I warn you, if you don't tell me that this means war,
if you still try to defend the infamies and horrors perpetrated by
that Antichrist- I really believe he is Antichrist- I will have
nothing more to do with you and you are no longer my friend, no longer
my 'faithful slave,' as you call yourself! But how do you do? I see
I have frightened you- sit down and tell me all the news.`

It was in July, 1805, and the speaker was the well-known Anna
Pavlovna Scherer, maid of honor and favorite of the Empress Marya
Fedorovna. With these words she greeted Prince Vasili Kuragin, a man
of high rank and importance, who was the first to arrive at her
reception. Anna Pavlovna had had a cough for some days. She was, as
she said, suffering from la grippe; grippe being then a new word in
St. Petersburg, used only by the elite."""
]
```

 接下来，我们选择小输入批次或大输入批次，预处理输入并测试模型。

```python
input_batch=big_input_batch

model_input = to_tensor(transform(input_batch), padding_value=1)
output = model(model_input)
output.shape
```

 最后，我们设置基准迭代计数：

```python
ITERATIONS=10
```

2.执行

 2.1 在具有和不具有 BT 快速路径的 CPU 上运行和基准测试推理(仅限本机 MHA)

 我们在 CPU 上运行模型，并收集配置文件信息：

* 第一次运行使用传统的 (“slow path”) 执行。
* 第二次运行通过使用模型将模型置于推理模式来启用 BT 快速路径执行 _model.eval()_,并使用 _torch.no_grad()_ 禁用梯度收集。

 当模型在 CPU 上执行时，您可以看到改进(其幅度取决于 CPU 型号)。请注意，快速路径配置文件显示了本机 TransformerEncoderLayer 实现 aten::_transformer_encoder_layer_fwd 的大部分执行时间。

```python
print("slow path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=False) as prof:
  for i in range(ITERATIONS):
    output = model(model_input)
print(prof)

model.eval()

print("fast path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=False) as prof:
  with torch.no_grad():
    for i in range(ITERATIONS):
      output = model(model_input)
print(prof)

```

 2.2 在具有和不具有 BT 快速路径的(可配置)设备上运行和基准测试推理(仅限本机 MHA)

 我们检查 BT 稀疏性设置：

```python
model.encoder.transformer.layers.enable_nested_tensor
```

 我们禁用 BT 稀疏性：

```python
model.encoder.transformer.layers.enable_nested_tensor=False
```

 我们在 DEVICE 上运行模型，并收集 DEVICE 上本机 MHA 执行的配置文件信息:

* 第一次运行使用传统的 (“slow path”) 执行。
* 第二次运行通过使用模型将模型置于推理模式来启用 BT 快速路径执行 model.eval() 并使用 torch.no_grad() 禁用梯度收集。

 在 GPU 上执行时，您应该会看到显着的加速，特别是对于小输入批量设置：

```python
model.to(DEVICE)
model_input = model_input.to(DEVICE)

print("slow path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  for i in range(ITERATIONS):
    output = model(model_input)
print(prof)

model.eval()

print("fast path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  with torch.no_grad():
    for i in range(ITERATIONS):
      output = model(model_input)
print(prof)
```

 2.3 在具有和不具有 BT 快速路径的(可配置)设备上运行和基准测试推理(本机 MHA + 稀疏性)

 我们启用稀疏性支持：

```python
model.encoder.transformer.layers.enable_nested_tensor = True
```

 我们在 DEVICE 上运行模型，并收集设备上本机 MHA 和稀疏性支持执行的配置文件信息：

* 第一次运行使用传统的 (“slow path”) 执行。
* 第二次运行通过使用模型将模型置于推理模式来启用 BT 快速路径执行 _model.eval()_,并使用 _torch.no_grad()_ 禁用梯度收集。

在 GPU 上执行时，您应该会看到显着的加速，特别是对于包含稀疏性的大输入批处理设置：

```python
model.to(DEVICE)
model_input = model_input.to(DEVICE)

print("slow path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  for i in range(ITERATIONS):
    output = model(model_input)
print(prof)

model.eval()

print("fast path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  with torch.no_grad():
    for i in range(ITERATIONS):
      output = model(model_input)
print(prof)

```

## 摘要 [¶](#summary "此标题的永久链接")

 在本教程中，我们介绍了快速转换器推理，
使用 PyTorch 核心在 torchtext 中实现了更好的 Transformer 快速路径执行
对 Transformer 编码器模型的更好的 Transformer 支持。我们已经展示了 Better Transformer 与在 BT 快速路径执行可用之前训练的模型的使用。我们已经演示了 BT 快速路径执行模式、本机 MHA 执行和 BT 稀疏加速的使用，
并对其进行了基准测试。
