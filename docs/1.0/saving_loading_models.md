

# 保存和加载模型

> 译者 [bruce1408](https://github.com/bruce1408)

**作者:** [Matthew Inkawhich](https://github.com/MatthewInkawhich)

本文提供有关Pytorch模型保存和加载的各种用例的解决方案。您可以随意阅读整个文档，或者只是跳转到所需用例的代码部分。

当保存和加载模型时，有三个核心功能需要熟悉：

1.  [torch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save): 将序列化对象保存到磁盘。 此函数使用 Python 的[pickle](https://docs.python.org/3/library/pickle.html)模块进行序列化。使用此函数可以保存如模型、tensor、字典等各种对象。
2.  [torch.load](https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load): 使用 [pickle](https://docs.python.org/3/library/pickle.html)的 unpickling 功能将pickle对象文件反序列化到内存。 此功能还可以有助于设备加载数据(详见 [Saving & Loading Model Across Devices](#saving-loading-model-across-devices)).
3.  [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/nn.html?highlight=load_state_dict#torch.nn.Module.load_state_dict): 使用反序列化函数 _state_dict_ 来加载模型的参数字典。更多有关 _state_dict_ 的信息，请参考[What is a state_dict?](#what-is-a-state-dict).

**内容:**

*   [什么是`状态字典`?](#what-is-a-state-dict)
*   [保存和加载推断模型](#saving-loading-model-for-inference)
*   [保存 和 加载 Checkpoint](#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training)
*   [在一个文件中保存多个模型](#saving-multiple-models-in-one-file)
*   [使用在不同模型参数下的热启动模式](#warmstarting-model-using-parameters-from-a-different-model)
*   [Saving & Loading Model Across Devices](#saving-loading-model-across-devices)

## 什么是 `状态字典`?

在Pytorch中，`torch.nn.Module` 模型的可学习参数(即权重和偏差)包含在模型的 _parameters_ 中，(使用`model.parameters()`可以进行访问)。 _state_dict_ 仅仅是python字典对象，它将每一层映射到其参数张量。注意，只有具有可学习参数的层(如卷积层、线性层等)的模型才具有 _state_dict_ 这一项。优化目标 `torch.optim` 也有 _state_dict_ 属性，它包含有关优化器的状态信息，以及使用的超参数。

因为 _state_dict_ 的对象是python字典，所以他们可以很容易的保存、更新、更改和恢复，为Pytorch模型和优化器添加了大量模块。

### 示例:

让我们从 简单模型[训练一个分类器](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)中了解一下 _state_dict_ 的使用。

```py
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

```

**输出:**

```py
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])

Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]

```

## 保存和加载推断模型

### 保存/加载 `state_dict` (推荐使用)

**保存:**

```py
torch.save(model.state_dict(), PATH)

```

**加载:**

```py
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

```

当保存好模型用来推断的时候，只需要保存模型学习到的参数，使用 `torch.save()` 函数来保存模型 _state_dict_ ,它会给模型恢复提供最大的灵活性，这就是为什么要推荐它来保存的原因。

在 Pytorch 中最常见的模型保存使用 ‘.pt’ 或者是 ‘.pth’ 作为模型文件扩展名。
  
请记住，在运行推理之前，务必调用 `model.eval()` 去设置 dropout 和 batch normalization 层为评估模式。如果不这么做，可能导致模型推断结果不一致。

注意

请注意 `load_state_dict()` 函数只接受字典对象，而不是保存对象的路径。这就意味着在你传给 `load_state_dict()` 函数之前，你必须反序列化你保存的 _state_dict_。例如，你无法通过 `model.load_state_dict(PATH)`来加载模型。

### 保存/加载完整模型

**保存:**

```py
torch.save(model, PATH)

```

**加载:**

```py
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()

```
此部分保存/加载过程使用最直观的语法并涉及最少量的代码。以Python[pickle](https://docs.python.org/3/library/pickle.html)模块的方式来保存模型。这种方法的缺点是序列化数据受限于某种特殊的类而且需要确切的字典结构。这是因为pickle无法保存模型类本身。相反，它保存包含类的文件的路径，该文件在加载时使用。因此，当在其他项目使用或者重构之后，您的代码可能会以各种方式中断。

在 Pytorch 中最常见的模型保存使用 ‘.pt’ 或者是 ‘.pth’ 作为模型文件扩展名。

请记住，在运行推理之前，务必调用 `model.eval()` 去设置 dropout 和 batch normalization 层为评估模式。如果不这么做，可能导致模型推断结果不一致。

## 保存 和 加载 Checkpoint 用于推理/继续训练

### 保存:

```py
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

```

### 加载:

```py
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()

```

当保存成 checkpoint 的时候，可用于推理或者是恢复训练，您保存的不仅仅是模型的 _state_dict_ 。 保存优化器的 _state_dict_ 也很重要, 因为它包含作为模型训练更新的缓冲区和参数。你也许想保存其他项目，比如最新记录的训练损失，外部的 `torch.nn.Embedding` 层等等。

要保存多个组件，请在字典中组织它们并使用 `torch.save()` 来序列化字典。 Pytorch 中常见的保存checkpoint 是使用 `.tar` 文件扩展名。

要加载项目，首先需要初始化模型和优化器，然后使用 `torch.load()` 来加载本地字典。 这里，您可以非常容易的通过简单查询字典来访问您所保存的项目。

请记住在运行推理之前，务必调用 `model.eval()` 去设置 dropout 和 batch normalization 为评估。如果不这样做，有可能得到不一致的推断结果。如果你想要恢复训练，请调用 `model.train()` 以确保这些层处于训练模式。

## 在一个文件中保存多个模型

### 保存:

```py
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)

```

### 加载:

```py
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()

```

当保存一个模型由多个 `torch.nn.Modules`组成时，例如GAN(对抗生成网络), sequence-to-sequence (序列到序列模型), 或者是多个模型融合, 您可以采用与保存常规检查点相同的方法。换句话说，保存每个模型的 _state_dict_ 的字典和相对应的优化器。如前所述，您可以通过简单地将它们附加到字典的方式来保存任何其他项目，这样有助于您恢复训练。

Pytorch 中常见的保存checkpoint 是使用 `.tar` 文件扩展名。

要加载项目，首先需要初始化模型和优化器，然后使用 `torch.load()` 来加载本地字典。 这里，您可以非常容易的通过简单查询字典来访问您所保存的项目。

请记住在运行推理之前，务必调用 `model.eval()` 去设置 dropout 和 batch normalization 为评估。如果不这样做，有可能得到不一致的推断结果。如果你想要恢复训练，请调用 `model.train()` 以确保这些层处于训练模式。

## 使用在不同模型参数下的热启动模式

### 保存:

```py
torch.save(modelA.state_dict(), PATH)

```

### 加载:

```py
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)

```

在迁移学习或训练新的复杂模型时， 部分加载模型或加载部分模型是常见的情况。利用训练好的参数，有助于热启动训练过程，并希望帮助您的模型比从头开始训练更快地收敛

无论是从缺少某些键的 _state_dict_ 加载还是从键数多于加载模型的 _state_dict_ , 您可以通过在`load_state_dict()`函数中将`strict`参数设置为 **False** 来忽略非匹配键的函数。

如果要将参数从一个层加载到另一个层，但是某些键不匹配，主要修改正在加载的 _state_dict_ 中的参数键的名称以匹配要在加载到模型中的键即可。

## 通过设备保存/加载模型

### 保存到 GPU, 加载到 CPU

**保存:**

```py
torch.save(model.state_dict(), PATH)

```

**加载:**

```py
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

```

当从CPU上加载模型在GPU上训练时, 将 `torch.device('cpu')` 传递给 `torch.load()` 函数中的 `map_location`参数.在这种情况下，使用`map_location` 参数将张量下的存储器动态的重新映射到CPU设备。

### 保存到 GPU, 加载到 GPU

**保存:**

```py
torch.save(model.state_dict(), PATH)

```

**加载:**

```py
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model

```

当在GPU上训练并把模型保存在GPU，只需要使用 `model.to(torch.device('cuda'))`，将初始化的 `model` 转换为CUDA优化模型。另外，请务必在所有模型输入上使用 `.to(torch.device('cuda'))` 函数来为模型准备数据。请注意，调用 `my_tensor.to(device)` 会在GPU上返回`my_tensor` 的副本。因此，请记住手动覆盖张量：`my_tensor= my_tensor.to(torch.device('cuda'))`。

### 保存到 CPU, 加载到 GPU

**保存:**

```py
torch.save(model.state_dict(), PATH)

```

**加载:**

```py
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model

```

在CPU上训练好并保存的模型加载到GPU时， 将`torch.load()` 函数中的 `map_location` 参数设置为 _cuda:device_id_。这会将模型加载到指定的GPU设备。接下来，请务必调用 `model.to(torch.device('cuda'))` 将模型的参数张量转换为 CUDA 张量。最后，确保在所有模型输入上使用 `.to(torch.device('cuda'))` 函数来为CUDA优化模型。请注意， 调用 `my_tensor.to(device)` 会在GPU上返回 `my_tensor` 的新副本。 它不会覆盖 `my_tensor`。因此， 请手动覆盖张量 `my_tensor = my_tensor.to(torch.device('cuda'))`。

### 保存 `torch.nn.DataParallel` 模型

**保存:**

```py
torch.save(model.module.state_dict(), PATH)

```

**加载:**

```py
# Load to whatever device you want

```

`torch.nn.DataParallel` 是一个模型封装，支持并行GPU使用。要一般性的保存 `DataParallel` 模型, 请保存 `model.module.state_dict()`。这样，您就可以非常灵活地以任何方式加载模型到您想要的设备中。
