# 保存和加载模型

> **作者**：[Matthew Inkawhich](https://github.com/MatthewInkawhich)
>
> 译者：[片刻](https://github.com/jiangzhonglian)
>
> 校验：[片刻](https://github.com/jiangzhonglian)

本文档为有关保存和加载PyTorch模型的各种用例提供​​了解决方案。随意阅读整个文档，或者只是跳到所需用例所需的代码。

关于保存和加载模型，有三个核心功能需要熟悉：

1. [torch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save)：将序列化的对象保存到磁盘。此函数使用Python的 [pickle](https://docs.python.org/3/library/pickle.html)实用程序进行序列化。使用此功能可以保存各种对象的模型，张量和字典。
2. [torch.load](https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load)：使用[pickle](https://docs.python.org/3/library/pickle.html)的解腌功能将腌制的目标文件反序列化到内存中。此功能还有助于设备将数据加载到其中(请参阅 [跨设备保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices)）。
3. [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/nn.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)：使用反序列化的*state_dict*加载模型的参数字典 。有关*state_dict的*更多信息，请参阅[什么是state_dict？](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)。

**内容：**

* [什么是state_dict？](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)
* [推理的保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)
* [保存和加载常规检查点](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training)
* [将多个模型保存在一个文件中](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-multiple-models-in-one-file)
* [使用来自不同模型的参数进行热启动模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html#warmstarting-model-using-parameters-from-a-different-model)
* [跨设备保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices)


## 什么是`state_dict`？

在PyTorch中，模型的可学习参数(即权重和偏差） `torch.nn.Module` 包含在模型的参数中 (通过访问`model.parameters()`）。甲state_dict是一个简单的Python字典对象，每个层映射到其参数张量。请注意，只有具有可学习参数的层(卷积层，线性层等）和已注册的缓冲区(batchnorm的running_mean）才在模型的state_dict中具有条目。优化器对象(`torch.optim`）还具有state_dict，其中包含有关优化器状态以及所用超参数的信息。

由于 *state_dict* 对象是Python词典，因此可以轻松地保存，更新，更改和还原它们，从而为PyTorch模型和优化器增加了很多模块化。


### 例如：

让我们从[训练分类器](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) 教程中使用的简单模型 看一下state_dict。


```python    
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
    

**输出：**

    
```python 
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

## 推理模型的保存和加载

### 保存/加载`state_dict`(推荐）

**Save:**

    
```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

保存模型以进行推理时，仅需要保存训练后的模型的学习参数。使用 `torch.save()` 函数保存模型的state_dict将为您提供最大的灵活性，以便以后恢复模型，这就是为什么推荐使用此方法来保存模型。

常见的PyTorch约定是使用`.pt`或 `.pth`文件扩展名保存模型。

请记住，`model.eval()`在运行推理之前，必须先调用以将退出和批处理规范化层设置为评估模式。不这样做将产生不一致的推断结果。


> Note
请注意，该`load_state_dict()`函数使用字典对象，而不是保存对象的路径。这意味着，在将保存的state_dict传递给`load_state_dict()`函数之前 ，必须对其进行反序列化。例如，您无法使用加载 `model.load_state_dict(PATH)`。

### 整个模型的保存和加载

**Save:**

```python
torch.save(model, PATH)
```  

**Load:**
    
```python
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()
```
    
此保存/加载过程使用最直观的语法，并且涉及最少的代码。以这种方式保存模型将使用Python的[pickle](https://docs.python.org/3/library/pickle.html)模块保存整个 模块。这种方法的缺点是序列化的数据绑定到特定的类，并且在保存模型时使用确切的目录结构。这样做的原因是因为pickle不会保存模型类本身。而是将其保存到包含类的文件的路径，该路径在加载时使用。因此，在其他项目中使用或重构后，您的代码可能会以各种方式中断。

常见的PyTorch约定是使用`.pt`或 `.pth`文件扩展名保存模型。

请记住，`model.eval()`在运行推理之前，必须先调用以将退出和批处理规范化层设置为评估模式。不这样做将产生不一致的推断结果。


## 保存和加载用于推理和/或继续训练的常规检查点

### Save:

    
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    ...
    }, PATH)
``` 


### Load:

```python
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

保存用于检查或继续训练的常规检查点时，您必须保存的不仅仅是模型的 *state_dict*。保存优化器的state_dict也是很重要的，因为它包含随着模型训练而更新的缓冲区和参数。您可能要保存的其他项目包括您未启用的时期，最新记录的训练损失，外部`torch.nn.Embedding` 图层等。

要保存多个组件，请将它们组织在字典中并用于 torch.save()序列化字典。常见的PyTorch约定是使用`.tar`文件扩展名保存这些检查点。

要加载项目，请首先初始化模型和优化器，然后使用本地加载字典`torch.load()`。从这里，您只需按期望查询字典即可轻松访问已保存的项目。

请记住，`model.eval()`在运行推理之前，必须先调用以将退出和批处理规范化层设置为评估模式。不这样做将产生不一致的推断结果。如果您希望恢复训练，请调用`model.train()`以确保这些层处于训练模式。

## 将多个模型保存在一个文件中

### Save:

```python 
torch.save({
    'modelA_state_dict': modelA.state_dict(),
    'modelB_state_dict': modelB.state_dict(),
    'optimizerA_state_dict': optimizerA.state_dict(),
    'optimizerB_state_dict': optimizerB.state_dict(),
    ...
    }, PATH)
```


### Load:

```python
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
    
保存由多个模型组成的模型时`torch.nn.Modules`，例如GAN，序列到序列模型或模型集成，您将采用与保存常规检查点相同的方法。换句话说，保存每个模型的*state_dict*和相应的优化器的字典。如前所述，您可以保存任何其他可以帮助您恢复训练的项目，只需将它们添加到字典中即可。

常见的PyTorch约定是使用`.tar`文件扩展名保存这些检查点。

要加载模型，请首先初始化模型和优化器，然后使用本地加载字典`torch.load()`。从这里，您只需按期望查询字典即可轻松访问已保存的项目。

请记住，`model.eval()`在运行推理之前，必须先调用以将退出和批处理规范化层设置为评估模式。不这样做将产生不一致的推断结果。如果您希望恢复训练，请调用`model.train()`将这些图层设置为训练模式。

## 使用来自不同模型的参数进行热启动模型

### Save:

```python
torch.save(modelA.state_dict(), PATH)
``` 
    

### Load:

```python
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
```   

在转移学习或训练新的复杂模型时，部分加载模型或加载部分模型是常见方案。利用经过训练的参数，即使只有少数几个可用的参数，也将有助于热启动训练过程，并希望与从头开始训练相比，可以更快地收敛模型。

无论是从缺少某些键的部分state_dict加载，还是要加载比要加载的模型更多的key 的`state_dict`，都可以 在函数中将strict参数设置为**False**，`load_state_dict()`以忽略不匹配的键。

如果要将参数从一层加载到另一层，但是某些键不匹配，只需更改要加载的`state_dict`中参数键的名称， 以匹配要加载到的模型中的键。

## 跨设备保存和加载模型

### 保存在GPU上，在CPU上加载

**Save:**

```python
torch.save(model.state_dict(), PATH)
```  

**Load:**

```
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```

在使用GPU训练的CPU上加载模型时，将其传递 `torch.device('cpu')`给`map_location`函数中的 `torch.load()`参数。在这种情况下，使用`map_location`参数将张量下面的存储动态地重新映射到CPU设备。

### 保存在GPU上，在GPU上加载

**Save:**

```python
torch.save(model.state_dict(), PATH)
``` 

**Load:**

```python
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
``` 


在经过训练并保存在GPU上的GPU上加载模型时，只需使用`model.to(torch.device('cuda')`将初始化后的模型转换为CUDA优化模型即可。 另外，请确保在所有模型输入上使用`.to(torch.device('cuda'))`函数，以为模型准备数据。 请注意，调用`my_tensor.to(device)`会在GPU上返回my_tensor的新副本。 它不会覆盖`my_tensor`。因此，请记住手动覆盖张量：`my_tensor = my_tensor.to(torch.device('cuda'))`。


### 保存CPU，加载在GPU

**Save:**

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python   
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

在经过训练并保存在CPU上的GPU上加载模型时，请将`torch.load()`函数中的`map_location`参数设置为cuda：*device_id*。 这会将模型加载到给定的GPU设备。 接下来，请确保调用`model.to(torch.device('cuda'))`将模型的参数张量转换为CUDA张量。 最后，请确保在所有模型输入上使用`.to(torch.device('cuda'))`函数，以为CUDA优化模型准备数据。 请注意，调用`my_tensor.to(device)`会在GPU上返回my_tensor的新副本。 它不会覆盖`my_tensor`。 因此，请记住手动覆盖张量：`my_tensor = my_tensor.to(torch.device('cuda'))`。

### 保存`torch.nn.DataParallel`模型

**Save:**

```python
torch.save(model.module.state_dict(), PATH)
```    
    
**Load:**

```python
# Load to whatever device you want
```    
    
`torch.nn.DataParallel`是支持并行GPU利用率的模型包装器。要以`DataParallel`一般方式保存模型，请保存 `model.module.state_dict()`。这样，您便可以灵活地将所需的模型加载到所需的任何设备。

**脚本的总运行时间：** (0分钟0.000秒）
