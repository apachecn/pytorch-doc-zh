# 保存和加载模型

**作者：** [马修Inkawhich ](https://github.com/MatthewInkawhich)

本文档提供解决方案，以各种关于PyTorch模型的保存和加载使用情况。随时阅读整个文档，或者只是跳到你需要一个期望的使用情况下的代码。

当涉及到保存和加载模型，有三个核心功能熟悉：

  1. [ torch.save ](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save)：保存一个序列化的对象到磁盘。此功能使用Python的[泡菜](https://docs.python.org/3/library/pickle.html)实用程序进行序列化。模型，张量，以及各类对象的字典可以使用该功能进行保存。
  2. [ torch.load ](https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load)：使用[泡菜的](https://docs.python.org/3/library/pickle.html)在unpickle设施到腌对象文件反序列化到存储器。该功能也有助于该装置加载数据到（见保存&安培;荷载模型跨设备）。
  3. [ torch.nn.Module.load_state_dict ](https://pytorch.org/docs/stable/nn.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)：使用反序列化 _state_dict_ 加载一个模型的参数字典。有关 _更多信息state_dict_ 参见什么是state_dict？ 。

**内容：**

  * 什么是state_dict？ 
  * 保存&安培;为推理荷载模型
  * 保存&安培;载入通用检查点
  * 在一个文件中保存多个模型
  * Warmstarting模型从一个不同的模型使用参数
  * 保存&安培;荷载模型跨设备

## 什么是`state_dict`？

在PyTorch中，可学习的参数（即重量和偏见）的`torch.nn.Module`模型中包含的模型 _参数_ （带有访问`
model.parameters（） `）。 A _state_dict_
仅仅是每一层映射到其参数张量Python字典对象。注意与可学习参数（卷积层，线性层等）和注册缓冲器（batchnorm的running_mean），只有层具有在条目模型的
_state_dict_ 。优化器对象（`torch.optim`）也有一个 _state_dict_
，它包含有关该优化程序的状态的信息，以及所使用的超参数。

因为 _state_dict_ 对象是Python字典，它们可以方便地保存，更新，修改和恢复，加上模块化的大量工作PyTorch模型和优化。

### 例如：

让我们来看看从[训练分类](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-
glr-beginner-blitz-cifar10-tutorial-py)教程中使用的简单模型 _state_dict [HTG1。_

    
    
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
    
    

**输出：**

    
    
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
    

## 节省&安培;为荷载模型推断

### 保存/加载`state_dict`（推荐）

**保存：**

    
    
    torch.save(model.state_dict(), PATH)
    
    

**负载：**

    
    
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    

当节省推理模型，只需要保存训练模型的参数得知。保存模型 _state_dict_ 与`torch.save（）
`功能会给你最大的灵活性后恢复模型，这就是为什么它是推荐的方法为保存模型。

一个常见的PyTorch惯例是使用一个`.PT`或`.pth`文件扩展名来保存模式。

请记住，你必须调用`model.eval（） `运行推论之前设置辍学率和批标准化层为评估模式。如果不这样做会产生不一致的推断结果。

Note

注意，`load_state_dict（） `函数采用一个字典对象，而不是路径保存的对象。这意味着你必须反序列化保存 _state_dict_
你将它传递给`load_state_dict前（） `功能。例如，你不能加载使用`model.load_state_dict（PATH） `。

### SAVE / LOAD整个模型

**Save:**

    
    
    torch.save(model, PATH)
    
    

**Load:**

    
    
    # Model class must be defined somewhere
    model = torch.load(PATH)
    model.eval()
    
    

此保存/加载处理使用最直观的语法和涉及的代码量最少。以这种方式保存的模型将使用Python的[泡菜](https://docs.python.org/3/library/pickle.html)模块保存整个模块。这种方法的缺点是串行数据绑定到特定的类和在保存的模型中使用的精确的目录结构。这样做的原因是因为泡菜不保存模型类本身。相反，它保存到包含类，这是在负载时所使用的文件的路径。正因为如此，你的代码可以在其他项目或refactors后使用时，以各种方式突破。

A common PyTorch convention is to save models using either a `.pt`or `.pth`
file extension.

Remember that you must call `model.eval()`to set dropout and batch
normalization layers to evaluation mode before running inference. Failing to
do this will yield inconsistent inference results.

## 保存&放;加载一般检查点推断和/或恢复训练

### 保存：

    
    
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                ...
                }, PATH)
    
    

### 负载：

    
    
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
    
    

当保存一般的检查点，以用于任何推理或恢复训练，你必须保存不仅仅是模型的 _state_dict [HTG1。它也保存重要的是优化的 _state_dict_
，因为这包含了更新的模型火车缓冲区和参数。你可能希望保存其他项目，你离开时，最新的培训记录丢失，外部`torch.nn.Embedding`层等时代_

保存多个组件，但在一个字典组织它们，并使用`torch.save（） `序列化的词典。一个常见的PyTorch约定是为了节省使用`的.tar
`文件扩展名，这些检查站。

要装入的物品，首先初始化模型和优化器，然后装入词典本地使用`torch.load（）
`。从这里，你可以很容易地通过简单的查询你所期望的字典访问保存的项目。

请记住，你必须调用`model.eval（） `运行推论之前设置辍学率和批标准化层为评估模式。如果不这样做会产生不一致的推断结果。如果你想恢复训练，调用`
model.train（） `，以确保这些层在训练模式。

## 在一个文件中保存多个模型

### 保存：

    
    
    torch.save({
                'modelA_state_dict': modelA.state_dict(),
                'modelB_state_dict': modelB.state_dict(),
                'optimizerA_state_dict': optimizerA.state_dict(),
                'optimizerB_state_dict': optimizerB.state_dict(),
                ...
                }, PATH)
    
    

### 负载：

    
    
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
    
    

当保存由多个`torch.nn.Modules
`，例如GAN，序列到序列模型或模型的集合的模型，就按照同样的方法，因为当您要保存一般的检查点。换言之，保存每个模型的 _state_dict_
和相应的优化的字典。正如前面提到的，你可以保存可以通过简单地追加他们的字典帮助您恢复训练其他任何物品。

一个常见的PyTorch约定是为了节省使用`的.tar`文件扩展名，这些检查站。

要加载模型中，首先初始化模型和优化器，然后装入词典本地使用`torch.load（）
`。从这里，你可以很容易地通过简单的查询你所期望的字典访问保存的项目。

请记住，你必须调用`model.eval（） `运行推论之前设置辍学率和批标准化层为评估模式。如果不这样做会产生不一致的推断结果。如果你想恢复训练，调用`
model.train（） `设置这些层的培训模式。

## Warmstarting模型中使用的参数从一个不同的模型

### 保存：

    
    
    torch.save(modelA.state_dict(), PATH)
    
    

### 负载：

    
    
    modelB = TheModelBClass(*args, **kwargs)
    modelB.load_state_dict(torch.load(PATH), strict=False)
    
    

部分加载模型或加载局部模型常见的场景时，转移学习或培训新的复杂的模型。凭借训练有素的参数，即使只有少数是可用的，将有助于WARMSTART训练过程，并希望能帮助你的模型收敛比从头训练快得多。

无论您是从装载部分 _state_dict_ ，它缺少一些键，或加载 _state_dict_ 与比要装载到模型更加按键，可以设置`严格 `参数为
**假** 在`load_state_dict（） `函数忽略非匹配密钥。

如果你想从一个层对其他负载参数，但有些键不匹配，只需更改 _参数键的名称state_dict_ 您加载以匹配你是模型的关键装入。

## 节省&安培;荷载模型跨设备

### 保存在GPU上，加载在CPU

**Save:**

    
    
    torch.save(model.state_dict(), PATH)
    

**Load:**

    
    
    device = torch.device('cpu')
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location=device))
    
    

当加载，将其与一个GPU培养了CPU上的模型，通过`torch.device（ 'CPU'） `到`map_location`在`
torch.load参数（） `功能。在这种情况下，张量基础的存储器使用`map_location`参数动态地重新映射到CPU的设备。

### 保存在GPU上，加载在GPU

**Save:**

    
    
    torch.save(model.state_dict(), PATH)
    

**Load:**

    
    
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model
    
    

当加载上进行训练，并且保存在GPU上GPU的模型，简单地转换初始化`模型 `，用`model.to（火炬CUDA优化模型。设备（ 'CUDA'））
`。另外，一定使用`。要（torch.device（ 'CUDA'）） `功能上的所有模型输入到该模型准备数据。请注意，调用`
my_tensor.to（设备）HTG14] `返回GPU的`my_tensor`新副本。它不会覆盖`my_tensor
[HTG23。因此，记得手动改写张量：`my_tensor  =  my_tensor.to（torch.device（ 'CUDA'）） `。`

### 节省CPU，装载在GPU

**Save:**

    
    
    torch.save(model.state_dict(), PATH)
    

**Load:**

    
    
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
    model.to(device)
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model
    
    

当加载上进行训练，并且保存在CPU一个GPU的模型，将`map_location`参数在`torch.load（） `函数为
_CUDA：DEVICE_ID_ 。这将加载模型给定的GPU设备。其次，一定要打电话`model.to（torch.device（ 'CUDA'））
`对模型的参数张量转换为CUDA张量。最后，一定要使用`。要（torch.device（ 'CUDA'））
`功能上的所有模型输入为CUDA优化模型准备数据。请注意，调用`my_tensor.to（设备）HTG20] `返回GPU的`my_tensor
`新副本。它不会覆盖`my_tensor  [HTG29。因此，记得手动改写张量：`my_tensor  =
my_tensor.to（torch.device（ 'CUDA'）） `。`

### 节省`torch.nn.DataParallel`模型

**Save:**

    
    
    torch.save(model.module.state_dict(), PATH)
    
    

**Load:**

    
    
    # Load to whatever device you want
    
    

`torch.nn.DataParallel`是一个模型包装，可以并行GPU利用率。要保存`数据并行 `模型一般，保存`
model.module.state_dict（） `。这样，您可以灵活地加载模型要你想要的任何设备的任何方式。

**脚本的总运行时间：** （0分钟0.000秒）

[`Download Python source code:
saving_loading_models.py`](../_downloads/4d2be551311e56235080ce6a019a2cc1/saving_loading_models.py)

[`Download Jupyter notebook:
saving_loading_models.ipynb`](../_downloads/b50843b19ff6d24140129232a11bcbff/saving_loading_models.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-orange.svg)](nn_tutorial.html "What
is torch.nn really?") [![](../_static/images/chevron-right-orange.svg)
Previous](../intermediate/tensorboard_tutorial.html "Visualizing Models, Data,
and Training with TensorBoard")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 保存和加载模型
    * 什么是`state_dict`？ 
      * [HTG0例：
    * 保存&安培;为推理荷载模型
      * 保存/加载`state_dict`（推荐）
      * 保存/加载整个模型
    * 保存&安培;载入通用检查点用于推断和/或恢复训练
      * 保存：
      * 负载：
    * 在一个文件中保存多个模型
      * 保存：
      * 负载：
    * Warmstarting模型从一个不同的模型使用参数
      * 保存：
      * 负载：
    * 保存&安培;荷载模型跨设备
      * 保存在GPU，CPU负荷
      * 节省GPU，GPU的负载
      * 保存在CPU，GPU上负载
      * 保存`torch.nn.DataParallel`模型

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)

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

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

