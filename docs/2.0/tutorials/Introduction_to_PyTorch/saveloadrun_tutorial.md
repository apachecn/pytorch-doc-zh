# 模型保存和加载

> 译者：[runzhi214](https://github.com/runzhi214)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/Introduction_to_PyTorch/saveloadrun_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html>

在这个章节我们会学习如何持久化模型状态来保存、加载和执行模型预测。

```py
import torch
import torchvision.models as models
```

## 模型权重的保存和加载

PyTorch模型将学习到的参数存储在一个内部状态字典中，叫`state_dict`。它们可以通过`torch.save`方法来持久化。

```py
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

Out:

```py
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /var/lib/jenkins/.cache/torch/hub/checkpoints/vgg16-397923af.pth

  0%|          | 0.00/528M [00:00<?, ?B/s]
  4%|4         | 22.5M/528M [00:00<00:02, 236MB/s]
  9%|8         | 46.5M/528M [00:00<00:02, 245MB/s]
 13%|#3        | 70.5M/528M [00:00<00:01, 248MB/s]
 18%|#7        | 94.4M/528M [00:00<00:01, 249MB/s]
 22%|##2       | 118M/528M [00:00<00:01, 250MB/s]
 27%|##6       | 142M/528M [00:00<00:01, 250MB/s]
 31%|###1      | 166M/528M [00:00<00:01, 249MB/s]
 36%|###5      | 190M/528M [00:00<00:01, 249MB/s]
 40%|####      | 214M/528M [00:00<00:01, 249MB/s]
 45%|####5     | 238M/528M [00:01<00:01, 250MB/s]
 50%|####9     | 262M/528M [00:01<00:01, 251MB/s]
 54%|#####4    | 286M/528M [00:01<00:01, 250MB/s]
 59%|#####8    | 310M/528M [00:01<00:00, 249MB/s]
 63%|######3   | 333M/528M [00:01<00:00, 249MB/s]
 68%|######7   | 357M/528M [00:01<00:00, 247MB/s]
 72%|#######2  | 381M/528M [00:01<00:00, 248MB/s]
 77%|#######6  | 405M/528M [00:01<00:00, 249MB/s]
 81%|########1 | 429M/528M [00:01<00:00, 250MB/s]
 86%|########5 | 453M/528M [00:01<00:00, 251MB/s]
 90%|######### | 477M/528M [00:02<00:00, 251MB/s]
 95%|#########5| 502M/528M [00:02<00:00, 253MB/s]
100%|##########| 528M/528M [00:02<00:00, 258MB/s]
100%|##########| 528M/528M [00:02<00:00, 251MB/s]
```

要加载模型权重，你需要首先创建一个模型（和要加载权重的模型一样）然后使用`load_state_dict()`方法加载参数。

```py
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

> 注意：
> 请确保在进行推理前调用`model.eval()`方法来将dropout层和batch normalization层设置为评估模式(evaluation模式)。如果不这么做的话会产生并不一致的推理结果。

## 保存和加载模型结构

在加载模型权重的时候，我们需要首先实例化一个模型类，因为类定义了神经网络的结构。我们也想把类结构和模型一起保存，那就可以通过将`model`传递给保存函数(而不是`model.state_dict())。

```py
torch.save(model, 'model.pth')
```

然后我们可以这么载入模型:

```py
model = torch.load('model.pth')
```

## 关联的教程

[在PyTorch中保存、加载一个Checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html) -- 译者注：该文档目前未完成翻译

**脚本总运行时间**: (0分7.704秒)
