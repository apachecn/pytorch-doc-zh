# 优化 Vision Transformer 模型以进行部署 [¶](#optimizing-vision-transformer-model-for-deployment "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
> 人工校正：[xiaoxstz](https://github.com/xiaoxstz)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/vt_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/vt_tutorial.html>

[Jeff Tang](https://github.com/jeffxtang) ,[Geeta Chauhan](https://github.com/gchauhan/)

 Vision Transformer 模型将自然语言处理中引入的基于注意力的尖端 Transformer 模型应用于计算机视觉任务，以实现各种最先进的 (SOTA) 结果。 Facebook 数据高效图像转换器 [DeiT](https://ai.facebook.com/blog/data-efficient-image-transformers-a-promising-new-technique-for-image-classification) 是在 ImageNet 上训练的 Vision Transformer 模型用于图像分类。

 在本教程中，我们将首先介绍 DeiT 是什么以及如何使用它，然后完成脚本编写、量化、优化以及在 iOS 和 Android 应用程序中使用模型的完整步骤。我们还将比较量化、优化和非量化、非优化模型的性能，并按步骤展示对模型应用量化和优化的好处。

## 什么是 DeiT [¶](#what-is-deit "此标题的永久链接")

自 2012 年深度学习兴起以来，卷积神经网络 (CNN) 一直是图像分类的主要模型，但 CNN 通常需要数亿张图像进行训练才能达到 SOTA 结果。 DeiT 是一种视觉变换器模型，需要更少的数据和计算资源进行训练，以便在执行图像分类方面与领先的CNN 竞争，这由 DeiT 的两个关键组件实现：

* 数据增强，模拟在更大的数据集上进行训练；
* 本机蒸馏，允许 Transformer 网络从 CNN’s 输出中学习。

DeiT 表明，Transformers 可以成功应用于计算机视觉任务，并且对数据和资源的访问受到限制。有关 DeiT 的更多详细信息，请参阅 [repo](https://github.com/facebookresearch/deit)  和 [论文](https://arxiv.org/abs/2012.12877) .

## 使用 DeiT 对图像进行分类 [¶](#classifying-images-with-deit "此标题的永久链接")

 按照 DeiT 存储库中的 `README.md` 获取有关如何使用 DeiT 对图像进行分类的详细信息，或者为了进行快速测试，请首先安装所需的软件包：

```bash
pip install torch torchvision timm pandas requests

```

 要在 Google Colab 中运行，请通过运行以下命令安装依赖项:

```python
!pip install timm pandas requests

```

 然后运行以下脚本：

```python
from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

print(torch.__version__)
# should be 1.8.0


model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()

transform = transforms.Compose(
    [transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
img = transform(img)[None,]
out = model(img)
clsidx = torch.argmax(out)
print(clsidx.item())

```

输出：

```txt
2.1.0+cu121
Downloading: "https://github.com/facebookresearch/deit/zipball/main" to /var/lib/jenkins/.cache/torch/hub/main.zip
Downloading: "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth" to /var/lib/jenkins/.cache/torch/hub/checkpoints/deit_base_patch16_224-b5f2ef4d.pth

  0%|          | 0.00/330M [00:00<?, ?B/s]
  4%|3         | 11.8M/330M [00:00<00:02, 124MB/s]
  8%|7         | 26.0M/330M [00:00<00:02, 138MB/s]
 12%|#1        | 39.6M/330M [00:00<00:02, 140MB/s]
 16%|#6        | 53.9M/330M [00:00<00:02, 144MB/s]
 21%|##        | 67.8M/330M [00:00<00:01, 145MB/s]
 25%|##4       | 81.5M/330M [00:00<00:02, 130MB/s]
 30%|##9       | 97.8M/330M [00:00<00:01, 142MB/s]
 34%|###3      | 112M/330M [00:00<00:01, 135MB/s]
 38%|###8      | 126M/330M [00:00<00:01, 140MB/s]
 43%|####2     | 141M/330M [00:01<00:01, 144MB/s]
 47%|####7     | 155M/330M [00:01<00:01, 147MB/s]
 51%|#####1    | 169M/330M [00:01<00:01, 146MB/s]
 56%|#####5    | 183M/330M [00:01<00:01, 147MB/s]
 60%|#####9    | 198M/330M [00:01<00:00, 145MB/s]
 64%|######4   | 211M/330M [00:01<00:00, 143MB/s]
 68%|######8   | 226M/330M [00:01<00:00, 146MB/s]
 73%|#######3  | 242M/330M [00:01<00:00, 153MB/s]
 78%|#######8  | 259M/330M [00:01<00:00, 159MB/s]
 83%|########3 | 275M/330M [00:01<00:00, 164MB/s]
 88%|########8 | 292M/330M [00:02<00:00, 166MB/s]
 93%|#########3| 308M/330M [00:02<00:00, 168MB/s]
 98%|#########8| 325M/330M [00:02<00:00, 169MB/s]
100%|##########| 330M/330M [00:02<00:00, 151MB/s]
269

```

 输出应为 269，根据类的 ImageNet 列表索引到 [标签文件](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)  ，映射到 `timber wolf`, `grey wolf`, `gray wolf`, `Canis lupus`.

 现在我们已经验证可以使用 DeiT 模型来分类图像，让我们看看如何修改模型，使其可以在 iOS 和Android 应用上运行。

## 脚本 DeiT [¶](#scripting-deit "此标题的永久链接")

 要在移动设备上使用模型，我们首先需要编写模型脚本。请参阅 [脚本和优化秘诀](https://pytorch.org/tutorials/recipes/script_optimized.html) 了解快速概述。运行下面的代码，将上一步中使用的 DeiT 模型转换为可在移动设备上运行的 TorchScript 格式。

```python
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("fbdeit_scripted.pt")

```

输出：

```txt
Using cache found in /var/lib/jenkins/.cache/torch/hub/facebookresearch_deit_main

```

 生成大小约为 346MB 的脚本化模型文件 `fbdeit_scripted.pt`。

## 量化 DeiT [¶](#quantizing-deit "此标题的固定链接")

 为了显着减小训练模型的大小，同时保持推理精度大致相同，可以对模型应用量化。得益于 DeiT 中使用的 Transformer 模型，我们可以轻松地将动态量化应用于模型，因为动态量化最适合 LSTM 和 Transformer 模型（请参阅 [此处](https://pytorch.org/docs/stable/quantization.html?highlight=quantization#dynamic-quantization) 了解更多详细信息)。

 现在运行以下代码：

```python
# Use 'x86' for server inference (the old 'fbgemm' is still available but 'x86' is the recommended default) and ``qnnpack`` for mobile inference.
backend = "x86" # replaced with ``qnnpack`` causing much worse inference speed for quantized model on this notebook
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
scripted_quantized_model = torch.jit.script(quantized_model)
scripted_quantized_model.save("fbdeit_scripted_quantized.pt")

```

输出：

```txt
/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/ao/quantization/observer.py:214: UserWarning:

Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.

```

 这会生成模型的脚本化和量化版本 `fbdeit_quantized_scripted.pt` ，大小约为 89MB，比非量化模型大小 346MB 减少了 74%！ 

 您可以使用 `脚本化_quantized_model` 来生成相同的推理结果：

```python
out = scripted_quantized_model(img)
clsidx = torch.argmax(out)
print(clsidx.item())
# The same output 269 should be printed

```

输出：

```txt
269

```

## 优化 DeiT [¶](#optimizing-deit "此标题的固定链接")

 在移动设备上使用量化和脚本化模型之前的最后一步是对其进行优化：

```python
from torch.utils.mobile_optimizer import optimize_for_mobile
optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
optimized_scripted_quantized_model.save("fbdeit_optimized_scripted_quantized.pt")
```

 生成的 `fbdeit_optimized_scripted_quantized.pt` 文件的大小与量化、脚本化但未优化的模型大致相同。 推理结果保持不变。

```python
out = optimized_scripted_quantized_model(img)
clsidx = torch.argmax(out)
print(clsidx.item())
# Again, the same output 269 should be printed

```

输出：

```txt
269

```

## 使用 Lite 解释器 [¶](#using-lite-interpreter "永久链接到此标题")

 要查看 Lite Interpreter 可以减少多少模型大小并加快推理速度，请让我们创建模型的 Lite 版本。

```python
optimized_scripted_quantized_model._save_for_lite_interpreter("fbdeit_optimized_scripted_quantized_lite.ptl")
ptl = torch.jit.load("fbdeit_optimized_scripted_quantized_lite.ptl")

```

 虽然精简版模型大小与非精简版相当，但在移动设备上运行精简版时，预计推理速度会加快。

## 比较推理速度 [¶](#comparing-inference-speed "永久链接到此标题")

 要查看四种模型（原始模型、脚本化模型、量化和脚本化模型、优化量化和脚本化模型）的推理速度有何不同，请运行以下代码：

```python
with torch.autograd.profiler.profile(use_cuda=False) as prof1:
    out = model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof2:
    out = scripted_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof3:
    out = scripted_quantized_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof4:
    out = optimized_scripted_quantized_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof5:
    out = ptl(img)

print("original model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))
print("scripted model: {:.2f}ms".format(prof2.self_cpu_time_total/1000))
print("scripted & quantized model: {:.2f}ms".format(prof3.self_cpu_time_total/1000))
print("scripted & quantized & optimized model: {:.2f}ms".format(prof4.self_cpu_time_total/1000))
print("lite model: {:.2f}ms".format(prof5.self_cpu_time_total/1000))

```

输出：

```txt
original model: 141.80ms
scripted model: 119.72ms
scripted & quantized model: 118.53ms
scripted & quantized & optimized model: 148.74ms
lite model: 113.89ms

```

 在 Google Colab 上运行的结果是：

输出：

```txt
original model: 1236.69ms
scripted model: 1226.72ms
scripted & quantized model: 593.19ms
scripted & quantized & optimized model: 598.01ms
lite model: 600.72ms

```

 以下结果总结了每个模型所花费的推理时间以及每个模型相对于原始模型减少的百分比。

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'Model': ['original model','scripted model', 'scripted & quantized model', 'scripted & quantized & optimized model', 'lite model']})
df = pd.concat(df, pd.DataFrame([
    ["{:.2f}ms".format([prof1.self_cpu_time_total/1000), "0%"],
    "{:.2f}ms".format([prof2.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof2.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
    "{:.2f}ms".format([prof3.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof3.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
    "{:.2f}ms".format([prof4.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof4.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
    "{:.2f}ms".format([prof5.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof5.self_cpu_time_total)/prof1.self_cpu_time_total*100)]],
    columns=['Inference Time', 'Reduction'])], axis=1)

print(df)

"""
 Model Inference Time Reduction
0 original model 1236.69ms 0%
1 scripted model 1226.72ms 0.81%
2 scripted & quantized model 593.19ms 52.03%
3 scripted & quantized & optimized model 598.01ms 51.64%
4 lite model 600.72ms 51.43%
"""

```

输出：

```txt
                                    Model  ... Reduction
0                          original model  ...        0%
1                          scripted model  ...    15.57%
2              scripted & quantized model  ...    16.41%
3  scripted & quantized & optimized model  ...    -4.90%
4                              lite model  ...    19.68%

[5 rows x 3 columns]

'        Model                             Inference Time    Reduction0\toriginal model                             1236.69ms           0%1\tscripted model                             1226.72ms        0.81%2\tscripted & quantized model                  593.19ms       52.03%3\tscripted & quantized & optimized model      598.01ms       51.64%4\tlite model                                  600.72ms       51.43%'

```

### 了解更多 [¶](#learn-more "此标题的永久链接")

* [Facebook 数据高效图像转换器](https://ai.facebook.com/blog/data-efficient-image-transformers-a-promising-new-technique-for-image-classification)
* [视觉转换器在 iOS 上使用 ImageNet 和 MNIST](https://github.com/pytorch/ios-demo-app/tree/master/ViT4MNIST)
* [在 Android 上使用 ImageNet 和 MNIST 的 Vision Transformer](https://github.com) com/pytorch/android-demo-app/tree/master/ViT4MNIST)

**脚本的总运行时间:** 
 ( 0 分 19.846 秒)
