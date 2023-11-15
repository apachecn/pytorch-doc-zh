


 (beta) PyTorch 中使用 Eager 模式的静态量化
 [¶](#beta-static-quantization-with-eager-mode-in-pytorch "永久链接到此标题")
==============================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/static_quantization_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>




**作者** 
 :
 [Raghuraman Krishnamoorthi](https://github.com/raghuramank100) 
**编辑** 
 :
 [Seth Weidman](https://github.com/SethHWeidman/) 
 ,
 [张杰](https:github.com/jerryzh168)




 本教程展示如何进行训练后静态量化，并说明
两种更先进的技术 - 每通道量化和量化感知训练 -
以进一步提高模型’s 的准确性。请注意，目前仅支持 CPU
量化，因此我们在本教程中不会使用 GPU/CUDA。
在本教程结束时，您将看到 PyTorch 中的量化
如何导致模型大小显着减小，同时增加模型大小速度。此外，您’将了解如何
轻松应用[此处](https://arxiv.org/abs/1806.08342)
所示的一些高级量化技术
，以便您的量化模型
花费更少
 
警告：我们使用其他 PyTorch 存储库中的大量样板代码来定义 
 `MobileNetV2`
 模型架构、定义数据加载器等。我们当然鼓励您阅读它；但如果您想了解量化功能，请随意跳到 “4。训练后静态量化” 部分。
我们’ 将首先进行必要的导入：






```
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)

```





 1. 模型架构
 [¶](#model-architecture "永久链接到此标题")
-------------------------------------------------------------------------------



 我们首先定义 MobileNetV2 模型架构，并进行一些显着的修改
以启用量化：



* 将加法替换为
 `nn.quantized.FloatFunctional`
* 在网络的开头和结尾插入
 `QuantStub`
 和
 `DeQuantStub`
。
* 将 ReLU6 替换为 ReLU



 注意：此代码取自
 [此处](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py) 
.






```
from torch.ao.quantization import QuantStub, DeQuantStub

def _make_divisible(v, divisor, min_value=None):
 """
 This function is taken from the original tf repo.
 It ensures that all layers have a channel number that is divisible by 8
 It can be seen here:
 https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
 :param v:
 :param divisor:
 :param min_value:
 :return:
 """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


类 ConvBNReLU(nn.Sequential):
 def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1 ):
 padding = (kernel_size - 1) //2
 super(ConvBNReLU, self).__init__(
 nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups,bias=False),
 nn.BatchNorm2d(out_planes, Momentum=0.1),
 # 替换为 ReLU
 nn.ReLU (原地=假)
)


class InvertedResidual(nn.Module):
 def __init__(self, inp, oup, stride, Expand_ratio):
 super(InvertedResidual, self)._ _init__()
 self.stride = stride
 在 [1, 2] 中断言步幅

 hide_dim = int(round(inp * Expand_ratio) )
 self.use_res_connect = self.stride == 1 且 inp == oup

 层 = []
 if Expand_ratio != 1:
 # pw
 层.append(ConvBNReLU(inp,hidden_dim,kernel_size=1))
 层.extend([
 # dw
 ConvBNReLU(hidden_dim,hidden_dim,stride=stride, groups =hidden_dim),
 # pw-线性
 nn.Conv2d(hidden_dim, oup, 1, 1, 0, 偏差=False),
 nn.BatchNorm2d(oup, 动量=0.1), 
 ])
 self.conv = nn.Sequential(*layers)
 # 将 torch.add 替换为 floatFunctional
 self.skip_add = nn.quantized.FloatFunctional()

 defforward (self, x):
 如果 self.use_res_connect:
 返回 self.skip_add.add(x, self.conv(x))
 else:
 返回 self.conv （X）


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
 """
 MobileNet V2 main class
 Args:
 num_classes (int): Number of classes
 width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
 inverted_residual_setting: Network structure
 round_nearest (int): Round the number of channels in each layer to be a multiple of this number
 Set to 1 to turn off rounding
 """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self, is_qat=False):
        fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

```






 2. 辅助函数
 [¶](#helper-functions "固定链接到此标题")
--------------------------------------------------------------------------- -



 接下来我们定义几个辅助函数来帮助模型评估。这些大多来自
 [此处](https://github.com/pytorch/examples/blob/master/imagenet/main.py)
 。






```
class AverageMeter(object):
 """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def precision(output, target, topk=(1,)):
 """使用 torch.no_grad() 计算指定 k 值的 k 个前预测的准确度"""
:\ n maxk = max(topk)
 batch_size = target.size(0)

 _, pred = output.topk(maxk, 1, True, True)
 pred = pred.t( )
 正确 = pred.eq(target.view(1, -1).expand_as(pred))

 res = []
 for k in topk:
 正确_k = 正确[:k].reshape(-1).float().sum(0, keepdim=True)
 res.append(正确_k.mul_(100.0 /batch_size))
 return资源


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

```






 3. 定义数据集和数据加载器
 [¶](#define-dataset-and-data-loaders "固定链接到此标题")
--------------------------------------------------------------------------------------------------------------------



 作为最后一个主要设置步骤，我们为训练和测试集定义数据加载器。




### 
 ImageNet 数据
 [¶](#imagenet-data "此标题的永久链接")



 要使用整个 ImageNet 数据集运行本教程中的代码，请首先按照此处的说明下载 imagenet
 [ImageNet Data](http://www.image-net.org/download) 
 。将下载的文件解压缩到 ‘data_path’ 文件夹中。



下载数据后，我们将显示下面的函数，这些函数定义我们’将用来读取此数据的数据加载器。这些函数主要来自
 [此处](https://github.com/pytorch/vision/blob/master/references/detection/train.py)
 。






```
def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

```




 接下来，我们’ 将加载预训练的 MobileNetV2 模型。我们提供下载模型的 URL
 [此处](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth) 
.






```
data_path = '~/.data/imagenet'
saved_model_dir = 'data/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 30
eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')

# Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
# while also improving numerical accuracy. While this can be used with any model, this is
# especially common with quantized models.

print(' Inverted Residual Block: Before fusion ', float_model.features[1].conv)
float_model.eval()

# Fuses modules
float_model.fuse_model()

# Note fusion of Conv+BN+Relu and Conv+Relu
print(' Inverted Residual Block: After fusion',float_model.features[1].conv)

```




 最后，为了获得 “baseline” 精度，让 ’s 查看我们的未量化模型
与融合模块的精度






```
num_eval_batches = 1000

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

```




 在整个模型上，我们在 50,000 张图像的评估数据集上获得了 71.9% 的准确率。




 这将是我们进行比较的基准。接下来让’s尝试不同的量化方法







 4. 训练后静态量化
 [¶](#post-training-static-quantization "固定链接到此标题")
-------------------------------------------------------------------------------------------------------------



 训练后静态量化不仅涉及动态量化中将权重从 float 转换为 int，
 还需要执行额外的步骤：首先通过网络馈送
数据并计算不同激活的结果分布
 （具体来说，这是通过在记录此数据的不同点插入观察者模块来完成的）。然后，使用这些分布来确定在推理时应如何具体量化不同的激活（一种简单的技术是将整个激活范围简单地分为 256 个级别，但我们也支持更复杂的方法）。重要的是，这个额外的步骤允许我们在操作之间传递量化值，而不是在每个操作之间将这些值转换为浮点数，然后再转换回整数，从而显着提高速度。






```
num_calibration_batches = 32

myModel = load_model(saved_model_dir + float_model_file).to('cpu')
myModel.eval()

# Fuse Conv, bn and relu
myModel.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myModel.qconfig = torch.ao.quantization.default_qconfig
print(myModel.qconfig)
torch.ao.quantization.prepare(myModel, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print(' Inverted Residual Block:After observer insertion ', myModel.features[1].conv)

# Calibrate with the training set
evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.ao.quantization.convert(myModel, inplace=True)
print('Post Training Quantization: Convert done')
print(' Inverted Residual Block: After fusion and quantization, note fused modules: ',myModel.features[1].conv)

print("Size of model after quantization")
print_size_of_model(myModel)

top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

```




 对于这个量化模型，我们在评估数据集上看到的准确率为 56.7%。这是因为我们使用简单的最小/最大观察器来确定量化参数。尽管如此，我们确实将模型的大小减少到略低于 3.6 MB，几乎减少了 4 倍。



此外，我们只需使用不同的量化配置即可显着提高准确性。我们使用 x86 架构量化的推荐配置重复相同的练习。此配置执行以下操作:



* 基于每个通道量化权重
* 使用直方图观察器收集激活直方图，然后以最佳方式选择
量化参数。





```
per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
# The old 'fbgemm' is still available but 'x86' is the recommended default.
per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
print(per_channel_quantized_model.qconfig)

torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)

```




仅更改此量化配置方法即可将准确度提高到
超过 67.3%！尽管如此，这仍然比上面达到的 71.9% 的基线差了 4%。
所以让我们尝试量化感知训练。






 5. 量化感知训练
 [¶](#quantization-aware-training "永久链接到此标题")
------------------------------------------------------------------------------------------------



 量化感知训练 (QAT) 是通常能获得最高准确度的量化方法。
使用 QAT，所有权重和激活在前向和后向传播过程中均被“fake 量化”训练：也就是说，浮点值被四舍五入以模拟 int8 值，但所有计算仍然使用浮点数完成。因此，训练期间的所有权重调整都是在模型最终被量化的情况下进行的；因此，在量化之后，此方法通常会
比动态量化或训练后静态量化产生更高的精度。




 实际执行 QAT 的整体工作流程与之前非常相似：



* 我们可以使用与之前相同的模型：量化感知
训练不需要额外的准备。
* 我们需要使用
 `qconfig`
 指定在后面插入哪种假量化权重
和激活，而不是指定观察者



 我们首先定义一个训练函数：






```
def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set: * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return

```




 我们像以前一样融合模块






```
qat_model = load_model(saved_model_dir + float_model_file)
qat_model.fuse_model(is_qat=True)

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
# The old 'fbgemm' is still available but 'x86' is the recommended default.
qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

```




 最后，
 `prepare_qat`
 执行 “fake 量化”，为量化感知训练准备模型






```
torch.ao.quantization.prepare_qat(qat_model, inplace=True)
print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules ',qat_model.features[1].conv)

```




 训练高精度量化模型需要在推理时对数值进行精确建模。因此，对于量化感知训练，我们通过以下方式修改训练循环：



* 在训练结束时切换批量归一化以使用运行均值和方差，以更好
匹配推理数值。
* 我们还冻结量化器参数（比例和零点）并微调权重。





```
num_train_batches = 20

# QAT takes time and one needs to train over a few epochs.
# Train and check accuracy after each epoch
for nepoch in range(8):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.ao.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))

```




 量化感知训练在整个 imagenet 数据集上的准确率超过 71.5%，接近浮点准确率 71.9%。




 有关量化感知训练的更多信息：



* QAT 是训练后量化技术的超集，允许进行更多调试。
例如，我们可以分析模型的准确性是否受到权重或激活
量化的限制。
* 我们还可以模拟浮点量化模型，因为
我们使用假量化来对实际量化算术的数值进行建模。
*我们也可以轻松模仿训练后量化。



### 
 量化加速
 [¶](#speedup-from-quantization "固定链接到此标题")



 最后，让’s 确认一下我们上面提到的事情：我们的量化模型实际上
执行推理速度更快吗？让’s 测试:






```
def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)

run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)

```




 在 MacBook Pro 上本地运行该模型，常规模型需要 61 毫秒，
量化模型只需 20 毫秒，这说明了与浮点模型相比，我们看到量化模型典型的 2-4 倍加速。







 结论
 [¶](#conclusion "此标题的永久链接")
------------------------------------------------------------------------------------



 在本教程中，我们展示了两种量化方法 - 训练后静态量化和量化感知训练 - 描述了它们在底层的用途 “” 以及如何在
中使用它们PyTorch。




感谢您的阅读！一如既往，我们欢迎任何反馈，因此请在[此处](https://github.com/pytorch/pytorch/issues)
 创建问题
（如果有任何反馈）。









