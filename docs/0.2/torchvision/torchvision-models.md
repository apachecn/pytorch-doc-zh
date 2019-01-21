# torchvision.models
`torchvision.models`模块的 子模块中包含以下模型结构。

- AlexNet
- VGG
- ResNet
- SqueezeNet
- DenseNet
You can construct a model with random weights by calling its constructor:

你可以使用随机初始化的权重来创建这些模型。
```python
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
squeezenet = models.squeezenet1_0()
densenet = models.densenet_161()
```
We provide pre-trained models for the ResNet variants and AlexNet, using the PyTorch torch.utils.model_zoo. These can constructed by passing pretrained=True:
对于`ResNet variants`和`AlexNet`，我们也提供了预训练(`pre-trained`)的模型。
```python
import torchvision.models as models
#pretrained=True就可以使用预训练的模型
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
```
ImageNet 1-crop error rates (224x224)

|Network	|Top-1 error  |Top-5 error|
|------|------|------|
|ResNet-18|	30.24|	10.92|
|ResNet-34|	26.70|	8.58|
|ResNet-50	|23.85	|7.13|
|ResNet-101|	22.63|	6.44|
|ResNet-152	|21.69	|5.94|
|Inception v3|	22.55|	6.44|
|AlexNet	|43.45	|20.91|
|VGG-11|	30.98|	11.37|
|VGG-13	|30.07	|10.75|
|VGG-16|	28.41|	9.62|
|VGG-19	|27.62	|9.12|
|SqueezeNet 1.0|	41.90|	19.58|
|SqueezeNet 1.1	|41.81	|19.38|
|Densenet-121|	25.35|	7.83|
|Densenet-169	|24.00	|7.00|
|Densenet-201|	22.80|	6.43|
|Densenet-161|	22.35	|6.20|

## torchvision.models.alexnet(pretrained=False, ** kwargs)
`AlexNet` 模型结构 [paper地址](https://arxiv.org/abs/1404.5997)

- pretrained (bool) – `True`, 返回在ImageNet上训练好的模型。

## torchvision.models.resnet18(pretrained=False, ** kwargs)
构建一个`resnet18`模型

- pretrained (bool) – `True`, 返回在ImageNet上训练好的模型。

## torchvision.models.resnet34(pretrained=False, ** kwargs)
构建一个`ResNet-34` 模型.

Parameters:	pretrained (bool) – `True`, 返回在ImageNet上训练好的模型。

## torchvision.models.resnet50(pretrained=False, ** kwargs)
构建一个`ResNet-50`模型

- pretrained (bool) – `True`, 返回在ImageNet上训练好的模型。

## torchvision.models.resnet101(pretrained=False, ** kwargs)
Constructs a ResNet-101 model.

- pretrained (bool) – `True`, 返回在ImageNet上训练好的模型。

## torchvision.models.resnet152(pretrained=False, ** kwargs)
Constructs a ResNet-152 model.

- pretrained (bool) – `True`, 返回在ImageNet上训练好的模型。

## torchvision.models.vgg11(pretrained=False, ** kwargs)
VGG 11-layer model (configuration “A”)
- pretrained (bool) – `True`, 返回在ImageNet上训练好的模型。

## torchvision.models.vgg11_bn(** kwargs)
VGG 11-layer model (configuration “A”) with batch normalization

## torchvision.models.vgg13(pretrained=False, ** kwargs)
VGG 13-layer model (configuration “B”)

- pretrained (bool) – `True`, 返回在ImageNet上训练好的模型。

## torchvision.models.vgg13_bn(** kwargs)
VGG 13-layer model (configuration “B”) with batch normalization

## torchvision.models.vgg16(pretrained=False, ** kwargs)
VGG 16-layer model (configuration “D”)

Parameters:	pretrained (bool) – If True, returns a model pre-trained on ImageNet
## torchvision.models.vgg16_bn(** kwargs)
VGG 16-layer model (configuration “D”) with batch normalization

## torchvision.models.vgg19(pretrained=False, ** kwargs)
VGG 19-layer model (configuration “E”)

- pretrained (bool) – `True`, 返回在ImageNet上训练好的模型。
## torchvision.models.vgg19_bn(** kwargs)
VGG 19-layer model (configuration ‘E’) with batch normalization
