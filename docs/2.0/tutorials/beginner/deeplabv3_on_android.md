


# Android 上的图像分割 DeepLabV3 [¶](#image-segmentation-deeplabv3-on-android "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/deeplabv3_on_android>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/deeplabv3_on_android.html>




**作者** 
 :
 [Jeff Tang](https://github.com/jeffxtang)




**审阅者** 
 :
 [Jeremiah Chung](https://github.com/jeremiahschung)





## 简介 [¶](#introduction "此标题的永久链接")




 语义图像分割是一项计算机视觉任务，它使用语义标签来标记输入图像的特定区域。 PyTorch 语义图像分割
 [DeepLabV3 模型](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101)
 可用于使用
 [20 个语义类](http://host.robots. ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html) 
 包括例如自行车、公共汽车、汽车、狗和人。图像分割模型在自动驾驶和场景理解等应用中非常有用。




 在本教程中，我们将提供有关如何在 Android 上准备和运行 PyTorch DeepLabV3 模型的分步指南，带您从开始拥有可能想要在 Android 上使用的模型到最终拥有使用该模型的完整 Android 应用程序。我们还将介绍如何检查您的下一个有利的预训练 PyTorch 模型是否可以在 Android 上运行以及如何避免陷阱的实用和一般技巧。





 注意




 在学习本教程之前，您应该查看
 [PyTorch Mobile for Android](https://pytorch.org/mobile/android/) 
 并为 PyTorch Android
 [Hello World](https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp) 
 示例应用程序快速尝试。本教程将超越图像分类模型，通常是部署在移动设备上的第一种模型。本教程的完整代码可在
 [此处](https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation)
 。





## 学习目标 [¶](#learning-objectives "永久链接到此标题")




 在本教程中，您将学习如何：



1. 转换 DeepLabV3 模型以进行 Android 部署。
2.在 Python 中获取示例输入图像的模型输出，并将其与 Android 应用程序的输出进行比较。
3.构建新的 Android 应用或重用 Android 示例应用来加载转换后的模型。
4.将输入准备为模型期望的格式并处理模型输出。
5.完成 UI、重构、构建并运行应用程序以查看图像分割的实际效果。





## 先决条件 [¶](#preconditions "永久链接到此标题")



* PyTorch 1.6 或 1.7
* torchvision 0.7 或 0.8
* 安装了 NDK 的 Android Studio 3.5.1 或更高版本





## 步骤 [¶](#steps "永久链接到此标题")




### 1. 转换 DeepLabV3 模型以进行 Android 部署 [¶](#convert-the-deeplabv3-model-for-android-deployment "永久链接到此标题")



 在 Android 上部署模型的第一步是将模型转换为
 [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) 
 格式。





 注意




 目前，并非所有 PyTorch 模型都可以转换为 TorchScript，因为模型定义可能使用 TorchScript 中不存在的语言功能，TorchScript 是 Python 的子集。请参阅
 [脚本和优化配方](../recipes/script_optimized.html)
 了解更多详细信息。





 只需运行以下脚本即可生成脚本化模型
 
 deeplabv3_scripted.pt
 
 :






```
import torch

# use deeplabv3_resnet50 instead of resnet101 to reduce the model size
model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

scriptedm = torch.jit.script(model)
torch.jit.save(scriptedm, "deeplabv3_scripted.pt")

```




 生成的
 
 deeplabv3_scripted.pt
 
 模型文件的大小应在 168MB 左右。理想情况下，模型在部署到 Android 应用程序之前还应该进行量化，以显着减小尺寸并加快推理速度。要对量化有一个总体了解，请参阅
 [量化配方](../recipes/quantization.html) 
 及其资源链接。我们将在未来的教程或秘籍中详细介绍如何将称为“训练后”的量化工作流程
 [静态量化](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html) 
 正确应用到 DeepLabV3 模型.





### 2. 在 Python 中获取模型的示例输入和输出 [¶](#get-example-input-and-output-of-the-model-in-python “此标题的永久链接”）



 现在我们有了一个脚本化的 PyTorch 模型，让’s 使用一些示例输入进行测试，以确保模型在 Android 上正常工作。首先，让’s 编写一个Python 脚本，使用该模型进行推理并检查输入和输出。对于 DeepLabV3 模型的此示例，我们可以重用步骤 1 和 [DeepLabV3 模型中心站点](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101) 中的代码。将以下代码片段添加到上面的代码中：






```
from PIL import Image
from torchvision import transforms
input_image = Image.open("deeplab.jpg")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
with torch.no_grad():
    output = model(input_batch)['out'][0]

print(input_batch.shape)
print(output.shape)

```




 从
 [此处]下载
 
 deeplab.jpg
 
(https://github.com/jeffxtang/android-demo-app/blob/new_demo_apps/ImageSegmentation/app/src/main /assets/deeplab.jpg) 
 ，然后运行上面的脚本，您将看到模型输入和输出的形状：






```
torch.Size([1, 3, 400, 400])
torch.Size([21, 400, 400])

```




 因此，如果您向 Android 上的模型提供相同的图像输入
 
 deeplab.jpg
 
 大小为 400x400，则模型的输出大小应为 [21, 400, 400]。您还应该至少打印出输入和输出的实际数据的开始部分，以便在下面的步骤 4 中使用，以便与模型在 Android 应用中运行时的实际输入和输出进行比较。





### 3. 构建新的 Android 应用程序或重用示例应用程序并加载模型 [¶](#build-a-new-android-app-or-reuse-an -example-app-and-load-the-model“此标题的永久链接”）



 首先，按照 [Android 模型准备食谱](../recipes/model_preparation_android.html#add-the-model-and-pytorch-library-on-android) 的步骤 3 
 使用我们的模型在启用 PyTorch Mobile 的 Android Studio 项目中。由于本教程中使用的 DeepLabV3 和 PyTorch Hello World Android 示例中使用的 MobileNet v2 都是计算机视觉模型，因此您还可以获取
 [Hello World 示例存储库](https://github.com/pytorch/android-demo -app/tree/master/HelloWorldApp) 
 以便更轻松地修改加载模型和处理输入和输出的代码。此步骤和步骤 4 的主要目标是确保步骤 1 中生成的模型
 
 deeplabv3_scripted.pt
 
 确实可以在 Android 上正常工作。




 现在让’s 将步骤 2 中使用的 
 
 deeplabv3_scripted.pt
 
 和 
 
 deeplab.jpg
 
 添加到 Android Studio 项目中，并修改\ n 
 MainActivity
 
 中的 onCreate
 
 方法类似于：






```
Module module = null;
try {
 module = Module.load(assetFilePath(this, "deeplabv3_scripted.pt"));
} catch (IOException e) {
 Log.e("ImageSegmentation", "Error loading model!", e);
 finish();
}

```




 然后在行
 
 finish()
 
 设置断点并构建并运行应用程序。如果应用没有’停在断点处，则表示步骤 1 中的脚本模型已成功加载到 Android 上。





### 4. 处理模型输入和输出以进行模型推理 [¶](#process-the-model-input-and-output-for-model-inference "Permalink to这个标题”）



 在上一步中加载模型后，让’s 验证它是否适用于预期输入并可以生成预期输出。由于 DeepLabV3 模型的模型输入是与 Hello World 示例中的 MobileNet v2 相同的图像，因此我们将重用
 [MainActivity.java](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/app/src/main/java/org/pytorch/helloworld/MainActivity.java) 
 来自 Hello World 的文件用于输入处理。替换
 [第 50 行](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/app/src/main/java/org/pytorch/helloworld/MainActivity 之间的代码片段。 java#L50) 
 和 73 位于
 
 MainActivity.java
 
 中，代码如下：






```
final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
 TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
 TensorImageUtils.TORCHVISION_NORM_STD_RGB);
final float[] inputs = inputTensor.getDataAsFloatArray();

Map<String, IValue> outTensors =
 module.forward(IValue.from(inputTensor)).toDictStringKey();

// the key "out" of the output tensor contains the semantic masks
// see https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101
final Tensor outputTensor = outTensors.get("out").toTensor();
final float[] outputs = outputTensor.getDataAsFloatArray();

int width = bitmap.getWidth();
int height = bitmap.getHeight();

```





 没有10



 模型输出是 DeepLabV3 模型的字典，因此我们使用
 
 toDictStringKey
 
 来正确提取结果。对于其他模型，模型输出也可能是单个张量或张量元组等。





 通过上面显示的代码更改，您可以在 
 
 Final float[] input
 
 和 
 
 Final float[]outputs
 
 之后设置断点，它们填充输入张量和将张量数据输出到浮点数组以便于调试。运行应用程序，当它在断点处停止时，将
 
 输入
 
 和
 
 输出
 
 中的数字与您在步骤 2 中看到的模型输入和输出数据进行比较，看看它们是否匹配。对于在 Android 和 Python 上运行的模型的相同输入，您应该获得相同的输出。





 警告




 由于某些 Android 模拟器’s 浮点实现问题，在 Android 模拟器上运行时，您可能会看到具有相同图像输入的不同模型输出。因此最好在真实的 Android 设备上测试该应用。





 到目前为止，我们所做的只是确认我们感兴趣的模型可以像在 Python 中一样在我们的 Android 应用程序中编写脚本并正确运行。到目前为止，我们在 iOS 应用中使用模型的步骤消耗了应用开发的大部分（如果不是大部分）时间，类似于数据预处理对于典型机器学习项目来说是最繁重的工作。





### 5. 完成 UI、重构、构建并运行应用程序 [¶](#complete-the-ui-refactor-build-and-run-the-app "此标题的永久链接”）



 现在我们已准备好完成应用程序和 UI，以实际将处理结果视为新图像。输出处理代码应如下所示，添加到步骤 4 中的代码片段末尾：






```
int[] intValues = new int[width * height];
// go through each element in the output of size [WIDTH, HEIGHT] and
// set different color for different classnum
for (int j = 0; j < width; j++) {
 for (int k = 0; k < height; k++) {
 // maxi: the index of the 21 CLASSNUM with the max probability
 int maxi = 0, maxj = 0, maxk = 0;
 double maxnum = -100000.0;
 for (int i=0; i < CLASSNUM; i++) {
 if (outputs[i*(width*height) + j*width + k] > maxnum) {
 maxnum = outputs[i*(width*height) + j*width + k];
 maxi = i; maxj = j; maxk= k;
 }
 }
 // color coding for person (red), dog (green), sheep (blue)
 // black color for background and other classes
 if (maxi == PERSON)
 intValues[maxj*width + maxk] = 0xFFFF0000; // red
 else if (maxi == DOG)
 intValues[maxj*width + maxk] = 0xFF00FF00; // green
 else if (maxi == SHEEP)
 intValues[maxj*width + maxk] = 0xFF0000FF; // blue
 else
 intValues[maxj*width + maxk] = 0xFF000000; // black
 }
}

```




 上面代码中使用的常量在类的开头定义
 
 MainActivity
 
 :






```
private static final int CLASSNUM = 21;
private static final int DOG = 12;
private static final int PERSON = 15;
private static final int SHEEP = 17;

```




 这里的实现是基于对 DeepLabV3 模型的理解，该模型针对 width*height 的输入图像输出大小为 [21, width, height] 的张量。 width*height 输出数组中的每个元素都是 0 到 20 之间的值（简介中描述的总共 21 个语义标签），该值用于设置特定颜色。此处分割的颜色编码基于概率最高的类，您可以扩展自己数据集中所有类的颜色编码。




 输出处理完成后，您还需要调用以下代码将 RGB
 
 intValues
 
 数组渲染为位图实例
 
 outputBitmap
 
 然后再将其显示在
 
 图像视图
 
 :






```
Bitmap bmpSegmentation = Bitmap.createScaledBitmap(bitmap, width, height, true);
Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0,
 outputBitmap.getWidth(), outputBitmap.getHeight());
imageView.setImageBitmap(outputBitmap);

```




 此应用程序的 UI 也与 Hello World 类似，只是您不需要 
 
 TextView
 
 来显示图像分类结果。您还可以添加两个按钮
 
 Segment
 
 和
 
 Restart
 
（如代码存储库中所示）来运行模型推理并在显示分割结果后显示原始图像。 




 现在，当您在 Android 模拟器或最好是实际设备上运行该应用程序时，您将看到如下屏幕：



[![https://pytorch.org/tutorials/_images/deeplabv3_android.png](https://pytorch.org/tutorials/_images/deeplabv3_android.png)](https://pytorch.org/tutorials/_images/deeplabv3_android.png)
[![https://pytorch.org/tutorials/_images/deeplabv3_android2.png](.. /_images/deeplabv3_android2.png)](https://pytorch.org/tutorials/_images/deeplabv3_android2.png)


## 回顾 [¶](#recap "此标题的永久链接")




 在本教程中，我们描述了如何转换适用于 Android 的预训练 PyTorch DeepLabV3 模型以及如何确保模型可以在 Android 上成功运行。我们的重点是帮助您了解确认模型确实可以在 Android 上运行的过程。完整的代码存储库可在
 [此处](https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation)
 。




 未来的演示应用和教程将很快涵盖更多高级主题，例如量化和通过迁移学习或您自己的 Android 模型使用模型。





## 了解更多 [¶](#learn-more "此标题的永久链接")



1. [PyTorch 移动网站](https://pytorch.org/mobile)
2. [DeepLabV3 模型](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101)
3. [DeepLabV3论文](https://arxiv.org/pdf/1706.05587.pdf)








