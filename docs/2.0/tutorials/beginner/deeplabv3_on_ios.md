


# iOS 上的图像分割 DeepLabV3 [¶](#image-segmentation-deeplabv3-on-ios "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/deeplabv3_on_ios>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/deeplabv3_on_ios.html>




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




 在本教程中，我们将提供有关如何在 iOS 上准备和运行 PyTorch DeepLabV3 模型的分步指南，带您从开始拥有可能想要在 iOS 上使用的模型到最终拥有使用该模型的完整 iOS 应用程序。我们还将介绍如何检查您下一个最喜欢的预训练 PyTorch 模型是否可以在 iOS 上运行以及如何避免陷阱的实用和一般技巧。





 注意




 在学习本教程之前，您应该查看
 [PyTorch Mobile for iOS](https://pytorch.org/mobile/ios/) 
 并提供 PyTorch iOS
 [HelloWorld](https://pytorch.org/mobile/ios/) /github.com/pytorch/ios-demo-app/tree/master/HelloWorld) 
 示例应用程序快速尝试。本教程将超越图像分类模型，通常是部署在移动设备上的第一种模型。本教程的完整代码可在
 [此处](https://github.com/pytorch/ios-demo-app/tree/master/ImageSegmentation)
 。





## 学习目标 [¶](#learning-objectives "永久链接到此标题")




 在本教程中，您将学习如何：



1. 转换 DeepLabV3 模型以进行 iOS 部署。
2.在 Python 中获取示例输入图像的模型输出，并将其与 iOS 应用程序的输出进行比较。
3.构建新的 iOS 应用程序或重用 iOS 示例应用程序来加载转换后的模型。
4.将输入准备为模型期望的格式并处理模型输出。
5.完成 UI、重构、构建并运行应用程序以查看图像分割的实际效果。





## 先决条件 [¶](#preconditions "永久链接到此标题")



* PyTorch 1.6 或 1.7
* torchvision 0.7 或 0.8
* Xcode 11 或 12





## 步骤 [¶](#steps "永久链接到此标题")




### 1. 转换 DeepLabV3 模型以进行 iOS 部署 [¶](#convert-the-deeplabv3-model-for-ios-deployment "永久链接到此标题")



 在 iOS 上部署模型的第一步是将模型转换为
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

# use deeplabv3_resnet50 instead of deeplabv3_resnet101 to reduce the model size
model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

scriptedm = torch.jit.script(model)
torch.jit.save(scriptedm, "deeplabv3_scripted.pt")

```




 生成的
 
 deeplabv3_scripted.pt
 
 模型文件的大小应在 168MB 左右。理想情况下，模型在部署到 iOS 应用程序之前还应该进行量化，以显着减小尺寸并加快推理速度。要对量化有一个总体了解，请参阅
 [量化配方](../recipes/quantization.html) 
 及其资源链接。我们将在未来的教程或秘籍中详细介绍如何将称为“训练后”的量化工作流程
 [静态量化](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html) 
 正确应用到 DeepLabV3 模型.





### 2. 在 Python 中获取模型的示例输入和输出 [¶](#get-example-input-and-output-of-the-model-in-python "此标题的永久链接")



 现在我们有了一个脚本化的 PyTorch 模型，让’s 使用一些示例输入进行测试，以确保模型在 iOS 上正常工作。首先，让’s 编写一个Python 脚本，使用该模型进行推理并检查输入和输出。对于 DeepLabV3 模型的此示例，我们可以重用步骤 1 和 [DeepLabV3 模型中心站点](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101) 中的代码。将以下代码片段添加到上面的代码中：






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




 下载
 
 deeplab.jpg
 
 从
 [此处](https://github.com/pytorch/ios-demo-app/blob/master/ImageSegmentation/ImageSegmentation/deeplab.jpg ) 
 并运行上面的脚本来查看模型输入和输出的形状：






```
torch.Size([1, 3, 400, 400])
torch.Size([21, 400, 400])

```




 因此，如果您向 iOS 上的模型提供相同的图像输入
 
 deeplab.jpg
 
 大小为 400x400，则模型的输出大小应为 [21, 400, 400]。您还应该至少打印出输入和输出的实际数据的开始部分，以便在下面的步骤 4 中使用，以便与模型在 iOS 应用中运行时的实际输入和输出进行比较。





### 3. 构建新的 iOS 应用程序或重用示例应用程序并加载模型 [¶](#build-a-new-ios-app-or-reuse-an-example-app-and-load-the-model "此标题的永久链接")



 首先，按照 [iOS 模型准备食谱](../recipes/model_preparation_ios.html#add-the-model-and-pytorch-library-on-ios) 的步骤 3 
 使用我们的模型在启用 PyTorch Mobile 的 Xcode 项目中。由于本教程中使用的 DeepLabV3 模型和 PyTorch Hello World iOS 示例中使用的 MobileNet v2 模型都是计算机视觉模型，因此您可以选择从
 [HelloWorld 示例存储库](https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld) 
 作为模板来重用加载模型并处理输入和输出的代码。




 现在让’s 将步骤 2 中使用的
 
 deeplabv3_scripted.pt
 
 和
 
 deeplab.jpg
 
 添加到 Xcode 项目中并修改
 \ n ViewController.swift
 
 类似于：






```
class ViewController: UIViewController {
    var image = UIImage(named: "deeplab.jpg")!

    override func viewDidLoad() {
        super.viewDidLoad()
    }

    private lazy var module: TorchModule = {
        if let filePath = Bundle.main.path(forResource: "deeplabv3_scripted",
              ofType: "pt"),
            let module = TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't load the model file!")
        }
    }()
}

```




 然后在行
 
 return module
 
 设置断点并构建并运行应用程序。应用应在断点处停止，这意味着步骤 1 中的脚本化模型已成功加载到 iOS 上。





### 4. 处理模型输入和输出以进行模型推理 [¶](#process-the-model-input-and-output-for-model-inference "Permalink to这个标题")



 在上一步中加载模型后，让’s 验证它是否适用于预期输入并可以生成预期输出。由于 DeepLabV3 模型的模型输入是图像，与 Hello World 示例中的 MobileNet v2 相同，因此我们将重用
 [TorchModule.mm](https://github.com/pytorch/ios-demo-app/blob/master/HelloWorld/HelloWorld/HelloWorld/TorchBridge/TorchModule.mm) 
 来自 Hello World 的文件用于输入处理。将
 
 TorchModule.mm
 
 中的
 
 PredictImage
 
 方法实现替换为以下代码:






```
- (unsigned char*)predictImage:(void*)imageBuffer {
 // 1. the example deeplab.jpg size is size 400x400 and there are 21 semantic classes
 const int WIDTH = 400;
 const int HEIGHT = 400;
 const int CLASSNUM = 21;

 at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, WIDTH, HEIGHT}, at::kFloat);
 torch::autograd::AutoGradMode guard(false);
 at::AutoNonVariableTypeMode non_var_type_mode(true);

 // 2. convert the input tensor to an NSMutableArray for debugging
 float* floatInput = tensor.data_ptr<float>();
 if (!floatInput) {
 return nil;
 }
 NSMutableArray* inputs = [[NSMutableArray alloc] init];
 for (int i = 0; i < 3 * WIDTH * HEIGHT; i++) {
 [inputs addObject:@(floatInput[i])];
 }

 // 3. the output of the model is a dictionary of string and tensor, as
 // specified at https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101
 auto outputDict = _impl.forward({tensor}).toGenericDict();

 // 4. convert the output to another NSMutableArray for easy debugging
 auto outputTensor = outputDict.at("out").toTensor();
 float* floatBuffer = outputTensor.data_ptr<float>();
 if (!floatBuffer) {
 return nil;
 }
 NSMutableArray* results = [[NSMutableArray alloc] init];
 for (int i = 0; i < CLASSNUM * WIDTH * HEIGHT; i++) {
 [results addObject:@(floatBuffer[i])];
 }

 return nil;
}

```





 没有10



 模型输出是 DeepLabV3 模型的字典，因此我们使用 
 
 toGenericDict
 
 来正确提取结果。对于其他模型，模型输出也可能是单个张量或张量元组等。





 通过上面显示的代码更改，您可以在填充
 
 输入
 
 和
 
 结果
 
 的两个for 循环之后设置断点，并将它们与模型输入和输出进行比较您在步骤 2 中看到的数据，看看它们是否匹配。对于在 iOS 和 Python 上运行的模型的相同输入，您应该获得相同的输出。




 到目前为止，我们所做的只是确认我们感兴趣的模型可以像在 Python 中一样在我们的 iOS 应用程序中编写脚本并正确运行。到目前为止，我们在 iOS 应用中使用模型的步骤消耗了应用开发的大部分（如果不是大部分）时间，类似于数据预处理对于典型机器学习项目来说是最繁重的工作。





### 5. 完成 UI、重构、构建并运行应用程序 [¶](#complete-the-ui-refactor-build-and-run-the-app "此标题的永久链接")



 现在我们已准备好完成应用程序和 UI，以实际将处理结果视为新图像。输出处理代码应该是这样的，添加到步骤 4 中代码片段的末尾
 
 TorchModule.mm
 
 - 记得先删除行
 
 return nil;
 
暂时放在那里以使代码构建并运行:






```
// see the 20 semantic classes link in Introduction
const int DOG = 12;
const int PERSON = 15;
const int SHEEP = 17;

NSMutableData* data = [NSMutableData dataWithLength:
 sizeof(unsigned char) * 3 * WIDTH * HEIGHT];
unsigned char* buffer = (unsigned char*)[data mutableBytes];
// go through each element in the output of size [WIDTH, HEIGHT] and
// set different color for different classnum
for (int j = 0; j < WIDTH; j++) {
 for (int k = 0; k < HEIGHT; k++) {
 // maxi: the index of the 21 CLASSNUM with the max probability
 int maxi = 0, maxj = 0, maxk = 0;
 float maxnum = -100000.0;
 for (int i = 0; i < CLASSNUM; i++) {
 if ([results[i * (WIDTH * HEIGHT) + j * WIDTH + k] floatValue] > maxnum) {
 maxnum = [results[i * (WIDTH * HEIGHT) + j * WIDTH + k] floatValue];
 maxi = i; maxj = j; maxk = k;
 }
 }
 int n = 3 * (maxj * width + maxk);
 // color coding for person (red), dog (green), sheep (blue)
 // black color for background and other classes
 buffer[n] = 0; buffer[n+1] = 0; buffer[n+2] = 0;
 if (maxi == PERSON) buffer[n] = 255;
 else if (maxi == DOG) buffer[n+1] = 255;
 else if (maxi == SHEEP) buffer[n+2] = 255;
 }
}
return buffer;

```




 这里的实现是基于对 DeepLabV3 模型的理解，该模型针对 width*height 的输入图像输出大小为 [21, width, height] 的张量。 width*height 输出数组中的每个元素都是 0 到 20 之间的值（简介中描述的总共 21 个语义标签），该值用于设置特定颜色。此处分割的颜色编码基于概率最高的类，您可以扩展自己数据集中所有类的颜色编码。




 输出处理后，您还需要调用辅助函数将 RGB
 
 缓冲区
 
 转换为
 
 UIImage
 
 实例以显示在
 
 UIImageView 上
 
 。您可以参考代码库中

 UIImageHelper.mm

中定义的示例代码

convertRGBBufferToUIImage

。




 此应用程序的 UI 也与 Hello World 类似，只是您不需要 
 
 UITextView
 
 来显示图像分类结果。您还可以添加两个按钮
 
 Segment
 
 和
 
 Restart
 
（如代码存储库中所示）来运行模型推理并在显示分割结果后显示原始图像。 




 运行应用程序之前的最后一步是将所有部分连接在一起。修改
 
 ViewController.swift
 
 文件以使用
 
 PredictImage
 
 ，该文件在存储库中被重构并更改为
 
 segmentImage
 
 ，并且您可以使用辅助函数如存储库中的示例代码所示构建，位于 
 
 ViewController.swift
 
 。将按钮连接到操作，您就可以开始了。




 现在，当您在 iOS 模拟器或实际 iOS 设备上运行该应用程序时，您将看到以下屏幕：



[![https://pytorch.org/tutorials/_images/deeplabv3_ios.png](https://pytorch.org/tutorials/_images/deeplabv3_ios.png)](https://pytorch.org/tutorials/_images/deeplabv3_ios.png)
[![https://pytorch.org/tutorials/_images/deeplabv3_ios2.png](.. /_images/deeplabv3_ios2.png)](https://pytorch.org/tutorials/_images/deeplabv3_ios2.png)


## 回顾 [¶](#recap "此标题的永久链接")




 在本教程中，我们描述了如何转换适用于 iOS 的预训练 PyTorch DeepLabV3 模型以及如何确保模型可以在 iOS 上成功运行。我们的重点是帮助您了解确认模型确实可以在 iOS 上运行的过程。完整的代码存储库可在
 [此处](https://github.com/pytorch/ios-demo-app/tree/master/ImageSegmentation)
 。




 未来的演示应用和教程将很快涵盖更多高级主题，例如量化和通过迁移学习或您自己的 iOS 模型使用模型。





## 了解更多 [¶](#learn-more "此标题的永久链接")



1. [PyTorch 移动网站](https://pytorch.org/mobile)
2. [DeepLabV3 模型](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101)
3. [DeepLabV3论文](https://arxiv.org/pdf/1706.05587.pdf)








