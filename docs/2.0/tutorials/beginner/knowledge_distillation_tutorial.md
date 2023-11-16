# 知识蒸馏教程 [¶](#knowledge-distillation-tutorial "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/knowledge_distillation_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html>




**作者** 
 :
 [Alexandros Chariton](https://github.com/AlexandrosChrtn)




 知识蒸馏是一种技术，可以将知识从计算成本较高的大型模型转移到较小的模型，而不会失去有效性。这允许在功能较弱的硬件上进行部署，
从而使评估更快、更高效。




 在本教程中，我们将进行一些实验，重点是提高
轻量级神经网络的准确性，使用更强大的网络作为教师。
轻量级网络的计算成本和速度将保持不受影响，
我们的干预只关注它的权重，而不是它的前向传递。
这项技术的应用可以在无人机或手机等设备中找到。
在本教程中，我们不使用任何外部包，因为我们需要的一切都可以在
 `torch`
 和
 `torchvision`
 。




 在本教程中，您将学习：



* 如何修改模型类以提取隐藏表示并将其用于进一步计算
* 如何修改 PyTorch 中的常规训练循环以在分类的交叉熵等之上包含额外损失
* 如何改进通过使用更复杂的模型作为教师来提高轻量级模型的性能


## 先决条件 [¶](#preconditions "永久链接到此标题")



* 1 个 GPU，4GB 内存
* PyTorch v2.0 或更高版本
* CIFAR-10 数据集（由脚本下载并保存在名为
 `/data`
 的目录中）





```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```




### 正在加载 CIFAR-10 [¶](#loading-cifar-10 "此标题的永久链接")



 CIFAR-10 是一个流行的图像数据集，有 10 个类别。我们的目标是为每个输入图像预测以下类别之一。




![https://pytorch.org/tutorials/_static/img/cifar10.png](https://pytorch.org/tutorials/_static/img/cifar10.png)


 CIFAR-10 图像示例
  [¶](#id1 "此图像的永久链接")





 输入图像为 RGB，因此它们有 3 个通道且为 32x32 像素。基本上，每个图像由 3 x 32 x 32 = 3072 个从 0 到 255 的数字来描述。
神经网络中的常见做法是对输入进行归一化，这样做的原因有多种，
包括避免常用激活函数中的饱和并提高数值稳定性。
我们的归一化过程包括减去平均值并除以每个通道的标准差。
张量 \xe2\x80\x9cmean=[0.485, 0.456, 0.406]\xe2\x80\x9d 和 \xe2 \x80\x9cstd=[0.229, 0.224, 0.225]\xe2\x80\x9d 已经计算出来，
它们表示作为训练集的 CIFAR-10 预定义子集中每个通道的平均值和标准差。 
请注意我们如何将这些值用于测试集，而无需从头开始重新计算平均值和标准差。
这是因为网络是根据上述数字的减法和除法产生的特征进行训练的，并且我们希望保持一致性。 
此外，在现实生活中，我们将无法计算测试集的平均值和标准差，因为
根据我们的假设，此时无法访问该数据。



作为结束点，我们通常将这个保留集称为验证集，并且在优化验证集上的模型’s 性能后，我们使用一个单独的集，
称为测试集。 
这样做是为了避免根据单个指标的贪婪和偏差优化来选择模型。






```
# Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
transforms_cifar = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

```






```
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz

  0%|          | 0/170498071 [00:00<?, ?it/s]
  0%|          | 491520/170498071 [00:00<00:34, 4904960.44it/s]
  4%|3         | 6422528/170498071 [00:00<00:04, 36690842.68it/s]
  8%|8         | 14221312/170498071 [00:00<00:02, 55010639.16it/s]
 13%|#3        | 22347776/170498071 [00:00<00:02, 65050191.73it/s]
 18%|#7        | 30375936/170498071 [00:00<00:01, 70486433.85it/s]
 22%|##2       | 38240256/170498071 [00:00<00:01, 73230999.45it/s]
 27%|##7       | 46235648/170498071 [00:00<00:01, 75416570.58it/s]
 32%|###1      | 54362112/170498071 [00:00<00:01, 77198065.50it/s]
 36%|###6      | 62095360/170498071 [00:00<00:01, 74575797.27it/s]
 41%|####      | 69599232/170498071 [00:01<00:01, 68381407.99it/s]
 45%|####4     | 76546048/170498071 [00:01<00:01, 64756721.59it/s]
 49%|####8     | 83132416/170498071 [00:01<00:01, 62428511.39it/s]
 52%|#####2    | 89456640/170498071 [00:01<00:01, 60843771.25it/s]
 56%|#####6    | 95584256/170498071 [00:01<00:01, 59907178.11it/s]
 60%|#####9    | 101613568/170498071 [00:01<00:01, 59123340.86it/s]
 63%|######3   | 107544576/170498071 [00:01<00:01, 58702798.00it/s]
 67%|######6   | 113442816/170498071 [00:01<00:00, 58440757.69it/s]
 70%|######9   | 119308288/170498071 [00:01<00:00, 58089494.00it/s]
 73%|#######3  | 125140992/170498071 [00:02<00:00, 57932132.45it/s]
 77%|#######6  | 130940928/170498071 [00:02<00:00, 57883290.88it/s]
 80%|########  | 136839168/170498071 [00:02<00:00, 58119038.21it/s]
 84%|########3 | 142671872/170498071 [00:02<00:00, 57947761.75it/s]
 87%|########7 | 148471808/170498071 [00:02<00:00, 57772107.48it/s]
 90%|######### | 154271744/170498071 [00:02<00:00, 57722742.68it/s]
 94%|#########3| 160071680/170498071 [00:02<00:00, 57730560.35it/s]
 97%|#########7| 165871616/170498071 [00:02<00:00, 57513696.67it/s]
100%|##########| 170498071/170498071 [00:02<00:00, 60789943.83it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified

```





 没有10



 本节仅适用于对快速结果感兴趣的 CPU 用户。仅当您’ 对小规模实验感兴趣时才使用此选项。请记住，使用任何 GPU 时代码都应该运行得相当快。仅从训练/测试数据集中选择前
 `num_images_to_keep`
 个图像






```
#from torch.utils.data import Subset
#num_images_to_keep = 2000
#train_dataset = Subset(train_dataset, range(min(num_images_to_keep, 50_000)))
#test_dataset = Subset(test_dataset, range(min(num_images_to_keep, 10_000)))

```







```
#Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

```





### 定义模型类和实用函数 [¶](#defining-model-classes-and-utility-functions "永久链接到此标题")



 接下来，我们需要定义模型类。这里需要设置几个用户定义的参数。我们使用两种不同的架构，在整个实验中保持固定的滤波器数量，以确保公平比较。
这两种架构都是卷积神经网络 (CNN)，具有不同数量的卷积层，用作特征提取器，后跟具有 10 个类别的分类器.
对于学生来说，过滤器和神经元的数量较少。






```
# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Lightweight neural network class to be used as student:
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

```




 我们使用 2 个函数来帮助我们生成和评估原始分类任务的结果。
其中一个函数称为
 `train`
 并采用以下参数：



* `model`
 ：通过此函数训练（更新其权重）的模型实例。
* `train_loader`
 ：我们在上面定义了
 `train_loader`
 ，及其工作是将数据输入到模型中。
* `epochs`
：我们循环数据集的次数。
* `learning_rate`
：学习率决定了我们收敛的步长应该。太大或太小的步长都可能有害。
* `device`
 ：确定运行工作负载的设备。可以是 CPU 或 GPU，具体取决于可用性。



 我们的测试函数类似，但它将使用
 `test_loader`
 调用以从测试集中加载图像。




![https://pytorch.org/tutorials/_static/img/knowledge_distillation/ce_only.png](https://pytorch.org/tutorials/_static/img/knowledge_distillation/ce_only.png)


 使用交叉熵训练两个网络。学生将用作基线：
  [¶](#id2 "Permalink to this image")







```
def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

```





### 交叉熵运行 [¶](#cross-entropy-runs "永久链接到此标题")



 为了重现性，我们需要设置火炬手动种子。我们使用不同的方法训练网络，因此为了公平地比较它们，
使用相同的权重初始化网络是有意义的。
首先使用交叉熵训练教师网络：






```
torch.manual_seed(42)
nn_deep = DeepNN(num_classes=10).to(device)
train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_deep = test(nn_deep, test_loader, device)

# Instantiate the lightweight network:
torch.manual_seed(42)
nn_light = LightNN(num_classes=10).to(device)

```






```
Epoch 1/10, Loss: 1.3308625350827756
Epoch 2/10, Loss: 0.868983477582712
Epoch 3/10, Loss: 0.6817919808580443
Epoch 4/10, Loss: 0.5415832494073511
Epoch 5/10, Loss: 0.42635518247666565
Epoch 6/10, Loss: 0.3240613896218712
Epoch 7/10, Loss: 0.2288390991975889
Epoch 8/10, Loss: 0.17136443598800913
Epoch 9/10, Loss: 0.14584973608346088
Epoch 10/10, Loss: 0.13291112373552055
Test Accuracy: 75.10%

```




 我们实例化一个更轻量级的网络模型来比较它们的性能。
反向传播对权重初始化很敏感，
因此我们需要确保这两个网络具有完全相同的初始化。






```
torch.manual_seed(42)
new_nn_light = LightNN(num_classes=10).to(device)

```




 为了确保我们已经创建了第一个网络的副本，我们检查其第一层的范数。
如果匹配，那么我们就可以确定网络确实是相同的。






```
# Print the norm of the first layer of the initial lightweight model
print("Norm of 1st layer of nn_light:", torch.norm(nn_light.features[0].weight).item())
# Print the norm of the first layer of the new lightweight model
print("Norm of 1st layer of new_nn_light:", torch.norm(new_nn_light.features[0].weight).item())

```






```
Norm of 1st layer of nn_light: 2.327361822128296
Norm of 1st layer of new_nn_light: 2.327361822128296

```




 打印每个模型中的参数总数:






```
total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
print(f"DeepNN parameters: {total_params_deep}")
total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
print(f"LightNN parameters: {total_params_light}")

```






```
DeepNN parameters: 1,186,986
LightNN parameters: 267,738

```




 使用交叉熵损失训练和测试轻量级网络：






```
train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_light_ce = test(nn_light, test_loader, device)

```






```
Epoch 1/10, Loss: 1.4697845640694698
Epoch 2/10, Loss: 1.1614691967244648
Epoch 3/10, Loss: 1.029334182934383
Epoch 4/10, Loss: 0.9275425788386703
Epoch 5/10, Loss: 0.8504851350698934
Epoch 6/10, Loss: 0.7837966749125429
Epoch 7/10, Loss: 0.7167237832418183
Epoch 8/10, Loss: 0.6593718075233957
Epoch 9/10, Loss: 0.6066458740502673
Epoch 10/10, Loss: 0.5534635244885369
Test Accuracy: 69.86%

```


正如我们所看到的，根据测试准确性，我们现在可以将用作教师的更深层次网络与我们假设的学生的轻量级网络进行比较。到目前为止，我们的学生还没有与老师进行干预，因此这个表现是由学生自己实现的。
到目前为止的指标可以通过以下几行看到：






```
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")

```






```
Teacher accuracy: 75.10%
Student accuracy: 69.86%

```





### 知识蒸馏运行 [¶](#knowledge-distillation-run "永久链接到此标题")



 现在让 ’s 尝试通过纳入教师来提高学生网络的测试准确性。
知识蒸馏是实现此目的的一种简单技术，
基于两个网络都输出我们的概率分布的事实
因此，两个网络共享相同数量的输出神经元。
该方法的工作原理是将额外的损失合并到传统的交叉熵损失中，
这是基于教师网络的 softmax 输出。
假设是经过适当训练的教师网络的输出激活携带了学生网络在训练期间可以利用的附加信息。
原始工作表明，利用软目标中较小概率的比率可以帮助实现深度神经网络的基本目标，\这是在数据上创建相似性结构，其中相似的对象被更紧密地映射在一起。
例如，在 CIFAR-10 中，如果卡车有轮子，则可能会被误认为是汽车或飞机，
但可能性较小被误认为是狗。
因此，假设有价值的信息不仅存在于经过适当训练的模型的顶部预测中，而且存在于整个输出分布中是有道理的。
但是，单独的交叉熵并不能充分利用这些信息，因为非预测类的激活
t往往非常小，以至于传播的梯度不会有意义地改变构建此所需向量空间的权重。




 当我们继续定义第一个引入师生动态的辅助函数时，我们需要包含一些额外的参数：



* `T`
 ：温度控制输出分布的平滑度。较大
 `T`
 会导致更平滑的分布，因此较小的概率会获得较大的提升。
* `soft_target_loss_weight`
 ：分配给额外目标 we\xe2\ 的权重x80\x99re 即将包含。
* `ce_loss_weight`
 ：分配给交叉熵的权重。调整这些权重可以推动网络针对任一目标进行优化。



![https://pytorch.org/tutorials/_static/img/knowledge_distillation/distillation_output_loss.png](https://pytorch.org/tutorials/_static/img/knowledge_distillation/distillation_output_loss.png)


 蒸馏损失是根据网络的对数计算的。它只向学生返回渐变：
  [¶](#id3 "Permalink to this image")







```
def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

# Compare the student test accuracy with and without the teacher, after distillation
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")

```






```
Epoch 1/10, Loss: 2.708952553132001
Epoch 2/10, Loss: 2.193634529552801
Epoch 3/10, Loss: 1.9721146946977777
Epoch 4/10, Loss: 1.8154662112750666
Epoch 5/10, Loss: 1.6874753056889604
Epoch 6/10, Loss: 1.575195058532383
Epoch 7/10, Loss: 1.4828760791617586
Epoch 8/10, Loss: 1.403088356527831
Epoch 9/10, Loss: 1.3256588410538481
Epoch 10/10, Loss: 1.2621631930246378
Test Accuracy: 70.80%
Teacher accuracy: 75.10%
Student accuracy without teacher: 69.86%
Student accuracy with CE + KD: 70.80%

```





### 余弦损失最小化运行 [¶](#cosine-loss-minimization-run "永久链接到此标题")



 随意调整控制 softmax 函数的软度和损失系数的温度参数。
在神经网络中，很容易在主要目标中包含额外的损失函数，以实现更好的泛化等目标。 
让\xe2\x80\x99s 尝试为学生添加一个目标，但现在让\xe2\x80\x99s 关注其隐藏状态而不是输出层。
我们的目标是传达来自教师\xe2\x80 的信息\x99 通过包含一个朴素损失函数来向学生表示，
其最小化意味着随着损失的减少，随后传递给分类器的展平向量变得更加
*相似*
。
当然，老师这样做不更新其权重，因此最小化仅取决于学生\xe2\x80\x99s 权重。
此方法背后的基本原理是，我们假设教师模型具有更好的内部表示，
不太可能学生在没有外部干预的情况下实现了这一目标，因此我们人为地促使学生模仿教师的内部表征。
但这是否最终会帮助学生并不简单，因为推动轻量级网络
达到这一点假设我们已经找到了一种可以提高测试准确性的内部表示，这可能是一件好事，但它也可能是有害的，因为网络具有不同的架构，并且学生不具备与老师相同的学习能力。
换句话说，学生\xe2\x80\x99s 和教师\xe2\x80\x99s 这两个向量没有理由匹配每个组件。
学生可以达到作为教师排列的内部表示\ xe2\x80\x99s ，它同样有效。
尽管如此，我们仍然可以运行一个快速实验来找出此方法的影响。
我们将使用
 `CosineEmbeddingLoss`
，它由公式如下：




[![https://pytorch.org/tutorials/_static/img/knowledge_distillation/cosine_embedding_loss.png](https://pytorch.org/tutorials/_static/img/knowledge_distillation/cosine_embedding_loss.png)](https://pytorch.org/tutorials/_static/img/knowledge_distillation/cosine_embedding_loss.png)


 CosineEmbeddingLoss 的公式
  [¶](#id4 "此图像的永久链接")





 显然，我们首先需要解决一件事。
当我们将蒸馏应用于输出层时，我们提到两个网络具有相同数量的神经元，等于类的数量。
但是，情况并非如此卷积层之后的层。在这里，在最终卷积层扁平化之后，老师比学生拥有更多的神经元。我们的损失函数接受两个相同维度的向量作为输入，因此我们需要以某种方式匹配它们。我们将通过在教师’s 卷积层之后包含一个平均池化层来解决这个问题，以降低其维度以匹配学生的维度。




 要继续，我们将修改模型类，或创建新的模型类。
现在，前向函数不仅返回网络的 logits，还返回卷积层之后的扁平化隐藏表示。我们为修改后的教师添加了上述池化内容。






```
class ModifiedDeepNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output, 2)
        return x, flattened_conv_output_after_pooling

# Create a similar student class where we return a tuple. We do not apply pooling after flattening.
class ModifiedLightNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLightNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        return x, flattened_conv_output

# We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
modified_nn_deep = ModifiedDeepNNCosine(num_classes=10).to(device)
modified_nn_deep.load_state_dict(nn_deep.state_dict())

# Once again ensure the norm of the first layer is the same for both networks
print("Norm of 1st layer for deep_nn:", torch.norm(nn_deep.features[0].weight).item())
print("Norm of 1st layer for modified_deep_nn:", torch.norm(modified_nn_deep.features[0].weight).item())

# Initialize a modified lightweight network with the same seed as our other lightweight instances. This will be trained from scratch to examine the effectiveness of cosine loss minimization.
torch.manual_seed(42)
modified_nn_light = ModifiedLightNNCosine(num_classes=10).to(device)
print("Norm of 1st layer:", torch.norm(modified_nn_light.features[0].weight).item())

```






```
Norm of 1st layer for deep_nn: 7.507150650024414
Norm of 1st layer for modified_deep_nn: 7.507150650024414
Norm of 1st layer: 2.327361822128296

```




 当然，我们需要更改训练循环，因为现在模型返回一个元组
 `(logits,
 

 hide_representation)`
 。使用样本输入张量
我们可以打印它们的形状。






```
# Create a sample input tensor
sample_input = torch.randn(128, 3, 32, 32).to(device) # Batch size: 128, Filters: 3, Image size: 32x32

# Pass the input through the student
logits, hidden_representation = modified_nn_light(sample_input)

# Print the shapes of the tensors
print("Student logits shape:", logits.shape) # batch_size x total_classes
print("Student hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

# Pass the input through the teacher
logits, hidden_representation = modified_nn_deep(sample_input)

# Print the shapes of the tensors
print("Teacher logits shape:", logits.shape) # batch_size x total_classes
print("Teacher hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

```






```
Student logits shape: torch.Size([128, 10])
Student hidden representation shape: torch.Size([128, 1024])
Teacher logits shape: torch.Size([128, 10])
Teacher hidden representation shape: torch.Size([128, 1024])

```




 在我们的例子中，
 `hidden_representation_size`
 是
 `1024`
 。这是学生最终卷积层的扁平化特征图，正如您所看到的，
it 是其分类器的输入。对于老师来说也是
 `1024`
，因为我们用
 `avg_pool1d`
 从
 `2048`
 做到了这一点。
这里应用的损失仅影响学生在损失计算之前。换句话说，它不会影响学生的分类器。
修改后的训练循环如下：




![https://pytorch.org/tutorials/_static/img/knowledge_distillation/cosine_loss_distillation.png](https://pytorch.org/tutorials/_static/img/knowledge_distillation/cosine_loss_distillation.png)


 在余弦损失最小化中，我们希望通过向学生返回梯度来最大化两个表示的余弦相似度：
  [¶](#id5 "Permalink to this image")







```
def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model and keep only the hidden representation
            with torch.no_grad():
                _, teacher_hidden_representation = teacher(inputs)

            # Forward pass with the student model
            student_logits, student_hidden_representation = student(inputs)

            # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device))

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

```




 出于同样的原因，我们需要修改我们的测试函数。这里我们忽略模型返回的隐藏表示。






```
def test_multiple_outputs(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs) # Disregard the second tensor of the tuple
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

```


在这种情况下，我们可以轻松地将知识蒸馏和余弦损失最小化包含在同一函数中。在师生范式中组合方法来实现更好的性能是很常见的。
现在，我们可以运行一个简单的训练测试会话。






```
# Train and test the lightweight network with cross entropy loss
train_cosine_loss(teacher=modified_nn_deep, student=modified_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_cosine_loss = test_multiple_outputs(modified_nn_light, test_loader, device)

```






```
Epoch 1/10, Loss: 1.3048383903015606
Epoch 2/10, Loss: 1.0683175657716248
Epoch 3/10, Loss: 0.9695691844386518
Epoch 4/10, Loss: 0.8942368797329076
Epoch 5/10, Loss: 0.8388044855478779
Epoch 6/10, Loss: 0.7944333541118885
Epoch 7/10, Loss: 0.75288055208333
Epoch 8/10, Loss: 0.7179904060290597
Epoch 9/10, Loss: 0.6782254899859124
Epoch 10/10, Loss: 0.6515725936426227
Test Accuracy: 71.50%

```





### 中间回归器运行 [¶](#intermediate-regressor-run "永久链接到此标题")



 由于多种原因，我们的朴素最小化并不能保证更好的结果，其中之一是向量的维度。
对于更高维度的向量，余弦相似度通常比欧几里德距离效果更好，
但我们处理的是每个向量具有 1024 个分量，因此提取有意义的相似性要困难得多。
此外，正如我们所提到的，推动教师和学生的隐藏表示的匹配并没有得到理论的支持。
没有充分的理由为什么我们应该以 1 为目标: 这些向量的 1 个匹配。
我们将通过包含一个称为回归器的额外网络来提供训练干预的最终示例。
目标是首先在卷积层之后提取教师的特征图，
然后提取
但是，这一次，我们将在网络之间引入一个回归器以促进匹配过程。
回归器将是可训练的，并且理想情况下会比我们的模型做得更好朴素余弦损失最小化方案。
它的主要工作是匹配这些特征图的维数，以便我们可以正确定义教师和学生之间的损失函数。
定义这样的损失函数提供了一个教学“路径,” 基本上是反向传播梯度的流程，它将改变学生’s 的权重。
关注原始网络每个分类器之前的卷积层的输出，我们有以下形状:






```
# Pass the sample input only from the convolutional feature extractor
convolutional_fe_output_student = nn_light.features(sample_input)
convolutional_fe_output_teacher = nn_deep.features(sample_input)

# Print their shapes
print("Student's feature extractor output shape: ", convolutional_fe_output_student.shape)
print("Teacher's feature extractor output shape: ", convolutional_fe_output_teacher.shape)

```






```
Student's feature extractor output shape:  torch.Size([128, 16, 8, 8])
Teacher's feature extractor output shape:  torch.Size([128, 32, 8, 8])

```




 我们为教师设置了 32 个过滤器，为学生设置了 16 个过滤器。
我们将包含一个可训练层，将学生的特征图转换为教师特征图的形状。
在实践中，我们修改轻量级类在与卷积特征图大小匹配的中间回归器之后返回隐藏状态
，教师类则返回最终卷积层的输出，而无需池化或展平。




![https://pytorch.org/tutorials/_static/img/knowledge_distillation/fitnets_knowledge_distill.png](https://pytorch.org/tutorials/_static/img/knowledge_distillation/fitnets_knowledge_distill.png)


 可训练层与中间张量的形状相匹配，并且均方误差 (MSE) 已正确定义：
  [¶](#id6 "Permalink to this image")







```
class ModifiedDeepNNRegressor(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        conv_feature_map = x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, conv_feature_map

class ModifiedLightNNRegressor(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLightNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Include an extra regressor (in our case linear)
        self.regressor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        regressor_output = self.regressor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, regressor_output

```




 之后，我们必须再次更新我们的火车循环。这次，我们提取学生的回归器输出，教师的特征图，
我们计算这些张量的
 `MSE`
（它们具有完全相同的形状，因此’s被正确定义），然后我们根据该损失反向传播梯度，
 除了分类任务的常规交叉熵损失之外。






```
def train_mse_loss(teacher, student, train_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Again ignore teacher logits
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)

            # Forward pass with the student model
            student_logits, regressor_feature_map = student(inputs)

            # Calculate the loss
            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Notice how our test function remains the same here with the one we used in our previous case. We only care about the actual outputs because we measure accuracy.

# Initialize a ModifiedLightNNRegressor
torch.manual_seed(42)
modified_nn_light_reg = ModifiedLightNNRegressor(num_classes=10).to(device)

# We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
modified_nn_deep_reg = ModifiedDeepNNRegressor(num_classes=10).to(device)
modified_nn_deep_reg.load_state_dict(nn_deep.state_dict())

# Train and test once again
train_mse_loss(teacher=modified_nn_deep_reg, student=modified_nn_light_reg, train_loader=train_loader, epochs=10, learning_rate=0.001, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_mse_loss = test_multiple_outputs(modified_nn_light_reg, test_loader, device)

```






```
Epoch 1/10, Loss: 1.7031925587398011
Epoch 2/10, Loss: 1.3212134987496964
Epoch 3/10, Loss: 1.1756496081876633
Epoch 4/10, Loss: 1.0840044809729241
Epoch 5/10, Loss: 1.0064094048326888
Epoch 6/10, Loss: 0.9451821583616155
Epoch 7/10, Loss: 0.8923830903704514
Epoch 8/10, Loss: 0.8415379870273269
Epoch 9/10, Loss: 0.7999916655938034
Epoch 10/10, Loss: 0.7622286942608826
Test Accuracy: 70.82%

```




 预计最终方法会比 
 `CosineLoss` 效果更好
 因为现在我们在教师和学生之间允许有一个可训练层，
这为学生在学习时提供了一些回旋余地，而不是迫使学生复制教师’s 表示。
包括额外的网络是基于提示的蒸馏背后的想法。






```
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
print(f"Student accuracy with CE + CosineLoss: {test_accuracy_light_ce_and_cosine_loss:.2f}%")
print(f"Student accuracy with CE + RegressorMSE: {test_accuracy_light_ce_and_mse_loss:.2f}%")

```






```
Teacher accuracy: 75.10%
Student accuracy without teacher: 69.86%
Student accuracy with CE + KD: 70.80%
Student accuracy with CE + CosineLoss: 71.50%
Student accuracy with CE + RegressorMSE: 70.82%

```





### 结论 [¶](#conclusion "此标题的永久链接")



 上述方法都不会增加网络参数数量或推理时间，
因此性能的提高是以在训练期间计算梯度的很少成本为代价的。
在 ML 应用中，我们最关心的是推理时间，因为训练发生在模型部署。
如果我们的轻量级模型对于部署来说仍然太重，我们可以应用不同的想法，例如训练后量化。
额外的损失可以应用于许多任务，而不仅仅是分类，并且您可以尝试像这样的数量系数、
温度或神经元数量。请随意调整上面教程中的任何数字，
但请记住，如果更改神经元/过滤器的数量，则可能会发生形状不匹配。




 有关详细信息，请参阅：



* [Hinton, G.、Vinyals, O.、Dean, J.：在神经网络中提炼知识。见：神经信息处理系统深度学习研讨会 (2015)](https://arxiv.org/abs/1503.02531)
* [Romero, A.、Ballas, N.、Kahou, S.E.、Chassang, A.、Gatta , C., Bengio, Y.：Fitnets：薄深网的提示。见：国际学习表征会议论文集（2015）](https://arxiv.org/abs/1412.6550)



**脚本总运行时间:** 
 ( 7 分 32.811 秒)

