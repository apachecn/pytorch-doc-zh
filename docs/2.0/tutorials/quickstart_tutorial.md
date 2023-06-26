# 快速入门
本节介绍机器学习中常见任务的 API。请参考每个部分中的链接，以便进一步深入。



## 处理数据
PyTorch有两个[处理数据的基本操作](https://pytorch.org/docs/stable/data.html)：``torch.utils.data.DataLoader``和``torch.utils.data.Dataset``。``Dataset``存储样本及其相应的标签，而``Dataset``则围绕``Dataset``包装了一个可迭代的数据。

```py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

PyTorch 提供特定于领域的库，如 [TorchText](https://pytorch.org/text/stable/index.html),
[TorchVision](https://pytorch.org/vision/stable/index.html) 和 [TorchAudio](https://pytorch.org/audio/stable/index.html),所有这些库都包含数据集。对于本教程，我们将使用 TorchVision 数据集。

``torchvision.datasets``模块包含了许多真实世界视觉数据的 ``Dataset``对象，比如 CIFAR、 COCO ([完整列表在这里](https://pytorch.org/vision/stable/datasets.html))。在本教程中，我们使用 FashionMNIST 数据集。每个 TorchVision ``Dataset``都包含两个参数: ``transform`` 和 ``target_transform``，分别用于转换样本和标签。


```py
# 从开源数据集下载训练数据。
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 从开源数据集下载测试数据。
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```
输出：
```py
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:12, 365718.31it/s]
  1%|          | 229376/26421880 [00:00<00:38, 685682.68it/s]
  3%|3         | 884736/26421880 [00:00<00:10, 2498938.52it/s]
  7%|7         | 1933312/26421880 [00:00<00:05, 4141475.37it/s]
 19%|#8        | 4915200/26421880 [00:00<00:01, 10854978.12it/s]
 26%|##5       | 6782976/26421880 [00:00<00:01, 11037400.65it/s]
 37%|###7      | 9797632/26421880 [00:01<00:01, 15568756.79it/s]
 44%|####4     | 11730944/26421880 [00:01<00:01, 14184748.16it/s]
 55%|#####5    | 14647296/26421880 [00:01<00:00, 17510568.70it/s]
 63%|######3   | 16777216/26421880 [00:01<00:00, 15834704.91it/s]
 75%|#######4  | 19693568/26421880 [00:01<00:00, 18759775.35it/s]
 83%|########2 | 21889024/26421880 [00:01<00:00, 16780435.96it/s]
 94%|#########3| 24772608/26421880 [00:01<00:00, 19391805.01it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 13914460.04it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 326673.50it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:12, 362354.20it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 684627.79it/s]
 21%|##        | 917504/4422102 [00:00<00:01, 2626211.85it/s]
 44%|####3     | 1933312/4422102 [00:00<00:00, 4103892.12it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6109664.51it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 61868988.52it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```



We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
in the dataloader iterable will return a batch of 64 features and labels.


我们将``Dataset``作为参数传递给``DataLoader``。这在我们的数据集上包裹了一个可迭代的数据集，并支持自动批处理、采样、随机打乱和多进程数据加载。
在这里，我们定义了一个大小为64的批处理，即dataloader迭代器中的每个元素都会返回一个由64个特征和标签组成的批次数据。



```py
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```
输出：
```py

Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```

阅读有关[在 PyTorch 中加载数据](data_tutorial.html).的更多信息。


## 创建模型




为了在 PyTorch 中定义一个神经网络，我们创建了一个继承自  [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)的类。我们在 ``__init__``函数中定义网络的层，并在 ``forward`` 函数中指定数据将如何通过网络。为了加速神经网络中的操作，我们将其移动到 GPU 或 MPS (如果有的话)。
```py
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```


输出：
```py
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)

```
了解更多关于[在 PyTorch 中构建神经网络](buildmodel_tutorial.html)的信息。


## 模型参数的优化
为了训练一个模型，我们需要一个[损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)和一个[优化器](https://pytorch.org/docs/stable/optim.html)。



```py
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

在单个训练循环中，模型对训练数据集进行预测(分批进行) ，并反向传播预测误差以调整模型的参数。
```py
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

我们还根据测试数据集检查模型的性能，以确保它正在学习。









```py
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```
训练过程是在几个迭代(epochs)中进行的。在每个epochs过程中，模型可以学习并更新参数，以便做出更好的预测。我们打印模型的准确性和损失在每个epochs， 我们希望看到在每个epochs中准确度的提高和损失的降低。

```py
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

输出：
```py
Epoch 1
-------------------------------
loss: 2.303494  [   64/60000]
loss: 2.294637  [ 6464/60000]
loss: 2.277102  [12864/60000]
loss: 2.269977  [19264/60000]
loss: 2.254234  [25664/60000]
loss: 2.237145  [32064/60000]
loss: 2.231056  [38464/60000]
loss: 2.205036  [44864/60000]
loss: 2.203239  [51264/60000]
loss: 2.170890  [57664/60000]
Test Error:
 Accuracy: 53.9%, Avg loss: 2.168587

Epoch 2
-------------------------------
loss: 2.177784  [   64/60000]
loss: 2.168083  [ 6464/60000]
loss: 2.114908  [12864/60000]
loss: 2.130411  [19264/60000]
loss: 2.087470  [25664/60000]
loss: 2.039667  [32064/60000]
loss: 2.054271  [38464/60000]
loss: 1.985452  [44864/60000]
loss: 1.996019  [51264/60000]
loss: 1.917239  [57664/60000]
Test Error:
 Accuracy: 60.2%, Avg loss: 1.920371

Epoch 3
-------------------------------
loss: 1.951699  [   64/60000]
loss: 1.919513  [ 6464/60000]
loss: 1.808724  [12864/60000]
loss: 1.846544  [19264/60000]
loss: 1.740612  [25664/60000]
loss: 1.698728  [32064/60000]
loss: 1.708887  [38464/60000]
loss: 1.614431  [44864/60000]
loss: 1.646473  [51264/60000]
loss: 1.524302  [57664/60000]
Test Error:
 Accuracy: 61.4%, Avg loss: 1.547089

Epoch 4
-------------------------------
loss: 1.612693  [   64/60000]
loss: 1.570868  [ 6464/60000]
loss: 1.424729  [12864/60000]
loss: 1.489538  [19264/60000]
loss: 1.367247  [25664/60000]
loss: 1.373463  [32064/60000]
loss: 1.376742  [38464/60000]
loss: 1.304958  [44864/60000]
loss: 1.347153  [51264/60000]
loss: 1.230657  [57664/60000]
Test Error:
 Accuracy: 62.7%, Avg loss: 1.260888

Epoch 5
-------------------------------
loss: 1.337799  [   64/60000]
loss: 1.313273  [ 6464/60000]
loss: 1.151835  [12864/60000]
loss: 1.252141  [19264/60000]
loss: 1.123040  [25664/60000]
loss: 1.159529  [32064/60000]
loss: 1.175010  [38464/60000]
loss: 1.115551  [44864/60000]
loss: 1.160972  [51264/60000]
loss: 1.062725  [57664/60000]
Test Error:
 Accuracy: 64.6%, Avg loss: 1.087372

Done!
```

阅读更多关于[训练你的模型](optimization_tutorial.html)。


## 保存模型
保存模型的一种常见方法是序列化内部状态字典(包含模型参数)。

```py
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```
## 加载模型
加载模型的过程包括重新创建模型结构并将状态字典加载到其中。

```py
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
```
输出：
```py
<All keys matched successfully>
```
这个模型现在可以用来做预测。
```py
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```
输出：
```py
Predicted: "Ankle boot", Actual: "Ankle boot"
```
阅读更多关于[保存和加载模型的内容](saveloadrun_tutorial.html)。

脚本的总运行时间: (0分钟42.070秒)