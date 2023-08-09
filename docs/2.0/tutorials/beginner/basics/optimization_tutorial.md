# 优化模型参数

> 译者：[runzhi214](https://github.com/runzhi214)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/basics/optimization_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>


既然我们已经有了模型和数据，现在现在应该训练、验证、测试我们的模型(基于我们的数据来优化参数)。训练一个模型也是一个遍历的过程；在每次遍历中，模型会对输出中进行一次猜想，计算这个猜想的错误程度(损失值),收集这些错误相对于参数的导数（像我们前面一章说的），然后通过梯度下降的方式来**优化**这些参数。关于这个过程的更详细的介绍，可以看3Blue1Brown制作的[《反向传播演算》](https://www.youtube.com/watch?v=tIeHLnjs5U8)这个视频。

## 前提代码

我们读取了之前的模块([数据集和数据加载器](datasets_and_dataloaders.md)、[构建模型](build_the_neural_network.md))的代码。

```py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

Out:

```py
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:11, 366175.29it/s]
  1%|          | 229376/26421880 [00:00<00:38, 685578.25it/s]
  3%|3         | 851968/26421880 [00:00<00:10, 2413401.26it/s]
  7%|7         | 1933312/26421880 [00:00<00:05, 4152542.15it/s]
 17%|#7        | 4521984/26421880 [00:00<00:02, 9852648.59it/s]
 25%|##5       | 6684672/26421880 [00:00<00:01, 11079775.27it/s]
 35%|###4      | 9142272/26421880 [00:01<00:01, 14258039.45it/s]
 44%|####3     | 11501568/26421880 [00:01<00:01, 14171893.75it/s]
 53%|#####2    | 13926400/26421880 [00:01<00:00, 16469591.54it/s]
 62%|######1   | 16351232/26421880 [00:01<00:00, 15727451.19it/s]
 71%|#######   | 18743296/26421880 [00:01<00:00, 17539641.69it/s]
 80%|########  | 21200896/26421880 [00:01<00:00, 16492622.63it/s]
 89%|########9 | 23592960/26421880 [00:01<00:00, 18100280.02it/s]
 99%|#########8| 26116096/26421880 [00:01<00:00, 16993447.90it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 13272484.55it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 328358.81it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:11, 365656.04it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 686664.05it/s]
 18%|#7        | 786432/4422102 [00:00<00:01, 2220917.26it/s]
 41%|####      | 1802240/4422102 [00:00<00:00, 3882085.99it/s]
 98%|#########7| 4325376/4422102 [00:00<00:00, 9553123.99it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6037898.77it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 39691685.65it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## 超参数

超参数是你用来控制模型优化过程的、可以调整的参数。不同的超参数取值能够影响模型训练和收敛的速度(更多[关于调整超参数的内容](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html))

我们定义以下用于训练的超参数:

- **Number of Epochs** - (时期、纪元) - 遍历数据集次数
- **Batch Size** - 在参数更新之前通过网络传播的数据样本数量。
- **Learning Rate** - 学习率 - 每个Batch/Epoch更新模型参数的幅度。较小的值会产生较慢的学习速度，较大的值可能会在训练过程中产生无法预料的行为。

```py
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## 优化循环

设置完超参数后，接下来我们在一个优化循环中训练并优化我们的模型。优化循环的每次遍历叫做一个**Epoch(时期、纪元)**。

每个Epoch由两个主要部分构成:

- **训练循环** 在训练数据集上遍历，尝试收敛到最优的参数。
- **验证/测试循环** 在测试数据集上遍历，来检查模型效果是否在提升。

我让我们大致的熟悉一下一些在训练循环中使用的概念。完整的优化循环代码可以直接跳到: [完整实现](#完整实现)

## 损失函数

拿到一些训练数据的时候，我们的模型不太可能给出正确答案。**损失函数**能测量获得的结果相对于目标值的偏离程度，我们希望在训练中能够最小化这个损失函数。我们对给定的数据样本做出预测然后和真实标签数据对比来计算损失。

常见的损失函数包括给回归任务用的`nn.MSELoss`(Mean Square Error,均方差)、给分类任务使用的`nn.NLLLoss`(Negative Log Likelihood,负对数似然)、`nn.CrossEntropyLoss`(交叉熵损失函数)结合了`nn.LogSoftmax`和`nn.NLLLoss`.

我们把模型输出的logits传递给 `nn.CrossEntropyLoss` -- 它会正则化llogits并计算预测误差。

```py
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

## 优化器

优化是在每一个训练步骤中调整模型参数来减小模型误差的过程。**优化算法**定义了这个过程应该如何进行（在这个例子中，我们使用Stochastic Gradient Descent-即SGD，随机梯度下降）。所有优化的逻辑都被封装在这个`optimizer`对象中。这里，我们使用SGD优化器。除此之外，在PyTorch中还有很多[其他可用的优化器](https://pytorch.org/docs/stable/optim.html)，比如ADAM和RMSProp -- 在不同类型的模型和数据上表现得更好。

我们通过注册需要训练的模型参数、然后传递学习率这个超参数来初始化优化器。

```py
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

在训练循环内部, 优化在三个步骤上发生：

* 调用`optimizer.zero_grad()`来重置模型参数的梯度。梯度会默认累加，为了防止重复计算（梯度），我们在每次遍历中显式的清空（梯度累加值）。
* 调用`loss.backward()`来反向传播预测误差。PyTorch对每个参数分别存储损失梯度。
* 我们获取到梯度后，调用`optimizer.step()`来根据反向传播中收集的梯度来调整参数。

## 完整实现

我们定义`train_loop`为优化循环的代码，`test_loop` 为根据测试数据来评估模型表现的代码

```py
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

我们初始化了损失函数和优化器，传递给`train_loop`和`test_loop`。你可以随意地修改epochs的数量来跟踪模型表现的进步情况。

Out:

```py
Epoch 1
-------------------------------
loss: 2.298730  [   64/60000]
loss: 2.289123  [ 6464/60000]
loss: 2.273286  [12864/60000]
loss: 2.269406  [19264/60000]
loss: 2.249603  [25664/60000]
loss: 2.229407  [32064/60000]
loss: 2.227368  [38464/60000]
loss: 2.204261  [44864/60000]
loss: 2.206193  [51264/60000]
loss: 2.166651  [57664/60000]
Test Error:
 Accuracy: 50.9%, Avg loss: 2.166725

Epoch 2
-------------------------------
loss: 2.176750  [   64/60000]
loss: 2.169595  [ 6464/60000]
loss: 2.117500  [12864/60000]
loss: 2.129272  [19264/60000]
loss: 2.079674  [25664/60000]
loss: 2.032928  [32064/60000]
loss: 2.050115  [38464/60000]
loss: 1.985236  [44864/60000]
loss: 1.987887  [51264/60000]
loss: 1.907162  [57664/60000]
Test Error:
 Accuracy: 55.9%, Avg loss: 1.915486

Epoch 3
-------------------------------
loss: 1.951612  [   64/60000]
loss: 1.928685  [ 6464/60000]
loss: 1.815709  [12864/60000]
loss: 1.841552  [19264/60000]
loss: 1.732467  [25664/60000]
loss: 1.692914  [32064/60000]
loss: 1.701714  [38464/60000]
loss: 1.610632  [44864/60000]
loss: 1.632870  [51264/60000]
loss: 1.514263  [57664/60000]
Test Error:
 Accuracy: 58.8%, Avg loss: 1.541525

Epoch 4
-------------------------------
loss: 1.616448  [   64/60000]
loss: 1.582892  [ 6464/60000]
loss: 1.427595  [12864/60000]
loss: 1.487950  [19264/60000]
loss: 1.359332  [25664/60000]
loss: 1.364817  [32064/60000]
loss: 1.371491  [38464/60000]
loss: 1.298706  [44864/60000]
loss: 1.336201  [51264/60000]
loss: 1.232145  [57664/60000]
Test Error:
 Accuracy: 62.2%, Avg loss: 1.260237

Epoch 5
-------------------------------
loss: 1.345538  [   64/60000]
loss: 1.327798  [ 6464/60000]
loss: 1.153802  [12864/60000]
loss: 1.254829  [19264/60000]
loss: 1.117322  [25664/60000]
loss: 1.153248  [32064/60000]
loss: 1.171765  [38464/60000]
loss: 1.110263  [44864/60000]
loss: 1.154467  [51264/60000]
loss: 1.070921  [57664/60000]
Test Error:
 Accuracy: 64.1%, Avg loss: 1.089831

Epoch 6
-------------------------------
loss: 1.166889  [   64/60000]
loss: 1.170514  [ 6464/60000]
loss: 0.979435  [12864/60000]
loss: 1.113774  [19264/60000]
loss: 0.973411  [25664/60000]
loss: 1.015192  [32064/60000]
loss: 1.051113  [38464/60000]
loss: 0.993591  [44864/60000]
loss: 1.039709  [51264/60000]
loss: 0.971077  [57664/60000]
Test Error:
 Accuracy: 65.8%, Avg loss: 0.982440

Epoch 7
-------------------------------
loss: 1.045165  [   64/60000]
loss: 1.070583  [ 6464/60000]
loss: 0.862304  [12864/60000]
loss: 1.022265  [19264/60000]
loss: 0.885213  [25664/60000]
loss: 0.919528  [32064/60000]
loss: 0.972762  [38464/60000]
loss: 0.918728  [44864/60000]
loss: 0.961629  [51264/60000]
loss: 0.904379  [57664/60000]
Test Error:
 Accuracy: 66.9%, Avg loss: 0.910167

Epoch 8
-------------------------------
loss: 0.956964  [   64/60000]
loss: 1.002171  [ 6464/60000]
loss: 0.779057  [12864/60000]
loss: 0.958409  [19264/60000]
loss: 0.827240  [25664/60000]
loss: 0.850262  [32064/60000]
loss: 0.917320  [38464/60000]
loss: 0.868384  [44864/60000]
loss: 0.905506  [51264/60000]
loss: 0.856353  [57664/60000]
Test Error:
 Accuracy: 68.3%, Avg loss: 0.858248

Epoch 9
-------------------------------
loss: 0.889765  [   64/60000]
loss: 0.951220  [ 6464/60000]
loss: 0.717035  [12864/60000]
loss: 0.911042  [19264/60000]
loss: 0.786085  [25664/60000]
loss: 0.798370  [32064/60000]
loss: 0.874939  [38464/60000]
loss: 0.832796  [44864/60000]
loss: 0.863254  [51264/60000]
loss: 0.819742  [57664/60000]
Test Error:
 Accuracy: 69.5%, Avg loss: 0.818780

Epoch 10
-------------------------------
loss: 0.836395  [   64/60000]
loss: 0.910220  [ 6464/60000]
loss: 0.668506  [12864/60000]
loss: 0.874338  [19264/60000]
loss: 0.754805  [25664/60000]
loss: 0.758453  [32064/60000]
loss: 0.840451  [38464/60000]
loss: 0.806153  [44864/60000]
loss: 0.830360  [51264/60000]
loss: 0.790281  [57664/60000]
Test Error:
 Accuracy: 71.0%, Avg loss: 0.787271

Done!
```

## 进一步阅读

- [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions) 损失函数
- [torch.optim](https://pytorch.org/docs/stable/optim.html) torch的优化器包
- [Warmstart Training a Model](https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html) 给模型的训练预热
