
 使用 Ray Tune 进行超参数调整
 [¶](#hyperparameter-tuning-with-ray-tune "永久链接到此标题")
===========================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/hyperparameter_tuning_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html>




 超参数调整可以区分平均模型和
高度准确的模型。通常，简单的事情（例如选择不同的学习率或更改
网络层大小）可能会对模型性能产生巨大影响。




 幸运的是，有一些工具可以帮助找到最佳参数组合。
 [Ray Tune](https://docs.ray.io/en/latest/tune.html) 
 是一个行业标准工具，用于分布式超参数调整。 Ray Tune 包含最新的超参数搜索\算法，与 TensorBoard 和其他分析库集成，
通过 [Ray’s 分布式机器学习引擎](https://ray.io/) 原生支持分布式训练
.




 在本教程中，我们将向您展示如何将 Ray Tune 集成到 PyTorch
训练工作流程中。我们将扩展
 [PyTorch 文档中的本教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
 用于训练
 CIFAR10 图像分类器。




 正如您将看到的，我们只需要添加一些细微的修改。特别是，我们
需要



1. 将数据加载和训练包装在函数中，
2.使一些网络参数可配置，
3.添加检查点（可选），
4.并定义模型调整的搜索空间









 要运行本教程，请确保安装了以下软件包:



* `ray[tune]`
 : 分布式超参数调整库
* `torchvision`
 : 用于数据转换器




 设置/导入
 [¶](#setup-imports "永久链接到此标题")
---------------------------------------------------------------------------------



 让’s 从导入开始：






```
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

```




 构建 PyTorch 模型需要大部分导入。只有最后三个
导入用于 Ray Tune。






 数据加载器
 [¶](#data-loaders "此标题的固定链接")
-------------------------------------------------------------------------



 我们将数据加载器包装在它们自己的函数中，并传递一个全局数据目录。
这样我们就可以在不同的试验之间共享数据目录。






```
def load_data(data_dir="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset

```






 可配置神经网络
 [¶](#configurable-neural-network "永久链接到此标题")
------------------------------------------------------------------------------------------------



 我们只能调整那些可配置的参数。
在此示例中，我们可以指定
全连接层的层大小：






```
class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```






 训练函数
 [¶](#the-train-function "永久链接到此标题")
---------------------------------------------------------------------------



 现在它变得有趣了，因为我们对示例进行了一些更改
 [来自 PyTorch
文档](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) 
 。




 我们将训练脚本包装在函数中
 `train_cifar(config,
 

 data_dir=None)`
 。

 `config`
 参数将接收我们想要
训练的超参数。 
 `data_dir`
 指定我们加载和存储数据的目录，
以便多次运行可以共享相同的数据源。
我们还在运行开始时加载模型和优化器状态，如果提供检查点。在本教程的后面，您将找到有关如何
保存检查点及其用途的信息。






```
net = Net(config["l1"], config["l2"])

checkpoint = session.get_checkpoint()

if checkpoint:
    checkpoint_state = checkpoint.to_dict()
    start_epoch = checkpoint_state["epoch"]
    net.load_state_dict(checkpoint_state["net_state_dict"])
    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
else:
    start_epoch = 0

```




 优化器的学习率也是可配置的：






```
optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

```


我们还将训练数据分为训练和验证子集。因此，我们对 80% 的数据进行训练，并计算剩余 20% 的验证损失。我们迭代训练和测试集的批量大小
也是可配置的。




### 
 使用 DataParallel 添加（多）GPU 支持
 [¶](#adding-multi-gpu-support-with-dataparallel "永久链接到此标题")



 图像分类很大程度上受益于 GPU。幸运的是，我们可以继续在 Ray Tune 中使用
PyTorch’s 抽象。因此，我们可以将模型包装在
 `nn.DataParallel`
 中以支持多个 GPU 上的数据并行训练：






```
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
net.to(device)

```




 通过使用 
 `device`
 变量，我们确保在没有可用 GPU 的情况下训练也能正常工作。 PyTorch 要求我们显式地将数据发送到 GPU 内存，
如下所示：






```
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

```




 该代码现在支持在 CPU、单个 GPU 和多个 GPU 上进行训练。值得注意的是，Ray
还支持
[分数 GPU](https://docs.ray.io/en/master/using-ray-with-gpus.html#fractional-gpus)
，因此我们可以在试验之间共享 GPU ，只要模型仍然适合 GPU 内存。我们’ 稍后再回来
讨论这个问题。





### 
 与 Ray Tune 通信
 [¶](#communicating-with-ray-tune "永久链接到此标题")



 最有趣的部分是与 Ray Tune 的通信：






```
checkpoint_data = {
    "epoch": epoch,
    "net_state_dict": net.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
checkpoint = Checkpoint.from_dict(checkpoint_data)

session.report(
    {"loss": val_loss / val_steps, "accuracy": correct / total},
    checkpoint=checkpoint,
)

```




 这里我们首先保存一个检查点，然后向 Ray Tune 报告一些指标。具体来说，
我们将验证损失和准确性发送回 Ray Tune。然后，Ray Tune 可以使用这些指标
来决定哪种超参数配置可以带来最佳结果。这些指标
还可以用于尽早停止表现不佳的试验，以避免
在这些试验上浪费资源。




 检查点保存是可选的，但是，如果我们想使用高级
s调度程序，例如
 [基于人群的训练](https://docs.ray.io/en/latest/tune/examples/pbt_guide. html) 
 。
此外，通过保存检查点，我们可以稍后加载经过训练的模型并在测试集上验证它们。最后，保存检查点对于容错很有用，并且允许
中断训练并在以后继续训练。





### 
 完整训练函数
 [¶](#full-training-function "永久链接到此标题")



 完整的代码示例如下所示：






```
def train_cifar(config, data_dir=None):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps, "accuracy": correct / total},
            checkpoint=checkpoint,
        )
    print("Finished Training")

```




 正如您所看到的，大部分代码直接改编自原始示例。







 测试集精度
 [¶](#test-set-accuracy "永久链接到此标题")
----------------------------------------------------------------------- -



 通常，机器学习模型的性能是在保留测试集上进行测试的，其中使用的数据尚未用于训练模型。我们还将其包装在
函数中：






```
def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

```




 该函数还需要一个
 `device`
 参数，因此我们可以
在 GPU 上进行测试集验证。






 配置搜索空间
 [¶](#configuring-the-search-space "固定链接到此标题")
------------------------------------------------------------------------------------------------



 最后，我们需要定义 Ray Tune’s 搜索空间。这是一个示例：






```
config = {
    "l1": tune.choice([2 ** i for i in range(9)]),
    "l2": tune.choice([2 ** i for i in range(9)]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
}

```




 `tune.choice()`
 接受统一采样的值列表。
在此示例中，
 `l1`
 和
 `l2`
 参数\ n 应该是 4 到 256 之间的 2 的幂，因此可以是 4、8、16、32、64、128 或 256。

 `lr`
 （学习率）应该在 0.0001 到 0.1 之间均匀采样。最后，
批量大小可以在 2、4、8 和 16 之间选择。




 在每次试验中，Ray Tune 现在将从这些搜索空间中随机采样参数组合。然后，它将并行训练多个模型，并找到其中性能最好的一个。我们还使用
 `ASHAScheduler`
 它将提前终止
不良执行试验。




 我们用
 `functools.partial` 包装
 `train_cifar`
 函数来设置常量
 
 `data_dir`
 参数。我们还可以告诉 Ray Tune 每次试验
应提供哪些资源：






```
gpus_per_trial = 2
# ...
result = tune.run(
    partial(train_cifar, data_dir=data_dir),
    resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    checkpoint_at_end=True)

```




 您可以指定可用的 CPU 数量，例如
以增加 PyTorch
 `DataLoader`
 实例的
 `num_workers`
 数量。在每次试验中，所选的
GPU 数量对 PyTorch 可见。试验无法访问
未请求’ 的 GPU - 因此您’ 不必关心
使用同一组资源的两个试验。




 这里我们还可以指定分数 GPU，因此像 
 `gpus_per_Trial=0.5`
 这样的内容是完全有效的。然后，这些试验将相互共享 GPU。
您只需确保模型仍然适合 GPU 内存即可。




 训练模型后，我们将找到性能最好的模型并从检查点文件加载经过训练的
网络。然后我们获得测试集的准确性并通过打印报告
一切。




 完整的 main 函数如下所示：






```
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)

```






```
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to /var/lib/jenkins/workspace/beginner_source/data/cifar-10-python.tar.gz

  0% 0/170498071 [00:00<?, ?it/s]
  0% 393216/170498071 [00:00<00:45, 3746334.21it/s]
  2% 3145728/170498071 [00:00<00:09, 17436737.90it/s]
  4% 6225920/170498071 [00:00<00:07, 23367611.59it/s]
  5% 9207808/170498071 [00:00<00:06, 25833973.42it/s]
  7% 12288000/170498071 [00:00<00:05, 27559162.59it/s]
  9% 15400960/170498071 [00:00<00:05, 28644180.15it/s]
 11% 18415616/170498071 [00:00<00:05, 29008808.83it/s]
 13% 21594112/170498071 [00:00<00:04, 29856262.02it/s]
 14% 24641536/170498071 [00:00<00:04, 29973490.66it/s]
 16% 27656192/170498071 [00:01<00:04, 30019521.27it/s]
 18% 30736384/170498071 [00:01<00:04, 30170808.41it/s]
 20% 33783808/170498071 [00:01<00:04, 30190525.05it/s]
 22% 36831232/170498071 [00:01<00:04, 30167201.58it/s]
 23% 39878656/170498071 [00:01<00:04, 29260428.25it/s]
 25% 42827776/170498071 [00:01<00:04, 26982577.57it/s]
 27% 45580288/170498071 [00:01<00:04, 25636421.39it/s]
 28% 48201728/170498071 [00:01<00:04, 24915725.82it/s]
 30% 50724864/170498071 [00:01<00:05, 23913970.15it/s]
 31% 53149696/170498071 [00:02<00:04, 23532339.49it/s]
 33% 55541760/170498071 [00:02<00:04, 23383498.24it/s]
 34% 57901056/170498071 [00:02<00:04, 22903012.20it/s]
 35% 60227584/170498071 [00:02<00:04, 22780849.92it/s]
 37% 62521344/170498071 [00:02<00:04, 22779328.41it/s]
 38% 64815104/170498071 [00:02<00:04, 22556584.11it/s]
 39% 67305472/170498071 [00:02<00:04, 23175939.68it/s]
 41% 69632000/170498071 [00:02<00:04, 22891094.15it/s]
 42% 71991296/170498071 [00:02<00:04, 22940283.94it/s]
 44% 74317824/170498071 [00:02<00:04, 22988041.55it/s]
 45% 76644352/170498071 [00:03<00:04, 22776567.73it/s]
 46% 78938112/170498071 [00:03<00:04, 22793542.87it/s]
 48% 81297408/170498071 [00:03<00:03, 22959098.16it/s]
 49% 83853312/170498071 [00:03<00:03, 23617265.17it/s]
 51% 86474752/170498071 [00:03<00:03, 24258118.52it/s]
 52% 89030656/170498071 [00:03<00:03, 24568422.57it/s]
 54% 91652096/170498071 [00:03<00:03, 24970464.06it/s]
 55% 94273536/170498071 [00:03<00:03, 25296320.63it/s]
 57% 96862208/170498071 [00:03<00:02, 25420331.92it/s]
 58% 99450880/170498071 [00:03<00:02, 25452213.25it/s]
 60% 102039552/170498071 [00:04<00:02, 25515213.67it/s]
 61% 104595456/170498071 [00:04<00:02, 25447731.73it/s]
 63% 107184128/170498071 [00:04<00:02, 25529738.58it/s]
 65% 110362624/170498071 [00:04<00:02, 27354838.87it/s]
 67% 113410048/170498071 [00:04<00:02, 28003287.49it/s]
 68% 116457472/170498071 [00:04<00:01, 28584688.95it/s]
 70% 119341056/170498071 [00:04<00:01, 28467744.40it/s]
 72% 122191872/170498071 [00:04<00:01, 27357032.69it/s]
 73% 124944384/170498071 [00:04<00:01, 27076107.47it/s]
 75% 128548864/170498071 [00:04<00:01, 29572523.94it/s]
 77% 131891200/170498071 [00:05<00:01, 30681280.76it/s]
 79% 135299072/170498071 [00:05<00:01, 31556313.85it/s]
 81% 138608640/170498071 [00:05<00:00, 31907727.24it/s]
 83% 141885440/170498071 [00:05<00:00, 32121077.41it/s]
 85% 145260544/170498071 [00:05<00:00, 32430257.49it/s]
 87% 148766720/170498071 [00:05<00:00, 33141756.31it/s]
 89% 152141824/170498071 [00:05<00:00, 33279231.96it/s]
 91% 155680768/170498071 [00:05<00:00, 33877287.66it/s]
 94% 159514624/170498071 [00:05<00:00, 35200983.04it/s]
 96% 163381248/170498071 [00:05<00:00, 36120545.90it/s]
 98% 167280640/170498071 [00:06<00:00, 36818707.01it/s]
100% 170498071/170498071 [00:06<00:00, 27589781.71it/s]
Extracting /var/lib/jenkins/workspace/beginner_source/data/cifar-10-python.tar.gz to /var/lib/jenkins/workspace/beginner_source/data
Files already downloaded and verified
2023-11-15 00:41:41,211 WARNING services.py:1816 -- WARNING: The object store is using /tmp instead of /dev/shm because /dev/shm has only 2147479552 bytes available. This will harm performance! You may be able to free up space by deleting files in /dev/shm. If you are inside a Docker container, you can increase /dev/shm size by passing '--shm-size=10.24gb' to 'docker run' (or add it to the run_options list in a Ray cluster config). Make sure to set this to more than 30% of available RAM.
2023-11-15 00:41:41,352 INFO worker.py:1625 -- Started a local Ray instance.
2023-11-15 00:41:42,538 INFO tune.py:218 -- Initializing Ray automatically. For cluster usage or custom Ray initialization, call `ray.init(...)` before `tune.run(...)`.
== Status ==
Current time: 2023-11-15 00:41:47 (running for 00:00:05.36)
Using AsyncHyperBand: num_stopped=0
Bracket: Iter 8.000: None | Iter 4.000: None | Iter 2.000: None | Iter 1.000: None
Logical resource usage: 2.0/16 CPUs, 0/1 GPUs (0.0/1.0 accelerator_type:M60)
Result logdir: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
Number of trials: 10/10 (9 PENDING, 1 RUNNING)
+-------------------------+----------+-----------------+--------------+------+------+-------------+
| Trial name              | status   | loc             |   batch_size |   l1 |   l2 |          lr |
|-------------------------+----------+-----------------+--------------+------+------+-------------|
| train_cifar_bf064_00000 | RUNNING  | 172.17.0.2:2681 |            2 |   16 |    1 | 0.00213327  |
| train_cifar_bf064_00001 | PENDING  |                 |            4 |    1 |    2 | 0.013416    |
| train_cifar_bf064_00002 | PENDING  |                 |            2 |  256 |   64 | 0.0113784   |
| train_cifar_bf064_00003 | PENDING  |                 |            8 |   64 |  256 | 0.0274071   |
| train_cifar_bf064_00004 | PENDING  |                 |            4 |   16 |    2 | 0.056666    |
| train_cifar_bf064_00005 | PENDING  |                 |            4 |    8 |   64 | 0.000353097 |
| train_cifar_bf064_00006 | PENDING  |                 |            8 |   16 |    4 | 0.000147684 |
| train_cifar_bf064_00007 | PENDING  |                 |            8 |  256 |  256 | 0.00477469  |
| train_cifar_bf064_00008 | PENDING  |                 |            8 |  128 |  256 | 0.0306227   |
| train_cifar_bf064_00009 | PENDING  |                 |            2 |    2 |   16 | 0.0286986   |
+-------------------------+----------+-----------------+--------------+------+------+-------------+


(func pid=2681) 文件已下载并验证
(func pid=2681) 文件已下载并验证
== 状态 ==
当前时间: 2023-11-15 00:41:53 (运行时间 00: 00:10.69)
使用 AsyncHyperBand：num_stopped=0
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000：无
逻辑资源使用：4.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录：/var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
编号试验次数：10/10（8 个待定，2 个正在运行）
+------------------------+-----------------------------+--------------+------+------+- ------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | lr |
|------------------------+----------+---------------+--------------+------+------+---------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 |
|火车_cifar_bf064_00002 |待定 | | 2 | 256 | 256 64 | 64 0.0113784 |
|火车_cifar_bf064_00003 |待定 | | 8 | 64 | 64 256 | 256 0.0274071 |
|火车_cifar_bf064_00004 |待定 | | 4 | 16 | 16 2 | 0.056666 |
|火车_cifar_bf064_00005 |待定 | | 4 | 8 | 64 | 64 0.000353097 |
|火车_cifar_bf064_00006 |待定 | | 8 | 16 | 16 4 | 0.000147684 |
|火车_cifar_bf064_00007 |待定 | | 8 | 256 | 256 256 | 256 0.00477469 |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 |
+-------------------------+---------+------------------+--------------+------+-----+------------ -+


(func pid=2752) 文件已下载并验证[跨集群重复 2 次]（Ray 默认情况下会删除重复日志。设置 RAY_DEDUP_LOGS=0 以禁用日志重复删除，或参阅 https://docs.ray.io/en/master/ray -observability/ray-logging.html#log-deduplication 了解更多选项。)
== 状态 ==
当前时间：2023-11-15 00:41:58（运行时​​间 00:00:16.14）
使用 AsyncHyperBand : num_stopped=0
括号: Iter 8.000: 无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: 无
逻辑资源使用: 6.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
Number试验次数：10/10（7 个待处理，3 个正在运行）
+------------------------+-----------------------------+--------------+------+------+- ------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | lr |
|------------------------+----------+---------------+--------------+------+------+---------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 |
|火车_cifar_bf064_00003 |待定 | | 8 | 64 | 64 256 | 256 0.0274071 |
|火车_cifar_bf064_00004 |待定 | | 4 | 16 | 16 2 | 0.056666 |
|火车_cifar_bf064_00005 |待定 | | 4 | 8 | 64 | 64 0.000353097 |
|火车_cifar_bf064_00006 |待定 | | 8 | 16 | 16 4 | 0.000147684 |
|火车_cifar_bf064_00007 |待定 | | 8 | 256 | 256 256 | 256 0.00477469 |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 |
+-------------------------+---------+------------------+--------------+------+-----+------------ -+


(func pid=2681) [1, 2000] 丢失：2.330
(func pid=3240) 文件已下载并验证[跨集群重复 2 次]
== 状态 ==
当前时间：2023-11-15 00 :42:04（运行 00:00:21.80）
使用 AsyncHyperBand：num_stopped=0
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: 无
逻辑资源使用: 8.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
Number试验次数：10/10（6 个待定，4 个正在运行）
+------------------------+-----------------------------+--------------+------+------+- ------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | lr |
|------------------------+----------+---------------+--------------+------+------+---------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 |
|火车_cifar_bf064_00004 |待定 | | 4 | 16 | 16 2 | 0.056666 |
|火车_cifar_bf064_00005 |待定 | | 4 | 8 | 64 | 64 0.000353097 |
|火车_cifar_bf064_00006 |待定 | | 8 | 16 | 16 4 | 0.000147684 |
|火车_cifar_bf064_00007 |待定 | | 8 | 256 | 256 256 | 256 0.00477469 |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 |
+-------------------------+---------+------------------+--------------+------+-----+------------ -+


(func pid=2752) [1, 2000] 丢失：2.313
(func pid=3728) 已下载并验证的文件 [跨集群重复 2 次]
(func pid=2681) [1, 4000] 丢失：1.152\ n== 状态 ==
当前时间：2023-11-15 00:42:11（运行时间为 00:00:28.43）
使用 AsyncHyperBand：num_stopped=0
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: 无
逻辑资源使用: 10.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
Number试验次数：10/10（5 个待定，5 个正在运行）
+------------------------+-----------------------------+--------------+------+------+- ------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | lr |
|------------------------+----------+---------------+--------------+------+------+---------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 |
|火车_cifar_bf064_00005 |待定 | | 4 | 8 | 64 | 64 0.000353097 |
|火车_cifar_bf064_00006 |待定 | | 8 | 16 | 16 4 | 0.000147684 |
|火车_cifar_bf064_00007 |待定 | | 8 | 256 | 256 256 | 256 0.00477469 |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 |
+-------------------------+---------+------------------+--------------+------+-----+------------ -+


(func pid=4217) 文件已下载并验证
(func pid=4217) 文件已下载并验证
(func pid=3240) 文件已下载并验证
(func pid=2752) [1, 4000]损失：1.155
== 状态 ==
当前时间：2023-11-15 00:42:18（运行时​​间为 00:00:35.41）
使用 AsyncHyperBand：num_stopped=0
Bracket：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: 无
逻辑资源使用: 12.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
Number试验次数：10/10（4 个待定，6 个正在运行）
+------------------------+-----------------------------+--------------+------+------+- ------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | lr |
|------------------------+----------+---------------+--------------+------+------+---------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 |
|火车_cifar_bf064_00006 |待定 | | 8 | 16 | 16 4 | 0.000147684 |
|火车_cifar_bf064_00007 |待定 | | 8 | 256 | 256 256 | 256 0.00477469 |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 |
+-------------------------+---------+------------------+--------------+------+-----+------------ -+


(func pid=4708) 文件已下载并验证
(func pid=3728) [1, 2000] 丢失：2.101 [跨集群重复 2x]
(func pid=4708) 文件已下载并验证
(func pid=2752) [1, 6000] 损失: 0.769 [跨集群重复 3 次]
== 状态 ==
当前时间: 2023-11-15 00:42:26 (运行 00:00:44.31)
使用AsyncHyperBand：num_stopped=0
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: 无
逻辑资源使用: 14.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
Number试验次数：10/10（3 个待定，7 个正在运行）
+------------------------+-----------------------------+--------------+------+------+- ------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | lr |
|------------------------+----------+---------------+--------------+------+------+---------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 |
|火车_cifar_bf064_00006 |跑步 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 |
|火车_cifar_bf064_00007 |待定 | | 8 | 256 | 256 256 | 256 0.00477469 |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 |
+-------------------------+---------+------------------+--------------+------+-----+------------ -+


(func pid=5199) 文件已下载并验证
(func pid=5199) 文件已下载并验证
(func pid=3728) [1, 4000] 损失：1.033 [跨集群重复 2 次]
==状态 ==
当前时间：2023-11-15 00:42:34（运行时间为 00:00:52.20）
使用 AsyncHyperBand：num_stopped=0
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: 无
逻辑资源使用: 16.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
Number试验次数：10/10（2 个待定，8 个正在运行）
+------------------------+-----------------------------+--------------+------+------+- ------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | lr |
|------------------------+----------+---------------+--------------+------+------+---------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 |
|火车_cifar_bf064_00006 |跑步 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 |
+-------------------------+---------+------------------+--------------+------+-----+------------ -+


(func pid=5692) 文件已下载并验证
(func pid=5692) 文件已下载并验证
== 状态 ==
当前时间: 2023-11-15 00:42:39 (运行时间 00: 00:57.21)
使用 AsyncHyperBand：num_stopped=0
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: 无
逻辑资源使用: 16.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
Number试验次数：10/10（2 个待定，8 个正在运行）
+------------------------+-----------------------------+--------------+------+------+- ------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | lr |
|------------------------+----------+---------------+--------------+------+------+---------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 |
|火车_cifar_bf064_00006 |跑步 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 |
+-------------------------+---------+------------------+--------------+------+-----+------------ -+


(func pid=3240) [1, 6000] 损失：0.772 [跨集群重复 5 次]
== 状态 ==
当前时间：2023-11-15 00:42:44（运行时间为 00:01:02.22） 
使用 AsyncHyperBand：num_stopped=0
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: 无
逻辑资源使用: 16.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42
Number试验次数：10/10（2 个待定，8 个正在运行）
+------------------------+-----------------------------+--------------+------+------+- ------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | lr |
|------------------------+----------+---------------+--------------+------+------+---------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 |
|火车_cifar_bf064_00006 |跑步 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 |
+-------------------------+---------+------------------+--------------+------+-----+------------ -+


(func pid=4708) [1, 4000] 损失：1.121 [跨集群重复 2 倍]
train_cifar_bf064_00003 的结果：
 准确度：0.2254
 日期：2023-11-15_00-42-47
 完成：false
主机名：47e6634b378a \ n iterations_since_restore：1 \ n损失：2.028742760181427 \ n node_ip：172.17.0.2 \ n pid：3728 \ n should_checkpoint：true \ n time_since_restore：43.234081745147705 \ n time_this_it er_s: 43.234081745147705
 time_total_s: 43.234081745147705
 时间戳: 1700008967 
 Training_iteration: 1
 Trial_id: bf064_00003

== 状态 ==
当前时间: 2023-11-15 00:42:52 (运行 00:01:10.06)
使用 AsyncHyperBand: num_stopped=0\ nBracket：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: -2.028742760181427
逻辑资源使用: 16.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（2 个待处理，8 个正在运行）
+------------------------+-----------+----------------+--------------+------+-----+ ----------+--------+--------------------+-------- -+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|-------------------------+----------+---------------+--------------+------+------+---------------+--------+--------------------+---------+-------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | | | | |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | | | | |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 1 | 43.2341 | 2.02874 | 0.2254 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 | | | |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 | | | |
|火车_cifar_bf064_00006 |跑步 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | | | | |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | | | | |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 | | | |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 | | | | |
+-------------------------+----------+------------------+--------------+------+------+---------------- +--------+------------------+---------+------------ -+


(func pid=5692) [1, 2000] 损失：1.855 [跨集群重复 4 次]
== 状态 ==
当前时间：2023-11-15 00:42:57（运行时间为 00:01:15.06） 
使用 AsyncHyperBand：num_stopped=0
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: -2.028742760181427
逻辑资源使用: 16.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（2 个待处理，8 个正在运行）
+------------------------+-----------+----------------+--------------+------+-----+ ----------+--------+--------------------+-------- -+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|-------------------------+----------+---------------+--------------+------+------+---------------+--------+--------------------+---------+-------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | | | | |
|火车_cifar_bf064_00001 |跑步 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | | | | |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 1 | 43.2341 | 2.02874 | 0.2254 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 | | | |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 | | | |
|火车_cifar_bf064_00006 |跑步 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | | | | |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | | | | |
|火车_cifar_bf064_00008 |待定 | | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 | | | |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 | | | | |
+-------------------------+----------+------------------+--------------+------+------+---------------- +--------+------------------+---------+------------ -+


train_cifar_bf064_00001 的结果：
 准确度：0.1008
 日期：2023-11-15_00-43-00
 完成：true
 主机名：47e6634b378a
 iterations_since_restore：1
 损失：2.3081862998008726
 node_ip：1 72.17.0.2\ n pid：2752 \ n should_checkpoint：true \ n time_since_restore：67.53077411651611 \ n time_this_iter_s：67.53077411651611 \ n time_total_s：67.53077411651611 \ n时间戳：1700008980 \ n Training_iteration : 1
 Trial_id: bf064_00001

试验 train_cifar_bf064_00001 已完成。
(func pid=2681) [1, 14000] 丢失：0.329 [跨集群重复 4 次]
(func pid=2752) 文件已下载并验证
(func pid=2752) 文件已下载并验证
== 状态 = =
当前时间：2023-11-15 00:43:05（运行时间为 00:01:23.27）
使用 AsyncHyperBand：num_stopped=1
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: -2.16846452999115
逻辑资源使用: 16.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 个待处理、8 个正在运行、1 个已终止）
+-------------------------+-------------+-----------------+--------------+------+-------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | | | | |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 1 | 43.2341 | 2.02874 | 0.2254 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 | | | |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 | | | |
|火车_cifar_bf064_00006 |跑步 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | | | | |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | | | | |
|火车_cifar_bf064_00008 |跑步 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 | | | |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=5692) [1, 4000] 损失：0.780 [跨集群重复 3 次]
== 状态 ==
当前时间：2023-11-15 00:43:10（运行时间 00:01:28.28） 
使用 AsyncHyperBand：num_stopped=1
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: -2.16846452999115
逻辑资源使用: 16.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 个待处理、8 个正在运行、1 个已终止）
+-------------------------+-------------+-----------------+--------------+------+-------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | | | | |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 1 | 43.2341 | 2.02874 | 0.2254 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 | | | |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 | | | |
|火车_cifar_bf064_00006 |跑步 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | | | | |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | | | | |
|火车_cifar_bf064_00008 |跑步 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 | | | |
|火车_cifar_bf064_00009 |待定 | | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00006 的结果：
 准确度：0.1689
 日期：2023-11-15_00-43-11
 完成：true
 主机名：47e6634b378a
 iterations_since_restore：1
 损失：2.2780231006622316
 node_ip：1 72.17.0.2\ n pid：5199 \ n should_checkpoint：true \ n time_since_restore：44.75676083564758 \ n time_this_iter_s：44.75676083564758 \ n time_total_s：44.75676083564758 \ n时间戳：1700008991 \ n Training_iteration : 1
 Trial_id: bf064_00006

试验 train_cifar_bf064_00006 已完成。
(func pid=5199) 文件已下载并验证
(func pid=5199) 文件已下载并验证
== 状态 ==
当前时间: 2023-11-15 00:43:16 (运行时间 00:01: 34.11)
使用 AsyncHyperBand：num_stopped=2
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: -2.2780231006622316
逻辑资源使用: 16.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（8 次运行，2 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | | | | |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 1 | 43.2341 | 2.02874 | 0.2254 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 | | | |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 | | | |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | | | | |
|火车_cifar_bf064_00008 |跑步 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 | | | |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=2752) [1, 2000] 损失：2.108 [跨集群重复 5 次]
== 状态 ==
当前时间：2023-11-15 00:43:21（运行时间为 00:01:39.12） 
使用 AsyncHyperBand：num_stopped=2
括号：Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: -2.2780231006622316
逻辑资源使用: 16.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（8 次运行，2 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | | | | |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 1 | 43.2341 | 2.02874 | 0.2254 |
|火车_cifar_bf064_00004 |跑步 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 | | | |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 | | | |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | | | | |
|火车_cifar_bf064_00008 |跑步 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 | | | |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=2681) [1, 18000] 损失：0.256 [跨集群重复 2 次]
train_cifar_bf064_00007 的结果：
 准确度：0.4756
 日期：2023-11-15_00-43-24
 完成：false
主机名：47e6634b378a \ n iterations_since_restore：1 \ n损失：1.451918194580078 \ n node_ip：172.17.0.2 \ n pid：5692 \ n should_checkpoint：true \ n time_since_restore：49.35653567314148 \ n time_this_iter _s: 49.35653567314148
 time_total_s: 49.35653567314148
 时间戳: 1700009004 
 Training_iteration: 1
 Trial_id: bf064_00007

train_cifar_bf064_00004 的结果:
 准确度: 0.0986
 日期: 2023-11-15_00-43-24
 完成: true
 主机名: 47e6634b378a
 iterations_since_restore: 1 
 损失：2.3359092045783996
 node_ip：172.17.0.2
 pid：4217
 should_checkpoint：true
 time_since_restore：73.20940923690796
 time_this_iter_s：73.20940923690796
 time_total_ s: 73.20940923690796
 时间戳: 1700009004
 训练迭代: 1
 Trial_id : bf064_00004

试用 train_cifar_bf064_00004 已完成。
== 状态 ==
当前时间: 2023-11-15 00:43:29 (运行时间 00:01:46.67)
使用 AsyncHyperBand: num_stopped=3
Bracket: Iter 8.000：无 | Iter 4.000：无 | Iter 2.000：无 | Iter 1.000: -2.2780231006622316
逻辑资源使用: 14.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（7 次运行，3 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | | | | |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 1 | 43.2341 | 2.02874 | 0.2254 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 | | | |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 1 | 49.3565 | 49.3565 1.45192 | 0.4756 |
|火车_cifar_bf064_00008 |跑步 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 | | | |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=3240) [1, 12000] 损失: 0.386 [跨集群重复 3 次]
train_cifar_bf064_00003 的结果:
 准确度: 0.2368
 日期: 2023-11-15_00-43-32
 完成: false
主机名：47e6634b378a \ n iterations_since_restore：2 \ n损失：2.0462085193634034 \ n node_ip：172.17.0.2 \ n pid：3728 \ n should_checkpoint：true \ n time_since_restore：87.77183723449707 \ n time_this_it er_s: 44.537755489349365
 time_total_s: 87.77183723449707
 时间戳: 1700009012 
 Training_iteration: 2
 Trial_id: bf064_00003

train_cifar_bf064_00005 的结果:
 准确度: 0.3659
 日期: 2023-11-15_00-43-33
 完成: false
 主机名: 47e6634b378a
 iterations_since_restore: 1 
 损失：1.6868873691082
 node_ip：172.17.0.2
 pid：4708
 should_checkpoint：true
 time_since_restore：75.3049852848053
 time_this_iter_s：75.3049852848053
 time_total_s：75.3049852848053
 时间戳：1700009013
 训练迭代：1
 Trial_id : bf064_00005

(func pid=5199) [1, 4000] 损失: 1.169 [跨集群重复 3 次]
== 状态 ==
当前时间: 2023-11-15 00:43:38 (运行时间00:01:55.74)
使用 AsyncHyperBand：num_stopped=3
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-2.0462085193634034 | Iter 1.000: -2.153382930421829
逻辑资源使用: 14.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（7 次运行，3 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | | | | |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 2 | 87.7718 | 2.04621 | 0.2368 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 1 | 49.3565 | 49.3565 1.45192 | 0.4756 |
|火车_cifar_bf064_00008 |跑步 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 | | | |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:43:43（运行时间为 00:02:00.76）
使用 AsyncHyperBand：num_stopped=3
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-2.0462085193634034 | Iter 1.000: -2.153382930421829
逻辑资源使用: 14.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（7 次运行，3 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | | | | |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 2 | 87.7718 | 2.04621 | 0.2368 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 1 | 49.3565 | 49.3565 1.45192 | 0.4756 |
|火车_cifar_bf064_00008 |跑步 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 | | | |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [2, 2000] 损失：1.656 [跨集群重复 2 倍]
train_cifar_bf064_00008 的结果：
 准确度：0.2189
 日期：2023-11-15_00-43-48
 完成：true
主机名：47e6634b378a \ n iterations_since_restore：1 \ n损失：2.180678374862671 \ n node_ip：172.17.0.2 \ n pid：2752 \ n should_checkpoint：true \ n time_since_restore：47.31365776062012 \ n time_this_iter _s: 47.31365776062012
 time_total_s: 47.31365776062012
 时间戳: 1700009028 
 Training_iteration：1
 Trial_id：bf064_00008

试验 train_cifar_bf064_00008 已完成。
train_cifar_bf064_00000 的结果：
 准确度：0.102
 日期：2023-11-15_00-43-50
 完成：true
 主机名：47e663 4b378a 
 iterations_since_restore: 1
 损失: 2.303914258527756
 node_ip: 172.17.0.2
 pid: 2681
 should_checkpoint: true
 time_since_restore: 123.00910973548889
 time_this_iter_s: 123.0091097 3548889
 总时间：123.00910973548889
 时间戳：1700009030
 训练迭代: 1
 Trial_id: bf064_00000

试验 train_cifar_bf064_00000 已完成。
== 状态 ==
当前时间: 2023-11-15 00:43:51 (运行时间 00:02:08.39)
使用 AsyncHyperBand: num_stopped= 5
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-2.0462085193634034 | Iter 1.000: -2.2293507377624513
逻辑资源使用: 12.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（6 次运行，4 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00000 |跑步 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 2 | 87.7718 | 2.04621 | 0.2368 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 1 | 49.3565 | 49.3565 1.45192 | 0.4756 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=5692) [2, 4000] 损失：0.692 [跨集群重复 4 次]
== 状态 ==
当前时间：2023-11-15 00:43:56（运行时间 00:02:13.41） 
使用 AsyncHyperBand：num_stopped=5
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-2.0462085193634034 | Iter 1.000: -2.2293507377624513
逻辑资源使用: 10.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（5 次运行，5 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 2 | 87.7718 | 2.04621 | 0.2368 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 1 | 49.3565 | 49.3565 1.45192 | 0.4756 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=3728) [3, 4000] 损失：1.050 [跨集群重复 3 次]
== 状态 ==
当前时间：2023-11-15 00:44:01（运行时间为 00:02:18.41） 
使用 AsyncHyperBand：num_stopped=5
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-2.0462085193634034 | Iter 1.000: -2.2293507377624513
逻辑资源使用: 10.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（5 次运行，5 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 2 | 87.7718 | 2.04621 | 0.2368 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 1 | 49.3565 | 49.3565 1.45192 | 0.4756 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00007 的结果：
 准确度：0.5254
 日期：2023-11-15_00-44-05
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：2
 损失：1.3188089747428895
 node_ip：1 72.17.0.2\ n pid：5692 \ n should_checkpoint：true \ n time_since_restore：91.16031050682068 \ n time_this_iter_s：41.8037748336792 \ n time_total_s：91.16031050682068 \ n时间戳：1700009045 \ n Training_iteration： 2
 Trial_id: bf064_00007

(func pid=5199) [ 1, 10000] 丢失：0.468 [跨集群重复 2 倍]
== 状态 ==
当前时间：2023-11-15 00:44:11（运行 00:02:28.38）
使用 AsyncHyperBand：num_stopped=5 
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-1.6825087470531463 | Iter 1.000: -2.2293507377624513
逻辑资源使用: 10.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（5 次运行，5 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 2 | 87.7718 | 2.04621 | 0.2368 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 2 | 91.1603 | 91.1603 1.31881 | 0.5254 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00003 的结果：
 准确度：0.2211
 日期：2023-11-15_00-44-12
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：3
 损失：2.0626959712028503
 node_ip：1 72.17.0.2\ n pid：3728 \ n should_checkpoint：true \ n time_since_restore：127.77058959007263 \ n time_this_iter_s：39.99875235557556 \ n time_total_s：127.77058959007263 \ n时间戳：1700009052 \ n Training_迭代：3
 Trial_id：bf064_00003

(func pid=3240) [ 1, 18000] 丢失：0.257 [跨集群重复 2 倍]
== 状态 ==
当前时间：2023-11-15 00:44:17（运行 00:02:34.58）
使用 AsyncHyperBand：num_stopped=5 
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-1.6825087470531463 | Iter 1.000: -2.2293507377624513
逻辑资源使用: 10.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（5 次运行，5 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 3 | 127.771 | 127.771 2.0627 | 2.0627 0.2211 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 2 | 91.1603 | 91.1603 1.31881 | 0.5254 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:44:22（运行时间为 00:02:39.59）
使用 AsyncHyperBand：num_stopped=5
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-1.6825087470531463 | Iter 1.000: -2.2293507377624513
逻辑资源使用: 10.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（5 次运行，5 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 3 | 127.771 | 127.771 2.0627 | 2.0627 0.2211 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 2 | 91.1603 | 91.1603 1.31881 | 0.5254 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=5199) [1, 14000] 损失：0.334 [跨集群重复 4 次]
== 状态 ==
当前时间：2023-11-15 00:44:27（运行时间为 00:02:44.60） 
使用 AsyncHyperBand：num_stopped=5
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-1.6825087470531463 | Iter 1.000: -2.2293507377624513
逻辑资源使用: 10.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（5 次运行，5 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 3 | 127.771 | 127.771 2.0627 | 2.0627 0.2211 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 2 | 91.1603 | 91.1603 1.31881 | 0.5254 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:44:32（运行时间为 00:02:49.61）
使用 AsyncHyperBand：num_stopped=5
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-1.6825087470531463 | Iter 1.000: -2.2293507377624513
逻辑资源使用: 10.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（5 次运行，5 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 3 | 127.771 | 127.771 2.0627 | 2.0627 0.2211 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 1 | 75.305 | 75.305 1.68689 | 0.3659 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 2 | 91.1603 | 91.1603 1.31881 | 0.5254 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=5692) [3, 4000] 损失: 0.629 [跨集群重复 4 次]
train_cifar_bf064_00005 的结果:
 准确度: 0.4406
 日期: 2023-11-15_00-44-36
 完成: false
主机名：47e6634b378a \ n iterations_since_restore：2 \ n损失：1.4916184381961823 \ n node_ip：172.17.0.2 \ n pid：4708 \ n should_checkpoint：true \ n time_since_restore：138.08053255081177 \ n time_this_ iter_s: 62.77554726600647
 time_total_s: 138.08053255081177
 时间戳: 1700009076 
 Training_iteration: 2
 Trial_id: bf064_00005

(func pid=3728) [4, 4000] 损失: 1.155 [跨集群重复 2 次]
== 状态 ==
当前时间: 2023-11-15 00:44:41（运行 00:02:58.50）
使用 AsyncHyperBand：num_stopped=5
括号：Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.2293507377624513
逻辑资源使用: 10.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（5 次运行，5 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00002 |跑步 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | | | | |
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 3 | 127.771 | 127.771 2.0627 | 2.0627 0.2211 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 2 | 138.081 | 1.49162 | 0.4406 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 2 | 91.1603 | 91.1603 1.31881 | 0.5254 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00002 的结果：
 准确度：0.1028
 日期：2023-11-15_00-44-44
 完成：true
 主机名：47e6634b378a
 iterations_since_restore：1
 损失：2.3167312343597413
 node_ip：1 72.17.0.2\ n pid：3240 \ n should_checkpoint：true \ n time_since_restore：165.89642000198364 \ n time_this_iter_s：165.89642000198364 \ n time_total_s：165.89642000198364 \ n时间戳：1700009084 \ n训练_iteration: 1
 Trial_id: bf064_00002

试验 train_cifar_bf064_00002 已完成。
train_cifar_bf064_00007 的结果:
 准确度：0.5636
 日期：2023-11-15_00-44-44
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：3
 损失：1.246082941901684
 node_ip：172.17.0.2
 pid： 5692
 should_checkpoint: true
 time_since_restore: 129.91541862487793
 time_this_iter_s: 38.75510811805725
 time_total_s: 129.91541862487793
 时间戳: 1700009084
 Training_iteration: 3
 试验 ID: bf064_00007

(func pid=4708) [3, 2000 ] 损失：1.495 [跨集群重复 2 倍]
== 状态 ==
当前时间：2023-11-15 00:44:49（运行时间 00:03:07.13）
使用 AsyncHyperBand：num_stopped=6
Bracket： Iter 8.000：无 | Iter 4.000：无 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.2780231006622316
逻辑资源使用: 8.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（4 次运行，6 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 3 | 127.771 | 127.771 2.0627 | 2.0627 0.2211 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 2 | 138.081 | 1.49162 | 0.4406 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 3 | 129.915 | 129.915 1.24608 | 0.5636 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00003 的结果：
 准确度：0.0984
 日期：2023-11-15_00-44-51
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：4
 损失：2.3100966709136963
 node_ip：1 72.17.0.2\ n pid：3728 \ n should_checkpoint：true \ n time_since_restore：166.9245798587799 \ n time_this_iter_s：39.153990268707275 \ n time_total_s：166.9245798587799 \ n时间戳：1700009091 \ n Training_it次数: 4
 Trial_id: bf064_00003

(func pid=5199) [ 1, 20000] 损失: 0.234
== 状态 ==
当前时间: 2023-11-15 00:44:56 (运行时间 00:03:13.74)
使用 AsyncHyperBand: num_stopped=6
Bracket: Iter 8.000:无 |迭代 4.000：-2.3100966709136963 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.2780231006622316
逻辑资源使用: 8.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（4 次运行，6 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 4 | 166.925 | 166.925 2.3101 | 2.3101 0.0984 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 2 | 138.081 | 1.49162 | 0.4406 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 3 | 129.915 | 129.915 1.24608 | 0.5636 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [3, 4000] 损失: 0.743
== 状态 ==
当前时间: 2023-11-15 00:45:01 (运行 00:03:18.75)
使用 AsyncHyperBand: num_stopped= 6
括号：Iter 8.000：无 |迭代 4.000：-2.3100966709136963 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.2780231006622316
逻辑资源使用: 8.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（4 次运行，6 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 4 | 166.925 | 166.925 2.3101 | 2.3101 0.0984 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 2 | 138.081 | 1.49162 | 0.4406 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 3 | 129.915 | 129.915 1.24608 | 0.5636 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=3728) [5, 2000] 损失：2.310 [跨集群重复 2 次]
== 状态 ==
当前时间：2023-11-15 00:45:06（运行 00:03:23.76） 
使用 AsyncHyperBand：num_stopped=6
括号：Iter 8.000：无 |迭代 4.000：-2.3100966709136963 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.2780231006622316
逻辑资源使用: 8.0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（4 次运行，6 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 4 | 166.925 | 166.925 2.3101 | 2.3101 0.0984 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 2 | 138.081 | 1.49162 | 0.4406 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 3 | 129.915 | 129.915 1.24608 | 0.5636 |
|火车_cifar_bf064_00009 |跑步 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | | | | |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00009 的结果：
 准确度：0.0994
 日期：2023-11-15_00-45-07
 完成：true
 主机名：47e6634b378a
 iterations_since_restore：1
 损失：2.3218609105825423
 node_ip：1 72.17.0.2\ n pid：5199
 should_checkpoint：true
 time_since_restore：116.12234020233154
 time_this_iter_s：116.12234020233154
 time_total_s：116.12234020233154
时间戳：1700009107
训练_iteration: 1
 Trial_id: bf064_00009

试验 train_cifar_bf064_00009 已完成。
(func pid=5692) [4, 4000] 损失: 0.598 [跨集群重复 2 次]
== 状态 ==
当前时间: 2023-11-15 00:45:12 (运行 00:03:30.25)
使用AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-2.3100966709136963 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 4 | 166.925 | 166.925 2.3101 | 2.3101 0.0984 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 2 | 138.081 | 1.49162 | 0.4406 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 3 | 129.915 | 129.915 1.24608 | 0.5636 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [3, 8000] 损失: 0.364
(func pid=3728) [5, 4000] 损失: 1.155
== 状态 ==
当前时间: 2023-11-15 00:45: 17（运行时间为 00:03:35.26）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-2.3100966709136963 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 4 | 166.925 | 166.925 2.3101 | 2.3101 0.0984 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 2 | 138.081 | 1.49162 | 0.4406 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 3 | 129.915 | 129.915 1.24608 | 0.5636 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00007 的结果：
 准确度：0.5565
 日期：2023-11-15_00-45-19
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：4
 损失：1.2592607187390328
 node_ip：1 72.17.0.2\ n pid：5692 \ n should_checkpoint：true \ n time_since_restore：164.99337315559387 \ n time_this_iter_s：35.07795453071594 \ n time_total_s：164.99337315559387 \ n时间戳：1700009119 \ n Training_迭代: 4
 Trial_id: bf064_00007

== 状态 ==
当前时间：2023-11-15 00:45:24（运行时间为 00:03:42.21）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 4 | 166.925 | 166.925 2.3101 | 2.3101 0.0984 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 2 | 138.081 | 1.49162 | 0.4406 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 4 | 164.993 | 164.993 1.25926 | 1.25926 0.5565 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [3, 10000] 损失: 0.289
train_cifar_bf064_00003 的结果:
 准确度: 0.0994
 日期: 2023-11-15_00-45-27
 完成: false
 主机名: 47e6634b378a
 iterations_since_restore : 5
 丢失: 2.3078094497680666
 node_ip: 172.17.0.2
 pid: 3728
 should_checkpoint: true
 time_since_restore: 202.74810004234314
 time_this_iter_s: 35.82352018356323
 time_total_s: 202.74810004234314
 时间戳: 1700009127
 训练迭代: 5\ n Trial_id: bf064_00003

(func pid=5692) [5, 2000] 损失: 1.068
== 状态 ==
当前时间: 2023-11-15 00:45:32 (运行时间 00:03: 49.56)
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 5 | 202.748 | 2.30781 | 2.30781 0.0994 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 2 | 138.081 | 1.49162 | 0.4406 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 4 | 164.993 | 164.993 1.25926 | 1.25926 0.5565 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00005 的结果：
 准确度：0.4593
 日期：2023-11-15_00-45-32
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：3
 损失：1.4651956122279166
 node_ip：1 72.17.0.2\ n pid：4708 \ n should_checkpoint：true \ n time_since_restore：194.22449851036072 \ n time_this_iter_s：56.14396595954895 \ n time_total_s：194.22449851036072 \ n时间戳：1700009132 \ n Training_迭代: 3
 Trial_id: bf064_00005

== 状态 ==
当前时间：2023-11-15 00:45:37（运行时间为 00:03:54.65）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 5 | 202.748 | 2.30781 | 2.30781 0.0994 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 3 | 194.224 | 194.224 1.4652 | 0.4593 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 4 | 164.993 | 164.993 1.25926 | 1.25926 0.5565 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=3728) [6, 2000] 损失: 2.310
(func pid=4708) [4, 2000] 损失: 1.408
== 状态 ==
当前时间: 2023-11-15 00:45: 42（运行 00:03:59.66）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 5 | 202.748 | 2.30781 | 2.30781 0.0994 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 3 | 194.224 | 194.224 1.4652 | 0.4593 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 4 | 164.993 | 164.993 1.25926 | 1.25926 0.5565 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:45:47（运行时间为 00:04:04.67）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 5 | 202.748 | 2.30781 | 2.30781 0.0994 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 3 | 194.224 | 194.224 1.4652 | 0.4593 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 4 | 164.993 | 164.993 1.25926 | 1.25926 0.5565 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [4, 4000] 损失：0.696 [跨集群重复 2 次]
== 状态 ==
当前时间：2023-11-15 00:45:52（运行时间 00:04:09.68） 
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 5 | 202.748 | 2.30781 | 2.30781 0.0994 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 3 | 194.224 | 194.224 1.4652 | 0.4593 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 4 | 164.993 | 164.993 1.25926 | 1.25926 0.5565 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00007 的结果：
 准确度：0.575
 日期：2023-11-15_00-45-52
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：5
 损失：1.2706222375750542
 node_ip：17 2.17.0.2\ n pid：5692 \ n should_checkpoint：true \ n time_since_restore：198.0124270915985 \ n time_this_iter_s：33.01905393600464 \ n time_total_s：198.0124270915985 \ n时间戳：1700009152 \ n Training_iteration : 5
 Trial_id: bf064_00007

== 状态 ==
当前时间：2023-11-15 00:45:57（运行时间为 00:04:15.22）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 5 | 202.748 | 2.30781 | 2.30781 0.0994 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 3 | 194.224 | 194.224 1.4652 | 0.4593 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 5 | 198.012 | 1.27062 | 0.575 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [4, 6000] 损失: 0.462 [跨集群重复 2 次]
train_cifar_bf064_00003 的结果:
 准确度: 0.0999
 日期: 2023-11-15_00-46-01
 完成: false
主机名：47e6634b378a \ n iterations_since_restore：6 \ n损失：2.3086511194229127 \ n node_ip：172.17.0.2 \ n pid：3728 \ n should_checkpoint：true \ n time_since_restore：236.95160698890686 \ n time_this_ iter_s: 34.20350694656372
 time_total_s: 236.95160698890686
 时间戳: 1700009161 
 Training_iteration: 6
 Trial_id: bf064_00003

(func pid=5692) [6, 2000] 损失: 1.030
== 状态 ==
当前时间: 2023-11-15 00:46:06 (运行时间为 00:04:23.76)
使用 AsyncHyperBand: num_stopped=7
Bracket: Iter 8.000: None |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 6 | 236.952 | 236.952 2.30865 | 2.30865 0.0999 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 3 | 194.224 | 194.224 1.4652 | 0.4593 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 5 | 198.012 | 1.27062 | 0.575 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [4, 8000] 损失: 0.340
== 状态 ==
当前时间: 2023-11-15 00:46:11 (运行 00:04:28.77)
使用 AsyncHyperBand: num_stopped= 7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 6 | 236.952 | 236.952 2.30865 | 2.30865 0.0999 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 3 | 194.224 | 194.224 1.4652 | 0.4593 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 5 | 198.012 | 1.27062 | 0.575 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=3728) [7, 2000] 损失: 2.310
(func pid=5692) [6, 4000] 损失: 0.543
== 状态 ==
当前时间: 2023-11-15 00:46: 16（运行 00:04:33.78）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 6 | 236.952 | 236.952 2.30865 | 2.30865 0.0999 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 3 | 194.224 | 194.224 1.4652 | 0.4593 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 5 | 198.012 | 1.27062 | 0.575 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:46:21（运行时间为 00:04:38.79）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.7846786948263644 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 6 | 236.952 | 236.952 2.30865 | 2.30865 0.0999 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 3 | 194.224 | 194.224 1.4652 | 0.4593 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 5 | 198.012 | 1.27062 | 0.575 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00005 的结果：
 准确度：0.4865
 日期：2023-11-15_00-46-24
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：4
 损失：1.4287443323016167
 node_ip：1 72.17.0.2\ n pid：4708 \ n should_checkpoint：true \ n time_since_restore：246.26441478729248 \ n time_this_iter_s：52.03991627693176 \ n time_total_s：246.26441478729248 \ n时间戳：1700009184 \ n Training_迭代：4
 Trial_id：bf064_00005

(func pid=3728) [ 7, 4000] 损失：1.155 [跨集群重复 2 倍]
train_cifar_bf064_00007 的结果：
 准确度：0.572
 日期：2023-11-15_00-46-26
 完成：false
 主机名：47e6634b378a
 iterations_since_restore： 6
 丢失：1.291559043586254
 node_ip：172.17.0.2
 pid：5692
 should_checkpoint：true
 time_since_restore：231.19213557243347
 time_this_iter_s：33.17970848083496
 time_to tal_s: 231.19213557243347
 时间戳: 1700009186
 训练迭代: 6
 Trial_id: bf064_00007

== 状态 ==
当前时间: 2023-11-15 00:46:31 (运行时间 00:04:48.41)
使用 AsyncHyperBand: num_stopped=7
Bracket: Iter 8.000: None |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 6 | 236.952 | 236.952 2.30865 | 2.30865 0.0999 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 4 | 246.264 | 246.264 1.42874 | 0.4865 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 6 | 231.192 | 231.192 1.29156 | 0.572 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [5, 2000] 损失: 1.306
train_cifar_bf064_00003 的结果:
 准确度: 0.0993
 日期: 2023-11-15_00-46-35
 完成: false
 主机名: 47e6634b378a
 iterations_since_restore : 7
 丢失: 2.306323892593384
 node_ip: 172.17.0.2
 pid: 3728
 should_checkpoint: true
 time_since_restore: 271.1397247314453
 time_this_iter_s: 34.18811774253845
 time_ Total_s: 271.1397247314453
 时间戳: 1700009195
 训练迭代: 7\ n Trial_id: bf064_00003

(func pid=5692) [7, 2000] 损失: 0.984
== 状态 ==
当前时间: 2023-11-15 00:46:40 (运行时间 00:04: 57.95)
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 7 | 271.14 | 271.14 2.30632 | 2.30632 0.0993 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 4 | 246.264 | 246.264 1.42874 | 0.4865 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 6 | 231.192 | 231.192 1.29156 | 0.572 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:46:45（运行时间为 00:05:02.96）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 7 | 271.14 | 271.14 2.30632 | 2.30632 0.0993 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 4 | 246.264 | 246.264 1.42874 | 0.4865 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 6 | 231.192 | 231.192 1.29156 | 0.572 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=3728) [8, 2000] 损失：2.310 [跨集群重复 2 次]
== 状态 ==
当前时间：2023-11-15 00:46:50（运行时间 00:05:07.97） 
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 7 | 271.14 | 271.14 2.30632 | 2.30632 0.0993 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 4 | 246.264 | 246.264 1.42874 | 0.4865 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 6 | 231.192 | 231.192 1.29156 | 0.572 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:46:55（运行时间为 00:05:12.97）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 7 | 271.14 | 271.14 2.30632 | 2.30632 0.0993 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 4 | 246.264 | 246.264 1.42874 | 0.4865 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 6 | 231.192 | 231.192 1.29156 | 0.572 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00007 的结果：
 准确度：0.5584
 日期：2023-11-15_00-46-59
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：7
 损失：1.3122931097507478
 node_ip：1 72.17.0.2\ n pid：5692 \ n should_checkpoint：true \ n time_since_restore：264.2365355491638 \ n time_this_iter_s：33.04439997673035 \ n time_total_s：264.2365355491638 \ n时间戳：1700009219 \ n Training_iteration : 7
 Trial_id: bf064_00007

(func pid=3728) [ 8, 4000] 丢失：1.155 [跨集群重复 3 倍]
== 状态 ==
当前时间：2023-11-15 00:47:04（运行 00:05:21.45）
使用 AsyncHyperBand：num_stopped=7 
括号：Iter 8.000：无 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 7 | 271.14 | 271.14 2.30632 | 2.30632 0.0993 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 4 | 246.264 | 246.264 1.42874 | 0.4865 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 7 | 264.237 | 264.237 1.31229 | 1.31229 0.5584 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:47:09（运行时间为 00:05:26.46）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：无 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 7 | 271.14 | 271.14 2.30632 | 2.30632 0.0993 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 4 | 246.264 | 246.264 1.42874 | 0.4865 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 7 | 264.237 | 264.237 1.31229 | 1.31229 0.5584 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00003 的结果：
 准确度：0.0983
 日期：2023-11-15_00-47-09
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：8
 损失：2.3119888971328737
 node_ip：1 72.17.0.2\ n pid：3728
 should_checkpoint：true
 time_since_restore：305.2927396297455
 time_this_iter_s：34.15301489830017
 time_total_s：305.2927396297455
时间戳：1700009229
training_iteration : 8
 Trial_id: bf064_00003

(func pid=4708) [ 5, 10000] 丢失：0.257 [跨集群重复 2 倍]
== 状态 ==
当前时间：2023-11-15 00:47:14（运行 00:05:32.10）
使用 AsyncHyperBand：num_stopped=7 
括号：Iter 8.000：-2.3119888971328737 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 8 | 305.293 | 305.293 2.31199 | 2.31199 0.0983 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 4 | 246.264 | 246.264 1.42874 | 0.4865 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 7 | 264.237 | 264.237 1.31229 | 1.31229 0.5584 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00005 的结果：
 准确度：0.5344
 日期：2023-11-15_00-47-16
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：5
 损失：1.2956238314509392
 node_ip：1 72.17.0.2\ n pid：4708 \ n should_checkpoint：true \ n time_since_restore：298.829829454422 \ n time_this_iter_s：52.56541466712952 \ n time_total_s：298.829829454422 \ n时间戳：1700009236 \ n Training_iteration： 5
 Trial_id: bf064_00005

== 状态 ==
当前时间：2023-11-15 00:47:21（运行时间为 00:05:39.26）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：-2.3119888971328737 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 8 | 305.293 | 305.293 2.31199 | 2.31199 0.0983 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 5 | 298.83 | 298.83 1.29562 | 0.5344 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 7 | 264.237 | 264.237 1.31229 | 1.31229 0.5584 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=3728) [9, 2000] 损失：2.309 [跨集群重复 2 次]
== 状态 ==
当前时间：2023-11-15 00:47:26（运行时间为 00:05:44.27） 
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：-2.3119888971328737 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 8 | 305.293 | 305.293 2.31199 | 2.31199 0.0983 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 5 | 298.83 | 298.83 1.29562 | 0.5344 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 7 | 264.237 | 264.237 1.31229 | 1.31229 0.5584 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:47:31（运行时间为 00:05:49.28）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：-2.3119888971328737 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 8 | 305.293 | 305.293 2.31199 | 2.31199 0.0983 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 5 | 298.83 | 298.83 1.29562 | 0.5344 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 7 | 264.237 | 264.237 1.31229 | 1.31229 0.5584 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00007 的结果：
 准确度：0.5579
 日期：2023-11-15_00-47-32
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：8
 损失：1.345710159623623
 node_ip：17 2.17.0.2\ n pid：5692 \ n should_checkpoint：true \ n time_since_restore：297.4871952533722 \ n time_this_iter_s：33.250659704208374 \ n time_total_s：297.4871952533722 \ n时间戳：1700009252 \ n Training_it次数: 8
 Trial_id: bf064_00007

(func pid=3728) [ 9, 4000] 丢失：1.155 [跨集群重复 3 倍]
== 状态 ==
当前时间：2023-11-15 00:47:37（运行 00:05:54.70）
使用 AsyncHyperBand：num_stopped=7 
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 8 | 305.293 | 305.293 2.31199 | 2.31199 0.0983 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 5 | 298.83 | 298.83 1.29562 | 0.5344 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 8 | 297.487 | 297.487 1.34571 | 0.5579 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:47:42（运行时间为 00:05:59.71）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 8 | 305.293 | 305.293 2.31199 | 2.31199 0.0983 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 5 | 298.83 | 298.83 1.29562 | 0.5344 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 8 | 297.487 | 297.487 1.34571 | 0.5579 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [6, 6000] 损失：0.422 [跨集群重复 2 次]
train_cifar_bf064_00003 的结果：
 准确度：0.0958
 日期：2023-11-15_00-47-44
 完成：false
主机名：47e6634b378a
 iterations_since_restore：9
 损失：2.312095869445801
 node_ip：172.17.0.2
 pid：3728
 should_checkpoint：true
 time_since_restore：339.6230595111847
 time_this_iter _s: 34.33031988143921
 time_total_s: 339.6230595111847
 时间戳: 1700009264 
 Training_iteration: 9
 Trial_id: bf064_00003

== 状态 ==
当前时间: 2023-11-15 00:47:49 (运行 00:06:06.43)
使用 AsyncHyperBand: num_stopped=7\ nBracket：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 9 | 339.623 | 339.623 2.3121 | 2.3121 0.0958 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 5 | 298.83 | 298.83 1.29562 | 0.5344 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 8 | 297.487 | 297.487 1.34571 | 0.5579 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [6, 8000] 损失：0.315 [跨集群重复 2 次]
== 状态 ==
当前时间：2023-11-15 00:47:54（运行时间 00:06:11.44） 
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 9 | 339.623 | 339.623 2.3121 | 2.3121 0.0958 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 5 | 298.83 | 298.83 1.29562 | 0.5344 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 8 | 297.487 | 297.487 1.34571 | 0.5579 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:47:59（运行时间为 00:06:16.45）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 9 | 339.623 | 339.623 2.3121 | 2.3121 0.0958 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 5 | 298.83 | 298.83 1.29562 | 0.5344 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 8 | 297.487 | 297.487 1.34571 | 0.5579 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [6, 10000] 损失：0.251 [跨集群重复 3 次]
== 状态 ==
当前时间：2023-11-15 00:48:04（运行时间为 00:06:21.46） 
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 9 | 339.623 | 339.623 2.3121 | 2.3121 0.0958 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 5 | 298.83 | 298.83 1.29562 | 0.5344 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 8 | 297.487 | 297.487 1.34571 | 0.5579 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00007 的结果：
 准确度：0.5614
 日期：2023-11-15_00-48-05
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：9
 损失：1.3895927495241165
 node_ip：1 72.17.0.2\ n pid：5692 \ n should_checkpoint：true \ n time_since_restore：330.4981036186218 \ n time_this_iter_s：33.010908365249634 \ n time_total_s：330.4981036186218 \ n时间戳：1700009285 \ n Training_it次数: 9
 Trial_id: bf064_00007

(func pid=3728) [ 10, 4000] 损失: 1.155
train_cifar_bf064_00005 的结果:
 准确度: 0.5396
 日期: 2023-11-15_00-48-08
 完成: false
 主机名: 47e6634b378a
 iterations_since_restore: 6
 损失: 1.27774 6639201045 
 node_ip: 172.17.0.2
 pid: 4708
 should_checkpoint: true
 time_since_restore: 350.8566966056824
 time_this_iter_s: 52.026867151260376
 time_total_s: 350.8566966056824
时间戳：1700009288
 训练迭代：6
 试验 ID：bf064_00005
\ n== 状态 ==
当前时间：2023-11-15 00:48:13（运行时间为 00:06:31.28）
使用 AsyncHyperBand：num_stopped=7
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 6.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（3 次运行，7 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00003 |跑步 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 9 | 339.623 | 339.623 2.3121 | 2.3121 0.0958 |
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 6 | 350.857 | 350.857 1.27775 | 0.5396 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 9 | 330.498 | 1.38959 | 1.38959 0.5614 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=5692) [10, 2000] 损失：0.909
train_cifar_bf064_00003 的结果：
 准确度：0.1031
 日期：2023-11-15_00-48-18
 完成：true
 主机名：47e6634b378a
 iterations_since_restore : 10
 丢失: 2.3079761926651
 node_ip: 172.17.0.2
 pid: 3728
 should_checkpoint: true
 time_since_restore: 373.88145756721497
 time_this_iter_s: 34.25839805603027
 time_ Total_s: 373.88145756721497
 时间戳: 1700009298
 训练迭代: 10\ n Trial_id: bf064_00003

试用 train_cifar_bf064_00003 已完成。
== 状态 ==
当前时间: 2023-11-15 00:48:23 (运行时间 00:06:40.70)
使用 AsyncHyperBand: num_stopped=8
Bracket ：迭代 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 4.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（2 次运行，8 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 6 | 350.857 | 350.857 1.27775 | 0.5396 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 9 | 330.498 | 1.38959 | 1.38959 0.5614 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [7, 4000] 损失：0.619 [跨集群重复 2 次]
== 状态 ==
当前时间：2023-11-15 00:48:28（运行时​​间为 00:06:45.71） 
使用 AsyncHyperBand：num_stopped=8
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 4.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（2 次运行，8 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 6 | 350.857 | 350.857 1.27775 | 0.5396 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 9 | 330.498 | 1.38959 | 1.38959 0.5614 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:48:33（运行时间为 00:06:50.72）
使用 AsyncHyperBand：num_stopped=8
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 4.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（2 次运行，8 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+--------------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 6 | 350.857 | 350.857 1.27775 | 0.5396 |
|火车_cifar_bf064_00007 |跑步 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 9 | 330.498 | 1.38959 | 1.38959 0.5614 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [7, 6000] 损失：0.403 [跨集群重复 2 次]
train_cifar_bf064_00007 的结果：
 准确度：0.5389
 日期：2023-11-15_00-48-37
 完成：true
主机名：47e6634b378a
 iterations_since_restore：10
 损失：1.4658892618656159
 node_ip：172.17.0.2
 pid：5692
 should_checkpoint：true
 time_since_restore：362.4083983898163
 time_this_ iter_s: 31.910294771194458
 time_total_s: 362.4083983898163
 时间戳: 1700009317 
 Training_iteration: 10
 Trial_id: bf064_00007

Trial train_cifar_bf064_00007 已完成。
== 状态 ==
当前时间：2023-11-15 00:48:42（运行时间为 00:06:59.63）
使用 AsyncHyperBand : num_stopped=9
括号: Iter 8.000: -1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 6 | 350.857 | 350.857 1.27775 | 0.5396 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [7, 8000] 损失: 0.307
== 状态 ==
当前时间: 2023-11-15 00:48:47 (运行 00:07:04.64)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 6 | 350.857 | 350.857 1.27775 | 0.5396 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [7, 10000] 损失: 0.243
== 状态 ==
当前时间: 2023-11-15 00:48:52 (运行 00:07:09.65)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 6 | 350.857 | 350.857 1.27775 | 0.5396 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:48:57（运行时间为 00:07:14.66）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 6 | 350.857 | 350.857 1.27775 | 0.5396 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00005 的结果：
 准确度：0.5466
 日期：2023-11-15_00-48-58
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：7
 损失：1.2611191002845765
 node_ip：1 72.17.0.2\ n pid：4708 \ n should_checkpoint：true \ n time_since_restore：400.0100951194763 \ n time_this_iter_s：49.153398513793945 \ n time_total_s：400.0100951194763 \ n时间戳：1700009338 \ n Training_it版本: 7
 Trial_id: bf064_00005

== 状态 ==
当前时间：2023-11-15 00:49:03（运行时间为 00:07:20.43）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 7 | 400.01 | 1.26112 | 0.5466 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [8, 2000] 损失: 1.206
== 状态 ==
当前时间: 2023-11-15 00:49:08 (运行 00:07:25.44)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 7 | 400.01 | 1.26112 | 0.5466 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:49:13（运行时间为 00:07:30.45）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 7 | 400.01 | 1.26112 | 0.5466 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [8, 4000] 损失: 0.590
== 状态 ==
当前时间: 2023-11-15 00:49:18 (运行 00:07:35.46)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 7 | 400.01 | 1.26112 | 0.5466 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [8, 6000] 损失: 0.403
== 状态 ==
当前时间: 2023-11-15 00:49:23 (运行 00:07:40.47)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 7 | 400.01 | 1.26112 | 0.5466 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:49:28（运行时​​间为 00:07:45.47）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 7 | 400.01 | 1.26112 | 0.5466 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [8, 8000] 损失: 0.297
== 状态 ==
当前时间: 2023-11-15 00:49:33 (运行 00:07:50.48)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 7 | 400.01 | 1.26112 | 0.5466 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:49:38（运行时​​间为 00:07:55.49）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 7 | 400.01 | 1.26112 | 0.5466 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [8, 10000] 损失: 0.238
== 状态 ==
当前时间: 2023-11-15 00:49:43 (运行 00:08:00.50)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.8288495283782482 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 7 | 400.01 | 1.26112 | 0.5466 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00005 的结果：
 准确度：0.5557
 日期：2023-11-15_00-49-45
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：8
 损失：1.2735305621564388
 node_ip：1 72.17.0.2\ n pid：4708 \ n should_checkpoint：true \ n time_since_restore：447.28948068618774 \ n time_this_iter_s：47.279385566711426 \ n time_total_s：447.28948068618774 \ n时间戳：1700009385 \ n训练_iteration: 8
 Trial_id: bf064_00005

== 状态 ==
当前时间：2023-11-15 00:49:50（运行时间为 00:08:07.71）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 8 | 447.289 | 447.289 1.27353 | 0.5557 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [9, 2000] 损失: 1.175
== 状态 ==
当前时间: 2023-11-15 00:49:55 (运行 00:08:12.72)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 8 | 447.289 | 447.289 1.27353 | 0.5557 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:50:00（运行时间为 00:08:17.73）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 8 | 447.289 | 447.289 1.27353 | 0.5557 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [9, 4000] 损失: 0.585
== 状态 ==
当前时间: 2023-11-15 00:50:05 (运行 00:08:22.74)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 8 | 447.289 | 447.289 1.27353 | 0.5557 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [9, 6000] 损失: 0.386
== 状态 ==
当前时间: 2023-11-15 00:50:10 (运行 00:08:27.74)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 8 | 447.289 | 447.289 1.27353 | 0.5557 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:50:15（运行时间为 00:08:32.75）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 8 | 447.289 | 447.289 1.27353 | 0.5557 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [9, 8000] 损失: 0.285
== 状态 ==
当前时间: 2023-11-15 00:50:20 (运行 00:08:37.76)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 8 | 447.289 | 447.289 1.27353 | 0.5557 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:50:25（运行时间为 00:08:42.77）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 8 | 447.289 | 447.289 1.27353 | 0.5557 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [9, 10000] 损失: 0.234
== 状态 ==
当前时间: 2023-11-15 00:50:30 (运行 00:08:47.78)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 8 | 447.289 | 447.289 1.27353 | 0.5557 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00005 的结果：
 准确度：0.5823
 日期：2023-11-15_00-50-32
 完成：false
 主机名：47e6634b378a
 iterations_since_restore：9
 损失：1.176440757805109
 node_ip：17 2.17.0.2\ n pid：4708 \ n should_checkpoint：true \ n time_since_restore：494.55520391464233 \ n time_this_iter_s：47.26572322845459 \ n time_total_s：494.55520391464233 \ n时间戳：1700009432 \ n Training_迭代: 9
 Trial_id: bf064_00005

== 状态 ==
当前时间：2023-11-15 00:50:37（运行时间为 00:08:54.98）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 9 | 494.555 | 494.555 1.17644 | 0.5823 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [10, 2000] 损失: 1.121
== 状态 ==
当前时间: 2023-11-15 00:50:42 (运行 00:08:59.99)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 9 | 494.555 | 494.555 1.17644 | 0.5823 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:50:47（运行时间为 00:09:05.00）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 9 | 494.555 | 494.555 1.17644 | 0.5823 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [10, 4000] 损失: 0.564
== 状态 ==
当前时间: 2023-11-15 00:50:52 (运行 00:09:10.00)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 9 | 494.555 | 494.555 1.17644 | 0.5823 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [10, 6000] 损失: 0.376
== 状态 ==
当前时间: 2023-11-15 00:50:57 (运行 00:09:15.02)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 9 | 494.555 | 494.555 1.17644 | 0.5823 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:51:02（运行时间为 00:09:20.02）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 9 | 494.555 | 494.555 1.17644 | 0.5823 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [10, 8000] 损失: 0.283
== 状态 ==
当前时间: 2023-11-15 00:51:07 (运行 00:09:25.03)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 9 | 494.555 | 494.555 1.17644 | 0.5823 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


== 状态 ==
当前时间：2023-11-15 00:51:12（运行时间为 00:09:30.04）
使用 AsyncHyperBand：num_stopped=9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 9 | 494.555 | 494.555 1.17644 | 0.5823 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


(func pid=4708) [10, 10000] 损失: 0.228
== 状态 ==
当前时间: 2023-11-15 00:51:17 (运行 00:09:35.05)
使用 AsyncHyperBand: num_stopped= 9
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 2.0/16 个 CPU, 0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（1 次运行，9 次终止）
+------------------------+------------+-----------------+--------------+------+--------+-------------+--------+-----------------+----------+------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00005 |跑步 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 9 | 494.555 | 494.555 1.17644 | 0.5823 |
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


train_cifar_bf064_00005 的结果：
 准确度：0.5684
 日期：2023-11-15_00-51-19
 完成：true
 主机名：47e6634b378a
 iterations_since_restore：10
 损失：1.2000473774269222
 node_ip： 172.17.0.2\ n pid：4708 \ n should_checkpoint：true \ n time_since_restore：541.2027287483215 \ n time_this_iter_s：46.6475248336792 \ n time_total_s：541.2027287483215 \ n时间戳：1700009479 \ n Training_iteration： 10
 Trial_id: bf064_00005

试用 train_cifar_bf064_00005 已完成。
==状态==
当前时间：2023-11-15 00:51:19（运行时间为00:09:36.63）
使用AsyncHyperBand：num_stopped=10
括号：Iter 8.000：-1.345710159623623 |迭代 4.000：-1.4287443323016167 |迭代 2.000：-1.4916184381961823 | Iter 1.000: -2.290968679594994
逻辑资源使用: 0/16 个 CPU、0/1 个 GPU (0.0/1.0 Accelerator_type:M60)
结果日志目录: /var/lib/jenkins/ray_results/train_cifar_2023-11-15_00-41-42\ n试验次数：10/10（10 次终止）
+------------------------+--------------------------------+--------------+------+------+- ----------+--------+--------------------+--------- +------------+
|试用名称 |状态 |洛克|批量大小 | l1 | l2 | LR |迭代器 |总时间（秒）|损失|准确度 |
|------------------------+------------+-----------------+--------------+------+-----+-------------+--------+--------------------+---------+--------------|
|火车_cifar_bf064_00000 |终止 | 172.17.0.2:2681 | 2 | 16 | 16 1 | 0.00213327 | 1 | 123.009 | 123.009 2.30391 | 2.30391 0.102 |
|火车_cifar_bf064_00001 |终止 | 172.17.0.2:2752 | 4 | 1 | 2 | 0.013416 | 1 | 67.5308 | 2.30819 | 0.1008 |
|火车_cifar_bf064_00002 |终止 | 172.17.0.2:3240 | 2 | 256 | 256 64 | 64 0.0113784 | 1 | 165.896 | 165.896 2.31673 | 0.1028 |
|火车_cifar_bf064_00003 |终止 | 172.17.0.2:3728 | 8 | 64 | 64 256 | 256 0.0274071 | 10 | 10 373.881 | 2.30798 | 0.1031 |
|火车_cifar_bf064_00004 |终止 | 172.17.0.2:4217 | 4 | 16 | 16 2 | 0.056666 | 0.056666 1 | 73.2094 | 2.33591 | 2.33591 0.0986 |
|火车_cifar_bf064_00005 |终止 | 172.17.0.2:4708 | 4 | 8 | 64 | 64 0.000353097 | 0.000353097 10 | 10 541.203 | 541.203 1.20005 | 0.5684 |
|火车_cifar_bf064_00006 |终止 | 172.17.0.2:5199 | 8 | 16 | 16 4 | 0.000147684 | 1 | 44.7568 | 2.27802 | 2.27802 0.1689 |
|火车_cifar_bf064_00007 |终止 | 172.17.0.2:5692 | 8 | 256 | 256 256 | 256 0.00477469 | 10 | 10 362.408 | 1.46589 | 1.46589 0.5389 |
|火车_cifar_bf064_00008 |终止 | 172.17.0.2:2752 | 8 | 128 | 128 256 | 256 0.0306227 | 0.0306227 1 | 47.3137 | 47.3137 2.18068 | 2.18068 0.2189 |
|火车_cifar_bf064_00009 |终止 | 172.17.0.2:5199 | 2 | 2 | 16 | 16 0.0286986 | 1 | 116.122 | 116.122 2.32186 | 2.32186 0.0994 |
+------------------------+------------+-------------------+--------------+------+-----+----------------+--------+--------------------+--------------------+-------------+


2023-11-15 00:51:19,257 INFO tune.py:945 -- Total run time: 576.72 seconds (576.62 seconds for the tuning loop).
Best trial config: {'l1': 8, 'l2': 64, 'lr': 0.0003530972286268149, 'batch_size': 4}
Best trial final validation loss: 1.2000473774269222
Best trial final validation accuracy: 0.5684
Files already downloaded and verified
Files already downloaded and verified
Best trial test set accuracy: 0.5715

```




 如果运行代码，示例输出可能如下所示：






```
Number of trials: 10/10 (10 TERMINATED)
+-----+--------------+------+------+-------------+--------+---------+------------+
| ... |   batch_size |   l1 |   l2 |          lr |   iter |    loss |   accuracy |
|-----+--------------+------+------+-------------+--------+---------+------------|
| ... |            2 |    1 |  256 | 0.000668163 |      1 | 2.31479 |     0.0977 |
| ... |            4 |   64 |    8 | 0.0331514   |      1 | 2.31605 |     0.0983 |
| ... |            4 |    2 |    1 | 0.000150295 |      1 | 2.30755 |     0.1023 |
| ... |           16 |   32 |   32 | 0.0128248   |     10 | 1.66912 |     0.4391 |
| ... |            4 |    8 |  128 | 0.00464561  |      2 | 1.7316  |     0.3463 |
| ... |            8 |  256 |    8 | 0.00031556  |      1 | 2.19409 |     0.1736 |
| ... |            4 |   16 |  256 | 0.00574329  |      2 | 1.85679 |     0.3368 |
| ... |            8 |    2 |    2 | 0.00325652  |      1 | 2.30272 |     0.0984 |
| ... |            2 |    2 |    2 | 0.000342987 |      2 | 1.76044 |     0.292  |
| ... |            4 |   64 |   32 | 0.003734    |      8 | 1.53101 |     0.4761 |
+-----+--------------+------+------+-------------+--------+---------+------------+

Best trial config: {'l1': 64, 'l2': 32, 'lr': 0.0037339984519545164, 'batch_size': 4}
Best trial final validation loss: 1.5310075663924216
Best trial final validation accuracy: 0.4761
Best trial test set accuracy: 0.4737

```




 为了避免浪费资源，大多数试验已提前停止。
表现最好的试验的验证准确度达到了约 47%，
这可以在测试集上得到证实。




 这样’ 就可以了！您现在可以调整 PyTorch 模型的参数。




**脚本的总运行时间:** 
 ( 10 分钟 0.422 秒)
