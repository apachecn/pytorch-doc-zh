#强化学习

> 译者：[Noahs212](https://github.com/Noahs212)
>
> project address：<https://pytorch.apachecn.org/2.0/tutorials/intermediate//>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/Reinforcement_Learning_(DQN)_Tutorial.html>

**作者**：
    -[Adam Paszke](https://github.com/apaszke)
    -[Mark Towers] (https://github.com/pseudo-rnd-thoughts)



这个教程展示了如何使用 PyTorch在 [Gymnasium](https://gymnasium.farama.org/index.html) 的CartPole-v1 任务上训练深度 Q 学习（DQN）代理。

**任务**

代理人必须在两种操作之间进行选择：向左或向右移动小车，以使其附着的杆保持垂直。您可以在[Gymnasium的网站](https://gymnasium.farama.org/environments/classic_control/cart_pole/)上找到有关该环境和其他更具挑战性的环境的更多信息。

<!--> insert cartpole image-->

随着代理人观察环境的当前状态并选择一个动作，环境将转变为新的状态，并返回一个奖励，表明该动作的后果。在此任务中，每增加一个时间步奖励就是+1，如果杆倒下太远或小车离中心超过2.4个单位，则环境会终止。这意味着表现更好的情景将持续更长的时间，累积更大的回报。

CartPole任务的设计是，代理人的输入是4个表示环境状态的实数（位置、速度等）。我们不进行任何缩放地获取这4个输入，并将它们通过一个具有2个输出的小型全连接网络，每个动作一个输出。该网络经过训练以预测给定输入状态下每个动作的期望值。然后选择期望值最高的动作。

##库
<!--> error with package and fork vocab-->
首先，让我们导入所需的库。首先，我们需要[gymnasium](https://gymnasium.farama.org/)用于环境，通过使用pip进行安装。这是原始OpenAI Gym项目的一个分支，并且自Gym v0.19以来由同一团队维护。如果你在Google Colab中运行此代码，运行：

```py
%%bash
pip3 install gymnasium[classic_control]
```


我们还将使用来自PyTorch的以下内容：

 - 神经网络 (```torch.nn```)

 - 优化 (```torch.optim```)

 - 自动微分 (```torch.autograd```)


```py
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

#设置matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# 如果用GPU的话
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


##重播内存
我们将使用重播内存(ER)来训练我们的 DQN。它存储了代理观察到的转换，使我们可以以后重用这些数据。通过从中随机抽取转换，构成批次的转换将被去相关化。已经显示，这会大大稳定和改进 DQN 的训练过程。

对于这个，我们将需要两个类

- ```Transition```  - 一个代表我们环境中单一转换的命名元组。它本质上将（状态，动作）配对映射到它们的（下一个状态，奖励）结果，其中状态是屏幕差异图像，稍后将进行描述。

- ```ReplayMemory``` -一个有界大小的循环缓冲区，用于保存最近观察到的转换。它还实现了一个 ```.sample()``` 方法，用于选择训练的随机批次的转换。


```py
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存转换"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
```
现在，让我们定义我们的模型。但首先，让我们快速回顾一下DQN是什么。

## DQN 算法

asdfadf \( R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t-t_0}r_t \)afasdfd

\(R_{t_0}\)

\(Q^* : State \times Action \rightarrow \mathbb{R}\)

\(\pi^*(s) = \underset{a}{\mathrm{argmax}}\, Q^*(s,a)\)

\(Q^*\)

\(Q^\pi(s,a) = r + \gamma Q^\pi(s', \pi(s'))\)

\(\delta = Q(s,a) - \left( r + \gamma \cdot \max_{a'} Q(s',a') \right)\)

\(\delta\)

\(L = \frac{1}{|B|} \sum\limits_{(s,a,s',r) \in B} L(\delta)\)


\(
    L(\delta) = 
\begin{cases} 
  \frac{1}{2} \delta^2 & \text{for } |\delta| \leq 1, \\
  |\delta| - \frac{1}{2} & \text{otherwise.}
\end{cases}
\)

