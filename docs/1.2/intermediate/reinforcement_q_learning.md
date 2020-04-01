# 强化学习(DQN）教程

> **作者**：[Adam Paszke](https://github.com/apaszke)
> 
> **译者**：[wutong Zhang](https://github.com/wutongzhang)
> 
> 校验：[wutong Zhang](https://github.com/wutongzhang)


本教程介绍了如何使用PyTorch训练一个Deep Q-learning(DQN）智能点(Agent）来完成[OpenAI Gym](https://gym.openai.com/)中的CartPole-V0任务。

**任务**

智能点需要决定两种动作：向左或向右来使其上的杆保持直立。
你可以在[OpenAI Gym](https://gym.openai.com/envs/CartPole-v0)找到一个有各种算法和可视化的官方排行榜。

![cartpole](https://pytorch.org/tutorials/_images/cartpole1.gif)

当智能点观察环境的当前状态并选择动作时，环境将转换为新状态，并返回指示动作结果的奖励。在这项任务中，每增加一个时间步，奖励+1，如果杆子掉得太远或大车移动距离中心超过2.4个单位，环境就会终止。这意味着更好的执行场景将持续更长的时间，积累更大的回报。

Cartpole任务的设计为智能点输入代表环境状态(位置、速度等）的4个实际值。然而，神经网络完全可以通过观察场景来解决这个任务，所以我们将使用以车为中心的一块屏幕作为输入。因此，我们的结果无法直接与官方排行榜上的结果相比——我们的任务更艰巨。不幸的是，这会减慢训练速度，因为我们必须渲染所有帧。

严格地说，我们将以当前帧和前一个帧之间的差异来呈现状态。这将允许代理从一张图像中考虑杆子的速度。

**包**

首先你需要导入必须的包。我们需要 [gym](https://gym.openai.com/docs) 作为环境 (使用 pip install gym 安装). 我们也需要 PyTorch 的如下功能:

  * 神经网络(`torch.nn`）
  * 优化(`torch.optim`）
  * 自动微分(`torch.autograd`）
  * 对于视觉任务工具(`torchvision`\- [一个单独的包](https://github.com/pytorch/vision)）
  
```python3
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 回放内存

我们将使用经验回放内存来训练DQN。它存储智能点观察到的转换，允许我们稍后重用此数据。通过从中随机抽样，组成批对象的转换将被取消相关性。结果表明，这大大稳定和改进了DQN训练过程。

因此，我们需要两个类别：

  * `Transition `\- 一个命名的元组，表示我们环境中的单个转换。它基本上将(状态、动作）对映射到它们的(下一个状态、奖励）结果，状态是屏幕差分图像，如后面所述。
  * `ReplayMemory`\-  一个有界大小的循环缓冲区，用于保存最近观察到的转换。它还实现了一个`.sample()`方法，用于选择一批随机转换进行训练。

    
```python3
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)  
```

现在我们来定义自己的模型。但首先来快速了解一下DQN。

## DQN算法

我们的环境是确定的，所以这里提出的所有方程也都是确定性的，为了简单起见。在强化学习文献中，它们还包含对环境中随机转换的期望。

我们的目标是制定一项策略，试图最大化折扣、累积奖励$$ R_{t_0}=\sum_{T=T_0}^{\infty}\gamma^{t-t_0}r_t $$ ，其中$$ R_{t_0} $$也被认为是返回值。 折扣， $$ \gamma $$应该是介于$$ 0 $$ 和$$ 1 $$ 之间的常量，以确保和收敛。它使来自不确定的遥远未来的回报对我们的智能点来说比它在不久的将来相当有信心的回报更不重要。

Q-Learning背后的主要思想是，如果我们有一个函数$$ Q ^* :State \times Action \rightarrow \mathbb{R} $$，则如果我们在特定的状态下采取行动，那么我们可以很容易地构建一个最大化回报的策略：

$$ \pi^*(s) = \arg\max_a \ Q^*(s, a) $$

然而，我们并不了解世界的一切，因此我们无法访问$$ Q^* $$。但是，由于神经网络是通用的函数逼近器，我们可以简单地创建一个并训练它类似于$$ Q^* $$。

对对于我们的训练更新规则，我们将假设某些策略的每个$$ Q $$ 函数都遵循Bellman方程：

$$ Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s')) $$ 

等式两边的差异被称为时间差误差，即$$ \delta $$：

$$ \delta = Q(s, a) - (r + \gamma \max_a Q(s', a)) $$

为了尽量减少这个错误，我们将使用[Huber loss](https://en.wikipedia.org/wiki/Huber_loss)。Huber损失在误差很小的情况下表现为均方误差，但在误差较大的情况下表现为平均绝对误差 —— 这使得当对$$ Q $$的估计噪音很大时，对异常值的鲁棒性更强。我们通过从重放内存中取样的一批转换来计算$$ B $$

$$ \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta) $$

$$ \text{where} \quad \mathcal{L}(\delta) = \begin{cases}\frac{1}{2}{\delta^2} & \text{for } |\delta| \le 1,\\\ |\delta| - \frac{1}{2} & \text{otherwise.} \end{cases} $$

### Q-网络

我们的模型将是一个卷积神经网络需要在当前和以前的屏幕补丁之间的差异。它具有两个输出端，表示$$ Q(S,\mathrm{left}) $$和 $$ Q(S,\mathrm{right}) $$(其中$$ S $$是输入到网络)。实际上，网络正试图预测在给定电流输入的情况下采取每项行动的预期回报。

```python3   
class DQN(nn.Module):

def __init__(self, h, w, outputs):
    super(DQN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.bn3 = nn.BatchNorm2d(32)

    # 线性输入连接的数量取决于conv2d层的输出，因此需要计算输入图像的大小。
    def conv2d_size_out(size, kernel_size = 5, stride = 2):
        return (size - (kernel_size - 1) - 1) // stride  + 1
    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
    linear_input_size = convw * convh * 32
    self.head = nn.Linear(linear_input_size, outputs)

# 使用一个元素调用以确定下一个操作，或在优化期间调用批处理。返回张量
def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    return self.head(x.view(x.size(0), -1))
```

### 获取输入
下面的代码是用于从环境中提取和处理渲染图像的实用程序。它使用了`torchvision`包，这样就可以很容易地组合图像转换。运行单元后，它将显示它提取的示例帧。

```python3
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # 返回 gym 需要的400x600x3 图片, 但有时会更大，如800x1200x3. 将其转换为torch (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # 车子在下半部分，因此请剥去屏幕的顶部和底部。
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # 去掉边缘，这样我们就可以得到一个以车为中心的正方形图像。
    screen = screen[:, :, slice_range]
    # 转化为 float, 重新裁剪, 转化为 torch 张量(这并不需要拷贝)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 重新裁剪,加入批维度 (BCHW)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()
```    

## 训练

### 超参数和配置

此单元实例化模型及其优化器，并定义一些实用程序：

  * `select_action`\- 将根据迭代次数贪婪策略选择一个行动。简单地说，我们有时会使用我们的模型来选择动作，有时我们只会对其中一个进行统一的采样。选择随机动作的概率将从 `EPS_START` 开始并以指数形式向 `EPS_END`衰减。 `EPS_DECAY` 控制衰减速率。
  * `plot_durations`\- 一个帮助绘制迭代次数持续时间，以及过去100迭代次数的平均值(官方评估中使用的度量）。迭代次数将在包含主训练循环的单元下方，并在每迭代之后更新。

```python3    
    
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# 获取屏幕大小，以便我们可以根据从ai-gym返回的形状正确初始化层。
# 这一点上的典型尺寸接近3x40x90，这是在get_screen(）中抑制和缩小的渲染缓冲区的结果。
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    
```

### 训练循环

最后，训练我们的模型的代码。

在这里，你可以找到执行最优化的一个步骤的`optimize_model`功能。它执行优化的一个步骤。它首先对一批数据进行采样，将所有张量连接成一个张量，计算出$$ Q(s_t,a_t) $$ 和 $$ V(s_ {t + 1})= \ max_a Q(s_ {t + 1},a) $$，并将它们组合成我们的损失。根据定义，如果 $$ s $$ 是结束状态，我们设置  $$ V(s) = 0 $$。我们还使用目标网络来计算 $$ V(s_{t+1}) \` $$ 以增加稳定性。目标网络的权重大部分时间保持不变，但每隔一段时间就会更新一次策略网络的权重。这通常是一组步骤，但为了简单起见，我们将使用迭代次数。
    
```python3
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # 转置批样本(有关详细说明，请参阅https://stackoverflow.com/a/19343/3343043）。这会将转换的批处理数组转换为批处理数组的转换。
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接批处理元素(最终状态将是模拟结束后的状态）
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t, a)-模型计算 Q(s_t)，然后选择所采取行动的列。这些是根据策略网络对每个批处理状态所采取的操作。
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算下一个状态的V(s_{t+1})。非最终状态下一个状态的预期操作值是基于“旧”目标网络计算的；选择max(1)[0]的最佳奖励。这是基于掩码合并的，这样当状态为最终状态时，我们将获得预期状态值或0。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 计算期望 Q 值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算 Huber 损失
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
```

接下来，你可以找到主训练循环。开始时，我们重置环境并初始化`state`张量。然后，我们对一个操作进行采样，执行它，观察下一个屏幕和奖励(总是1），并对我们的模型进行一次优化。当 episode 结束(我们的模型失败）时，我们重新启动循环。

`num_episodes`设置得很小。你可以下载并运行更多的`epsiodes`，比如300+来进行有意义的持续时间改进。

```python3
num_episodes = 50
for i_episode in range(num_episodes):
    # 初始化环境和状态
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # 选择并执行动作
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 观察新状态
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 在内存中储存当前参数
        memory.push(state, action, next_state, reward)

        # 进入下一状态
        state = next_state

        # 记性一步优化 (在目标网络)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    #更新目标网络, 复制在 DQN 中的所有权重偏差
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
```
    

下面是一个图表，它说明了整个结果数据流。

![img/reinforcement_learning_diagram.jpg](https://pytorch.org/tutorials/_images/reinforcement_learning_diagram.jpg)

动作可以是随机选择的，也可以是基于一个策略，从gym环境中获取下一步的样本。我们将结果记录在回放内存中，并在每次迭代中运行优化步骤。优化从重放内存中随机抽取一批来训练新策略。
“旧的”target_net也用于优化计算预期的Q值；它偶尔会更新以保持其最新。


**脚本的总运行时间：** (0分钟0.000秒）

