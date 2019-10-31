# 强化学习（DQN）教程

**作者** ：[亚当Paszke ](https://github.com/apaszke)

**翻译** ：[wutong Zhang](https://github.com/wutongzhang)


本教程介绍了如何使用PyTorch训练一个Deep Q-learning（DQN）智能点（Agent）来完成[OpenAI Gym](https://gym.openai.com/)中的CartPole-V0任务。

**任务**

智能点需要决定两种动作：向左或向右来使其上的杆保持直立。
你可以在[OpenAI Gym](https://gym.openai.com/envs/CartPole-v0)找到一个有各种算法和可视化的官方排行榜。

![cartpole](https://pytorch.org/tutorials/_images/cartpole1.gif)

当智能点观察环境的当前状态并选择动作时，环境将转换为新状态，并返回指示动作结果的奖励。在这项任务中，每增加一个时间步，奖励+1，如果杆子掉得太远或大车移动距离中心超过2.4个单位，环境就会终止。这意味着更好的执行场景将持续更长的时间，积累更大的回报。

Cartpole任务的设计为智能点输入代表环境状态（位置、速度等）的4个实际值。然而，神经网络完全可以通过观察场景来解决这个任务，所以我们将使用以车为中心的一块屏幕作为输入。因此，我们的结果无法直接与官方排行榜上的结果相比——我们的任务更艰巨。不幸的是，这会减慢训练速度，因为我们必须渲染所有帧。

严格地说，我们将以当前帧和前一个帧之间的差异来呈现状态。这将允许代理从一张图像中考虑杆子的速度。

**包**

首先你需要导入必须的包。我们需要 [gym](https://gym.openai.com/docs) 作为环境 (使用 pip install gym 安装). 我们也需要 PyTorch 的如下功能:

  * 神经网络（`torch.nn`）
  * 优化（`torch.optim`）
  * 自动微分（`torch.autograd`）
  * 对于视觉任务工具（`torchvision`\- [一个单独的包](https://github.com/pytorch/vision)）
  
  
  ```
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

  * `Transition `\- 一个命名的元组，表示我们环境中的单个转换。它基本上将（状态、动作）对映射到它们的（下一个状态、奖励）结果，状态是屏幕差分图像，如后面所述。
  * `ReplayMemory`\-  一个有界大小的循环缓冲区，用于保存最近观察到的转换。它还实现了一个`.sample（） `方法，用于选择一批随机转换进行训练。

    
```
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

我们的目标是制定一项策略，试图最大化折扣、累积奖励 \（R_ {T_0} = \ sum_ {T = T_0} ^ {\ infty} \伽马^ {吨 -
T_0} r_t \） ，其中 \（R_ {T_0} \）也被称为 _返回_ 。折扣， \（\伽玛\），应为（0 \）和 \（1 \）确保的总和收敛之间
\常数。这使得从不确定的遥远的未来回报我们的代理比在不久的将来，那些它可以相当自信更重要。

背后Q学习的主要思想是，如果我们有一个函数 \（Q ^ *：国家\次行动\ RIGHTARROW \ mathbb {R}
\），即能告诉我们什么退货会，如果我们采取在给定的状态的动作，那么我们就可以轻松构建最大化我们的奖励政策：

\\[\pi^*(s) = \arg\\!\max_a \ Q^*(s, a)\\]

但是，我们不知道世界上的一切，所以我们没有进入 \（Q ^ * \）HTG1。但是，由于神经网络是通用的函数逼近，我们可以简单地创建一个和训练它类似于
\（Q ^ * \）HTG3。

对于我们的培训更新规则，我们将使用一个事实，即每 \（Q \）HTG1对于一些政策功能服从Bellman方程：

\\[Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))\\]

平等的两侧之间的差异被称为时间差误差， \（\增量\）：

\\[\delta = Q(s, a) - (r + \gamma \max_a Q(s', a))\\]

为了减少这种错误，我们将使用[胡贝尔损失[HTG1。胡伯损失的行为像均方误差时的误差小，但喜欢当误差较大平均绝对误差 - 这使得它更加坚固，以当 \（Q
\）的估计值是异常区十分吵闹。我们计算这在批次转变， \（B
\），从重放存储器采样：](https://en.wikipedia.org/wiki/Huber_loss)

\\[\mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B}
\mathcal{L}(\delta)\\]

\\[\begin{split}\text{where} \quad \mathcal{L}(\delta) = \begin{cases}
\frac{1}{2}{\delta^2} & \text{for } |\delta| \le 1, \\\ |\delta| - \frac{1}{2}
& \text{otherwise.} \end{cases}\end{split}\\]

### Q-网络

我们的模型将是一个卷积神经网络需要在当前和以前的屏幕补丁之间的差异。它具有两个输出端，表示\(Q(S，\ mathrm {左})\)和 \(Q(S，\mathrm {右})\)(其中 \(S \)是输入到网络)。实际上，网络试图预测 _预期收益_ 的同时考虑到当前输入的每个动作。

```    
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

```
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
  * `plot_durations`\- 一个帮助绘制迭代次数持续时间，以及过去100迭代次数的平均值（官方评估中使用的度量）。迭代次数将在包含主训练循环的单元下方，并在每迭代之后更新。

```    
    
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    
    # 获取屏幕大小，以便我们可以根据从ai-gym返回的形状正确初始化层。
    # 这一点上的典型尺寸接近3x40x90，这是在get_screen（）中抑制和缩小的渲染缓冲区的结果。
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

在这里，你可以找到执行最优化的一个步骤的`optimize_model`功能。它首先样品批次符，连接所有的张量成一个单一的一个，计算
\（Q（S_T，A_T）\）和 \（V（S_ {T + 1}）= \ max_a Q（S_ {T +
1}，一）\），并且将它们组合成我们的损失。通过defition我们将 \（V（S）= 0 \）如果 \（S \）是终端状态。我们还使用的目标网络来计算
\（V（S_ {T +
1}）\），以增加稳定性。目标网络有其权冷冻保存的大部分时间，但随着政策网络的权重，每隔一段时间更新一次。这通常是步一组数字，但我们将使用事件为简单起见。

    
    
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    

下面，你可以找到主要的训练循环。在开始的时候，我们重置环境和初始化`状态
`张量。然后，我们样本的行动，执行它，观察下一屏幕和奖励（始终为1），一旦优化我们的模型。当情节结束（我们的模型没有），我们重新开始循环。

下面， num_episodes 设置为小。你应该下载笔记本电脑和运行更多的epsiodes，如300多个有意义的持续改进。

    
    
    num_episodes = 50
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
    
            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
    
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
    
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
    

下面是示出了整体得到的数据流的示图。

![img/reinforcement_learning_diagram.jpg](img/reinforcement_learning_diagram.jpg)

操作被选择随机地或者基于策略，获得从健身房环境下一步样品。我们记录回放存储器中的结果，也运行在每个迭代优化步骤。优化选取一个随机批量从重放内存做新政策的培训。
“旧版” target_net也用于优化计算的预期的Q值;它不时地被更新，以保持它的电流。

**脚本的总运行时间：** （0分钟0.000秒）

[`Download Python source code:
reinforcement_q_learning.py`](../_downloads/b8954cc7b372cac10a92b8c6183846a3/reinforcement_q_learning.py)

[`Download Jupyter notebook:
reinforcement_q_learning.ipynb`](../_downloads/2b3f06b04b5e96e4772746c20fcb4dcc/reinforcement_q_learning.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](flask_rest_api_tutorial.html "1. Deploying PyTorch in Python via
a REST API with Flask") [![](../_static/images/chevron-right-orange.svg)
Previous](../beginner/transformer_tutorial.html "Sequence-to-Sequence Modeling
with nn.Transformer and TorchText")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * 强化学习（DQN）教程
    * 重放存储器
    * DQN算法
      * Q-网络
      * 输入提取
    * 培训
      * 超参数和效用
      * 培训环

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



