# 强化学习（DQN）教程

**作者** ：[亚当Paszke ](https://github.com/apaszke)

本教程介绍了如何使用PyTorch从[
OpenAI健身房](https://gym.openai.com/)培养出深层Q学习（DQN）代理的CartPole-V0任务。

**任务**

代理有两个动作之间做出选择 - 向左或向右移动的车 -
这样附加了极点保持直立。你可以找到在[健身房网站](https://gym.openai.com/envs/CartPole-v0)各种算法和可视化的官方排行榜。

![cartpole](../_images/cartpole1.gif)

cartpole

为展开剂观察环境的当前状态并选择一个动作，环境 _跃迁_
到一个新的状态，并且还返回一个奖励，指示行动的后果。在此任务中，奖励是+1，每增量时间步长，如果极倒下太远或车从中心移动多于2.4单位处环境终止。这意味着性能更好的方案将用于持续时间较长的运行，积累了较大的回报。

所述CartPole任务被设计成使得输入到该代理是代表环境状态（位置，速度等）4的实数值。然而，神经网络可以完全通过看现场解决的任务，所以我们将使用集中在车作为输入屏幕的补丁。正因为如此，我们的结果没有直接可比性从官方排行榜中的
- 我们的任务是要困难得多。不幸的是这并减慢培训，因为我们必须使所有的帧。

严格地说，我们将目前的状态为当前屏幕补丁和前之间的差异。这将允许代理采取极的速度考虑从一个图像。

**封装**

首先，让我们导入所需的软件包。首先，我们需要[健身房](https://gym.openai.com/docs)对环境（安装使用
PIP安装健身房）。我们也将使用从PyTorch如下：

  * 神经网络（`torch.nn`）
  * 优化（`torch.optim`）
  * 自动微分（`torch.autograd`）
  * 对于视觉任务工具（`torchvision`\- [一个单独的包](https://github.com/pytorch/vision)）。

    
    
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
    

## 重放存储器

我们将使用经验重播记忆训练我们DQN。它存储的转换，代理观察，让我们在以后重新使用此数据。通过从随机抽样，即建立一个批次的转换是去相关。它已被证明，这极大地稳定和提高了DQN训练过程。

对于这一点，我们将需要两个类：

  * `过渡 `\- 表示在我们的环境中的单个过渡命名元组。它本质上（状态，行动）对映射到它们（next_state，奖励）结果，在状态如后述的屏幕差分图像。
  * `ReplayMemory`\- 保持最近观察到的转变界大小的循环缓冲区。它也实现了一个`。样品（） `方法用于选择随机批次转换的进行训练。

    
    
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
    

现在，让我们来定义模型。首先，让我们快速回顾一下一个DQN是什么。

## DQN算法

我们的环境是确定性的，所以这里提出的所有公式确定性也制定了简单起见。在强化学习文学，他们也将在包含在环境中随机转换的预期。

我们的目标是培训试图最大化打折，累积奖励策略 \（R_ {T_0} = \ sum_ {T = T_0} ^ {\ infty} \伽马^ {吨 -
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

我们的模型将是一个卷积神经网络需要在当前和以前的屏幕补丁之间的差异。它具有两个输出端，表示 \（Q（S，\ mathrm {左}）\）和 \（Q（S，\
mathrm {右}）\）（其中 \（S \）是输入到网络）。实际上，网络试图预测 _预期收益_ 的同时考虑到当前输入的每个动作。

    
    
    class DQN(nn.Module):
    
        def __init__(self, h, w, outputs):
            super(DQN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
            self.bn3 = nn.BatchNorm2d(32)
    
            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size, kernel_size = 5, stride = 2):
                return (size - (kernel_size - 1) - 1) // stride  + 1
            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
            linear_input_size = convw * convh * 32
            self.head = nn.Linear(linear_input_size, outputs)
    
        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return self.head(x.view(x.size(0), -1))
    

### 输入的提取

下面的代码是用于提取和处理从环境中渲染的图像的工具。它使用`torchvision
`包，这使得它易于撰写图像变换。一旦运行了电池它会显示它提取的例子补丁。

    
    
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])
    
    
    def get_cart_location(screen_width):
        world_width = env.x_threshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
    
    def get_screen():
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
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
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(device)
    
    
    env.reset()
    plt.figure()
    plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    

## 培训

### 超参数和效用

这种细胞实例我们的模型及其优化，并定义了一些实用程序：

  * `select_action`\- 将相应地选择一个动作的ε贪婪政策。简单地说，我们有时会用我们的模型选择的动作，有时我们只品尝一个均匀。选择一个随机行动的概率将开始在`EPS_START`和将成倍朝`EPS_END`衰减。 `EPS_DECAY`控制衰减速率。
  * `plot_durations`\- 密谋发作的持续时间，平均在过去的100个集（官方评价中使用的指标）沿帮手。该地块将是一个包含主要的训练循环单元的下方，并且每一集后，将更新。

    
    
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    
    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
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

![../_images/reinforcement_learning_diagram.jpg](../_images/reinforcement_learning_diagram.jpg)

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

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

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

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

