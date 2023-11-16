


# 分布式 RPC 框架入门 [¶](#getting-started-with-distributed-rpc-framework "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/rpc_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>




**作者** 
 :
 [沉力](https://mrshenli.github.io/)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/intermediate_source/rpc_tutorial.rst) 
.





 先决条件:



* [PyTorch 分布式概述](../beginner/dist_overview.html)
* [RPC API 文档](https://pytorch.org/docs/master/rpc.html)



 本教程使用两个简单的示例来演示如何使用第一个包
 [torch.distributed.rpc](https://pytorch.org/docs/stable/rpc.html) 构建分布式训练
作为 PyTorch v1.4 中的实验性功能引入。
这两个示例的源代码可以在
 [PyTorch 示例](https://github.com/pytorch/examples) 中找到
 。




 以前的教程，
 [分布式数据并行入门](ddp_tutorial.html) 
 和
 [使用 PyTorch 编写分布式应用程序](dist_tuto.html) 
 、
描述
 [DistributedDataParallel](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html) 
 它支持特定的训练范例，其中模型在多个进程之间复制，并且每个进程处理输入数据的分割。 
有时，您可能会遇到需要不同训练
范式的场景。例如：



1. 在强化学习中，从环境中获取训练数据可能相对昂贵，而模型本身可能非常小。在这种情况下，生成并行运行的多个观察者并共享单个代理可能会很有用。在这种情况下，代理负责本地训练，但应用程序仍需要库在观察者和训练者之间发送和接收数据。
2.您的模型可能太大，无法适应单台计算机上的 GPU，因此
需要一个库来帮助将模型拆分到多台计算机上。或者您可能正在实现一个[参数服务器](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)训练框架，其中模型参数和训练器位于不同的机器上。



 [torch.distributed.rpc](https://pytorch.org/docs/stable/rpc.html) 
 包可以帮助解决上述场景。在情况 1 中，
 [RPC](https://pytorch.org/docs/stable/rpc.html#rpc) 
 和
 [RRef](https://pytorch.org/docs/stable/rpc.html#rref)
 允许将数据
从一个工作人员发送到另一个工作人员，同时轻松引用远程数据对象。在情况 2 中，
 [分布式 autograd](https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework) 
 和 
 [分布式优化器](https://pytorch. org/docs/stable/rpc.html#module-torch.distributed.optim) 
 使执行向后传递和优化器步骤就像本地训练一样。在接下来的两节中，我们将使用强化学习示例和语言模型来演示 [torch.distributed.rpc](https://pytorch.org/docs/stable/rpc.html) 的 API例子。请注意，本教程的目的并不是建立最准确或最有效的模型来解决给定的问题，相反，这里的主要目标是展示如何使用 [torch.distributed.rpc](https://pytorch.org/docs/stable/rpc.html) 
 用于构建分布式训练应用程序的包。





## 使用 RPC 和 RRef 的分布式强化学习 [¶](#distributed-reinforcement-learning-using-rpc-and-rref "永久链接到此标题")




 本节介绍使用 RPC 构建玩具分布式强化学习模型的步骤，以解决来自 [OpenAI Gym](https://gym.openai.com) 的 CartPole-v1 
 。
策略代码大部分是借用的来自现有的单线程
 [示例](https://github.com/pytorch/examples/blob/master/reinforcement_learning)
 如下所示。我们将跳过
“策略”
 设计的细节，并重点关注 RPC
 的用法。






```
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

```




 我们已准备好介绍观察者。在此示例中，每个观察者创建
自己的环境，并等待agent’s 命令运行情节。在每一集中，一个观察者最多循环 n `n_steps`
 次迭代，并且在每次迭代中，它使用 RPC 将其环境状态传递给代理并获取返回的操作。然后，它将该操作应用于其环境，并从环境中获取奖励
和下一个状态。之后，观察者使用另一个
RPC 向代理报告奖励。再次请注意，这显然不是最有效的观察者实现。例如，一个简单的优化可以将当前状态和最后的奖励打包在一个 RPC 中，以减少通信开销。然而，我们的目标是演示 RPC API\而不是为 CartPole 构建最佳求解器。因此，让’s 保持逻辑简单，并在本示例中明确两个步骤。






```
import argparse
import gym
import torch.distributed.rpc as rpc

parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--world_size', default=2, type=int, metavar='W',
                    help='number of workers')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='how much to value future rewards')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed for reproducibility')
args = parser.parse_args()

class Observer:

    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make('CartPole-v1')
        self.env.seed(args.seed)

    def run_episode(self, agent_rref):
        state, ep_reward = self.env.reset(), 0
        for _ in range(10000):
            # send the state to the agent to get an action
            action = agent_rref.rpc_sync().select_action(self.id, state)

            # apply the action to the environment, and get the reward
            state, reward, done, _ = self.env.step(action)

            # report the reward to the agent for training purpose
            agent_rref.rpc_sync().report_reward(self.id, reward)

            # finishes after the number of self.env._max_episode_steps
            if done:
                break

```




 代理的代码稍微复杂一些，我们将其分成多个
部分。在此示例中，代理既充当训练器又充当主控器，
因此它向多个分布式观察者发送命令来运行剧集，
并且它还在本地记录所有动作和奖励，这些动作和奖励将在每个训练阶段之后的训练阶段使用。插曲。下面的代码显示
 `Agent`
 构造函数，其中大多数行正在初始化各种组件。最后的循环在其他工作线程上远程初始化观察者，并在本地保存这些观察者的“RRefs”。代理稍后将使用这些观察者
 `RRefs`
 来发送命令。应用程序不需要担心
 `RRefs`
 的生命周期。
每个
 `RRef`
 的所有者维护一个引用计数映射来跟踪其
生命周期，并保证远程只要存在该数据对象的任何实时用户，数据对象就不会被删除。
 `RRef`
 。请参阅
 `RRef`
[设计文档](https://pytorch.org/docs/master/notes/rref.html)
 了解详细信息。






```
import gym
import numpy as np

import torch
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from torch.distributions import Categorical

class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

```


接下来，代理向观察者公开两个 API，用于选择操作和报告奖励。这些函数仅在代理上本地运行，但
将由观察者通过 RPC 触发。






```
class Agent:
    ...
    def select_action(self, ob_id, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        self.rewards[ob_id].append(reward)

```




 让’s 在代理上添加
 `run_episode`
 函数，告诉所有观察者
执行一个情节。在此函数中，它首先创建一个列表来从异步 RPC 中收集
futures，然后循环所有观察者
 `RRefs`
 以
创建异步 RPC。在这些 RPC 中，代理还将自己的
 `RRef`
 传递给观察者，以便观察者也可以
调用代理上的函数。如上所示，每个观察者都会向代理发出 RPC，这是
嵌套的 RPC。每集结束后，
“保存的_log_probs”和
“奖励”将
包含记录的操作概率和奖励。






```
class Agent:
    ...
    def run_episode(self):
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    ob_rref.rpc_sync().run_episode,
                    args=(self.agent_rref,)
                )
            )

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

```




 最后，在一集之后，代理需要训练模型，该模型在下面的
 `finish_episode`
 函数中实现。该函数中没有 RPC，主要是从单线程借用的 [示例](https://github.com/pytorch/examples/blob/master/reinforcement_learning) 。
因此，我们跳过描述其内容。






```
class Agent:
    ...
    def finish_episode(self):
      # joins probs and rewards from different observers into lists
      R, probs, rewards = 0, [], []
      for ob_id in self.rewards:
          probs.extend(self.saved_log_probs[ob_id])
          rewards.extend(self.rewards[ob_id])

      # use the minimum observer reward to calculate the running reward
      min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
      self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

      # clear saved probs and rewards
      for ob_id in self.rewards:
          self.rewards[ob_id] = []
          self.saved_log_probs[ob_id] = []

      policy_loss, returns = [], []
      for r in rewards[::-1]:
          R = r + args.gamma * R
          returns.insert(0, R)
      returns = torch.tensor(returns)
      returns = (returns - returns.mean()) / (returns.std() + self.eps)
      for log_prob, R in zip(probs, returns):
          policy_loss.append(-log_prob * R)
      self.optimizer.zero_grad()
      policy_loss = torch.cat(policy_loss).sum()
      policy_loss.backward()
      self.optimizer.step()
      return min_reward

```




 通过
 `Policy`
 、
 `Observer`
 和
 `Agent`
 类，我们准备启动
多个进程来执行分布式训练。在此示例中，所有进程都运行相同的“run_worker”函数，并且它们使用级别来区分其角色。等级 0 始终是代理，所有其他等级都是观察者。代理通过重复调用
 `run_episode`
 和
 `finish_episode`
 来充当主机，直到运行奖励超过
环境指定的奖励阈值
。所有观察者都被动地等待来自代理的命令。代码由
 [rpc.init_rpc](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.init_rpc) 
 和
 [rpc.shutdown] 包装(https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.shutdown) 
 ，
分别初始化和终止 RPC 实例。更多详细信息请参见
 [API 页面](https://pytorch.org/docs/stable/rpc.html)
 。






```
import os
from itertools import count

import torch.multiprocessing as mp

AGENT_NAME = "agent"
OBSERVER_NAME="obs{}"

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        agent = Agent(world_size)
        print(f"This will run until reward threshold of {agent.reward_threshold}"
                " is reached. Ctrl+C to exit.")
        for i_episode in count(1):
            agent.run_episode()
            last_reward = agent.finish_episode()

            if i_episode % args.log_interval == 0:
                print(f"Episode {i_episode}\tLast reward: {last_reward:.2f}\tAverage reward: "
                    f"{agent.running_reward:.2f}")
            if agent.running_reward > agent.reward_threshold:
                print(f"Solved! Running reward is now {agent.running_reward}!")
                break
    else:
        # other ranks are the observer
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # observers passively waiting for instructions from the agent

    # block until all rpcs finish, and shutdown the RPC instance
    rpc.shutdown()


mp.spawn(
    run_worker,
    args=(args.world_size, ),
    nprocs=args.world_size,
    join=True
)

```




 以下是使用
 
 world_size=2
 
 进行训练时的一些示例输出。






```
This will run until reward threshold of 475.0 is reached. Ctrl+C to exit.
Episode 10      Last reward: 26.00      Average reward: 10.01
Episode 20      Last reward: 16.00      Average reward: 11.27
Episode 30      Last reward: 49.00      Average reward: 18.62
Episode 40      Last reward: 45.00      Average reward: 26.09
Episode 50      Last reward: 44.00      Average reward: 30.03
Episode 60      Last reward: 111.00     Average reward: 42.23
Episode 70      Last reward: 131.00     Average reward: 70.11
Episode 80      Last reward: 87.00      Average reward: 76.51
Episode 90      Last reward: 86.00      Average reward: 95.93
Episode 100     Last reward: 13.00      Average reward: 123.93
Episode 110     Last reward: 33.00      Average reward: 91.39
Episode 120     Last reward: 73.00      Average reward: 76.38
Episode 130     Last reward: 137.00     Average reward: 88.08
Episode 140     Last reward: 89.00      Average reward: 104.96
Episode 150     Last reward: 97.00      Average reward: 98.74
Episode 160     Last reward: 150.00     Average reward: 100.87
Episode 170     Last reward: 126.00     Average reward: 104.38
Episode 180     Last reward: 500.00     Average reward: 213.74
Episode 190     Last reward: 322.00     Average reward: 300.22
Episode 200     Last reward: 165.00     Average reward: 272.71
Episode 210     Last reward: 168.00     Average reward: 233.11
Episode 220     Last reward: 184.00     Average reward: 195.02
Episode 230     Last reward: 284.00     Average reward: 208.32
Episode 240     Last reward: 395.00     Average reward: 247.37
Episode 250     Last reward: 500.00     Average reward: 335.42
Episode 260     Last reward: 500.00     Average reward: 386.30
Episode 270     Last reward: 500.00     Average reward: 405.29
Episode 280     Last reward: 500.00     Average reward: 443.29
Episode 290     Last reward: 500.00     Average reward: 464.65
Solved! Running reward is now 475.3163778435275!

```




 在此示例中，我们展示了如何使用 RPC 作为通信工具在工作人员之间传递
数据，以及如何使用 RRef 来引用远程对象。确实，您可以直接在 `ProcessGroup`
`send`
 和 
 `recv`
 API 之上构建整个结构，或者使用其他通信/RPC 库。但是，
通过使用
 
 torch.distributed.rpc
 
 ，您可以获得本机支持和
持续优化的性能。




 接下来，我们将展示如何将 RPC 和 RRef 与分布式 autograd 和
分布式优化器结合起来执行分布式模型并行训练。





## 使用分布式 Autograd 和分布式优化器的分布式 RNN [¶](#distributed-rnn-using-distributed-autograd-and-distributed-optimizer "永久链接到此标题")




 在本节中，我们使用 RNN 模型来展示如何使用 RPC API 构建分布式模型
并行训练。示例 RNN 模型非常小，
可以轻松装入单个 GPU，但我们仍然将其层划分为两个
不同的工作线程来演示这一想法。开发人员可以应用类似的
技术在多个设备和
机器上分发更大的模型。




 RNN 模型设计借鉴了 PyTorch 中的单词语言模型
 [示例](https://github.com/pytorch/examples/tree/master/word_language_model) 
 存储库，其中包含三个主要组件，嵌入表、
 `LSTM`
 层和解码器。下面的代码将嵌入表和解码器包装到子模块中，以便它们的构造函数可以传递给 RPCAPI。在“EmbeddingTable”子模块中，我们特意将“Embedding”层放在 GPU 上以覆盖用例。在 v1.4 中，RPC 始终在目标工作线程上创建
CPU 张量参数或返回值。如果函数
采用 GPU 张量，则需要将其显式移动到正确的设备。






```
class EmbeddingTable(nn.Module):
 r"""
 Encoding layers of the RNNModel
 """
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp).cuda()
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return self.drop(self.encoder(input.cuda()).cpu()


class Decoder(nn.Module):
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output))

```




 有了上述子模块，我们现在可以使用 RPC 将它们拼凑在一起，
创建一个 RNN 模型。在下面的代码中，
 `ps`
 表示参数服务器，
它托管嵌入表和解码器的参数。构造函数
使用[remote](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.remote)
 API来创建
 `EmbeddingTable`
对象并参数服务器上的
 `Decoder`
 对象，并在本地创建
 `LSTM`
 子模块。在正向传递期间，训练器使用
 `EmbeddingTable`
`RRef`
 查找远程子模块，并使用 RPC
 将输入数据传递到
 `EmbeddingTable`
 并获取查找结果。然后，它通过本地
 `LSTM`
 层运行嵌入，最后使用另一个 RPC 将输出发送到
 `Decoder`
 子模块。一般来说，要实现分布式模型并行训练，开发者可以将模型划分为子模块，调用RPC远程创建子模块实例，并在需要时使用on`RRef`来查找它们。
如您可以在下面的代码中看到，它看起来与单机模型
并行训练非常相似。主要区别在于用
RPC 函数替换
 `Tensor.to(device)`
。






```
class RNNModel(nn.Module):
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()

        # setup embedding table remotely
        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))
        # setup LSTM locally
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        # setup decoder remotely
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))

    def forward(self, input, hidden):
        # pass input to the remote embedding table and fetch emb tensor back
        emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)
        output, hidden = self.rnn(emb, hidden)
        # pass output to the rremote decoder and get the decoded output back
        decoded = _remote_method(Decoder.forward, self.decoder_rref, output)
        return decoded, hidden

```




 在引入分布式优化器之前，让’s 添加一个辅助函数来
生成模型参数的 RRef 列表，
该列表将由分布式优化器使用。在本地训练中，应用程序可以调用
 `Module.parameters()`
 来获取对所有参数张量的引用，并将其
传递给本地优化器以进行后续更新。但是，相同的 API 不适用于分布式训练场景，因为某些参数位于远程计算机上。因此，分布式优化器不是采用参数列表
 `Tensors`
 ，而是采用
 `RRefs`
 列表，每个模型
一个
参数用于本地和远程模型参数。辅助函数非常简单，只需调用
 `Module.parameters()`
 并在每个参数上
 创建一个本地
 `RRef`。






```
def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs

```




 然后，由于
 `RNNModel`
 包含三个子模块，我们需要调用
 `_parameter_rrefs`
 三次，并将其包装到另一个辅助函数中。






```
class RNNModel(nn.Module):
    ...
    def parameter_rrefs(self):
        remote_params = []
        # get RRefs of embedding table
        remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))
        # create RRefs for local parameters
        remote_params.extend(_parameter_rrefs(self.rnn))
        # get RRefs of decoder
        remote_params.extend(_remote_method(_parameter_rrefs, self.decoder_rref))
        return remote_params

```




 现在，我们准备好实施训练循环了。初始化模型\参数后，我们创建
 `RNNModel`
 和
 `DistributedOptimizer`
 。分布式优化器将采用参数列表 `RRefs`
 ，找到所有不同的
所有者工作线程，并创建给定的本地优化器（即，
 `SGD`
 在这种情况下，
您可以使用其他本地优化器）在每个所有者工作线程上使用
给定的参数（即
 `lr=0.05`
 ）。




 在训练循环中，它首先创建一个分布式 autograd 上下文，这将帮助分布式 autograd 引擎查找梯度和涉及的 RPC
send/recv 函数。分布式autograd引擎的设计细节可以在其[设计说明](https://pytorch.org/docs/master/notes/distributed_autograd.html)中找到。
然后，它开始前进就好像它是本地模型一样传递，并运行分布式向后传递。对于分布式向后，您
只需要指定一个根列表，在本例中，它是损失
 `Tensor`
 。
分布式autograd引擎将自动遍历分布式图
并正确写入梯度。接下来，它在分布式优化器上运行“step”函数，该函数将联系所有涉及的本地优化器来更新模型参数。与本地训练相比，
一个微小的区别是您不需要运行
 `zero_grad()`
 因为每个
autograd 上下文都有专用的空间来存储梯度，并且当我们创建
每次迭代都有一个上下文，来自不同迭代的那些梯度不会
累积到同一组
 `Tensors`
 。






```
def run_trainer():
    batch = 5
    ntoken = 10
    ninp = 2

    nhid = 3
    nindices = 3
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch():
        for _ in range(5):
            data = torch.LongTensor(batch, nindices) % ntoken
            target = torch.LongTensor(batch, ntoken) % nindices
            yield data, target

    # train for 10 iterations
    for epoch in range(10):
        for data, target in get_next_batch():
            # create distributed autograd context
            with dist_autograd.context() as context_id:
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                # run distributed backward pass
                dist_autograd.backward(context_id, [loss])
                # run distributed optimizer
                opt.step(context_id)
                # not necessary to zero grads since they are
                # accumulated into the distributed autograd context
                # which is reset every iteration.
        print("Training epoch {}".format(epoch))

```




 最后，让’s 添加一些粘合代码来启动参数服务器和训练器
进程。






```
def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 1:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        _run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)

```









