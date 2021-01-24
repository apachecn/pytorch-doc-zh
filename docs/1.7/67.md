# 使用异步执行实现批量 RPC 处理

> 原文：<https://pytorch.org/tutorials/intermediate/rpc_async_execution.html>

**作者**：[Shen Li](https://mrshenli.github.io/)

先决条件：

*   [PyTorch 分布式概述](../beginner/dist_overview.html)
*   [分布式 RPC 框架](rpc_tutorial.html)入门
*   [使用分布式 RPC 框架](rpc_param_server_tutorial.html)实现参数服务器
*   [RPC 异步执行装饰器](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)

本教程演示了如何使用[`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)装饰器来构建批量 RPC 应用，该装饰器通过减少阻止的 RPC 线程数和合并被调用方上的 CUDA 操作来帮助加快训练速度。 这使用 TorchServer 的相同想法进行[批量推断](https://pytorch.org/serve/batch_inference_with_ts.html)。

注意

本教程需要 PyTorch v1.6.0 或更高版本。

## 基础知识

先前的教程显示了使用[`torch.distributed.rpc`](https://pytorch.org/docs/stable/rpc.html)构建分布式训练应用的步骤，但并未详细说明在处理 RPC 请求时被调用方发生的情况。 从 PyTorch v1.5 开始，每个 RPC 请求都会在被调用方上阻塞一个线程，以在该请求中执行该函数，直到该函数返回为止。 这适用于许多用例，但有一个警告。 如果用户函数例如通过嵌套 RPC 调用在 IO 上阻塞，或者例如在等待其他 RPC 请求解除阻塞的信号时阻塞，则被调用方上的 RPC 线程将必须空闲，直到 IO 完成或发生信令事件为止。 结果，RPC 被调用者可能使用了不必要的更多线程。 造成此问题的原因是 RPC 将用户函数视为黑盒，并且几乎不了解该函数会发生什么。 为了允许用户函数产生和释放 RPC 线程，需要向 RPC 系统提供更多提示。

从 v1.6.0 开始，PyTorch 通过引入两个新概念来解决此问题：

*   [`torch.futures.Future`](https://pytorch.org/docs/master/futures.html) 类型封装了异步执行，还支持安装回调函数。
*   一个[`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)装饰器，允许应用告诉被调用方目标函数将返回将来的函数，并且在执行期间可以暂停并产生多次。

使用这两个工具，应用代码可以将用户函数分解为多个较小的函数，将它们作为`Future`对象上的回调链接在一起，然后返回包含最终结果的`Future`。 在被调用方，当获取`Future`对象时，它还将安装后续的 RPC 响应准备和通讯作为回调，这将在最终结果准备好时触发。 这样，被调用者不再需要阻塞一个线程并等待直到最终返回值准备就绪。 有关简单示例，请参考[`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 的 API 文档。

除了减少被调用方上的空闲线程数之外，这些工具还有助于使批量 RPC 处理更容易，更快捷。 本教程的以下两节演示了如何使用[`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)装饰器来构建分布式批更新参数服务器和批量强化学习应用。

## 批量更新参数服务器

考虑具有一个参数服务器（PS）和多个训练器的同步参数服务器训练应用。 在此应用中，PS 保留参数并等待所有训练器报告坡度。 在每次迭代中，它都会等到收到所有训练者的梯度后，再一次更新所有参数。 下面的代码显示 PS 类的实现。 `update_and_fetch_model`方法是用`@rpc.functions.async_execution`装饰的，将由训练器调用。 每次调用都会返回一个`Future`对象，该对象将填充有更新的模型。 大多数训练器发起的调用仅将梯度累积到`.grad`字段，立即返回，并在 PS 上产生 RPC 线程。 最后到达的训练器将触发优化器步骤，并消耗所有先前报告的梯度。 然后，它使用更新的模型设置`future_model`，该模型又通过`Future`对象通知其他训练器的所有先前请求，并将更新后的模型发送给所有训练器。

```py
import threading
import torchvision
import torch
import torch.distributed.rpc as rpc
from torch import optim

num_classes, batch_update_size = 30, 5

class BatchUpdateParameterServer(object):
    def __init__(self, batch_update_size=batch_update_size):
        self.model = torchvision.models.resnet50(num_classes=num_classes)
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self):
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        # Using the RRef to retrieve the local PS instance
        self = ps_rref.local_value()
        with self.lock:
            self.curr_update_size += 1
            # accumulate gradients into .grad field
            for p, g in zip(self.model.parameters(), grads):
                p.grad += g

            # Save the current future_model and return it to make sure the
            # returned Future object holds the correct model even if another
            # thread modifies future_model before this thread returns.
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                # update the model
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                # by settiing the result on the Future object, all previous
                # requests expecting this updated model will be notified and
                # the their responses will be sent accordingly.
                fut.set_result(self.model)
                self.future_model = torch.futures.Future()

        return fut

```

对于训练器，它们都使用来自 PS 的相同参数集进行初始化。 在每次迭代中，每位训练器首先进行前进和后退操作，以局部生成梯度。 然后，每个训练器都使用 RPC 向 PS 报告其梯度，并通过同一 RPC 请求的返回值取回更新的参数。 在训练器的实现中，目标函数是否标记有`@rpc.functions.async_execution`都没有关系。 训练器只需使用`rpc_sync`调用`update_and_fetch_model`，这会阻塞训练器，直到返回更新的模型。

```py
batch_size, image_w, image_h  = 20, 64, 64

class Trainer(object):
    def __init__(self, ps_rref):
        self.ps_rref, self.loss_fn = ps_rref, torch.nn.MSELoss()
        self.one_hot_indices = torch.LongTensor(batch_size) \
                                    .random_(0, num_classes) \
                                    .view(batch_size, 1)

    def get_next_batch(self):
        for _ in range(6):
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                        .scatter_(1, self.one_hot_indices, 1)
            yield inputs.cuda(), labels.cuda()

    def train(self):
        name = rpc.get_worker_info().name
        # get initial model parameters
        m = self.ps_rref.rpc_sync().get_model().cuda()
        # start training
        for inputs, labels in self.get_next_batch():
            self.loss_fn(m(inputs), labels).backward()
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
            ).cuda()

```

在本教程中，我们将跳过启动多个进程的代码，有关完整实现，请参考[示例](https://github.com/pytorch/examples/tree/master/distributed/rpc)回购。 请注意，可以在没有[`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)装饰器的情况下实现批量。 但是，这将需要在 PS 上阻塞更多的 RPC 线程，或者使用另一轮 RPC 来获取更新的模型，后者将增加代码的复杂性和通信开销。

本节使用一个简单的参数服务器训练示例来说明如何使用[`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)装饰器实现批量 RPC 应用。 在下一节中，我们将使用批量重新实现上一[分布式 RPC 框架](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)入门指南中的强化学习示例，并演示其对训练速度的影响。

## 批量 CartPole 求解器

本节以 [OpenAI Gym](https://gym.openai.com/) 中的 CartPole-v1 为例，说明批量 RPC 的性能影响。 请注意，我们的目标是演示[`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)的用法，而不是构建最佳的 CartPole 求解器或解决大多数不同的 RL 问题，我们使用非常简单的策略和奖励计算策略，并将重点放在多观察者单智能体的批量 RPC 实现。 我们使用与前面的教程类似的`Policy`模型，如下所示。 与上一教程相比，不同之处在于其构造器使用了一个附加的`batch`参数来控制`F.softmax`的`dim`参数，因为进行批量时，`forward`函数中的`x`参数包含来自多个观察者的状态，因此尺寸需要适当更改。 其他所有内容保持不变。

```py
import argparse
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch RPC Batch RL example')
parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                    help='discount factor (default: 1.0)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--num-episode', type=int, default=10, metavar='E',
                    help='number of episodes (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self, batch=True):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.dim = 2 if batch else 1

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=self.dim)

```

`Observer`的构造器也会相应地进行调整。 它还带有`batch`参数，该参数控制用于选择动作的`Agent`函数。 在批量模式下，它将调用`Agent`上的`select_action_batch`函数，该函数将很快出现，并且该函数将以[`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)装饰。

```py
import gym
import torch.distributed.rpc as rpc

class Observer:
    def __init__(self, batch=True):
        self.id = rpc.get_worker_info().id - 1
        self.env = gym.make('CartPole-v1')
        self.env.seed(args.seed)
        self.select_action = Agent.select_action_batch if batch else Agent.select_action

```

与之前的教程[分布式 RPC 框架入门](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)相比，观察者的行为略有不同。 它不会在环境停止时退出，而是始终在每个剧集中运行`n_steps`迭代。 当环境返回时，观察者只需重置环境并重新开始。 通过这种设计，智能体将从每个观察者那里收到固定数量的状态，因此可以将它们打包成固定大小的张量。 在每个步骤中，`Observer`使用 RPC 将其状态发送到`Agent`，并通过返回值获取操作。 在每个剧集的结尾，它将所有步骤的奖励返还给`Agent`。 注意，`Agent`将使用 RPC 调用此`run_episode`函数。 因此，此函数中的`rpc_sync`调用将是嵌套的 RPC 调用。 我们也可以将此函数标记为`@rpc.functions.async_execution`，以避免阻塞`Observer`上的一个线程。 但是，由于瓶颈是`Agent`而不是`Observer`，因此可以在`Observer`进程中阻塞一个线程。

```py
import torch

class Observer:
    ...

    def run_episode(self, agent_rref, n_steps):
        state, ep_reward = self.env.reset(), NUM_STEPS
        rewards = torch.zeros(n_steps)
        start_step = 0
        for step in range(n_steps):
            state = torch.from_numpy(state).float().unsqueeze(0)
            # send the state to the agent to get an action
            action = rpc.rpc_sync(
                agent_rref.owner(),
                self.select_action,
                args=(agent_rref, self.id, state)
            )

            # apply the action to the environment, and get the reward
            state, reward, done, _ = self.env.step(action)
            rewards[step] = reward

            if done or step + 1 >= n_steps:
                curr_rewards = rewards[start_step:(step + 1)]
                R = 0
                for i in range(curr_rewards.numel() -1, -1, -1):
                    R = curr_rewards[i] + args.gamma * R
                    curr_rewards[i] = R
                state = self.env.reset()
                if start_step == 0:
                    ep_reward = min(ep_reward, step - start_step + 1)
                start_step = step + 1

        return [rewards, ep_reward]

```

`Agent`的构造器还采用`batch`参数，该参数控制如何对动作概率进行批量。 在批量模式下，`saved_log_probs`包含一张张量列表，其中每个张量包含一个步骤中所有观察者的动作抢夺。 如果不进行批量，则`saved_log_probs`是字典，其中的键是观察者 ID，值是该观察者的动作概率列表。

```py
import threading
from torch.distributed.rpc import RRef

class Agent:
    def __init__(self, world_size, batch=True):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.policy = Policy(batch).cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.running_reward = 0

        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(rpc.remote(ob_info, Observer, args=(batch,)))
            self.rewards[ob_info.id] = []

        self.states = torch.zeros(len(self.ob_rrefs), 1, 4)
        self.batch = batch
        self.saved_log_probs = [] if batch else {k:[] for k in range(len(self.ob_rrefs))}
        self.future_actions = torch.futures.Future()
        self.lock = threading.Lock()
        self.pending_states = len(self.ob_rrefs)

```

非批量`select_acion`只需运行状态抛出策略，保存动作概率，然后立即将动作返回给观察者。

```py
from torch.distributions import Categorical

class Agent:
    ...

    @staticmethod
    def select_action(agent_rref, ob_id, state):
        self = agent_rref.local_value()
        probs = self.policy(state.cuda())
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

```

使用批量时，状态以观察者 id 为行 ID 存储在 2D 张量`self.states`中。 然后，它通过将回调函数安装到批量生成的`self.future_actions` `Future`对象上来链接`Future`，该对象将使用使用该观察者 ID 索引的特定行进行填充。 最后到达的观察者一口气通过策略运行所有批量状态，并相应地设置`self.future_actions`。 发生这种情况时，将触发安装在`self.future_actions`上的所有回调函数，并使用它们的返回值来填充链接的`Future`对象，该对象进而通知`Agent`为所有先前的 RPC 请求准备和传达来自其他观察者的响应。

```py
class Agent:
    ...

    @staticmethod
    @rpc.functions.async_execution
    def select_action_batch(agent_rref, ob_id, state):
        self = agent_rref.local_value()
        self.states[ob_id].copy_(state)
        future_action = self.future_actions.then(
            lambda future_actions: future_actions.wait()[ob_id].item()
        )

        with self.lock:
            self.pending_states -= 1
            if self.pending_states == 0:
                self.pending_states = len(self.ob_rrefs)
                probs = self.policy(self.states.cuda())
                m = Categorical(probs)
                actions = m.sample()
                self.saved_log_probs.append(m.log_prob(actions).t()[0])
                future_actions = self.future_actions
                self.future_actions = torch.futures.Future()
                future_actions.set_result(actions.cpu())
        return future_action

```

现在，让我们定义如何将不同的 RPC 函数结合在一起。 `Agent`控制每个剧集的执行。 它首先使用`rpc_async`在所有观察者上开始该剧集，并阻止将由观察者奖励填充的返还期货。 请注意，以下代码使用 RRef 帮助器`ob_rref.rpc_async()`在具有提供的参数的`ob_rref` RRef 的所有者上启动`run_episode`函数。 然后将保存的动作概率和返回的观察者奖励转换为期望的数据格式，并开始训练步骤。 最后，它将重置所有状态并返回当前剧集的奖励。 此函数是运行一集的入口。

```py
class Agent:
    ...

    def run_episode(self, n_steps=0):
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(ob_rref.rpc_async().run_episode(self.agent_rref, n_steps))

        # wait until all obervers have finished this episode
        rets = torch.futures.wait_all(futs)
        rewards = torch.stack([ret[0] for ret in rets]).cuda().t()
        ep_rewards = sum([ret[1] for ret in rets]) / len(rets)

        # stack saved probs into one tensor
        if self.batch:
            probs = torch.stack(self.saved_log_probs)
        else:
            probs = [torch.stack(self.saved_log_probs[i]) for i in range(len(rets))]
            probs = torch.stack(probs)

        policy_loss = -probs * rewards / len(rets)
        policy_loss.sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # reset variables
        self.saved_log_probs = [] if self.batch else {k:[] for k in range(len(self.ob_rrefs))}
        self.states = torch.zeros(len(self.ob_rrefs), 1, 4)

        # calculate running rewards
        self.running_reward = 0.5 * ep_rewards + 0.5 * self.running_reward
        return ep_rewards, self.running_reward

```

其余代码是正常的进程启动和日志记录，与其他 RPC 教程类似。 在本教程中，所有观察者都被动地等待来自智能体的命令。 有关完整的实现，请参考[示例](https://github.com/pytorch/examples/tree/master/distributed/rpc)回购。

```py
def run_worker(rank, world_size, n_episode, batch, print_log=True):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        agent = Agent(world_size, batch)
        for i_episode in range(n_episode):
            last_reward, running_reward = agent.run_episode(n_steps=NUM_STEPS)

            if print_log:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, last_reward, running_reward))
    else:
        # other ranks are the observer
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # observers passively waiting for instructions from agents
    rpc.shutdown()

def main():
    for world_size in range(2, 12):
        delays = []
        for batch in [True, False]:
            tik = time.time()
            mp.spawn(
                run_worker,
                args=(world_size, args.num_episode, batch),
                nprocs=world_size,
                join=True
            )
            tok = time.time()
            delays.append(tok - tik)

        print(f"{world_size}, {delays[0]}, {delays[1]}")

if __name__ == '__main__':
    main()

```

批量 RPC 有助于将操作推断合并为较少的 CUDA 操作，从而减少了摊销的开销。 上面的`main`函数使用不同数量的观察者（从 1 到 10）在批量和无批量模式下运行相同的代码。下图使用默认参数值绘制了不同世界大小的执行时间。 结果证实了我们的期望，即批量有助于加快训练速度。

![](img/f5504c7ed93640f2bed4d2a606c015ba.png)

## 了解更多

*   [批量更新参数服务器的源代码](https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/parameter_server.py)
*   [批量 CartPole 求解器](https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/reinforce.py)
*   [分布式 Autograd](https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework)
*   [分布式管道并行性](dist_pipeline_parallel_tutorial.html)