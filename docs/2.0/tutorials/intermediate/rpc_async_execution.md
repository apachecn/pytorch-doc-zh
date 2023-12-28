


# 使用异步执行实现批量 RPC 处理 [¶](#implementing-batch-rpc-processing-using-asynchronous-executions "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/rpc_async_execution>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/rpc_async_execution.html>




**作者** 
 :
 [沉力](https://mrshenli.github.io/)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/intermediate_source/rpc_async_execution.rst) 
.





 先决条件:



* [PyTorch 分布式概述](../beginner/dist_overview.html)
* [分布式 RPC 框架入门](rpc_tutorial.html)
* [使用分布式 RPC 框架实现参数服务器](rpc_param_server_tutorial.html) 
* [RPC 异步执行装饰器](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)



 本教程演示如何使用 [@rpc.functions.async_execution](https://pytorch.org/docs/master/rpc.html#torch.distributed) 构建批处理 RPC 应用程序。 rpc.functions.async_execution) 
 装饰器，它通过减少阻塞
RPC 线程的数量并在被调用者上整合 CUDA 操作来帮助加快训练速度。这与 [使用 TorchServe 进行批量推理](https://pytorch.org/serve/batch_inference_with_ts.html) 具有相同的想法。





 注意




 本教程需要 PyTorch v1.6.0 或更高版本。





## 基础知识 [¶](#basics "此标题的永久链接")




 之前的教程已经展示了使用 [torch.distributed.rpc](https://pytorch.org/docs/stable/rpc.html) 构建分布式训练
应用程序的步骤
 ，
但他们没有\xe2 \x80\x99t 详细说明处理 RPC 请求时被调用方发生的情况。从 PyTorch v1.5 开始，每个 RPC 请求都会阻塞被调用者上的一个线程来执行该请求中的函数，直到该函数返回。这适用于许多用例，但有一个警告。如果用户函数在 IO 上阻塞，例如，使用嵌套的 RPC 调用或信号发送，例如等待不同的 RPC 请求解除阻塞，则被调用方上的 RPC 线程将不得不等待，直到 IO 完成或发送信号发送事件发生。因此，
RPC 被调用者可能会使用超出必要数量的线程。造成此问题的原因是 RPC 将用户函数视为黑匣子，并且对函数中发生的情况知之甚少。为了允许用户函数产生并释放
RPC 线程，需要向 RPC 系统提供更多提示。




 从 v1.6.0 开始，PyTorch 通过引入两个新概念来解决这个问题：



* A
 [torch.futures.Future](https://pytorch.org/docs/master/futures.html) 
 封装异步执行的类型，也支持安装回调函数。
* An 
 [@rpc.functions.async_execution](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 
 允许应用程序告诉被调用者的装饰器目标函数
将返回一个 future，并且可以在执行过程中多次暂停和让出。



 使用这两个工具，应用程序代码可以将用户函数分解为
多个较小的函数，将它们作为回调链接在一起
 `Future`
 对象，并返回包含最终结果的
 `Future`
结果。在被调用方，当获取“Future”对象时，它也会将后续的 RPC 响应准备和通信作为回调安装，当最终结果准备好时将被触发。这样，被调用者就不再需要阻塞线程并等待，直到最终的返回值准备好。请参阅
 [@rpc.functions.async_execution](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 的
API 文档
简单的例子。




 除了减少被调用者上的空闲线程数量之外，这些工具还有助于
使批处理 RPC 处理更轻松、更快。本教程的以下两节演示如何使用 [@rpc.functions.async_execution](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 
 装饰器。





## 批量更新参数服务器 [¶](#batch-updating-parameter-server "永久链接到此标题")




 考虑一个具有一个参数
服务器 (PS) 和多个训练器的同步参数服务器训练应用程序。在此应用程序中，PS 保存参数并等待所有训练器报告梯度。在每次迭代中，
都会等待，直到接收到来自所有训练器的梯度，然后一次性更新所有
参数。下面的代码显示了 PS 类的实现。
`update_and_fetch_model`
 方法使用
 `@rpc.functions.async_execution`
 进行修饰，并且将被训练师召唤。每个调用都会返回一个“Future”对象，该对象将使用更新后的模型进行填充。大多数训练器启动的调用只是将梯度累积到
 `.grad`
 字段，立即返回，并在 PS 上产生 RPC 线程。最后到达的训练器将触发优化器步骤并消耗所有先前报告的梯度。然后，它使用更新后的模型设置
 `future_model`，
进而通过
 `Future`
 对象通知其他训练器之前的所有请求，并将更新后的模型发送给所有训练器。 






```
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




 对于训练器，它们都是使用 PS 中的同一组
参数进行初始化的。在每次迭代中，每个训练器首先运行前向和后向传递以在本地生成梯度。然后，每个训练器使用 RPC 将其梯度报告给 PS，并通过同一 RPC 请求的返回值取回更新的参数。在trainer’s
实现中，目标函数是否带有
 `@rpc.functions.async_execution`
 标记没有区别。 
训练器只需使用
 `rpc_sync`
 调用
 `update_and_fetch_model`
，这将阻止训练器
直到返回更新的模型。






```
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




 我们跳过本教程中启动多个进程的代码，请参阅
 [示例](https://github.com/pytorch/examples/tree/master/distributed/rpc) 
 存储库为全面落实。请注意，可以在没有 [@rpc.functions.async_execution](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 
 装饰器。但是，这需要在 PS 上阻塞更多 RPC 线程，
或使用另一轮 RPC 来获取更新的模型，而后者
会增加更多的代码复杂性和更多的通信开销。




 本节使用一个简单的参数服务器训练示例来展示如何
使用 [@rpc.functions.async_execution](https://pytorch.org/docs/master/rpc. html#torch.distributed.rpc.functions.async_execution) 
 装饰器。在下一节中，我们将重新实现上一节中的强化学习
示例
 [分布式 RPC 框架入门](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)
 使用批处理的教程，并展示其对训练
速度的影响。





## 批处理 CartPole 求解器 [¶](#batch-processing-cartpole-solver "永久链接到此标题")




 本节使用 [OpenAI Gym](https://gym.openai.com/) 中的 CartPole-v1 作为示例来展示批处理 RPC 的性能影响。请注意
因为目标是演示
 [@rpc.functions.async_execution](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 
 我们没有构建最好的 CartPole 求解器或解决大多数不同的 RL 问题，而是使用非常简单的策略和奖励计算策略，并
专注于多观察者单代理批量 RPC 实现。我们使用与之前教程类似的“策略”模型，如下所示。与之前的教程相比，不同之处在于它的构造函数需要一个额外的
 `batch`
 参数来控制
 `F.softmax`
 的
 `dim`
 参数，因为使用批处理， 
 `forward` 函数中的
 `x`
 参数包含来自多个观察者的状态，因此维度需要
正确更改。其他一切都保持不变。






```
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




 `Observer` 的构造函数也会相应地进行调整。它还需要一个
 `batch`
 参数，该参数控制它使用哪个
 `Agent`
 函数来选择
操作。在批处理模式下，它调用
 `Agent`上的
 `select_action_batch`
 函数，该函数将很快呈现，并且该函数将用
 [@rpc.functions.async 修饰_execution](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 
.






```
import gym
import torch.distributed.rpc as rpc

class Observer:
    def __init__(self, batch=True):
        self.id = rpc.get_worker_info().id - 1
        self.env = gym.make('CartPole-v1')
        self.env.seed(args.seed)
        self.select_action = Agent.select_action_batch if batch else Agent.select_action

```




 与之前的教程相比
 [分布式 RPC 框架入门](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html) 
 ，
 观察者的行为略有不同。当环境停止时，它不会退出，而是在每个情节中始终运行
 `n_steps`
 次迭代。当环境返回时，观察者只需重置环境并重新开始。通过这种设计，代理将从每个观察者接收固定数量的状态，因此可以将它们打包到固定大小的tensor中。在每一步中，“观察者”都使用 RPC 将其状态发送给“代理”，并通过返回值获取操作。在每一集结束时，它都会将所有步骤的奖励返回给
 `Agent`
 。请注意，此
 `run_episode`
 函数将由
 `Agent`
 使用RPC 调用。因此此函数中的
 `rpc_sync`
 调用将是一个嵌套的 RPC 调用。我们也可以将此函数标记为
 `@rpc.functions.async_execution`
 以避免阻塞
 `Observer`
 上的一个线程。然而，由于瓶颈是
 `Agent`
 而不是
 `Observer`
 ，因此阻塞
 `Observer` 进程上的一个
线程应该是可以的。






```
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




 `Agent` 的构造函数还采用
 `batch`
 参数，该参数控制
如何对操作概率进行批处理。在批处理模式下，
 `saved_log_probs`
 包含一个tensor列表，其中每个tensor包含一步中所有观察者的动作 robs。如果没有批处理，
 `saved_log_probs`
 是一个字典，
其中的键是观察者 ID，值是该观察者的操作概率列表。






```
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




 非批处理
 `select_acion`
 只是运行状态抛出策略，保存
操作概率，并立即将操作返回给观察者。






```
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




 通过批处理，状态存储在 2D tensor
 `self.states`
 中，使用观察者 ID 作为行 ID。然后，它通过将回调函数安装到批量生成的
 `self.future_actions`
`Future`
 对象来链接
 `Future`
，该对象
将填充特定的使用该观察者的 ID 进行索引的行。
最后到达的观察者通过策略一次性运行所有批处理状态，并相应地设置
 `self.future_actions`
 。发生这种情况时，
`self.future_actions` 上安装的所有回调函数都将被触发，
它们的返回值将用于填充链接的
 `Future`
 对象，
 turn 通知
“代理”
 为来自其他观察者的所有先前 RPC 请求准备响应并进行通信。






```
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




 现在让’s 定义如何将不同的 RPC 函数拼接在一起。 
 `Agent`
 控制每个情节的执行。它首先使用
 `rpc_async`
 在所有观察者上启动
这一事件，并阻止返回的 future，
将填充观察者奖励。请注意，下面的代码使用 RRef 帮助程序
 `ob_rref.rpc_async()`
 在
 `ob 的所有者上
 启动
 `run_episode`
 函数_rref`
 RRef 以及提供的参数。
然后它将保存的动作概率和返回的观察者奖励转换为
预期的数据格式，并启动训练步骤。最后，它重置所有
状态并返回当前剧集的奖励。此函数是运行一集的
入口点。






```
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




 其余代码是正常进程启动和日志记录，
与其他 RPC 教程类似。在本教程中，所有观察者都被动
等待来自代理的命令。请参阅
 [示例](https://github.com/pytorch/examples/tree/master/distributed/rpc) 
 存储库以了解完整实现。






```
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
 对于范围 (2, 12) 中的 world_size:
 延迟 = []
 对于 [True, False] 中的批次：
 tik = time.time()
 mp. spawn(
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




 批量 RPC 有助于将动作推理整合为更少的 CUDA 操作，
从而减少摊销开销。上面
 `main`
 函数使用不同数量的观察者在批处理和非批处理模式下运行
相同的代码，
范围从 1 到 10。下图使用默认值绘制了不同
世界大小的执行时间参数值。结果证实了我们的预期
批处理有助于加快训练速度。




![](https://pytorch.org/tutorials/_images/batch.png)


## 了解更多 [¶](#learn-more "此标题的永久链接")



* [批量更新参数服务器源代码](https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/parameter_server.py)
* [批量处理CartPole Solver](https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/reinforce.py)
* [分布式 Autograd](https://pytorch.org/docs/master/rpc.html#distributed- autograd-framework)
* [分布式管道并行性](dist_pipeline_parallel_tutorial.html)








