# PyTorch Pendulum：使用 TorchRL 进行物理环境模拟

> 译者：[先天亏钱圣体](https://github.com/sanxincao)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/pendulum>
>
> 原始地址：<https://pytorch.org/tutorials//advanced/pendulum.html>


创建环境（模拟器或物理控制系统的接口）是强化学习和控制工程的集成部分。  

TorchRL 提供了一组工具来在执行此操作。本教程演示如何使用 PyTorch 和 TorchRL 从头开始​​编写钟摆模拟器。 它的灵感来自[OpenAI-Gym/Farama-Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 控制库的 Pendulum-v1 实现。

### 主要学习内容： 

* 如何在 TorchRL 中设计环境模拟： - 编写规则（输入、观察和奖励）； - 实现行为：初始化、重置和步进。 
* 转换您的环境输入和输出，并编写您自己的转换。
* 如何使用TensorDict来携带任何结构的数据通过codebase，在此过程中，我们将接触 TorchRL 的三个关键组成部分：
* [环境](https://pytorch.org/rl/reference/envs.html) 
* [转换](https://pytorch.org/rl/reference/transforms.html) 
* [策略](https://pytorch.org/rl/reference/modules.html) (策略和价值函数) 

为了了解 TorchRL 环境可以实现的目标，我们将设计一个无状态环境。有状态环境会跟踪遇到的最新物理状态并依靠它来模拟状态到状态的转换，而无状态环境则希望在每个步骤中向它们提供当前状态以及所采取的操作。（想吐槽）TorchRL 支持这两种类型的环境，但无状态环境更通用，因此涵盖了 TorchRL 中更广泛的功能的环境 API。 

对无状态环境进行建模使用户可以完全控制模拟器的输入和输出：可以在任何阶段重置实验或主动从外部修改动态。然而，它假设我们可以完全控制任务，或者可以有极强的控制能力。但情况可能并非总是如此：解决我们无法控制当前状态的问题更具挑战性，而且具有更广泛的应用范围。  

无状态环境的另一个优点是它们可以批量执行转换模拟。如果后端和实现允许，可以在标量、向量或张量上无缝执行代数运算。本教程给出了这样的例子。

本教程的结构如下：
* 我们首先要熟悉环境属性：它的形状（batch_size），它的方法（主要是step()， reset()和set_seed()），最后是它的规则。 
* 写好模拟器后，我们将演示如何在转换训练期间使用它。
* 我们将探索 TorchRL API 的新途径，包括：转换输入的可能性、模拟的矢量化执行以及通过模拟图反向传播的可能性。
* 最后，我们将训练一个简单的策略来解决我们实现的系统。

```
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

DEFAULT_X = np.pi
DEFAULT_Y = 1.0
```
设计新环境类时必须注意四件事： 
* EnvBase._reset()，该代码用于将模拟器重置为（可能是随机的）初始状态；
* EnvBase._step()状态转换动态代码；
* EnvBase._set_seed`()实施种子机制；
* 环境规则 

让我们首先描述当前的问题：我们想要模拟一个简单的钟摆，通过它我们可以控制施加在其固定点上的扭矩。我们的目标是将摆锤置于向上位置（按照惯例，角度位置为 0）并使其静止在该位置。为了设计我们的动态系统，我们需要定义两个方程：动作（施加的扭矩）后的运动方程和构成我们目标函数的奖励方程。

对于运动方程，我们将更新角速度如下：
$$
\dot{\theta}_{t+1} = \dot{\theta}_t + dt \left( \frac{G}{L} \sin(\theta) + \frac{u}{mL^2} \right)
$$
公式中𝜃点是以 rad/sec 为单位的角速度，𝐺是引力，𝐿是摆的长度，m是它的质量， 𝜃是它的角位置并且
u是扭矩。然后根据以下公式更新角位置：
$$
\theta_{t+1} = \theta + dt \cdot \dot{\theta}_{t+1}
$$  
(不会写下标，凑活看)

定义奖励方程：
$$
r = -({\theta}^2+0.1*\dot{\theta}^2+0.001*u^2)
$$

当角度接近 0（摆处于向上位置）、角速度接近 0（无运动）且扭矩也为 0 时，该值将最大化。

## 对动作的效果进行编码：_step()
步进的方法是首要考虑的事，因为他是对我们感兴趣事务的编码。在 TorchRL 中， EnvBase该类有一个EnvBase.step() 方法，用于接收一个tensordict.TensorDict 实例，该"action"实例带有一个指示要采取什么操作的条目。  

为了在tensordict中方便读取和写入，确保密钥与库所期望的一致，模拟部分已委托给一个私有抽象方法_step()， 该方法从一个tensordict 读取输入数据，并使用输出数据写入新的 tensordict数据。  

该_step()方法应该执行以下操作：
* 读取输入按键（如"action"）并据此执行模拟 
* 检索观察结果、完成状态和奖励
* 将观察值集以及奖励和完成状态写入新的相关TensorDict中  

接下来step()方法将合并tensordict的输入与step()的输出，确保输入输出的一致性。  
通常，对于有状态环境，这将如下所示： 
```
>>> policy(env.reset())
>>> print(tensordict)
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        observation: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([]),
    device=cpu,
    is_shared=False)
>>> env.step(tensordict)
>>> print(tensordict)
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False),
        observation: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([]),
    device=cpu,
    is_shared=False)
``` 
请注意，根tensordict没有变化,唯一的修改是出现了一个新的‘next’条目，其中包含了新信息。  
在钟摆示例中，我们的_step()方法将从输入中读取相关条目，并在将按键编码（"action" key 行为按键）的力施加到钟摆上tensordict后计算钟摆的位置和速度。我们将摆锤的新角位置计算 为先前位置加上一段时间间隔内的新速度"new_th"的结果。 

由于我们的目标是将钟摆向上转动并将其保持在该位置，因此cost对于接近目标的位置和低速，我们的（负奖励）函数较低。事实上，我们希望阻止远离“向上”的位置和/或远离 0 的速度。

在我们的示例中，EnvBase._step()由于我们的环境是无状态的，因此被编码为静态方法。在有状态设置中，self需要该参数，因为需要从环境中读取状态。
```
def _step(tensordict):
    th, thdot = tensordict["th"], tensordict["thdot"]  # th := theta

    g_force = tensordict["params", "g"]
    mass = tensordict["params", "m"]
    length = tensordict["params", "l"]
    dt = tensordict["params", "dt"]
    u = tensordict["action"].squeeze(-1)
    u = u.clamp(-tensordict["params", "max_torque"], tensordict["params", "max_torque"])
    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

    new_thdot = (
        thdot
        + (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length**2) * u) * dt
    )
    new_thdot = new_thdot.clamp(
        -tensordict["params", "max_speed"], tensordict["params", "max_speed"]
    )
    new_th = th + new_thdot * dt
    reward = -costs.view(*tensordict.shape, 1)
    done = torch.zeros_like(reward, dtype=torch.bool)
    out = TensorDict(
        {
            "th": new_th,
            "thdot": new_thdot,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
    )
    return out


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi
```
## 重置模拟器：_reset()
我们需要关心的第二个方法是 _reset()方法。就像 _step()，它应该在它的输出中写入观察条目和可能的完成状态（如果省略完成状态，它将由父方法 tensordict填充失败）。在某些情况下，reset方法需要从调用它的函数接收命令（例如，在多智能体设置中，我们可能想要指明哪些智能体需要被重置）。 这也是为什么_reset同样需要一个tensordict作为输入参数。，尽管它可能完全为空。  
父级EnvBase.reset()会像EnvBase.step()一样执行一些简单的检查 ，例如确保"done"输出中返回状态tensordict以及形状与规则中的预期相匹配。   
对于我们来说，唯一需要考虑的重要事情是是否 EnvBase._reset()包含所有预期的观察结果。由于我们正在使用无状态环境，因此我们将钟摆的配置传递到嵌套的tensordict名为"params".   
在这个例子中，我们没有传递完成状态，因为这不是强制性的，_reset()而且我们的环境是非终止的，所以我们总是期望它是False。
```
def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        # if no ``tensordict`` is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input ``tensordict`` contains all the relevant
        # parameters to get started.
        tensordict = self.gen_params(batch_size=self.batch_size)

    high_th = torch.tensor(DEFAULT_X, device=self.device)
    high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
    low_th = -high_th
    low_thdot = -high_thdot

    # for non batch-locked environments, the input ``tensordict`` shape dictates the number
    # of simulators run simultaneously. In other contexts, the initial
    # random state's shape will depend upon the environment batch-size instead.
    th = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_th - low_th)
        + low_th
    )
    thdot = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_thdot - low_thdot)
        + low_thdot
    )
    out = TensorDict(
        {
            "th": th,
            "thdot": thdot,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    return out
```  
## 环境元数据 env.*_spec
规则定义了环境的输入和输出域。规则准确定义将在运行时接收的张量非常重要，因为它们通常用于携带有关多处理和多进程中的环境的信息。它们还可以用于实例化延迟定义的神经网络和测试脚本，而无需实际查询环境（例如，对于现实世界的物理系统来说，这可能成本高昂）。  
我们必须在我们的环境中编写四个规范：
* EnvBase.observation_spec: 这将是一个CompositeSpec 实例，其中每个键都是一个观察值（CompositeSpec可以被视为规则字典）。
* EnvBase.action_spec: 可以是任意类型的spec，但要求与"action"input中的条目对应tensordict；
* EnvBase.reward_spec:提供有关奖励空间的信息；
* EnvBase.done_spec:提供有关完成状态的信息。
TorchRL 规则被组织在两个通用容器中：input_spec其中包含步骤函数读取的信息规则（分为action_spec包含操作和state_spec包含所有其余内容）。以及output_spec对步骤输出的规则进行编码（observation_spec、reward_spec和done_spec）。一般来说，您不应该直接与output_spec和 input_spec的交互，而是应该和他们的内容交互: observation_spec, reward_spec, done_spec, action_spec and state_spec。原因是规则可能是以非正常方式构建的，output_spec and input_spec都不应该被直接修改。  

换句话说，observation_spec和相关属性是输出和输入规范容器内容的便捷快捷方式。  
TorchRL 提供多个TensorSpec 子类来编码环境的输入和输出特征。  

## Specs shape
环境规则主要尺寸必须与环境批量大小相匹配。这样做是为了强制环境的每个组件（包括其转换）都能够准确表示预期的输入和输出形状。这是应该在有状态设置中准确编码的内容。  

对于非批量锁定环境，例如我们示例中的环境（见下文），这是无关紧要的，因为环境批量大小很可能为空。
```
def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = CompositeSpec(
        th=BoundedTensorSpec(
            low=-torch.pi,
            high=torch.pi,
            shape=(),
            dtype=torch.float32,
        ),
        thdot=BoundedTensorSpec(
            low=-td_params["params", "max_speed"],
            high=td_params["params", "max_speed"],
            shape=(),
            dtype=torch.float32,
        ),
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    self.action_spec = BoundedTensorSpec(
        low=-td_params["params", "max_torque"],
        high=td_params["params", "max_torque"],
        shape=(1,),
        dtype=torch.float32,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite
```
## 可重复的实验：播种  
初始化环境时播种环境是一项常见操作。唯一的目标EnvBase._set_seed()是设置所包含模拟器的种子。如果可能，此操作不应调用环境执行reset()或与环境执行交互。父EnvBase.set_seed()方法采用了一种机制，允许使用不同的伪随机且可重现的种子播种多个环境。  
```
def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng
```

## 将事物包装在一起：EnvBase类 

我们终于可以将各个部分组合在一起并设计我们的环境类。环境构建过程中需要进行specs初始化，因此我们必须注意_make_spec()调用PendulumEnv.__init__().

我们添加一个静态方法PendulumEnv.gen_params()，它确定性地生成一组在执行期间使用的超参数： 
```
def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_speed": 8,
                    "max_torque": 2.0,
                    "dt": 0.05,
                    "g": g,
                    "m": 1.0,
                    "l": 1.0,
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td
```
batch_locked我们通过将homonymous 属性转换为 来将环境定义为非False。这意味着我们不会强制输入 tensordict与batch-size环境相匹配。

下面的代码将把我们上面编码的部分组合在一起。  
```
class PendulumEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed
```
## 测试我们的环境 
TorchRL 提供了一个简单的函数check_env_specs() 来检查（转换后的）环境是否具有与其规范规定的输入/输出结构相匹配的输入/输出结构。让我们尝试一下：
```
env = PendulumEnv()
check_env_specs(env)
```
我们可以查看我们的规范，以直观地表示环境签名：  
```
print("observation_spec:", env.observation_spec)
print("state_spec:", env.state_spec)
print("reward_spec:", env.reward_spec)
```
我们也可以执行几个命令来检查输出结构是否符合预期。
```
td = env.reset()
print("reset tensordict", td)
```
我们可以运行env.rand_step()从域中随机生成一个动作action_spec。由于我们的环境是无状态的，因此必须传递tensordict包含超参数和当前状态的A 。在有状态的上下文中，env.rand_step()也能完美工作。
```
td = env.rand_step(td)
print("random step tensordict", td)
```

## 转换环境
为无状态模拟器编写环境转换比有状态模拟器稍微复杂一些：转换需要在以下迭代中读取的输出条目需要在meth.step()下一步调用之前应用逆变换。这是展示 TorchRL 变换的所有功能的理想场景！ 
例如，在下面的转换环境中，我们的unsqueeze条目 能够沿着最后一个维度堆叠它们。一旦它们作为下一次迭代的输入传递，我们还将它们传递为将它们压缩回原始形状。 
```
env = TransformedEnv(
    env,
    # ``Unsqueeze`` the observations that we will concatenate
    UnsqueezeTransform(
        unsqueeze_dim=-1,
        in_keys=["th", "thdot"],
        in_keys_inv=["th", "thdot"],
    ),
)
```

## 编写自定义转换
TorchRL 的转换可能无法涵盖环境执行后想要执行的所有操作。编写转换并不需要太多努力。至于环境设计，编写转换有两个步骤：
* 获得正确的动作（正向和反向）
* 能够适应环境的规则  
转换可以在两种设置中使用：就其本身而言，它可以用作 Module.它也可以附加到 TransformedEnv.类的结构允许自定义不同上下文中的行为。一个Transform骨架可以概括如下：
```
class Transform(nn.Module):
    def forward(self, tensordict):
        ...
    def _apply_transform(self, tensordict):
        ...
    def _step(self, tensordict):
        ...
    def _call(self, tensordict):
        ...
    def inv(self, tensordict):
        ...
    def _inv_apply_transform(self, tensordict):
        ...
```
共有三个入口点（forward()、_step()和inv()），它们都接收tensordict.TensorDict实例。前两个最终将遍历由 指示的键in_keys 并调用_apply_transform()其中的每一个。如果提供的话，结果将写入所指向的条目中Transform.out_keys（如果没有，in_keys将使用转换后的值进行更新）。如果需要执行逆变换，则将执行类似的数据流，但使用Transform.inv()和 Transform._inv_apply_transform()方法并跨键in_keys_inv 和out_keys_inv列表。下图总结了环境和重播缓冲区的流程。  
(文档并没有给图哥们)  
在某些情况下，转换不会以单一方式处理键的子集，而是会在父环境上执行某些操作或处理整个输入tensordict。在这些情况下，应该重写_call()和方法，并且可以跳过该方法。forward()_apply_transform()

让我们编写新的变换来计算位置角度的sin和cos 值，因为这些值比原始角度值更有助于我们学习策略。
```
class SinTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.sin()

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


class CosTransform(Transform):
    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs.cos()

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=-1,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


t_sin = SinTransform(in_keys=["th"], out_keys=["sin"])
t_cos = CosTransform(in_keys=["th"], out_keys=["cos"])
env.append_transform(t_sin)
env.append_transform(t_cos)
```
将观察结果连接到“观察”条目。 del_keys=False确保我们为下一次迭代保留这些值。
```
cat_transform = CatTensors(
    in_keys=["sin", "cos", "thdot"], dim=-1, out_key="observation", del_keys=False
)
env.append_transform(cat_transform)
```
让我们再次检查我们的环境规格是否与收到的规格相符：
```
check_env_specs(env)
```
## 执行 
执行需要几个简单的步骤
* 重置环境
* 当某些条件不满足时：
  * 计算给定策略的操作
  * 执行给定此操作的步骤
  * 收集数据
  * 迈出MDP一步(?)
* 收集数据并返回  
这些操作已方便地包装在rollout() 方法中，我们在下面提供了一个简化版本。
```
def simple_rollout(steps=100):
    # preallocate:
    data = TensorDict({}, [steps])
    # reset
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand()
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data


print("data from rollout:", simple_rollout(100))
```
## Batching 计算
我们教程的最后一个未探索的部分是我们必须在 TorchRL 中进行批量计算的能力。由于我们的环境不对输入数据形状做出任何假设，因此我们可以在批量数据上无缝执行它。更好的是：对于非批量锁定的环境（例如我们的 Pendulum），我们可以动态更改批量大小，而无需重新创建环境。为此，我们只需生成具有所需形状的参数。  
```
batch_size = 10  # number of environments to be executed in batch
td = env.reset(env.gen_params(batch_size=[batch_size]))
print("reset (batch size of 10)", td)
td = env.rand_step(td)
print("rand step (batch size of 10)", td)
```
使用一批数据执行 rollout 需要我们重置 rollout 函数中的环境，因为我们需要动态定义 batch_size 而这不受以下支持rollout()：
```
rollout = env.rollout(
    3,
    auto_reset=False,  # we're executing the reset out of the ``rollout`` call
    tensordict=env.reset(env.gen_params(batch_size=[batch_size])),
)
print("rollout of len 3 (batch size of 10):", rollout)
```

## 训练一个简单的策略
在此示例中，我们将使用奖励作为可微目标（例如负损失）来训练一个简单的策略。我们将利用动态系统完全可微的事实，通过轨迹返回进行反向传播，并调整我们的策略权重以直接最大化该值。当然，在许多设置中，我们所做的许多假设都不成立，例如可微分系统和对底层机制的完全访问。  

尽管如此，这仍然是一个非常简单的示例，展示了如何使用 TorchRL 中的自定义环境对训练循环进行编码。

我们先来写一下策略网络：
```
torch.manual_seed(0)
env.set_seed(0)

net = nn.Sequential(
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(1),
)
policy = TensorDictModule(
    net,
    in_keys=["observation"],
    out_keys=["action"],
)
```
和我们的优化器：
```
optim = torch.optim.Adam(policy.parameters(), lr=2e-3)
```

## 训练循环
我们将陆续：
* 生成轨迹
* 总结奖励
* 通过这些操作定义的图进行反向传播
* 裁剪梯度范数并进行优化步骤
* 重复

在训练循环结束时，我们应该得到接近 0 的最终奖励，这表明钟摆是向上的并且仍然符合预期。
```
batch_size = 32
pbar = tqdm.tqdm(range(20_000 // batch_size))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20_000)
logs = defaultdict(list)

for _ in pbar:
    init_td = env.reset(env.gen_params(batch_size=[batch_size]))
    rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
    traj_return = rollout["next", "reward"].mean()
    (-traj_return).backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    pbar.set_description(
        f"reward: {traj_return: 4.4f}, "
        f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
    )
    logs["return"].append(traj_return.item())
    logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
    scheduler.step()


def plot():
    import matplotlib
    from matplotlib import pyplot as plt

    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    with plt.ion():
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(logs["return"])
        plt.title("returns")
        plt.xlabel("iteration")
        plt.subplot(1, 2, 2)
        plt.plot(logs["last_reward"])
        plt.title("last reward")
        plt.xlabel("iteration")
        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        plt.show()


plot()
```

<img src='https://pytorch.org/tutorials/_images/sphx_glr_pendulum_001.png' width=20% />
