


# 使用完全分片数据并行 (FSDP) 进行高级模型训练 [¶](#advanced-model-training-with-complete-sharded-data-parallel-fsdp "此标题的永久链接")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/FSDP_adavnced_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html>




**作者** 
 :
 [Hamid Shojanazeri](https://github.com/HamidShojanazeri) 
 ,
 [Less
Wright](https://github.com/lessw2020) 
 ,
 [Rohan Varma](https://github.com/rohan-varma/) 
 ,
 [赵艳丽](https://github.com/zhaojuanmao)




 本教程介绍了作为 PyTorch 1.12 版本一部分的完全分片数据并行 (FSDP) 的更多高级功能。要熟悉 FSDP，请参阅
[FSDP 入门教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
 。




 在本教程中，我们将使用 FSDP 微调 HuggingFace (HF) T5 模型以进行文本
摘要作为工作示例。




 该示例使用 Wikihow，为简单起见，我们将展示在具有 8 个 A100 GPU 的单节点 P4dn 实例上的训练。我们很快就会发布一篇关于多节点集群上的大规模 FSDP 训练的博文，请继续关注 PyTorch 媒体频道。




 FSDP 是一个生产就绪的软件包，重点关注易用性、性能和
长期支持。 FSDP 的主要优点之一是减少每个 GPU 上的内存
占用空间。与 DDP 相比，这使得能够以较低的总内存来训练较大的模型，并利用计算和通信的重叠来高效地训练模型。
这种减少的内存压力可用于训练较大的模型或
增加批量大小，从而可能有助于整体训练吞吐量。您可以
在[此处](https://pytorch.org/blog/introducing-pytorch-filled-sharded-data-parallel-api/)了解有关 PyTorch FSDP 的更多信息
 。





## 本教程中的 FSDP 功能 [¶](#fsdp-features-in-this-tutorial "永久链接到此标题")



* 变压器自动换行策略
* 混合精度
* 在设备上初始化 FSDP 模型
* 分片策略
* 向后预取
* 通过流式传输到 CPU 保存模型检查点





## 回顾 FSDP 的工作原理 [¶](#recap-on-how-fsdp-works "永久链接到此标题")




 在高层 FDSP 的工作原理如下：




*在构造函数中*



* 分片模型参数，每个等级只保留自己的分片



*前向传递*



* 运行
 
 all_gather
 
 从所有级别收集所有分片，以恢复此 FSDP 单元的完整
参数 运行前向计算
* 丢弃它刚刚收集到的非拥有参数分片空闲内存



*向后传递*



* 运行
 
 all_gather
 
 以收集所有等级的所有分片，以恢复此 FSDP 单元中的完整参数
运行反向计算
* 丢弃非拥有的参数以释放内存。
 * 运行reduce_scatter来同步梯度





## 微调 HF T5 [¶](#fine-tuning-hf-t5 "固定链接到此标题")



HF T5 预训练模型有四种不同的尺寸，从具有 6000 万个参数的小型模型到具有 110 亿个参数的 XXL 模型。在本教程中，我们演示了使用 WikiHow 数据集对带有 FSDP 的 T5 3B 进行微调以进行文本摘要。本教程的主要重点是
强调 FSDP 中的不同可用功能，这些功能有助于训练
超过 3B 参数的大型模型。此外，我们还介绍了基于 Transformer 的模型的特定功能。本教程的代码可在
 [Pytorch
示例](https://github.com/pytorch/examples/tree/main/distributed/FSDP/) 中找到
 。




*设置*




 1.1 安装 PyTorch Nightlies




 我们将安装 PyTorch nightlies，因为一些功能（如激活
检查点）在 nightlies 中可用，并将在 1.12 之后的下一个 PyTorch
版本中添加。






```
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html

```




 1.2 数据集设置




 请创建
 
 data
 
 文件夹，从 [wikihowAll.csv](https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358) 下载 WikiHow 数据集
 和
 [wikihowSep.cs](https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag) 
 ，
并将它们放在
 
 data
 
 文件夹中。我们将使用来自
 [summarization_dataset](https://github.com/pytorch/examples/blob/main/distributed/FSDP/summarization_dataset.py)的wikihow数据集
。




 接下来，我们将以下代码片段添加到 Python 脚本 “T5_training.py” 中。





 注意




 本教程的完整源代码可在 [PyTorch 示例](https://github.com/pytorch/examples/tree/main/distributed/FSDP/) 中找到
 。





 1.3 导入必要的包：






```
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.t5.modeling_t5 import T5Block

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
 checkpoint_wrapper,
 CheckpointImpl,
 apply_activation_checkpointing_wrapper)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from summarization_dataset import *
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
import tqdm
from datetime import datetime

```




 1.4 分布式训练设置。
这里我们使用两个辅助函数来初始化分布式
训练过程，然后在训练完成后进行清理。在本教程中，我们将使用 torch elastic，使用 [torchrun](https://pytorch.org/docs/stable/elastic/run.html) 
 ，这将设置 
worker
 \ n 自动排名
 
 和
 
 WORLD_SIZE
 
。






```
def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

```




 2.1 设置 HuggingFace T5 模型：






```
def setup_model(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer =  T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

```




 我们还在此处添加了几个用于日期和格式化内存
指标的辅助函数。






```
def get_date_of_run():
 """create date and time for file save uniqueness
 example: 2022-05-07-08:31:12_PM'
 """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

def format_metrics_to_gb(item):
 """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num

```




 2.2 定义训练函数:






```
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy

```




 2.3 定义验证函数:






```
def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

```




 2.4 定义一个将模型包装在 FSDP 中的分布式训练函数：






```
def fsdp_main(args):

    model, tokenizer = setup_model("t5-base")

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])


 dataset = load_dataset('wikihow', 'all', data_dir='data/')
 print(dataset.keys())
 print("训练数据集的大小: ", dataset['train'].shape)
 print("验证数据集大小: ", dataset['validation'].shape)


 #wikihow(tokenizer, type_path, num_samples, input_length, output_length, print_text=False)
 train_dataset = wikihow(tokenizer, 'train', 1500, 512 , 150, False)
 val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False)

 Sampler1 = DistributedSampler(train_dataset,rank=rank, num_replicas= world_size，shuffle=True)
 Sampler2 = DistributedSampler(val_dataset，rank=rank，num_replicas=world_size)

 setup()


 train_kwargs = {'batch_size'：args.batch_size，'sampler'：sampler1}
 test_kwargs = {'batch_size'：args.test_batch_size , 'sampler':sampler2}
 cuda_kwargs = {'num_workers': 2,
 'pin_memory': True,
 'shuffle': False}
 train_kwargs.更新(cuda_kwargs)
 测试_kwargs.update(cuda_kwargs)

 train_loader = torch.utils.data.DataLoader(train_dataset,**train _kwargs)
 val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

 t5_auto_wrap_policy = functools.partial(
 Transformer_auto_wrap_policy,
 Transformer_layer_cls={
 T5Block,
 },
 )
 sharding_strategy: ShardingStrategy = ShardingStrategy. SHARD_GRAD_OP #for Zero2 和 FULL_SHARD for Zero3
 torch.cuda.set_device(local_rank)


 #init_start_event = torch.cuda.Event(enable_timing=True)
 #init_end_event = torch.cuda.Event(enable_timing=True)

 #init_start_event.record()

 bf16_ready = (
 torch.version.cuda
 和 torch.cuda.is_bf16_supported()
 和 LooseVersion( torch.version.cuda) >= "11.0"
 和 dist.is_nccl_available()
 和 nccl.version() >= (2, 10)
 )

 如果 bf16\ \_ready:
 mp_policy = bfSixteen
 else:
 mp_policy = None # 默认为 fp32

 # 在输入 FSDP 之前模型已在 CPU 上
 model = FSDP(model,
 auto_wrap_policy=t5_auto_wrap_policy，
 mix_ precision=mp_policy，
 #sharding_strategy=sharding_strategy，
 device_id =torch.cuda.current_device())

 优化器 = optim.AdamW(model.parameters(), lr=args.lr)

 调度程序 = StepLR(optimizer, step_size=1 , gamma=args.gamma)
 best_val_loss = float("inf")
 curr_val_loss = float("inf")
 file_save_name = " T5-模型-"

 如果rank == 0:
 time_of_run = get_date_of_run()
 dur = []
 train_acc\ \_tracking = []
 val_acc_tracking = []
 Training_start_time = time.time()

 如果rank == 0 且args.track_memory:\ n mem_alloc_tracker = []
 mem_reserved_tracker = []

 for epoch in range(1, args.epochs + 1):
 t0 = time.time() 
 train_accuracy = train(args、model、rank、world_size、train_loader、optimizer、epoch、sampler=sampler1)
 if args.run_validation:
 curr_val\ \_loss = 验证(模型、排名、世界大小、val\_loader)
 Scheduler.step()

 如果排名 == 0:

 print(f"--> epoch {epoch } 已完成...进入保存和统计区域")

 dur.append(time.time() - t0)
 train_acc_tracking.append(train_accuracy.item())\ n
 if args.run_validation:
 val_acc_tracking.append(curr_val_loss.item())

 if args.track_memory:
 mem _alloc_tracker.append(
 format_metrics_to_gb(torch.cuda.memory_allocated())
 )
 mem_reserved_tracker.append(
 format_metrics_to_gb(torch.cuda.memory_reserved())
 )
 print(f"已完成保存和统计区域...")

 if args.save\ \_model 和 curr_val_loss < best_val_loss:

 # save
 ifrank == 0:
 print(f"--> 进入保存模型状态")
 
 save_policy = FullStateDictConfig(offload_to_cpu=True,rank0_only=True)
 with FSDP.state_dict_type(
 model, StateDictType.FULL_STATE\ \_DICT, save_policy
 ):
 cpu_state = model.state_dict()
 #print(f"保存过程：rank {rank} 完成 w state_dict")


            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                print(f"--> attempting to save model prefix {currEpoch}")
                save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                print(f"--> saving as model name {save_name}")

                torch.save(cpu_state, save_name)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()

```




 2.5 解析参数并设置main函数:






```
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=.002, metavar='LR',
                        help='learning rate (default: .002)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    fsdp_main(args)

```




 要使用 torchrun 运行训练：






```
torchrun --nnodes 1 --nproc_per_node 4 T5_training.py

```








## 变压器包装政策 [¶](#transformer-wrapping-policy "永久链接到此标题")




 正如在
 [上一篇教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) 中所讨论的，
auto_wrap_policy 是 FSDP 功能之一，可以轻松实现这一点自动对给定模型进行分片，并将模型、优化器和梯度分片放入不同的 FSDP 单元中。



对于某些架构（例如 Transformer 编码器-解码器），模型的某些部分（例如嵌入表）与编码器和解码器共享。在这种情况下，我们需要将嵌入表放置在外部 FSDP 单元中，以便可以从编码器和解码器访问。此外，通过注册变压器的层类，可以使分片计划的通信效率更高。在 PyTorch 1.12 中，FSDP 添加了此支持，现在我们
为转换器提供了包装策略。




 可以按如下方式创建，其中 T5Block 代表 T5 转换器
层类（包含 MHSA 和 FFN）。






```
t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )
torch.cuda.set_device(local_rank)


model = FSDP(model,
    fsdp_auto_wrap_policy=t5_auto_wrap_policy)

```




 要查看包装的模型，您可以轻松打印模型并目视检查
分片和 FSDP 单元。





## 混合精度 [¶](#mixed- precision "永久链接到此标题")



FSDP 支持灵活的混合精度训练，允许任意降低精度类型（例如 fp16 或 bfloat16）。目前BFloat16仅适用于Ampere GPU，因此在使用之前需要确认本机支持。例如，在 V100 上，BFloat16 仍然可以运行，但由于它非本机运行，
可能会导致速度显着降低。




 要检查 BFloat16 是否原生支持，您可以使用以下命令:






```
bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and LooseVersion(torch.version.cuda) >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)

```




 FSDP 中混合精度的优点之一是为参数、梯度和缓冲区提供
不同精度级别的粒度控制，
如下所示：






```
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    # Gradient communication precision.
    reduce_dtype=torch.float32,
    # Buffer precision.
    buffer_dtype=torch.float32,
)

```




 请注意，如果未指定某种类型（参数、reduce、buffer），则它们根本不会被强制转换。



这种灵活性允许用户进行细粒度控制，例如仅设置梯度通信以降低的精度进行，并且所有参数/缓冲区计算都以全精度完成。在节点内通信是主要瓶颈并且参数缓冲区必须完全精确以避免准确性问题的情况下，这可能很有用。这可以通过
以下策略来完成:






```
grad_bf16 = MixedPrecision(reduce_dtype=torch.bfloat16)

```




 在 2.4 中，我们只是将相关的混合精度策略添加到 FSDP 包装器中：






```
model = FSDP(model,
       auto_wrap_policy=t5_auto_wrap_policy,
       mixed_precision=bfSixteen)

```




 在我们的实验中，我们观察到，通过使用 BFloat16 进行训练，速度提高了 4 倍，并且在某些可用于增加批量大小的实验中
内存减少了约 30%。





## 正在设备上初始化 FSDP 模型 [¶](#intializing-fsdp-model-on-device "永久链接到此标题")




 在 1.12 中，FSDP 支持 
 
 device_id
 
 参数，用于初始化由 
 
 device_id
 
 给定的设备上的输入 CPU
模块。当整个模型不适合单个 GPU，但适合主机’s CPU 内存时，这非常有用。当指定
 
 device_id
 
 时，FSDP 将以每个 FSDP
单元为基础将模型移动到指定设备，避免 GPU OOM 问题，同时初始化速度比基于 CPU 的初始化快几倍： 






```
torch.cuda.set_device(local_rank)

 model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=torch.cuda.current_device())

```





## 分片策略 [¶](#sharding-strategy "固定链接到此标题")




 FSDP 分片策略默认设置为完全分片模型参数，
梯度和优化器状态在所有等级上进行分片。 （也称为 Zero3
s 分片）。如果您对 Zero2 分片策略感兴趣，
仅对优化器状态和梯度进行分片，FSDP 通过使用 \xe2\x80\x9cShardingStrategy.SHARD_GRAD_OP\xe2 传递分片策略来支持此功能\x80\x9d，\而不是 \xe2\x80\x9cShardingStrategy.FULL_SHARD\xe2\x80\x9d 到 FSDP 初始化，如下所示：






```
torch.cuda.set_device(local_rank)

 model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP # ZERO2)

```




 这将减少 FSDP 中的通信开销，在这种情况下，它在前向传递和后向传递之后保留完整
参数。




 这可以在向后过程中保存 all_gather，因此可以减少通信，但
代价是增加内存占用。请注意，完整的模型参数在向后传递结束时被释放，并且 all_gather 将在下一次向前传递时发生。





## 向后预取 [¶](#backward-prefetch "永久链接到此标题")




 向后预取设置控制应请求下一个 FSDP 单元’s
参数的时间。通过将其设置为

 BACKWARD_PRE

，可以开始请求下一个
FSDP’s 单元参数，并在当前单元的计算开始之前更快到达
。这与
 
 all _gather
 
 通信和梯度计算重叠，可以提高训练速度，以换取稍高的内存消耗。它可以在 2.4 中的 FSDP
wrapper 中使用，如下所示：






```
torch.cuda.set_device(local_rank)

 model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=torch.cuda.current_device(),
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE)

```





 backward_prefetch
 
 有两种模式，
 
 BACKWARD_PRE
 
 和
 
 BACKWARD_POST
 
 。
 
 BACKWARD_POST
 
意味着在当前 FSDP 单元处理完成之前，不会请求下一个 FSDP 单元’s 参数，从而最大限度地减少内存
开销。在某些情况下，使用
 
 BACKWARD_PRE
 
 可以将模型训练速度
提高高达 2-10%，对于较大的模型，速度改进甚至更高。





## 模型检查点保存，通过流式传输到 Rank0 CPU [¶](#model-checkpoint- saving-by-streaming-to-the-rank0-cpu "永久链接到此标题")




 为了使用 FULL_STATE_DICT 保存（以与本地模型相同的方式保存模型）来保存模型检查点，PyTorch 1.12 提供了一些实用程序来支持
较大模型的保存。




 首先，可以指定 FullStateDictConfig，允许 state_dict 只
填充到 0 级并卸载到 CPU。




 使用此配置时，FSDP 将全部收集模型参数，将它们一一卸载到 CPU，仅在 Rank 0 上。当 state_dict 最终
保存时，它将仅在Rank 0 上填充并包含 CPU张量。这可以避免
大于单个 GPU 内存的模型可能出现 OOM，并允许用户
检查大小大致等于
用户’s 计算机上的可用 CPU RAM 的模型。




 此功能可以按如下方式运行:






```
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            cpu_state = model.state_dict()
if rank == 0:
 save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
 torch.save(cpu_state, save_name)

```





## 摘要 [¶](#summary "此标题的永久链接")




 在本教程中，我们介绍了 Pytorch 1.12 中提供的许多 FSDP 新功能，并使用 HF T5 作为运行示例。使用正确的包装策略，特别是对于变压器模型，以及混合精度和向后预取，应该可以加快您的训练运行速度。此外，
在设备上初始化模型以及通过流式传输到 CPU 保存检查点等功能
应有助于避免处理大型模型时出现 OOM 错误。




 我们正在积极努力为下一个版本的 FSDP 添加新功能。如果您有反馈、功能请求、疑问或在使用 FSDP 时遇到问题，请随时通过在 [PyTorch Github 存储库](https://github.com/pytorch/pytorch) 中提出问题来与我们联系
.









