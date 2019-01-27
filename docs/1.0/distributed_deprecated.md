

# Distributed communication package (deprecated) - torch.distributed.deprecated

Warning

torch.distributed.deprecated is the older version of torch.distributed and currently deprecated. It will be removed soon. Please use and refer the doc for torch.distributed, which is the latest distributed communication package for PyTorch

torch.distributed.deprecated provides an MPI-like interface for exchanging tensor data across multi-machine networks. It supports a few different backends and initialization methods.

Currently torch.distributed.deprecated supports four backends, each with different capabilities. The table below shows which functions are available for use with CPU / CUDA tensors. MPI supports cuda only if the implementation used to build PyTorch supports it.

| Backend | `tcp` | `gloo` | `mpi` | `nccl` |
| --- | --- | --- | --- | --- |
| Device | CPU | GPU | CPU | GPU | CPU | GPU | CPU | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| send | ✓ | ✘ | ✘ | ✘ | ✓ | ? | ✘ | ✘ |
| recv | ✓ | ✘ | ✘ | ✘ | ✓ | ? | ✘ | ✘ |
| broadcast | ✓ | ✘ | ✓ | ✓ | ✓ | ? | ✘ | ✓ |
| all_reduce | ✓ | ✘ | ✓ | ✓ | ✓ | ? | ✘ | ✓ |
| reduce | ✓ | ✘ | ✘ | ✘ | ✓ | ? | ✘ | ✓ |
| all_gather | ✓ | ✘ | ✘ | ✘ | ✓ | ? | ✘ | ✓ |
| gather | ✓ | ✘ | ✘ | ✘ | ✓ | ? | ✘ | ✘ |
| scatter | ✓ | ✘ | ✘ | ✘ | ✓ | ? | ✘ | ✘ |
| barrier | ✓ | ✘ | ✓ | ✓ | ✓ | ? | ✘ | ✘ |

## Basics

The &lt;cite&gt;torch.distributed.deprecated&lt;/cite&gt; package provides PyTorch support and communication primitives for multiprocess parallelism across several computation nodes running on one or more machines. The class `torch.nn.parallel.deprecated.DistributedDataParallel()` builds on this functionality to provide synchronous distributed training as a wrapper around any PyTorch model. This differs from the kinds of parallelism provided by [Multiprocessing package - torch.multiprocessing](multiprocessing.html) and [`torch.nn.DataParallel()`](nn.html#torch.nn.DataParallel "torch.nn.DataParallel") in that it supports multiple network-connected machines and in that the user must explicitly launch a separate copy of the main training script for each process.

In the single-machine synchronous case, &lt;cite&gt;torch.distributed.deprecated&lt;/cite&gt; or the `torch.nn.parallel.deprecated.DistributedDataParallel()` wrapper may still have advantages over other approaches to data-parallelism, including [`torch.nn.DataParallel()`](nn.html#torch.nn.DataParallel "torch.nn.DataParallel"):

*   Each process maintains its own optimizer and performs a complete optimization step with each iteration. While this may appear redundant, since the gradients have already been gathered together and averaged across processes and are thus the same for every process, this means that no parameter broadcast step is needed, reducing time spent transferring tensors between nodes.
*   Each process contains an independent Python interpreter, eliminating the extra interpreter overhead and “GIL-thrashing” that comes from driving several execution threads, model replicas, or GPUs from a single Python process. This is especially important for models that make heavy use of the Python runtime, including models with recurrent layers or many small components.

## Initialization

The package needs to be initialized using the [`torch.distributed.deprecated.init_process_group()`](#torch.distributed.deprecated.init_process_group "torch.distributed.deprecated.init_process_group") function before calling any other methods. This blocks until all processes have joined.

```py
torch.distributed.deprecated.init_process_group(backend, init_method='env://', **kwargs)
```

Initializes the distributed package.

| Parameters: | 

*   **backend** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – Name of the backend to use. Depending on build-time configuration valid values include: `tcp`, `mpi`, `gloo` and `nccl`.
*   **init_method** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")_,_ _optional_) – URL specifying how to initialize the package.
*   **world_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Number of processes participating in the job.
*   **rank** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Rank of the current process.
*   **group_name** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")_,_ _optional_) – Group name. See description of init methods.

 |
| --- | --- |

To enable `backend == mpi`, PyTorch needs to built from source on a system that supports MPI. If you want to use Open MPI with CUDA-aware support, please use Open MPI major version 2 and above.

Note

This method initializes CUDA context. Therefore, if multiple processes run on a single machine but use different GPUs, make sure to use [`torch.cuda.set_device()`](cuda.html#torch.cuda.set_device "torch.cuda.set_device") before this method to avoid unnecessarily creating context on the first visible device.

```py
torch.distributed.deprecated.get_rank()
```

Returns the rank of current process.

Rank is a unique identifier assigned to each process within a distributed group. They are always consecutive integers ranging from `0` to `world_size - 1` (inclusive).

```py
torch.distributed.deprecated.get_world_size()
```

Returns the number of processes in the distributed group.

* * *

Currently three initialization methods are supported:

### TCP initialization

There are two ways to initialize using TCP, both requiring a network address reachable from all processes and a desired `world_size`. The first way requires specifying an address that belongs to the rank 0 process. This initialization method requires that all processes have manually specified ranks.

Alternatively, the address has to be a valid IP multicast address, in which case ranks can be assigned automatically. Multicast initialization also supports a `group_name` argument, which allows you to use the same address for multiple jobs, as long as they use different group names.

```py
import torch.distributed.deprecated as dist

# Use address of one of the machines
dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)

# or a multicast address - rank will be assigned automatically if unspecified
dist.init_process_group(backend, init_method='tcp://[ff15:1e18:5d4c:4cf0:d02d:b659:53ba:b0a7]:23456',
                        world_size=4)

```

### Shared file-system initialization

Another initialization method makes use of a file system that is shared and visible from all machines in a group, along with a desired `world_size`. The URL should start with `file://` and contain a path to a non-existent file (in an existing directory) on a shared file system. This initialization method also supports a `group_name` argument, which allows you to use the same shared file path for multiple jobs, as long as they use different group names.

Warning

This method assumes that the file system supports locking using `fcntl` - most local systems and NFS support it.

```py
import torch.distributed.deprecated as dist

# Rank will be assigned automatically if unspecified
dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                        world_size=4, group_name=args.group)

```

### Environment variable initialization

This method will read the configuration from environment variables, allowing one to fully customize how the information is obtained. The variables to be set are:

*   `MASTER_PORT` - required; has to be a free port on machine with rank 0
*   `MASTER_ADDR` - required (except for rank 0); address of rank 0 node
*   `WORLD_SIZE` - required; can be set either here, or in a call to init function
*   `RANK` - required; can be set either here, or in a call to init function

The machine with rank 0 will be used to set up all connections.

This is the default method, meaning that `init_method` does not have to be specified (or can be `env://`).

## Groups

By default collectives operate on the default group (also called the world) and require all processes to enter the distributed function call. However, some workloads can benefit from more fine-grained communication. This is where distributed groups come into play. [`new_group()`](#torch.distributed.deprecated.new_group "torch.distributed.deprecated.new_group") function can be used to create new groups, with arbitrary subsets of all processes. It returns an opaque group handle that can be given as a `group` argument to all collectives (collectives are distributed functions to exchange information in certain well-known programming patterns).

```py
torch.distributed.deprecated.new_group(ranks=None)
```

Creates a new distributed group.

This function requires that all processes in the main group (i.e., all processes that are part of the distributed job) enter this function, even if they are not going to be members of the group. Additionally, groups should be created in the same order in all processes.

| Parameters: | **ranks** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]_) – List of ranks of group members. |
| --- | --- |
| Returns: | A handle of distributed group that can be given to collective calls. |
| --- | --- |

## Point-to-point communication

```py
torch.distributed.deprecated.send(tensor, dst)
```

Sends a tensor synchronously.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Tensor to send.
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Destination rank.

 |
| --- | --- |

```py
torch.distributed.deprecated.recv(tensor, src=None)
```

Receives a tensor synchronously.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Tensor to fill with received data.
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Source rank. Will receive from any process if unspecified.

 |
| --- | --- |
| Returns: | Sender rank. |
| --- | --- |

[`isend()`](#torch.distributed.deprecated.isend "torch.distributed.deprecated.isend") and [`irecv()`](#torch.distributed.deprecated.irecv "torch.distributed.deprecated.irecv") return distributed request objects when used. In general, the type of this object is unspecified as they should never be created manually, but they are guaranteed to support two methods:

*   `is_completed()` - returns True if the operation has finished
*   `wait()` - will block the process until the operation is finished. `is_completed()` is guaranteed to return True once it returns.

When using the MPI backend, [`isend()`](#torch.distributed.deprecated.isend "torch.distributed.deprecated.isend") and [`irecv()`](#torch.distributed.deprecated.irecv "torch.distributed.deprecated.irecv") support non-overtaking, which has some guarantees on supporting message order. For more detail, see [http://mpi-forum.org/docs/mpi-2.2/mpi22-report/node54.htm#Node54](http://mpi-forum.org/docs/mpi-2.2/mpi22-report/node54.htm#Node54)

```py
torch.distributed.deprecated.isend(tensor, dst)
```

Sends a tensor asynchronously.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Tensor to send.
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Destination rank.

 |
| --- | --- |
| Returns: | A distributed request object. |
| --- | --- |

```py
torch.distributed.deprecated.irecv(tensor, src)
```

Receives a tensor asynchronously.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Tensor to fill with received data.
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Source rank.

 |
| --- | --- |
| Returns: | A distributed request object. |
| --- | --- |

## Collective functions

```py
torch.distributed.deprecated.broadcast(tensor, src, group=<object object>)
```

Broadcasts the tensor to the whole group.

`tensor` must have the same number of elements in all processes participating in the collective.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Data to be sent if `src` is the rank of current process, and tensor to be used to save received data otherwise.
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Source rank.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

```py
torch.distributed.deprecated.all_reduce(tensor, op=<object object>, group=<object object>)
```

Reduces the tensor data across all machines in such a way that all get the final result.

After the call `tensor` will be bitwise identical in all processes.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Input and output of the collective. The function operates in-place.
*   **op** (_optional_) – One of the values from `torch.distributed.deprecated.reduce_op` enum. Specifies an operation used for element-wise reductions.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

```py
torch.distributed.deprecated.reduce(tensor, dst, op=<object object>, group=<object object>)
```

Reduces the tensor data across all machines.

Only the process with rank `dst` is going to receive the final result.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Input and output of the collective. The function operates in-place.
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Destination rank
*   **op** (_optional_) – One of the values from `torch.distributed.deprecated.reduce_op` enum. Specifies an operation used for element-wise reductions.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

```py
torch.distributed.deprecated.all_gather(tensor_list, tensor, group=<object object>)
```

Gathers tensors from the whole group in a list.

| Parameters: | 

*   **tensor_list** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – Output list. It should contain correctly-sized tensors to be used for output of the collective.
*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Tensor to be broadcast from current process.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

```py
torch.distributed.deprecated.gather(tensor, **kwargs)
```

Gathers a list of tensors in a single process.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Input tensor.
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Destination rank. Required in all processes except the one that is receiveing the data.
*   **gather_list** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – List of appropriately-sized tensors to use for received data. Required only in the receiving process.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

```py
torch.distributed.deprecated.scatter(tensor, **kwargs)
```

Scatters a list of tensors to all processes in a group.

Each process will receive exactly one tensor and store its data in the `tensor` argument.

| Parameters: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Output tensor.
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Source rank. Required in all processes except the one that is sending the data.
*   **scatter_list** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – List of tensors to scatter. Required only in the process that is sending the data.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

```py
torch.distributed.deprecated.barrier(group=<object object>)
```

Synchronizes all processes.

This collective blocks processes until the whole group enters this function.

| Parameters: | **group** (_optional_) – Group of the collective. |
| --- | --- |

## Multi-GPU collective functions

If you have more than one GPU on each node, when using the NCCL backend, [`broadcast_multigpu()`](#torch.distributed.deprecated.broadcast_multigpu "torch.distributed.deprecated.broadcast_multigpu") [`all_reduce_multigpu()`](#torch.distributed.deprecated.all_reduce_multigpu "torch.distributed.deprecated.all_reduce_multigpu") [`reduce_multigpu()`](#torch.distributed.deprecated.reduce_multigpu "torch.distributed.deprecated.reduce_multigpu") and [`all_gather_multigpu()`](#torch.distributed.deprecated.all_gather_multigpu "torch.distributed.deprecated.all_gather_multigpu") support distributed collective operations among multiple GPUs within each node. These functions can potentially improve the overall distributed training performance and be easily used by passing a list of tensors. Each Tensor in the passed tensor list needs to be on a separate GPU device of the host where the function is called. Note that the length of the tensor list needs to be identical among all the distributed processes. Also note that currently the multi-GPU collective functions are only supported by the NCCL backend.

For example, if the system we use for distributed training has 2 nodes, each of which has 8 GPUs. On each of the 16 GPUs, there is a tensor that we would like to all-reduce. The following code can serve as a reference:

Code running on Node 0

```py
import torch
import torch.distributed.deprecated as dist

dist.init_process_group(backend="nccl",
                        init_method="file:///distributed_test",
                        world_size=2,
                        rank=0)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)

```

Code running on Node 1

```py
import torch
import torch.distributed.deprecated as dist

dist.init_process_group(backend="nccl",
                        init_method="file:///distributed_test",
                        world_size=2,
                        rank=1)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)

```

After the call, all 16 tensors on the two nodes will have the all-reduced value of 16

```py
torch.distributed.deprecated.broadcast_multigpu(tensor_list, src, group=<object object>)
```

Broadcasts the tensor to the whole group with multiple GPU tensors per node.

`tensor` must have the same number of elements in all the GPUs from all processes participating in the collective. each tensor in the list must be on a different GPU.

Note

Only NCCL backend is currently supported. `tensor_list` should only contain GPU tensors.

| Parameters: | 

*   **tensor_list** (_List__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – Tensors that participate in the collective operation. if `src` is the rank, then the first element of `tensor_list` (`tensor_list[0]`) will be broadcasted to all other tensors (on different GPUs) in the src process and all tensors in `tensor_list` of other non-src processes. You also need to make sure that `len(tensor_list)` is the same for all the distributed processes calling this function.
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Source rank.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

```py
torch.distributed.deprecated.all_reduce_multigpu(tensor_list, op=<object object>, group=<object object>)
```

Reduces the tensor data across all machines in such a way that all get the final result. This function reduces a number of tensors on every node, while each tensor resides on a different GPU. Therefore, the input tensor in the tensor list needs to be GPU tensors. Also, each tensor in the tensor list needs to reside on a different GPU.

After the call, all tensors in `tensor_list` will be bitwise identical in all processes.

Note

Only NCCL backend is currently supported. `tensor_list` should only contain GPU tensors.

| Parameters: | 

*   **tensor_list** (_List__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – List of input and output tensors of the collective. The function operates in-place and requires that each tensor to be a GPU tensor on different GPUs. You also need to make sure that `len(tensor_list)` is the same for all the distributed processes calling this function.
*   **op** (_optional_) – One of the values from `torch.distributed.deprecated.reduce_op` enum. Specifies an operation used for element-wise reductions.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

```py
torch.distributed.deprecated.reduce_multigpu(tensor_list, dst, op=<object object>, group=<object object>)
```

Reduces the tensor data on multiple GPUs across all machines. Each tensor in :attr`tensor_list` should reside on a separate GPU.

Only the GPU of `tensor_list[0]` on the process with rank `dst` is going to receive the final result.

Note

Only NCCL backend is currently supported. `tensor_list` should only contain GPU tensors.

| Parameters: | 

*   **tensor_list** (_List__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – Input and output GPU tensors of the collective. The function operates in-place. You also need to make sure that `len(tensor_list)` is the same for all the distributed processes calling this function.
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Destination rank
*   **op** (_optional_) – One of the values from `torch.distributed.deprecated.reduce_op` enum. Specifies an operation used for element-wise reductions.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

```py
torch.distributed.deprecated.all_gather_multigpu(output_tensor_lists, input_tensor_list, group=<object object>)
```

Gathers tensors from the whole group in a list. Each tensor in `input_tensor_list` should reside on a separate GPU.

Note

Only NCCL backend is currently supported. `output_tensor_lists` and `input_tensor_list` should only contain GPU tensors.

| Parameters: | 

*   **output_tensor_lists** (_List__[__List__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]__]_) – Output lists. It should contain correctly-sized tensors on each GPU to be used for output of the collective. e.g. `output_tensor_lists[i]` contains the all_gather result that resides on the GPU of `input_tensor_list[i]`. Note that each element of `output_tensor_lists[i]` has the size of `world_size * len(input_tensor_list)`, since the function all gathers the result from every single GPU in the group. To interpret each element of `output_tensor_list[i]`, note that `input_tensor_list[j]` of rank k will be appear in `output_tensor_list[i][rank * world_size + j]` Also note that `len(output_tensor_lists)`, and the size of each element in `output_tensor_lists` (each element is a list, therefore `len(output_tensor_lists[i])`) need to be the same for all the distributed processes calling this function.
*   **input_tensor_list** (_List__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – List of tensors (on different GPUs) to be broadcast from current process. Note that `len(input_tensor_list)` needs to be the same for all the distributed processes calling this function.
*   **group** (_optional_) – Group of the collective.

 |
| --- | --- |

## Launch utility

The &lt;cite&gt;torch.distributed.deprecated&lt;/cite&gt; package also provides a launch utility in &lt;cite&gt;torch.distributed.deprecated.launch&lt;/cite&gt;.

&lt;cite&gt;torch.distributed.launch&lt;/cite&gt; is a module that spawns up multiple distributed training processes on each of the training nodes.

The utility can be used for single-node distributed training, in which one or more processes per node will be spawned. The utility can be used for either CPU training or GPU training. If the utility is used for GPU training, each distributed process will be operating on a single GPU. This can achieve well-improved single-node training performance. It can also be used in multi-node distributed training, by spawning up multiple processes on each node for well-improved multi-node distributed training performance as well. This will especially be benefitial for systems with multiple Infiniband interfaces that have direct-GPU support, since all of them can be utilized for aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed training, this utility will launch the given number of processes per node (`--nproc_per_node`). If used for GPU training, this number needs to be less or euqal to the number of GPUs on the current system (`nproc_per_node`), and each process will be operating on a single GPU from _GPU 0 to GPU (nproc_per_node - 1)_.

**How to use this module:**

1.  Single-Node multi-process distributed training

```py
>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
 arguments of your training script)

```

1.  Multi-Node multi-process distributed training: (e.g. two nodes)

Node 1: _(IP: 192.168.1.1, and has a free port: 1234)_

```py
>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
 --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
 and all other arguments of your training script)

```

Node 2:

```py
>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
 --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
 and all other arguments of your training script)

```

1.  To look up what optional arguments this module offers:

```py
>>> python -m torch.distributed.launch --help

```

**Important Notices:**

1\. This utilty and multi-process distributed (single-node or multi-node) GPU training currently only achieves the best performance using the NCCL distributed backend. Thus NCCL backend is the recommended backend to use for GPU training.

2\. In your training program, you must parse the command-line argument: `--local_rank=LOCAL_PROCESS_RANK`, which will be provided by this module. If your training program uses GPUs, you should ensure that your code only runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:

Parsing the local_rank argument

```py
>>> import argparse
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument("--local_rank", type=int)
>>> args = parser.parse_args()

```

Set your device to local rank using either

```py
>>> torch.cuda.set_device(arg.local_rank)  # before your code runs

```

or

```py
>>> with torch.cuda.device(arg.local_rank):
>>>    # your code to run

```

3\. In your training program, you are supposed to call the following function at the beginning to start the distributed backend. You need to make sure that the init_method uses `env://`, which is the only supported `init_method` by this module.

```py
torch.distributed.init_process_group(backend='YOUR BACKEND',
                                     init_method='env://')

```

4\. In your training program, you can either use regular distributed functions or use [`torch.nn.parallel.DistributedDataParallel()`](nn.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") module. If your training program uses GPUs for training and you would like to use [`torch.nn.parallel.DistributedDataParallel()`](nn.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") module, here is how to configure it.

```py
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[arg.local_rank],
                                                  output_device=arg.local_rank)

```

Please ensure that `device_ids` argument is set to be the only GPU device id that your code will be operating on. This is generally the local rank of the process. In other words, the `device_ids` needs to be `[args.local_rank]`, and `output_device` needs to be `args.local_rank` in order to use this utility

Warning

`local_rank` is NOT globally unique: it is only unique per process on a machine. Thus, don’t use it to decide if you should, e.g., write to a networked filesystem. See [https://github.com/pytorch/pytorch/issues/12042](https://github.com/pytorch/pytorch/issues/12042) for an example of how things can go wrong if you don’t do this correctly.

