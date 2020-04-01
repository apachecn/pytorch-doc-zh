# torch.utils.data

> 译者：[BXuan694](https://github.com/BXuan694)

```py
class torch.utils.data.Dataset
```
表示数据集的抽象类。

所有用到的数据集都必须是其子类。这些子类都必须重写以下方法：`__len__`：定义了数据集的规模；`__getitem__`：支持0到len(self)范围内的整数索引。

```py
class torch.utils.data.TensorDataset(*tensors)
```

用于张量封装的Dataset类。

张量可以沿第一个维度划分为样例之后进行检索。

| 参数： | ***tensors** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 第一个维度相同的张量。 |
| --- | --- |

```py
class torch.utils.data.ConcatDataset(datasets)
```

用于融合不同数据集的Dataset类。目的：组合不同的现有数据集，鉴于融合操作是同时执行的，数据集规模可以很大。

| 参数： | **datasets**(_序列_）– 要融合的数据集列表。 |
| --- | --- |

```py
class torch.utils.data.Subset(dataset, indices)
```

用索引指定的数据集子集。

参数： 

*   **dataset**([_Dataset_](#torch.utils.data.Dataset "torch.utils.data.Dataset")）– 原数据集。
*   **indices**(_序列_）– 全集中选择作为子集的索引。

```py
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
```
数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。

参数： 
*   **dataset**([_Dataset_](#torch.utils.data.Dataset "torch.utils.data.Dataset")) – 要加载数据的数据集。
*   **batch_size**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选_) – 每一批要加载多少数据(默认：`1`）。
*   **shuffle**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 如果每一个epoch内要打乱数据，就设置为`True`(默认：`False`）。
*   **sampler**([_Sampler_](#torch.utils.data.Sampler "torch.utils.data.Sampler")_,_ _可选_）– 定义了从数据集采数据的策略。如果这一选项指定了，`shuffle`必须是False。
*   **batch_sampler**([_Sampler_](#torch.utils.data.Sampler "torch.utils.data.Sampler")_,_ _可选_）– 类似于sampler，但是每次返回一批索引。和`batch_size`，`shuffle`，`sampler`，`drop_last`互相冲突。
*   **num_workers**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选_) – 加载数据的子进程数量。0表示主进程加载数据(默认：`0`）。
*   **collate_fn**(_可调用_ _,_ _可选_）– 归并样例列表来组成小批。
*   **pin_memory**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为`True`，数据加载器会在返回前将张量拷贝到CUDA锁页内存。
*   **drop_last**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果数据集的大小不能不能被批大小整除，该选项设为`True`后不会把最后的残缺批作为输入；如果设置为`False`，最后一个批将会稍微小一点。(默认：`False`）
*   **timeout**(_数值_ _,_ _可选_） – 如果是正数，即为收集一个批数据的时间限制。必须非负。(默认：`0`）
*   **worker_init_fn**(_可调用_ _,_ _可选_）– 如果不是`None`，每个worker子进程都会使用worker id(在`[0, num_workers - 1]`内的整数）进行调用作为输入，这一过程发生在设置种子之后、加载数据之前。(默认：`None`）



注意：

默认地，每个worker都会有各自的PyTorch种子，设置方法是`base_seed + worker_id`，其中`base_seed`是主进程通过随机数生成器生成的long型数。而其它库(如NumPy）的种子可能由初始worker复制得到, 使得每一个worker返回相同的种子。(见FAQ中的[My data loader workers return identical random numbers](notes/faq.html#dataloader-workers-random-seed)部分。）你可以用[`torch.initial_seed()`](torch.html#torch.initial_seed "torch.initial_seed")查看`worker_init_fn`中每个worker的PyTorch种子，也可以在加载数据之前设置其他种子。

警告：

如果使用了`spawn`方法，那么`worker_init_fn`不能是不可序列化对象，如lambda函数。

```py
torch.utils.data.random_split(dataset, lengths)
```

以给定的长度将数据集随机划分为不重叠的子数据集。

参数：
*   **dataset** ([_Dataset_](#torch.utils.data.Dataset "torch.utils.data.Dataset")) – 要划分的数据集。
*   **lengths**(_序列_）– 要划分的长度。



```py
class torch.utils.data.Sampler(data_source)
```

所有采样器的基类。

每个Sampler子类必须提供__iter__方法，以便基于索引迭代数据集元素，同时__len__方法可以返回数据集大小。

```py
class torch.utils.data.SequentialSampler(data_source)
```
以相同的顺序依次采样。

| 参数： | **data_source** ([_Dataset_](#torch.utils.data.Dataset "torch.utils.data.Dataset")) – 要从中采样的数据集。 |
| --- | --- |

```py
class torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None)
```

随机采样元素。如果replacement不设置，则从打乱之后的数据集采样。如果replacement设置了，那么用户可以指定`num_samples`来采样。

参数：

*   **data_source** ([_Dataset_](#torch.utils.data.Dataset "torch.utils.data.Dataset")) – 要从中采样的数据集。
*   **num_samples** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 采样的样本数，默认为len(dataset)。
*   **replacement** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为`True`，替换采样。默认False。

```py
class torch.utils.data.SubsetRandomSampler(indices)
```

从给定的索引列表中采样，不替换。

| 参数： | **indices**(_序列_）– 索引序列 |
| --- | --- |

```py
class torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)
```

样本元素来自[0,..,len(weights)-1]，，给定概率(权重)。

参数：

*   **weights**(_序列_) – 权重序列，不需要和为1。
*   **num_samples** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 采样数。
*   **replacement** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果是`True`，替换采样。否则不替换，即：如果某个样本索引已经采过了，那么不会继续被采。

```py
class torch.utils.data.BatchSampler(sampler, batch_size, drop_last)
```

打包采样器来获得小批。

参数： 

*   **sampler**([_Sampler_](#torch.utils.data.Sampler "torch.utils.data.Sampler")）– 基采样器。
*   **batch_size**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 小批的规模。
*   **drop_last**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")）– 如果设置为`True`，采样器会丢弃最后一个不够`batch_size`的小批(如果存在的话）。

示例

```py
>>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
>>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

```py
class torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None)
```

将数据加载限制到数据集子集的采样器。

和[`torch.nn.parallel.DistributedDataParallel`](nn.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel")同时使用时尤其有效。在这中情况下，每个进程会传递一个DistributedSampler实例作为DataLoader采样器，并加载独占的原始数据集的子集。

注意：

假设数据集的大小不变。

参数： 

*   **dataset** – 采样的数据集。
*   **num_replicas**(_可选_）– 参与分布式训练的进程数。
*   **rank**(_可选_）– num_replicas中当前进程的等级。
