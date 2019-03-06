# torch.utils.data

> 译者：[@之茗](https://github.com/mayuanucas)
> 
> 校对者：[@aleczhang](http://community.apachecn.org/?/people/aleczhang)

```py
class torch.utils.data.Dataset
```

表示 Dataset 的抽象类.

所有其它数据集都应继承该类. 所有子类都应该重写 `__len__`, 提供数据集大小的方法, 和 `__getitem__`, 支持从 0 到 len(self) 整数索引的方法.

```py
class torch.utils.data.TensorDataset(data_tensor, target_tensor)
```

包装数据和目标张量的数据集.

通过沿着第一个维度索引两个张量来恢复每个样本.

参数：

*   `data_tensor (Tensor)` – 包含样本数据.
*   `target_tensor (Tensor)` – 包含样本目标 (标签).



```py
class torch.utils.data.ConcatDataset(datasets)
```

用以连结多个数据集的数据集. 目的: 对于组装不同的现有数据集非常有帮助, 可能是 大规模的数据集, 因为串联操作是以即时方式完成的.

参数：`datasets (iterable)` – 需要连结的数据集列表


```py
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate at 0x4316c08>, pin_memory=False, drop_last=False)
```

数据加载器. 组合数据集和采样器,并在数据集上提供单进程或多进程迭代器.

参数：

*   `dataset (Dataset)` – 从该数据集中加载数据.
*   `batch_size (int, 可选)` – 每个 batch 加载多少个样本 (默认值: 1).
*   `shuffle (bool, 可选)` – 设置为 `True` 时, 会在每个 epoch 重新打乱数据 (默认值: False).
*   `sampler (Sampler, 可选)` – 定义从数据集中提取样本的策略. 如果指定, `shuffle` 值必须为 False.
*   `batch_sampler (Sampler, 可选)` – 与 sampler 相似, 但一次返回一批指标. 与 batch_size, shuffle, sampler, and drop_last 互斥.
*   `num_workers (int, 可选)` – 用多少个子进程加载数据. 0表示数据将在主进程中加载 (默认值: 0)
*   `collate_fn (callable, 可选)` – 合并样本列表以形成一个 mini-batch.
*   `pin_memory (bool, 可选)` – 如果为 `True`, 数据加载器会将张量复制到 CUDA 固定内存中, 然后再返回它们.
*   `drop_last (bool, 可选)` – 设定为 `True` 以丢掉最后一个不完整的 batch, 如果数据集大小不能被 batch size整除. 设定为 `False` 并且数据集的大小不能被 batch size整除, 则最后一个 batch 将会更小. (default: False)



```py
class torch.utils.data.sampler.Sampler(data_source)
```

所有采样器的基类.

每一个 Sampler 的子类都必须提供一个 __iter__ 方法, 提供一种 迭代数据集元素的索引的方法, 以及一个 __len__ 方法, 用来返回 迭代器的长度.

```py
class torch.utils.data.sampler.SequentialSampler(data_source)
```

总是以相同的顺序, 依次对元素进行采样.

参数：`data_source (Dataset)` – 采样的数据集


```py
class torch.utils.data.sampler.RandomSampler(data_source)
```

采用无放回采样法, 随机对样本元素采样.

参数：`data_source (Dataset)` – 采样的数据集


```py
class torch.utils.data.sampler.SubsetRandomSampler(indices)
```

采用无放回采样法, 样本元素从指定的索引列表中随机抽取.

参数：`indices (list)` – 索引的列表


```py
class torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples, replacement=True)
```

使用给定的概率 (权重) 对 [0,..,len(weights)-1] 范围的元素进行采样.

参数：

*   `weights (list)` – 权重列表, 没必要加起来等于 1
*   `num_samples (int)` – 抽样数量
*   `replacement (bool)` – 设定为 `True`, 使用有放回采样法. 设定为 `False`, 采用无放回采样法, 这意味着对于一行来说,当一个 样本索引被取到后, 对于改行, 这个样本索引不能再次被取到.



```py
class torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None)
```

将数据加载限制为数据集子集的采样器.

当与 `torch.nn.parallel.DistributedDataParallel` 组合使用时, 特别有用. 在这种情况下, 每个进程都可以将分布式采样器实例作为Data Loader采样器, 并且加载一个原始数据集的子集并独占该数据子集.

注解：

数据集被假定为不变的大小.

参数：

*   `dataset` – 采样的数据集.
*   `num_replicas (optional)` – 参与分布式训练的进程数量.
*   `rank (optional)` – 在 num_replicas 中, 当前进程的等级.

