# torch.utils.data
```python
class torch.utils.data.Dataset
```

表示Dataset的抽象类。

所有其他数据集都应该进行子类化。所有子类应该override`__len__`和`__getitem__`，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)。

```python
class torch.utils.data.TensorDataset(data_tensor, target_tensor)
```
包装数据和目标张量的数据集。

通过沿着第一个维度索引两个张量来恢复每个样本。

**参数：**

- **data_tensor** (*Tensor*) －　包含样本数据
- **target_tensor** (*Tensor*) －　包含样本目标(标签）

```python
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
```
数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。

**参数：**

- **dataset** (*Dataset*) – 加载数据的数据集。
- **batch_size** (*int*, optional) – 每个batch加载多少个样本(默认: 1)。
- **shuffle** (*bool*, optional) – 设置为`True`时会在每个epoch重新打乱数据(默认: False).
- **sampler** (*Sampler*, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略`shuffle`参数。
- **num_workers** (*int*, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
- **collate_fn** (*callable*, optional) –
- **pin_memory** (*bool*, optional) –
- **drop_last** (*bool*, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)

```python
class torch.utils.data.sampler.Sampler(data_source)
```
所有采样器的基础类。

每个采样器子类必须提供一个`__iter__`方法，提供一种迭代数据集元素的索引的方法，以及返回迭代器长度的`__len__`方法。

```python
class torch.utils.data.sampler.SequentialSampler(data_source)
```
样本元素顺序排列，始终以相同的顺序。

**参数：**
- **data_source** (*Dataset*) – 采样的数据集。

```python
class torch.utils.data.sampler.RandomSampler(data_source)
```
样本元素随机，没有替换。

**参数：**
- **data_source** (*Dataset*) – 采样的数据集。


```python
class torch.utils.data.sampler.SubsetRandomSampler(indices)
```
样本元素从指定的索引列表中随机抽取，没有替换。

**参数：**
- **indices** (*list*) – 索引的列表

```python
class torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples, replacement=True)
```
样本元素来自于[0,..,len(weights)-1]，给定概率(weights）。

**参数：**
- **weights** (*list*) – 权重列表。没必要加起来为1
- **num_samples** (*int*) – 抽样数量
