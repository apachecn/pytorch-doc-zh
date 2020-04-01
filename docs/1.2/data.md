# torch.utils.data

> 译者：[shuziP](https://github.com/shuziP)
> 
> 校验：[shuziP](https://github.com/shuziP)

PyTorch数据加载程序的核心是 `torch.utils.data.DataLoader` 类。它表示在数据集上可迭代的Python，并支持

  * 映射样式和迭代样式的数据集([map-style and iterable-style datasets](https://pytorch.org/docs/stable/data.html#dataset-types)）

  * 自定义数据加载顺序([customizing data loading order](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)）

  * 自动批次([automatic batching](https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data)）

  * 单进程和多进程数据加载([single- and multi-process data loading](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)）

  * 自动内存锁([automatic memory pinning](https://pytorch.org/docs/stable/data.html#memory-pinning)）

这些选项是由`DataLoader`的构造函数参数配置的，具有签名:


​    
​    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
​               batch_sampler=None, num_workers=0, collate_fn=None,
​               pin_memory=False, drop_last=False, timeout=0,
​               worker_init_fn=None)


下面几节将详细描述这些选项的功能和用法。

## 数据集类型

 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)构造函数最重要的参数是dataset，它表示要从中加载数据的`dataset`对象。PyTorch支持两种不同类型的数据集:

  * [版图式数据集](https://pytorch.org/docs/stable/data.html#map-style-datasets),
  * [迭代式的数据集](https://pytorch.org/docs/stable/data.html#iterable-style-datasets).

### 版图式数据集

版图式数据集实现了 `__getitem__()` 和 `__len__()` 协议，并表示从(可能不是完整的)索引/键到数据样本的映射。

例如，当使用 `dataset[idx]`访问这样的数据集时，可以从磁盘上的文件夹中读取 `idx`-th i图像及其对应的标签。

参见[`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)了解更多细节。

### 可迭代式的数据集

可迭代式数据集的 `一个子类的实例IterableDataset`实现了 `__iter__()` 协议和代表了数据样本可迭代。这种类型的数据集特别适合这样的情况：随机读取非常高代价，甚至是不可能的，并且批大小取决于获取的数据。

例如，这样的数据集在被访问 `iter(dataset)`，可以返回从数据库、远程服务器甚至实时生成的日志读取的数据流。

参见 `IterableDataset`了解更多详情。

注意

当使用 [multi-process data loading](https://pytorch.org/docs/stable/data.html#multi-process-data-loading). 的[`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) 时。在每个工作进程上复制相同的数据集对象，因此必须对副本进行不同的配置，以避免重复数据。有关如何实现此目的，请参见[`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) 文档。

## 数据加载顺序和采样器

对于[iterable风格的数据集](https://pytorch.org/docs/stable/data.html#iterable风格的数据集)，数据加载顺序完全由用户定义的iterable控制。这允许更容易地实现块读取和动态批处理大小(例如，每次生成一个批处理样例)。

本节的其余部分涉及[map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets)。[`torch.utils.data.Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) 类用于指定数据加载中使用的索引/键的顺序。它们表示数据集索引上的可迭代对象。例如，在随机梯度像样(SGD)的常见情况下，一个 [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) 可以随机排列一个索引列表，并一次产生一个，或产生一小部分用于小型批量SGD的索引。

顺序采样器或打乱采样器将根据 DataLoader 的' shuffle '参数自动构建。或者，用户可以使用‘sampler’参数来指定一个自定义的[`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) 对象，该对象每次都会生成下一个要获取的索引/键。

一个自定义的 [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) ，一次生成一批索引的列表，可以作为' batch_sampler '参数传递。自动批处理也可以通过“batch_size”和“drop_last”参数启用。参见[下一节](https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data) 获得更多的细节。

请注意

“sampler”和“batch_sampler”都与迭代式数据集不兼容，因为这样的数据集没有键或索引的概念。

## 加载批处理和非批处理数据

DataLoader支持通过参数batch_size、drop_last和batch_sampler将单个获取的数据样本自动整理成批。

### 自动批处理(默认)

这是最常见的情况，它对应于获取少量数据并将其整理成成批的样本，即，包含一个维度为批处理维度(通常是第一个维度)的张量。

当“batch_size”(默认为“1”)不是“None”时，数据加载器将生成成批的样本，而不是单个样本。“batch_size”和“drop_last”参数用于指定数据加载器如何获取批量数据集键。对于地图样式的数据集，用户也可以指定“batch_sampler”，它一次生成一个键列表。

请注意



“batch_size”和“drop_last”参数主要用于从“sampler”构造“batch_sampler”。对于地图样式的数据集，“采样器”要么由用户提供，要么基于“shuffle”参数构造。对于迭代式数据集，“采样器”是一个虚拟的无限数据集。有关采样器的更多信息，请参见[本节](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler) 

请注意

当从具有多个处理的迭代式数据集中获取数据时，drop_last参数将删除每个工作区的数据集副本的最后一批未完成的数据。

使用来自sampler的索引获取样本列表之后，作为collate_fn参数传递的函数被用来将样本列表整理成批量。

在这种情况下，从一个地图样式的数据集加载大致相当于:

```
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

和从一个迭代式数据集加载大致相当于:

```
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```

自定义 `collate_fn`可用于自定义排序规则，例如，将顺序数据填充到批处理的最大长度。参见[本节](https://pytorch.org/docs/stable/data.html#dataloader-collate-fn) 了解更多关于 `collate_fn`.的信息。

### 禁用自动批处理

在某些情况下，用户可能希望在数据集代码中手动处理批处理，或者只加载单个示例。例如，直接加载成批数据(例如，从数据库中批量读取数据或读取连续的内存块)，或者批量大小依赖于数据，或者程序设计用于处理单个样本，这样做的成本更低。在这些场景下，最好不要使用自动批处理(其中使用' collate_fn '对样本进行排序)，而是让数据加载器直接返回' dataset '对象的每个成员。

当“batch_size”和“batch_sampler”都是“None”(batch_sampler的默认值已经是“None”)时，自动批处理将被禁用。从' dataset '获得的每个样例都使用作为' collate_fn '参数传递的函数进行处理。

**当自动批处理被禁用**时，默认的' collate_fn '只是将NumPy数组转换为PyTorch张量，而不改变其他内容。

In this case, loading from a map-style dataset is roughly equivalent with:

在这种情况下，从一个map-style dataset加载大致相当于:

```
for index in sampler:
    yield collate_fn(dataset[index])
```

从一个iterable-style dataset集加载大致相当于:

```
for data in iter(dataset):
    yield collate_fn(data)
```

见[这一节](https://pytorch.org/docs/stable/data.html#dataloader-collate-fn)更多关于collate_fn。

### Working with `collate_fn`

启用或禁用自动批处理时，' collate_fn '的使用略有不同。

**当自动批处理被禁用**，' collate_fn '与每个单独的数据样本一起被调用，输出由数据加载器迭代器产生。在本例中，默认' collate_fn '只是转换PyTorch张量中的NumPy数组。

**启用自动批处理**时，每次使用数据样本列表调用' collate_fn '。预期它会将输入样例整理成一个批，以便从数据加载器迭代器生成。本节的其余部分将在本例中描述默认' collate_fn '的行为。

例如，如果每个数据样本包含一个3通道图像和一个完整的类标签，即，数据集的每个元素都返回一个元组' (image, class_index) '，默认的' collate_fn '将这样的元组列表整理成成批处理的图像张量和成批处理的类标签张量的一个元组。特别是，默认的“collate_fn”具有以下属性:

- 它总是预先添加一个新的维度作为批处理维度。
- 它自动将NumPy数组和Python数值转换为PyTorch张量。
- 它保留了数据结构，例如，如果每个样本是一个字典，它将输出一个字典，该字典具有相同的一组键，但将批量张量作为值(如果不能将值转换为张量，则输出列表)。列表s、元组s、名称元组s也是如此。

用户可以使用自定义的“collate_fn”来实现自定义的批处理，例如，根据第一个维度以外的维度进行排序，填充不同长度的序列，或者添加对自定义数据类型的支持。

## Single- and Multi-process Data Loading

一个[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 默认使用单进程数据加载。



在Python进程中，[全局解释器锁(GIL)](https://wiki.python.org/moin/globalexpressionterlock)会阻止真正的跨线程完全并行化Python代码。为了避免使用数据加载阻塞计算代码，PyTorch提供了一个简单的开关来执行多进程数据加载，只需将参数' num_workers '设置为正整数。

### 单进程数据加载(默认)

在这种模式下，在初始化[' DataLoader '](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)的过程中完成数据获取。因此，数据加载可能会阻塞计算。但是，当用于在进程之间共享数据的资源(例如，共享内存、文件描述符)有限时，或者当整个数据集很小并且可以完全加载到内存中时，这种模式可能是首选的。此外，单进程加载通常显示更多可读的错误跟踪，因此对于调试非常有用。

### Multi-process data loading多进程数据加载

将参数' num_workers '设置为正整数将打开多进程数据加载，并使用指定的加载工作进程数量。

在这种模式下，每次创建[' DataLoader '](https://pytorch.org/docs/stable/data.html# torch.utils.dataloader)的迭代器(例如，当您调用' enumerate(DataLoader) ')时，就会创建' num_workers '工作者进程。此时，' dataset '、' collate_fn '和' worker_init_fn '被传递给每个worker，它们用于初始化和获取数据。这意味着数据集访问及其内部IO、转换(包括' collate_fn ')在工作进程中运行。

[`torch.utils.data.get_worker_info()`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info)返回工作进程中的各种有用信息(包括工作进程id、数据集副本、初始种子等)，并在主进程中返回' None '。用户可以在数据集代码和/或‘worker_init_fn’中使用这个函数来单独配置每个数据集副本，并确定代码是否在工作进程中运行。例如，这对于数据集分片特别有帮助。

对于 map-style 数据集，主进程使用 `sampler` 生成索引并将它们发送给工作者。因此，任何随机洗牌都是在主进程中完成的，它通过为load分配索引来引导装载。

For iterable-style datasets, since each worker process gets a replica of the `dataset` object, naive multi-process loading will often result in duplicated data. Using [`torch.utils.data.get_worker_info()`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info) and/or `worker_init_fn`, users may configure each replica independently. (See [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) documentations for how to achieve this. ) For similar reasons, in multi-process loading, the `drop_last` argument drops the last non-full batch of each worker’s iterable-style dataset replica.

对于迭代风格的数据集，由于每个工作进程都获得一个“dataset”对象的副本，所以简单的多进程加载通常会导致重复的数据。使用[`torch.utils.data.get_worker_info()`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info)'](https://pytorch.org/docs/stable/data.html# torch.utille/get_worker_info)或 `worker_init_fn`,，用户可以独立配置每个副本。(参见 [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) 出于类似的原因，在多进程加载过程中，' drop_last '参数会删除每个worker的迭代式数据集副本的最后一批非完整数据。

一旦到达迭代的末尾，或者当迭代器变成垃圾收集时，Workers就会被关闭。

警告

它一般不建议恢复在多进程加载CUDA张量，因为许多微妙之处使用CUDA和多分享CUDA张量(见并行处理 [
CUDA）。相反，我们建议使用自动存储器钉扎(即，设置`pin_memory =真
`），这使得能够快速数据传输到支持CUDA的GPU。](notes/multiprocessing.html#multiprocessing-cuda-
note)

#### 特定于平台的行为

由于工人依靠Python的[`多重处理 `
](https://docs.python.org/3/library/multiprocessing.html#module-
multiprocessing "\(in Python v3.7\)")，工人启动在Windows上U不同于nix。

  * 在Unix上，`fork()` 是默认的[`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing) 启动方法。使用“fork()”，儿童工作者通常可以通过克隆的地址空间直接访问 `dataset` 和Python参数函数。
  * 在Windows中，`产卵(） `为默认[ `并行处理 `](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "\(in Python v3.7\)")启动方法。使用`重生(） `，另一种解释是推出是运行在主脚本，然后由接收`数据集 `内部职工功能， `collat​​e_fn`和通过[ `泡菜 `](https://docs.python.org/3/library/pickle.html#module-pickle "\(in Python v3.7\)")序列的其它参数。
  * 在Windows上，spawn()是默认的并行处理启动方法([`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing))。使用`spawn()` ，启动另一个解释器，它运行主脚本，然后启动内部的worker函数，它通过 [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle) 序列化接收数据集、collate_fn和其他参数。

这种独立的序列化意味着，你应该采取两个步骤，以确保与Windows兼容，同时使用多进程数据加载:

  * 将主脚本的大部分代码封装在 `if __name__ == '__main__':` block, 中，以确保在启动每个工作进程时不会再次运行(很可能会产生错误)。您可以将数据集和[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 实例创建逻辑放在这里，因为它不需要在workers中重新执行。
  * 确保任何自定义的`collate_fn`, `worker_init_fn` 或数据集代码都被声明为顶层定义，并在 `__main__` 检查之外。这确保它们在工作进程中可用。(这是必需的，因为函数仅作为引用进行pickle，而不是作为字节码。)

#### 多进程数据加载的随机性

默认情况下，每个worker将其PyTorch种子设置为base_seed + worker_id，其中base_seed是由使用其RNG的主进程生成的长种子(因此，强制使用RNG状态)。但是，其他库的种子可能在初始化worker (w.g.)时被复制。，导致每个worker返回相同的随机数。(参见FAQ中的这个 [部分](https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed))。

In `worker_init_fn`, you may access the PyTorch seed set for each worker with either [`torch.utils.data.get_worker_info().seed`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info) or [`torch.initial_seed()`](https://pytorch.org/docs/stable/torch.html#torch.initial_seed), and use it to seed other libraries before data loading.

在`worker_init_fn`,你可以访问PyTorch种子为每个工具人与 [`torch.utils.data.get_worker_info().seed`](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info) 或 [`torch.initial_seed()`](https://pytorch.org/docs/stable/torch.html#torch.initial_seed),并使用它的种子数据加载之前其他库。

## Memory Pinning

当来自固定(页面锁定)内存时，GPU副本的主机速度要快得多。参见[使用固定内存缓冲区](https://pytorch.org/docs/stable/notes/cuda.html# cuda-memory-)了解更多关于何时以及如何使用固定内存的细节。

对于数据加载，将' pin_memory=True '传递给[ DataLoader ](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)将自动将获取的数据张量放入固定内存中，从而能够更快地将数据传输到支持cuda的gpu。

默认的内存固定逻辑只识别张量、映射和包含张量的迭代器。默认情况下,如果把逻辑看到一批自定义类型(这将发生如果你有一批“collate_fn”,返回一个自定义类型),或者如果你批的每个元素是一个自定义类型,将逻辑不会认出他们,它会返回这批没有固定的内存(或这些元素)。要为自定义批处理或数据类型启用内存固定，请在自定义类型上定义' pin_memory() '方法。

See the example below.

请参见下面的例子。

例：

```python
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
```

*CLASS*`torch.utils.data.``DataLoader`(*dataset*, *batch_size=1*, *shuffle=False*, *sampler=None*, *batch_sampler=None*, *num_workers=0*, *collate_fn=None*, *pin_memory=False*, *drop_last=False*, *timeout=0*, *worker_init_fn=None*, *multiprocessing_context=None*)

​    

数据加载程序。组合一个数据集和一个采样器，并在给定的数据集上提供一个可迭代的。

 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)支持地图样式和迭代样式的数据集，支持单进程或多进程加载、自定义加载顺序以及可选的自动批处理(排序)和内存固定。

看[`torch.utils.data`](https://pytorch.org/docs/stable/data.html#module-torch.utils.data) 。有关更多详细信息，请参阅数据文档页。

Parameters

​    

  * **dataset** ([*Dataset*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)) - 从该数据集到加载数据。

  * **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*)) - 如何每批许多样品加载(默认值：`1`）。

  * **shuffle** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*)) - 设置为`真 `为具有在每个历元改组的数据(默认值：`假 `）。

  * **sampler** ([*Sampler*](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)*,* *optional*)) - 定义从数据集中得出样品的策略。如果指定，`洗牌 `必须`假 [HTG17。`

  * **batch_sampler**  (_取样_ _，_ _可选_ ) - 象`取样 `，但在同一时间返回一批指标。互斥与`的batch_size`，`洗牌 `，`取样 `和`drop_last`。

  * **num_workers** ([ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ) - 多少子过程用于数据加载。 `0`意味着数据将在主处理加载。 (默认值：`0`）

  * **collat​​e_fn**  (_可调用_ _，_ _可选_ ) - 合并的样本的列表，以形成小批量张量(S）的。使用从图式集装批处理时使用。

  * **pin_memory** ([ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ) - 如果`真 `，数据装载将在返回之前复制到张量CUDA固定内存。如果数据元素是一个自定义类型，或你的`collat​​e_fn`返回一批即自定义类型，见下面的例子。

  * **drop_last** ([ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ) - 设置为`真 `放弃最后一批不全，如果数据集大小不是由批量大小整除。如果`假 `和数据集的大小是不是批量大小整除，则最后一批将较小。 (默认值：`假 `）

  * **timeout** (_数字_ _，_ _可选_ ) - 如果为正，则为从工作者收集批的超时值。应该是非负的。(默认值:0)

  * **worker_init_fn**  (_可调用_ _，_ _可选_ ) - 如果不是' None '，则在播种之后和数据加载之前，以工作者id (' [0, num_workers - 1] '中的int)作为输入，在每个工作者子进程上调用它。(默认:“没有一个”)

Warning

如果使用 `spawn` 启动方法，则`worker_init_fn` 不能是一个不可修改的对象，例如lambda函数。有关PyTorch中并行处理的更多细节，请参见[Multiprocessing best practices](https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-best-practices) 。

Note

`len(dataloader)` 启发式是基于所用采样器的长度。当“dataset”是一个[`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)时，将使用一个无限采样器，它的 `__len__()` 没有实现，因为实际长度取决于可迭代和多进程加载配置。因此，除非使用地图样式的数据集，否则不应该查询此方法。有关这两种数据集的详细信息，请参见 [Dataset Types](https://pytorch.org/docs/stable/data.html#dataset-types) 

*CLASS*`torch.utils.data.``Dataset`

表示数据集的抽象类。

所有表示从键到数据样本的映射的数据集都应该继承它。所有的子类都应该覆盖`__getitem__()`，支持为给定的键获取数据样本。子类也可以选择性地覆盖 `__len__()`预计返回数据集的大小由许多[`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) 实现和默认选项[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).



Note

`的DataLoader`缺省构建一个索引采样能产生整数指数。为了使它与地图式的数据集与非整指数/键的作用，必须提供自定义采样。

[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 默认情况下构造一个索引采样器，生成完整的索引。要使它与具有非完整索引/键的地图样式数据集一起工作，必须提供自定义采样器。

_class_`torch.utils.data.``IterableDataset`[[source]](_modules/torch/utils/data/dataset.html#IterableDataset)

​    

可迭代的数据集。

代表数据样本的迭代所有数据集应该继承它。当数据来自一个数据集流的这种形式是特别有用的。

所有子类应该overrite `__iter __(） `，这将返回样本的迭代在该数据集。

当一个子类使用具有 `的DataLoader`，在数据集中的每个项目将被从得到的 `的DataLoader`迭代器。当`
num_workers  & GT ;  0`，每个工作进程将具有数据集对象的不同拷贝，因此通常希望独立地配置每个拷贝，以避免从工人返回重复数据。`
get_worker_info(） `，在一个工作进程调用时，返回关于工人的信息。它可以在任一使用的数据集的`__iter __(） `方法或 `
的DataLoader`的`worker_init_fn`选项来修改每个副本的行为。

实施例1：在所有工人分裂工作量`__iter __(） `：


​    

```python
>>> class MyIterableDataset(torch.utils.data.IterableDataset):
...     def __init__(self, start, end):
...         super(MyIterableDataset).__init__()
...         assert end > start, "this example code only works with end >= start"
...         self.start = start
...         self.end = end
...
...     def __iter__(self):
...         worker_info = torch.utils.data.get_worker_info()
...         if worker_info is None:  # single-process data loading, return the full iterator
...             iter_start = self.start
...             iter_end = self.end
...         else:  # in a worker process
...             # split workload
...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
...             worker_id = worker_info.id
...             iter_start = self.start + worker_id * per_worker
...             iter_end = min(iter_start + per_worker, self.end)
...         return iter(range(iter_start, iter_end))
...
>>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
>>> ds = MyIterableDataset(start=3, end=7)

>>> # Single-process loading
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
[3, 4, 5, 6]

>>> # Mult-process loading with two worker processes
>>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
[3, 5, 4, 6]

>>> # With even more workers
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
[3, 4, 5, 6]
```


实施例2：使用worker_init_fn在所有工人之间分配工作负载:

```python
>>> class MyIterableDataset(torch.utils.data.IterableDataset):
...     def __init__(self, start, end):
...         super(MyIterableDataset).__init__()
...         assert end > start, "this example code only works with end >= start"
...         self.start = start
...         self.end = end
...
...     def __iter__(self):
...         return iter(range(self.start, self.end))
...
>>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
>>> ds = MyIterableDataset(start=3, end=7)

>>> # Single-process loading
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
[3, 4, 5, 6]
>>>
>>> # Directly doing multi-process loading yields duplicate data
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
[3, 3, 4, 4, 5, 5, 6, 6]

>>> # Define a `worker_init_fn` that configures each dataset copy differently
>>> def worker_init_fn(worker_id):
...     worker_info = torch.utils.data.get_worker_info()
...     dataset = worker_info.dataset  # the dataset copy in this worker process
...     overall_start = dataset.start
...     overall_end = dataset.end
...     # configure the dataset to only process the split workload
...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
...     worker_id = worker_info.id
...     dataset.start = overall_start + worker_id * per_worker
...     dataset.end = min(dataset.start + per_worker, overall_end)
...

>>> # Mult-process loading with the custom `worker_init_fn`
>>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
[3, 5, 4, 6]

>>> # With even more workers
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
[3, 4, 5, 6]
```


_class_`torch.utils.data.``TensorDataset`( _*tensors_
)[[source]](_modules/torch/utils/data/dataset.html#TensorDataset)

​    

数据集包装张量。

每个样品将沿所述第一维度的索引张量进行检索。

Parameters

​    

***tensors** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor))  -
具有所述第一尺寸的大小相同张量。

_class_`torch.utils.data.``ConcatDataset`( _datasets_
)[[source]](_modules/torch/utils/data/dataset.html#ConcatDataset)

​    

数据集作为多个数据集的串联。

这个类是组装不同的现有数据集是有用的。

Parameters

​    

**datasets** (*sequence*) **数据集**  (_序列_ ) - 数据集的列表要连接

_class_`torch.utils.data.``ChainDataset`( _datasets_
)[[source]](_modules/torch/utils/data/dataset.html#ChainDataset)

​    

数据集chainning多个 `IterableDataset`秒。

这个类是组装不同的现有数据集流是有用的。该chainning操作上即时完成的，因此串联与此类大型数据集将是有效的。

Parameters

​    

**数据集**  (_IterableDataset_ 的迭代） - 数据集链接在一起

_class_`torch.utils.data.``Subset`( _dataset_ , _indices_
)[[source]](_modules/torch/utils/data/dataset.html#Subset)

​    

在指定的索引数据集的子集。

Parameters

​    

  * **数据集**  (_数据集_ ) - 整个数据集

  * **指数**  (_序列_ ) - 在整个组索引选择的子集

`torch.utils.data.``get_worker_info`()[[source]](_modules/torch/utils/data/_utils/worker.html#get_worker_info)

​    

返回当前 `的DataLoader`迭代工作进程的信息。

当一个工人叫，这将返回保证具有以下属性的对象：

  * `ID`：当前作业人员ID。

  * `num_workers`：工人的总数。

  * `种子 `：当前工人随机种子集。此值由主进程RNG和工人的ID来确定。参见 `的DataLoader`的更多细节的文档。

  * `数据集 `：数据集对象在 **这里** 过程的副本。请注意，这将是在不同的进程比一个主处理不同的对象。

当主过程调用，这将返回`无 `。

Note

用于worker_init_fn经过DataLoader时,这种方法可能是有用的设置每个工作进程不同,例如,使用worker_id配置数据集对象只读取一个特定部分的分片数据集,或其他使用种子种子库中使用数据集的代码(例如,NumPy)。

`torch.utils.data.``random_split`( _dataset_ , _lengths_
)[[source]](_modules/torch/utils/data/dataset.html#random_split)

​    

随机分割数据集到给定长度的非重叠的新的数据集。

Parameters

​    

  * **dataset**  (_数据集_ ) - 数据集要被分割

  * **lengths**  (_序列_ ) - 要产生裂缝的长度

_class_`torch.utils.data.``Sampler`( _data_source_
)[[source]](_modules/torch/utils/data/sampler.html#Sampler)

​    

基类的所有取样。

每采样的子类必须提供一个 `__iter__()` 的方法，提供一种方式来迭代数据集的元素的索引，和 `__len__()` 方法，它返回所返回的迭代器的长度。

Note

的`__len __(） `方法并不严格 `的DataLoader`必需的，但在涉及任何计算预期的 `的DataLoader`的长度。

_class_`torch.utils.data.``SequentialSampler`( _data_source_
)[[source]](_modules/torch/utils/data/sampler.html#SequentialSampler)

​    

顺序地将样品的元素，总是以相同的顺序。

Parameters

​    

**DATA_SOURCE**  (_数据集_ ) - 数据集以从采样

_class_`torch.utils.data.``RandomSampler`( _data_source_ , _replacement=False_
, _num_samples=None_
)[[source]](_modules/torch/utils/data/sampler.html#RandomSampler)

​    

样品元件中随机。如果不更换，然后从一个洗牌的数据集进行采样。如果具有置换，然后用户可指定`num_samples`绘制。

Parameters

​    

  * **data_source** ( _Dataset_) – dataset to sample from

  * **replacement**([ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 样品绘制替换如果`真 `，默认=``False``

  * **num_samples** ([ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 样本的数目来绘制，默认=`LEN(数据集）`。该参数应该当替换是`真 `仅被指定。

_class_`torch.utils.data.``SubsetRandomSampler`( _indices_
)[[source]](_modules/torch/utils/data/sampler.html#SubsetRandomSampler)

​    

随机样本元素从指数的定列表，无需更换。

Parameters

​    

**indices** (*sequence*)  - 索引的序列

_class_`torch.utils.data.``WeightedRandomSampler`( _weights_ , _num_samples_ ,
_replacement=True_
)[[source]](_modules/torch/utils/data/sampler.html#WeightedRandomSampler)

​    

从`样品元素 `[0,..,len(weights)-1]` 与给定的概率(权重）。

Parameters

​    

  * **weights** (_序列_ ) - 权重的顺序，没有必要总结到一个

  * **num_samples** ([ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 样本的数目来绘制

  * **replacement** ([ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `，样品绘制更换。如果不是，他们绘制无需更换，这意味着当指数样本绘制为行，不能再为该行画出。

例

```python
>>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
[0, 0, 0, 1, 0]
>>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
[0, 1, 4, 3, 2]
```

包装另一个采样，以产生小批量指数。

Parameters

​    

  * **sampler** (_取样_ ) - 基采样器。

  * **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int))  - 小批量的大小。

  * **drop_last** ([*bool*](https://docs.python.org/3/library/functions.html#bool))  - 如果`真 `，采样器将下降的最后一批，如果它的规模将是小于`的batch_size`

Example

```python
>>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
>>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

限制数据加载到数据集子集的采样器。

它与[`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel).特别有用。在这种情况下，每个进程可以将DistributedSampler实例作为DataLoader采样器传递，并加载原始数据集的一个子集，该子集是它独有的。

Note

数据集被认为是恒定的大小。

Parameters

​    

  * **dataset** \- 数据集用于采样。

  * **num_replicas**  (_可选_ ) - 的参与分布式训练的进程数。

  * **rank** (_可选_ ) - num_replicas内的当前过程的秩。

  * **shuffle** (_可选_ ) - 如果为true(默认值），采样器将会洗牌指数

[Next ![](_static/images/chevron-right-orange.svg)](dlpack.html
"torch.utils.dlpack") [![](_static/images/chevron-right-orange.svg)
Previous](cpp_extension.html "torch.utils.cpp_extension")

* * *

©版权所有2019年，Torch 贡献者。