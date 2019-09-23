# torch.utils.data

在PyTorch数据加载工具的心脏是 `torch.utils.data.DataLoader`类。它代表了一个数据集的一个Python迭代，与支持

  * 图式和可迭代式的数据集

  * 定制数据加载顺序

  * 自动配料

  * 单和多处理数据加载

  * 自动存储器钉扎。

这些选项由的构造器参数构成的 `的DataLoader`，其具有签名：

    
    
    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)
    

下面的章节详细描述了影响和这些选项的用法。

## 数据集类型

的 `的DataLoader`构造的最重要的参数是`数据集 `，其指示数据集对象从加载数据。 PyTorch支持两种不同类型的数据集：

  * 图式数据集

  * 迭代式的数据集[HTG1。

### 地图式的数据集

一种地图风格数据集是一个用于实现`__getitem __（） `和`__len __（） `协议，以及表示从（可能是一个地图非一体）索引/键数据样本。

例如，这样的数据集，当与`访问数据集[IDX]`，可以读取`IDX`个图像和其相应的标签从磁盘上的文件夹。

参见 `数据集 `了解更多详情。

### 可迭代式的数据集

可迭代式数据集的 `一个子类的实例IterableDataset`实现了`__iter __（）
`协议和代表了数据样本可迭代。这种类型的数据集的特别适合于情况下随机读取是昂贵的，甚至不可能的，并且其中所述批量大小取决于所取的数据。

例如，这样的数据集，称为`当ITER（数据集） `，可以返回数据从数据库中，远程服务器读取的流，或甚至原木实时生成。

参见 `IterableDataset`了解更多详情。

注意

当使用 `IterableDataset`与多进程数据加载。相同的数据集对象被复制在每个工作进程，因此副本必须被不同地配置，以避免重复的数据。参见
`IterableDataset`如何实现这个单证。

## 数据加载顺序和 `取样 `

用户定义的迭代为可迭代式的数据集，数据加载顺序完全由控制。这允许数据块读取和动态批量大小的更容易实现（例如，通过产生在每个时间成批样品）。

本节的其余部分涉及与图式的数据集的情况。`torch.utils.data.Sampler`
[HTG7类可用于指定在数据加载用于索引/键的序列。他们代表了索引到的数据集可迭代的对象。例如，与随机梯度下降（SGD）的常见情况下， `取样 `
可以随机置换指数列表和屈服每一次一个，或产生一个少数人的小批量SGD。

一种顺序或改组采样器将基于所述`洗牌 `参数向 `的DataLoader`自动构造。可替换地，用户可以使用`取样 `参数来指定自定义 `取样
`对象在每个时间产生的下一个索引/键获取。

自定义 `取样 `为在一个时间产生一批指数列表可以作为`batch_sampler`参数传递。自动配料，也可以通过`的batch_size
`和`drop_last`参数启用。参见下一节本更多细节。

Note

既不`取样 `也不`batch_sampler`是具有可迭代式的数据集兼容，因为这样的数据集没有一个键或索引的概念。

## 装载成批和非成批数据

`的DataLoader`支持单个取出的数据样本自动整理成批次经由参数`的batch_size`，`drop_last`和`
batch_sampler`。

### 自动配料（默认）HTG0]

这是最常见的情况，并且对应于提取数据的minibatch并将它们整理成批处理样品，即，含有与张量一个维度是所述批料尺寸（通常是第一个）。

当`的batch_size`（默认`1`）不是`无 `，数据加载器的产率批量样品，而不是个别样品。 `的batch_size`和`
drop_last`参数用于指定数据加载器如何获得数据集密钥的批次。在地图风格数据集，用户可以另外指定`batch_sampler
`，它在一个时间产生密钥的列表。

Note

的`的batch_size`和`drop_last`参数基本上被用于一个`batch_sampler`从构建`取样
`。在地图式的数据集时，`取样 `或者由用户提供的或根据`洗牌 `参数构成。对于迭代式的数据集时，`取样 `是伪无限之一。参见本节上采样的更多细节。

Note

当从迭代式的数据集与取多处理，在`drop_last`参数下降到最后的非整批生产的每个员工的数据集复制品。

取使用从采样器的索引的样本的列表之后，函数作为`collat​​e_fn`参数被用来校核样本列表成批通过。

在这种情况下，从图式集装是大致相当于：

    
    
    for indices in batch_sampler:
        yield collate_fn([dataset[i] for i in indices])
    

并从迭代式集装是大致相当于：

    
    
    dataset_iter = iter(dataset)
    for indices in batch_sampler:
        yield collate_fn([next(dataset_iter) for _ in indices])
    

自定义`collat​​e_fn`可以被用于定制的归类，例如，填充顺序数据至一批最大长度。查看更多关于`collat​​e_fn`
本节[HTG5。

### 禁用自动配料

在某些情况下，用户可能希望将在数据集代码手动处理配料，或简单地装载单个样品。例如，它可能更便宜直接加载成批数据（例如，批量从数据库中读取或读取的存储器大块连续）或批量大小是依赖于数据的，或者程序被设计为在单个样品工作。在这些情况下，很可能更好，不使用自动配料（其中`
collat​​e_fn`被用来校核的样品），但让所述数据加载器直接返回的`每个成员数据集 `对象。

当两个`的batch_size`和`batch_sampler`是`无 `为（默认值`batch_sampler`已经`无
`），自动配料被禁用。从`数据集获得的每个样品 `与作为`collat​​e_fn`参数传递的功能进行处理。

[HTG0当自动配料被禁用，默认`collat​​e_fn`简单地NumPy的阵列转换成PyTorch张量，并保持所有其他不变。

In this case, loading from a map-style dataset is roughly equivalent with:

    
    
    for index in sampler:
        yield collate_fn(dataset[index])
    

and loading from an iterable-style dataset is roughly equivalent with:

    
    
    for data in iter(dataset):
        yield collate_fn(data)
    

查看更多关于`collat​​e_fn`本节[HTG1。

### 与`collat​​e_fn`工作

利用`collat​​e_fn`当自动配料被启用或禁用略有不同。

[HTG0当自动配料被禁用，`collat​​e_fn`被称为与每个单独的数据样本，并且输出从所述数据加载器的迭代得到。在这种情况下，默认`
collat​​e_fn`简单地转换在PyTorch张量NumPy的阵列。

[HTG0当自动配料使能，`collat​​e_fn
`调用与各时刻的数据样本的一个列表。预计到输入样本整理成批处理从数据加载器的迭代得到。本节的其余部分描述了在这种情况下，默认的`collat​​e_fn
`的行为。

例如，如果每个数据样本包括3通道图像和积分类别标签，即，该数据集的每个元素返回一个元组`（图像， class_index） `，默认`
collat​​e_fn`核对这样元组的列表成批处理图像张量的一个元组和批处理类别标签张量。具体地，默认`collat​​e_fn`具有以下性质：

  * 它总是预先考虑一个新的维度批次尺寸。

  * 它自动NumPy的阵列和Python数值转换成PyTorch张量。

  * 它保留的数据结构，例如，如果每个样本是一个字典，它输出具有相同的密钥集合，但分批张量作为值（或列表，如果值不能被转换成张量）的字典。同样为`列表 `S，`元组 `S，`namedtuple`S等

用户可以使用定制`collat​​e_fn`以实现自定义配料，例如，沿除各种长度，或增加对自定义数据类型支撑件的第一，填充序列以外的尺寸核对。

## 单和多进程数据载入

A`的DataLoader`缺省使用单进程数据加载。

内一个Python过程中，[全局解释器锁（GIL）](https://wiki.python.org/moin/GlobalInterpreterLock)防止真正完全并行跨线程Python代码。为了避免与数据加载阻断计算代码，PyTorch提供了一个简单开关通过简单地将参数`
num_workers`设置为一个正整数，以执行多处理数据加载。

### 单进程的数据加载（默认）

在此模式下，数据被取在相同的工艺做了 `的DataLoader`
被初始化。因此，数据加载可能会阻止计算。然而，这种模式可被当处理（例如，共享存储器，文件描述符）之间使用共享数据资源（多个）是有限的，或者当整个数据集是小，并且可以完全在内存加载优选的。另外，单进程加载经常显示更加可读的错误的痕迹，因此对于调试是有用的。

### 多进程数据加载

设置参数`num_workers`作为正整数将接通的多进程数据加载与装载机的工作进程指定的次数。

在这种模式下，每次迭代一个 `的DataLoader`，创建（例如，当调用`枚举（的DataLoader） `），`num_workers
`被创建工作进程。在这一点上，`数据集 `，`collat​​e_fn`和`worker_init_fn
`被传递到每个工人，在那里它们被用来初始化，并获取数据。这意味着，数据集访问其内部IO一起，变换（包括`collat​​e_fn`）在工作进程中运行。

`torch.utils.data.get_worker_info（） `
在一个工作进程返回各种有用的信息（包括工人ID，数据集的副本，初始种子等）在主处理中，并返回`无 `。用户可以在数据集中代码中使用此功能和/或`
worker_init_fn`单独配置每个数据集的副本，并确定该代码是否在工作进程运行。例如，这可以是在分片数据集特别有用。

在地图风格数据集，主处理使用`取样 `产生的索引，并将它们发送给工人。因此，任何洗牌随机化，其中通过分配指标来加载引导加载主进程完成。

对于迭代式的数据集，因为每个工作进程得到`数据集 `对象的副本，幼稚多进程加载通常将导致复制的数据。使用 `
torch.utils.data.get_worker_info（） `和/或`worker_init_fn`中，用户可以配置每个复制品独立。
（参见 `IterableDataset`单证如何实现这一点。）对于类似的原因，在多进程加载时，`drop_last
`参数下降到最后的非整批生产的每个工人的迭代式的数据集副本。

工人被关闭一旦达到迭代结束时，或者当迭代器将变为垃圾收集。

警告

它一般不建议恢复在多进程加载CUDA张量，因为许多微妙之处使用CUDA和多分享CUDA张量（见多处理 [
CUDA）。相反，我们建议使用自动存储器钉扎（即，设置`pin_memory =真
`），这使得能够快速数据传输到支持CUDA的GPU。](notes/multiprocessing.html#multiprocessing-cuda-
note)

#### 特定于平台的行为

由于工人依靠Python的[ `多重处理 `
](https://docs.python.org/3/library/multiprocessing.html#module-
multiprocessing "\(in Python v3.7\)")，工人发射行为是在Windows上的不同比的Unix。

  * 在Unix，`叉（） `为默认[ `多处理 `](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "\(in Python v3.7\)")启动方法。使用`叉（） `，童工通常可以直接通过克隆地址空间中的`数据集 `和Python参数的函数访问。

  * 在Windows中，`产卵（） `为默认[ `多处理 `](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "\(in Python v3.7\)")启动方法。使用`重生（） `，另一种解释是推出是运行在主脚本，然后由接收`数据集 `内部职工功能， `collat​​e_fn`和通过[ `泡菜 `](https://docs.python.org/3/library/pickle.html#module-pickle "\(in Python v3.7\)")序列的其它参数。

这个单独的序列化意味着你应该采取两个步骤，以确保您与Windows兼容，同时使用多进程数据加载：

  * 包裹内`你们中的大多数主要脚本代码，如果 __name__  ==  '__main__'： `块，使确保它不会再次运行（最有可能产生误差）时，每个工作进程启动。您可以将您的数据集和 `的DataLoader`实例创建逻辑在这里，因为它并不需要在工人重新执行。

  * 确保任何自定义`collat​​e_fn`，`worker_init_fn`或`数据集 `代码声明顶层定义，`__main__`检查之外。这确保了他们在工作进程可用。 （这是需要，因为功能酸洗作为参考而已，不是`字节码 `）。

#### 随机性在多进程数据加载

默认情况下，每个工人将具有其PyTorch种子设为`base_seed  +  worker_id`，其中`base_seed
`是一个长期的，通过使用其RNG主过程中产生的（从而，消耗了RNG状态强制）。但是，对于其他种子库可以在初始化工人（W.G.，NumPy的），使每个工人返回相同的随机数被复制。
（参见[ 本部分 ](notes/faq.html#dataloader-workers-random-seed)在FAQ）。

在`worker_init_fn`，则可以访问PyTorch种子集对每个工人用任一 `
torch.utils.data.get_worker_info（）。种子 `或[ `torch.initial_seed（） `
](torch.html#torch.initial_seed "torch.initial_seed")，并用它的数据加载之前种子其他库。

## 存储器钢钉

主机到GPU副本要快得多，当他们从固定（锁定页）内存起源。参见[ 使用固定的内存缓冲区 ](notes/cuda.html#cuda-memory-
pinning)有关何时以及如何一般采用固定内存的更多细节。

为数据加载，使`pin_memory =真 `对 `的DataLoader`
将自动把所获取的数据张量在钉扎存储器，并因此能够更快的数据传输到支持CUDA的GPU。

默认存储器锁定逻辑仅识别张量和地图以及包含张量iterables。默认情况下，如果锁定逻辑看到一个批次是一个自定义类型（这将如果您有发生`
collat​​e_fn
返回一个自定义的间歇式`），或者如果每个元素的批量是一个自定义的类型，钉扎逻辑将无法识别它们，并且它会返回该批次（或那些元件）而没有钉扎的存储器。为了使存储器钉扎定制间歇或数据类型，定义上的自定义类型（多个）`
pin_memory`（）方法。

请参见下面的例子。

例：

    
    
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
    

_class_`torch.utils.data.``DataLoader`( _dataset_ , _batch_size=1_ ,
_shuffle=False_ , _sampler=None_ , _batch_sampler=None_ , _num_workers=0_ ,
_collate_fn=None_ , _pin_memory=False_ , _drop_last=False_ , _timeout=0_ ,
_worker_init_fn=None_ , _multiprocessing_context=None_
)[[source]](_modules/torch/utils/data/dataloader.html#DataLoader)

    

数据加载。结合了数据集和采样，并提供了在给定数据集的迭代。

的 `的DataLoader`同时支持地图风格和迭代式的数据集与单或多进程加载，定制加载顺序和可选的自动配料（对照）和内存牵制。

有关详细信息，请参见 `torch.utils.data`文档页面。

Parameters

    

  * **数据集** （ _数据集_ ） - 从该数据集到加载数据。

  * **的batch_size** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 如何每批许多样品加载（默认值：`1`）。

  * **洗牌** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 设置为`真 `为具有在每个历元改组的数据（默认值：`假 `）。

  * **取样** （ _取样_ _，_ _可选_ ） - 定义从数据集中得出样品的策略。如果指定，`洗牌 `必须`假 [HTG17。`

  * **batch_sampler** （ _取样_ _，_ _可选_ ） - 象`取样 `，但在同一时间返回一批指标。互斥与`的batch_size`，`洗牌 `，`取样 `和`drop_last`。

  * **num_workers** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 多少子过程用于数据加载。 `0`意味着数据将在主处理加载。 （默认值：`0`）

  * **collat​​e_fn** （ _可调用_ _，_ _可选_ ） - 合并的样本的列表，以形成小批量张量（S）的。使用从图式集装批处理时使用。

  * **pin_memory** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，数据装载将在返回之前复制到张量CUDA固定内存。如果数据元素是一个自定义类型，或你的`collat​​e_fn`返回一批即自定义类型，见下面的例子。

  * **drop_last** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 设置为`真 `放弃最后一批不全，如果数据集大小不是由批量大小整除。如果`假 `和数据集的大小是不是批量大小整除，则最后一批将较小。 （默认值：`假 `）

  * **超时** （ _数字_ _，_ _可选_ ） - 如果是阳性的，对于从工人收集一批的超时值。应始终非负。 （默认值：`0`）

  * **worker_init_fn** （ _可调用_ _，_ _可选_ ） - 如果未`无 `，这将是叫上与工人ID每个工人子（在`一个int [0， num_workers  -  1]`）作为输入，在播种之后和数据加载之前。 （默认值：`无 `）

Warning

如果使用`菌种 `启动方法，`worker_init_fn`不能是unpicklable对象，例如，lambda函数。参见[ 多处理最佳实践
](notes/multiprocessing.html#multiprocessing-best-
practices)在PyTorch到多处理相关的更多细节。

Note

`LEN（的DataLoader） `启发式是基于所使用的取样器的长度。当`数据集 `是 `IterableDataset`
，将使用一个无限采样器，其`__len__ （）
`未实现，因为实际的长度取决于两个可迭代以及多进程加载构造。所以，除非他们有地图式的数据集工作，一个不应该查询该方法。参见数据集类型关于这两种类型的数据集的更多细节。

_class_`torch.utils.data.``Dataset`[[source]](_modules/torch/utils/data/dataset.html#Dataset)

    

表示 `数据集 `的抽象类。

表示从键数据样本的地图所有数据集应该继承它。所有子类应该overrite `__getitem __（）
`，支持获取对于给定的密钥数据样本。子类还可以任选地覆盖`__len __（） `，预计由返回的数据集的大小许多 `取样 `实施方式和的 `
的DataLoader`的默认选项。

Note

`的DataLoader`缺省构建一个索引采样能产生整数指数。为了使它与地图式的数据集与非整指数/键的作用，必须提供自定义采样。

_class_`torch.utils.data.``IterableDataset`[[source]](_modules/torch/utils/data/dataset.html#IterableDataset)

    

可迭代的数据集。

代表数据样本的迭代所有数据集应该继承它。当数据来自一个数据集流的这种形式是特别有用的。

所有子类应该overrite `__iter __（） `，这将返回样本的迭代在该数据集。

当一个子类使用具有 `的DataLoader`，在数据集中的每个项目将被从得到的 `的DataLoader`迭代器。当`
num_workers  & GT ;  0`，每个工作进程将具有数据集对象的不同拷贝，因此通常希望独立地配置每个拷贝，以避免从工人返回重复数据。`
get_worker_info（） `，在一个工作进程调用时，返回关于工人的信息。它可以在任一使用的数据集的`__iter __（） `方法或 `
的DataLoader`的`worker_init_fn`选项来修改每个副本的行为。

实施例1：在所有工人分裂工作量`__iter __（） `：

    
    
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
    

实施例2：使用`在所有工人分裂工作量worker_init_fn`：

    
    
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
    
    >>> # Define a `worker_init_fn`that configures each dataset copy differently
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
    

_class_`torch.utils.data.``TensorDataset`( _*tensors_
)[[source]](_modules/torch/utils/data/dataset.html#TensorDataset)

    

数据集包装张量。

每个样品将沿所述第一维度的索引张量进行检索。

Parameters

    

***张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） -
具有所述第一尺寸的大小相同张量。

_class_`torch.utils.data.``ConcatDataset`( _datasets_
)[[source]](_modules/torch/utils/data/dataset.html#ConcatDataset)

    

数据集作为多个数据集的串联。

这个类是组装不同的现有数据集是有用的。

Parameters

    

**数据集** （ _序列_ ） - 数据集的列表要连接

_class_`torch.utils.data.``ChainDataset`( _datasets_
)[[source]](_modules/torch/utils/data/dataset.html#ChainDataset)

    

数据集chainning多个 `IterableDataset`秒。

这个类是组装不同的现有数据集流是有用的。该chainning操作上即时完成的，因此串联与此类大型数据集将是有效的。

Parameters

    

**数据集** （ _IterableDataset_ 的迭代） - 数据集链接在一起

_class_`torch.utils.data.``Subset`( _dataset_ , _indices_
)[[source]](_modules/torch/utils/data/dataset.html#Subset)

    

在指定的索引数据集的子集。

Parameters

    

  * **数据集** （ _数据集_ ） - 整个数据集

  * **指数** （ _序列_ ） - 在整个组索引选择的子集

`torch.utils.data.``get_worker_info`()[[source]](_modules/torch/utils/data/_utils/worker.html#get_worker_info)

    

返回当前 `的DataLoader`迭代工作进程的信息。

当一个工人叫，这将返回保证具有以下属性的对象：

  * `ID`：当前作业人员ID。

  * `num_workers`：工人的总数。

  * `种子 `：当前工人随机种子集。此值由主进程RNG和工人的ID来确定。参见 `的DataLoader`的更多细节的文档。

  * `数据集 `：数据集对象在 **这里** 过程的副本。请注意，这将是在不同的进程比一个主处理不同的对象。

当主过程调用，这将返回`无 `。

Note

当所使用的`worker_init_fn`传递到 `的DataLoader`，该方法可以是设置每个工人有用过程不同，例如，使用`
worker_id`配置`数据集 `目的是只读分片数据集的特定部分，或使用`种子 `种子中的数据集的代码（例如，NumPy的）使用其他文库。

`torch.utils.data.``random_split`( _dataset_ , _lengths_
)[[source]](_modules/torch/utils/data/dataset.html#random_split)

    

随机分割数据集到给定长度的非重叠的新的数据集。

Parameters

    

  * **数据集** （ _数据集_ ） - 数据集要被分割

  * **长度** （ _序列_ ） - 要产生裂缝的长度

_class_`torch.utils.data.``Sampler`( _data_source_
)[[source]](_modules/torch/utils/data/sampler.html#Sampler)

    

基类的所有取样。

每采样的子类必须提供一个`__iter __（） `的方法，提供一种方式来迭代数据集的元素的索引，和`__len __（）
`方法，它返回所返回的迭代器的长度。

Note

的`__len __（） `方法并不严格 `的DataLoader`必需的，但在涉及任何计算预期的 `的DataLoader`的长度。

_class_`torch.utils.data.``SequentialSampler`( _data_source_
)[[source]](_modules/torch/utils/data/sampler.html#SequentialSampler)

    

顺序地将样品的元素，总是以相同的顺序。

Parameters

    

**DATA_SOURCE** （ _数据集_ ） - 数据集以从采样

_class_`torch.utils.data.``RandomSampler`( _data_source_ , _replacement=False_
, _num_samples=None_
)[[source]](_modules/torch/utils/data/sampler.html#RandomSampler)

    

样品元件中随机。如果不更换，然后从一个洗牌的数据集进行采样。如果具有置换，然后用户可指定`num_samples`绘制。

Parameters

    

  * **data_source** ( _Dataset_) – dataset to sample from

  * **替换** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 样品绘制替换如果`真 `，默认=``False``

  * **num_samples** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 样本的数目来绘制，默认=`LEN（数据集）`。该参数应该当替换是`真 `仅被指定。

_class_`torch.utils.data.``SubsetRandomSampler`( _indices_
)[[source]](_modules/torch/utils/data/sampler.html#SubsetRandomSampler)

    

随机样本元素从指数的定列表，无需更换。

Parameters

    

**指数** （ _序列_ ） - 索引的序列

_class_`torch.utils.data.``WeightedRandomSampler`( _weights_ , _num_samples_ ,
_replacement=True_
)[[source]](_modules/torch/utils/data/sampler.html#WeightedRandomSampler)

    

从`样品元素[0，..，LEN（权重）-1]`与给定的概率（权重）。

Parameters

    

  * **权重** （ _序列_ ） - 权重的顺序，没有必要总结到一个

  * **num_samples** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 样本的数目来绘制

  * **替换** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `，样品绘制更换。如果不是，他们绘制无需更换，这意味着当指数样本绘制为行，不能再为该行画出。

例

    
    
    >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
    [0, 0, 0, 1, 0]
    >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
    [0, 1, 4, 3, 2]
    

_class_`torch.utils.data.``BatchSampler`( _sampler_ , _batch_size_ ,
_drop_last_ )[[source]](_modules/torch/utils/data/sampler.html#BatchSampler)

    

包装另一个采样，以产生小批量指数。

Parameters

    

  * **取样** （ _取样_ ） - 基采样器。

  * **的batch_size** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 小批量的大小。

  * **drop_last** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `，采样器将下降的最后一批，如果它的规模将是小于`的batch_size`

Example

    
    
    >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    

_class_`torch.utils.data.distributed.``DistributedSampler`( _dataset_ ,
_num_replicas=None_ , _rank=None_ , _shuffle=True_
)[[source]](_modules/torch/utils/data/distributed.html#DistributedSampler)

    

取样器，限制数据加载到数据集的一个子集。

它与[ `torch.nn.parallel.DistributedDataParallel`
](nn.html#torch.nn.parallel.DistributedDataParallel
"torch.nn.parallel.DistributedDataParallel")结合特别有用。在这种情况下，每个过程可以通过一个DistributedSampler实例作为的DataLoader采样器，并加载原始数据集即排它的一个子集。

Note

数据集被认为是恒定的大小。

Parameters

    

  * **数据集** \- 数据集用于采样。

  * **num_replicas** （ _可选_ ） - 的参与分布式训练的进程数。

  * **秩** （ _可选_ ） - num_replicas内的当前过程的秩。

  * **洗牌** （ _可选_ ） - 如果为true（默认值），采样器将会洗牌指数

[Next ![](_static/images/chevron-right-orange.svg)](dlpack.html
"torch.utils.dlpack") [![](_static/images/chevron-right-orange.svg)
Previous](cpp_extension.html "torch.utils.cpp_extension")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * torch.utils.data 
    * 数据集类型
      * 地图式的数据集
      * 可迭代式的数据集
    * 数据加载顺序和`取样 `
    * 装载成批和非成批数据
      * 自动配料（默认值）
      * 禁止自动配料
      * 与`工作collat​​e_fn`
    * 单和多过程数据载入
      * 单处理的数据加载（默认）
      * 多进程数据加载
        * 特定于平台的行为
        * 随机性在多进程数据加载
    * [HTG0存储器钢钉

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

