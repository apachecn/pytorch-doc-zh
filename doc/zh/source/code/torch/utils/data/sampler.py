import torch


class Sampler(object):
    """所有采样器的基类.

    每一个 Sampler 的子类都必须提供一个  __iter__ 方法, 提供一种
    迭代数据集元素的索引的方法, 以及一个 __len__ 方法, 用来返回
    迭代器的长度.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    """总是以相同的顺序, 依次对元素进行采样.

    Args:
        data_source (Dataset): 采样的数据集
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """采用无放回采样法, 随机对样本元素采样.

    Args:
        data_source (Dataset): 采样的数据集
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long())

    def __len__(self):
        return len(self.data_source)


class SubsetRandomSampler(Sampler):
    """采用无放回采样法, 样本元素从指定的索引列表中随机抽取.

    Args:
        indices (list): 索引的列表
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    """使用给定的概率 (权重) 对 [0,..,len(weights)-1] 范围的元素进行采样.

    Args:
        weights (list)   : 权重列表, 没必要加起来等于 1
        num_samples (int): 抽样数量
        replacement (bool): 设定为 ``True``, 使用有放回采样法.
            设定为 ``False``, 采用无放回采样法, 这意味着对于一行来说,当一个
            样本索引被取到后, 对于改行, 这个样本索引不能再次被取到.
    """

    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples


class BatchSampler(object):
    """包装另一个采样器以迭代产生一个索引的 mini-batch.

    Args:
        sampler (Sampler): 基采样器.
        batch_size (int): mini-batch 的大小.
        drop_last (bool): 设定为 ``True``, 如果最后一个 batch 的大小
            比 ``batch_size`` 小, 则采样器会丢掉最后一个 batch .

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
