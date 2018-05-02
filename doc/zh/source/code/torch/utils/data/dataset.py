import bisect


class Dataset(object):
    """表示 Dataset 的抽象类.

    所有其它数据集都应继承该类(进行子类化). 所有子类应该 override
    ``__len__`` 和 ``__getitem__``, 前者返回数据集大小，后者提供了
    支持从 0 到 len(self) 整数索引的方法.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class TensorDataset(Dataset):
    """包装数据和目标张量的数据集.

    通过沿着第一个维度索引两个张量来恢复每个样本.

    Args:
        data_tensor (Tensor): 包含样本数据.
        target_tensor (Tensor): 包含样本目标 (标签).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class ConcatDataset(Dataset):
    """
    用以连结多个数据集的数据集.  
    目的: 因为串联操作是以即时方式完成的对于组装不同的现有数据集(可能是
    大规模的数据集)非常有帮助

    Args:
        datasets (iterable): 需要连结的数据集列表
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cummulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cummulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cummulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cummulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
