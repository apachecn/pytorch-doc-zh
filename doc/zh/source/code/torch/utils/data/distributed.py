import math
import torch
from .sampler import Sampler
from torch.distributed import get_world_size, get_rank


class DistributedSampler(Sampler):
    """将数据加载限制为数据集子集的采样器.

    当与 :class:`torch.nn.parallel.DistributedDataParallel` 组合使用时，效果较好.
    在这种情况下, 每个进程都可以将分布式采样器实例作为Data Loader采样器,
    并且加载一个原始数据集的子集并独占该数据子集.

    .. note::
        数据集被假定为不变的大小.

    Args:
        dataset: 采样的数据集.
        num_replicas (optional): 参与分布式训练的进程数量.
        rank (optional): 在 num_replicas 中, 当前进程的等级.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
