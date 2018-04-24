import torch
from ._utils import _type, _cuda, _range


class _StorageBase(object):
    is_cuda = False
    is_sparse = False

    def __str__(self):
        content = ' ' + '\n '.join(str(self[i]) for i in _range(len(self)))
        return content + '\n[{} of size {}]'.format(torch.typename(self), len(self))

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], _range(self.size())))

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def __reduce__(self):
        return type(self), (self.tolist(),)

    def clone(self):
        """返回此存储的一个副本"""
        return type(self)(self.size()).copy_(self)

    def tolist(self):
        """返回一个包含此存储中的元素的列表"""
        return [v for v in self]

    def cpu(self):
        """如果当前此存储不在CPU上 , 则返回一个它的CPU副本 . """
        return self.type(getattr(torch, self.__class__.__name__))

    def double(self):
        """将此存储转换为 double 类型"""
        return self.type(type(self).__module__ + '.DoubleStorage')

    def float(self):
        """将此存储转换为 float 类型"""
        return self.type(type(self).__module__ + '.FloatStorage')

    def half(self):
        """将此存储转换为 half 类型"""
        return self.type(type(self).__module__ + '.HalfStorage')

    def long(self):
        """将此存储转换为 long 类型"""
        return self.type(type(self).__module__ + '.LongStorage')

    def int(self):
        """将此存储转换为 int 类型"""
        return self.type(type(self).__module__ + '.IntStorage')

    def short(self):
        """将此存储转换为 short 类型"""
        return self.type(type(self).__module__ + '.ShortStorage')

    def char(self):
        """将此存储转换为 char 类型"""
        return self.type(type(self).__module__ + '.CharStorage')

    def byte(self):
        """将此存储转换为 byte 类型"""
        return self.type(type(self).__module__ + '.ByteStorage')

    def pin_memory(self):
        """如果此存储当前未被锁定 , 则将它复制到锁定内存中 . """
        if self.is_cuda:
            raise TypeError("cannot pin '{0}' only CPU memory can be pinned"
                            .format(self.type()))
        import torch.cuda
        allocator = torch.cuda._host_allocator()
        return type(self)(self.size(), allocator=allocator).copy_(self)

    def share_memory_(self):
        """将存储移动到共享内存中 . 

        这对于已经存在于共享内存中的存储或者 CUDA 存储无效 , 它们不需要移动就能在进程间共享 . 
        共享内存中的存储不能调整大小 . 

        Returns: self
        """
        from torch.multiprocessing import get_sharing_strategy
        if self.is_cuda:
            pass  # CUDA doesn't use POSIX shared memory
        elif get_sharing_strategy() == 'file_system':
            self._share_filename_()
        else:
            self._share_fd_()
        return self

    @classmethod
    def _new_shared(cls, size):
        """在共享内存中创建一个新的相同类型的存储"""
        from torch.multiprocessing import get_sharing_strategy
        if cls.is_cuda:
            return cls(size)
        elif get_sharing_strategy() == 'file_system':
            return cls._new_using_filename(size)
        else:
            return cls._new_using_fd(size)


_StorageBase.type = _type
_StorageBase.cuda = _cuda
