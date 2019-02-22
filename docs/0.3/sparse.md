# torch.sparse

> 译者：[@王帅](https://github.com/sirwangshuai)
> 
> 校对者：[@Timor](https://github.com/timors)

警告：

此 API 目前是实验性的 , 可能会在不久的将来发生变化 .

Torch 支持 COO(rdinate) 格式的稀疏张量 , 还能高效地存储和处理大多数元素为零的 张量 .

一个稀疏张量可以表示为一对稠密张量 : 一个张量的值和一个二维张量的指数 . 通过提供这两个张量以及稀疏张量的大小 (不能从这些张量推断!) , 可以构造一个稀疏张量 . 假设我们要在位置 (0,2) 处定义条目3 , 位置 (1,0) 的条目4 , 位置 (1,2) 的条目5的 稀疏张量 , 我们可以这样写 :

```py
>>> i = torch.LongTensor([[0, 1, 1],
 [2, 0, 2]])
>>> v = torch.FloatTensor([3, 4, 5])
>>> torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense()
 0  0  3
 4  0  5
[torch.FloatTensor of size 2x3]

```

请注意 , LongTensor 的传入参数不是索引元组的列表 . 如果你想用这种方式编写索引 , 你应该在 将它们传递给稀疏构造函数之前进行转换 :

```py
>>> i = torch.LongTensor([[0, 2], [1, 0], [1, 2]])
>>> v = torch.FloatTensor([3,      4,      5    ])
>>> torch.sparse.FloatTensor(i.t(), v, torch.Size([2,3])).to_dense()
 0  0  3
 4  0  5
[torch.FloatTensor of size 2x3]

```

你还可以构造混合稀疏张量 , 其中只有第一个n维是稀疏的 , 而其余维度是密集的 .

```py
>>> i = torch.LongTensor([[2, 4]])
>>> v = torch.FloatTensor([[1, 3], [5, 7]])
>>> torch.sparse.FloatTensor(i, v).to_dense()
 0  0
 0  0
 1  3
 0  0
 5  7
[torch.FloatTensor of size 5x2]

```

一个空的稀疏张量可以通过指定它的大小来构造 :

```py
>>> torch.sparse.FloatTensor(2, 3)
SparseFloatTensor of size 2x3 with indices:
[torch.LongTensor with no dimension]
and values:
[torch.FloatTensor with no dimension]

```

注解：

我们的稀疏张量格式允许非聚合稀疏张量 , 索引可能对应有重复的坐标 ; 在这 种情况下 , 该索引处的值代表所有重复条目值的总和 . 非聚合张量允许我们更 有效地实现确定的操作符 .

在大多数情况下 , 你不必关心稀疏张量是否聚合 , 因为大多数操作在聚合或 不聚合稀疏张量的情况下都会以相同的方式工作 . 但是 , 你可能需要关心两种情况 .

首先 , 如果你反复执行可以产生重复条目的操作 (例如 , `torch.sparse.FloatTensor.add()`) , 则应适当聚合稀疏张量以防止它们变得太大.

其次 , 一些操作符将根据是否聚合 (例如 , `torch.sparse.FloatTensor._values()` 和 `torch.sparse.FloatTensor._indices()` , 还有 `torch.Tensor._sparse_mask()`) 来生成不同的值 . 这些运算符前面加下划线表示它们揭示 内部实现细节 , 因此应谨慎使 , 因为与聚合的稀疏张量一起工作的代码可能不适用于未聚合的稀疏张量 ; 一般来说 , 在运用这些运算符之前 , 最安全的就是确保是聚合的 .

例如 , 假设我们想直接通过 `torch.sparse.FloatTensor._values()` 来实现一个操作 . 随着乘法分布的增加 , 标量的乘法可以轻易实现 ; 然而 , 平方根不能直接实现 , `sqrt(a + b) != sqrt(a) +sqrt(b)` (如果给定一个非聚合张量 , 这将被计算出来 . )

```py
class torch.sparse.FloatTensor
```

```py
add()
```

```py
add_()
```

```py
clone()
```

```py
dim()
```

```py
div()
```

```py
div_()
```

```py
get_device()
```

```py
hspmm()
```

```py
mm()
```

```py
mul()
```

```py
mul_()
```

```py
resizeAs_()
```

```py
size()
```

```py
spadd()
```

```py
spmm()
```

```py
sspaddmm()
```

```py
sspmm()
```

```py
sub()
```

```py
sub_()
```

```py
t_()
```

```py
toDense()
```

```py
transpose()
```

```py
transpose_()
```

```py
zero_()
```

```py
coalesce()
```

```py
is_coalesced()
```

```py
_indices()
```

```py
_values()
```

```py
_nnz()
```