# 广播语义 [¶](#broadcasting-semantics "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/broadcasting>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/broadcasting.html>


 许多 PyTorch 操作支持 NumPy 的广播语义。有关详细信息，请参阅 <https://numpy.org/doc/stable/user/basics.broadcasting.html>。


 简而言之，如果 PyTorch 操作支持广播，那么它的 Tensor 参数可以自动扩展为相同的大小(无需复制数据)。


## 通用语义 [¶](#general-semantics "此标题的永久链接")


 如果满足以下规则，则两个张量是“可广播的”：



* 每个张量至少有一个维度。 
* 当迭代维度大小时，从尾随维度开始，维度大小必须相等，其中之一为 1，或者其中之一不存在。


 例如：


```
>>> x=torch.empty(5,7,3)
>>> y=torch.empty(5,7,3)
# same shapes are always broadcastable (i.e. the above rules always hold)

>>> x=torch.empty((0,))
>>> y=torch.empty(2,2)
# x and y are not broadcastable, because x does not have at least 1 dimension

# can line up trailing dimensions
>>> x=torch.empty(5,3,4,1)
>>> y=torch.empty(  3,1,1)
# x and y are broadcastable.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist

# but:
>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(  3,1,1)
# x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3

```


 如果两个张量“x”、“y”是“可广播的”，则结果张量大小计算如下：



* 如果`x`和`y`的维数不相等，则在维数较少的张量的维数前面加上1，使它们的长度相等。*然后，对于每个维数大小，得到的维数大小是以下的最大值沿该维度的“x”和“y”的大小。


 例如：


```
# can line up trailing dimensions to make reading easier
>>> x=torch.empty(5,1,4,1)
>>> y=torch.empty(  3,1,1)
>>> (x+y).size()
torch.Size([5, 3, 4, 1])

# but not necessary:
>>> x=torch.empty(1)
>>> y=torch.empty(3,1,7)
>>> (x+y).size()
torch.Size([3, 1, 7])

>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(3,1,1)
>>> (x+y).size()
RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1

```


## 就地语义 [¶](#in-place-semantics "此标题的永久链接")


 一个复杂之处是就地操作不允许就地张量因广播而改变形状。


 例如：


```
>>> x=torch.empty(5,3,4,1)
>>> y=torch.empty(3,1,1)
>>> (x.add_(y)).size()
torch.Size([5, 3, 4, 1])

# but:
>>> x=torch.empty(1,3,1)
>>> y=torch.empty(3,1,7)
>>> (x.add_(y)).size()
RuntimeError: The expanded size of the tensor (1) must match the existing size (7) at non-singleton dimension 2.

```


## 向后兼容性 [¶](#backwards-compatibility "此标题的永久链接")


 PyTorch 的早期版本允许在不同形状的张量上执行某些逐点函数，只要每个张量中的元素数量相等。然后通过将每个张量视为一维来执行逐点运算。 PyTorch 现在支持广播，并且“一维”逐点行为被视为已弃用，并且在张量不可广播但具有相同数量的元素的情况下将生成 Python 警告。


 请注意，在两个张量不具有相同形状，但可广播且具有相同数量元素的情况下，引入广播可能会导致向后不兼容的更改。例如：


```
>>> torch.add(torch.ones(4,1), torch.randn(4))

```


 以前会生成大小为 torch.Size([4,1]) 的张量，但现在生成大小为 torch.Size([4,4]) 的张量。为了帮助识别代码中向后不兼容的情况可能存在广播引入的情况，您可以将 torch.utils.backcompat.broadcast_warning.enabled 设置为 True ，在这种情况下会生成 python 警告。


 例如：


```
>>> torch.utils.backcompat.broadcast_warning.enabled=True
>>> torch.add(torch.ones(4,1), torch.ones(4))
__main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, and have the same number of elements.
Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 1-dimensional.

```