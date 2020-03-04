# 广播语义  

> 译者：[冯宝宝](https://github.com/PEGASUS1993)
>
> 校验：[AlexJakin](https://github.com/AlexJakin)

许许多多的PyTorch操作都支持[`NumPy Broadcasting Semantics`](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#module-numpy.doc.broadcasting "(in NumPy v1.15)")。  

简而言之，如果PyTorch操作支持广播，那么它的Tensor参数可以自动扩展为相同的类型大小(不需要复制数据）。  

## 一般语义  

如果遵守以下规则，则两个张量是“可广播的”：  

* 每个张量至少有一个维度；
* 遍历张量维度大小时，从末尾随开始遍历，两个张量的维度满足以下条件之一：大小必须相等、它们其中一个为1、他们其中一个不存在。  

例如：  

```py
>>> x=torch.empty(5,7,3)
>>> y=torch.empty(5,7,3)
# 相同形状的张量可以被广播(上述规则总是成立的)

>>> x=torch.empty((0,))
>>> y=torch.empty(2,2)
# x和y不能被广播,因为x没有维度

# can line up trailing dimensions
>>> x=torch.empty(5,3,4,1)
>>> y=torch.empty(  3,1,1)
# x和y能够广播.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist

# 但是:
>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(  3,1,1)
# x和y不能被广播   (  )  

```  

如果x,y两个张量是可以广播的，则通过计算得到的张量大小遵循以下原则：   

* 如果x和y的维数不相等，则在维度较小的张量的前面增加1个维度，使它们的长度相等。  
* 然后,生成新张量维度的大小是x和y在每个维度上的最大值。  
   
例如：   

```py
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

## In - place 语义   

一个复杂因素是in-place操作不允许in-place张量像广播那样改变形状。  

例如：  

```py
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

## 向后兼容性  

PyTorch的早期版本允许某些逐点函数在具有不同形状的张量上执行，只要每个张量中的元素数量相等即可。 然后通过将每个张量视为1维来执行逐点运算。PyTorch现在支持广播，并且“1维”逐点行为被认为已弃用，并且在张量不可广播但具有相同数量的元素的情况下将生成Python警告。  

注意，在两个张量不具有相同形状但是可广播并且具有相同数量元素的情况下，广播的引入可能导致向后不兼容。例如：    

```py
>>> torch.add(torch.ones(4,1), torch.randn(4))

```  

以前可能会产生一个torch.Size([4,1]）的Tensor，但现在会产生一个torch.Size([4,4]）这样的Tensor。 为了帮助识别代码中可能存在广播引起的向后不兼容性的情况，您可以设置`torch.utils.backcompat.broadcast_warning.enabled` 为 `True`，在这种情况下会产生python警告。  

```py
>>> torch.utils.backcompat.broadcast_warning.enabled=True
>>> torch.add(torch.ones(4,1), torch.ones(4))
__main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, and have the same number of elements.
Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 1-dimensional.

```
