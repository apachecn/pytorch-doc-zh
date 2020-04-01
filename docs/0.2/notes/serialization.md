# 序列化语义
## 最佳实践
### 保存模型的推荐方法
这主要有两种方法序列化和恢复模型。

第一种(推荐）只保存和加载模型参数：
```python
torch.save(the_model.state_dict(), PATH)
```
然后：
```python
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```
第二种保存和加载整个模型：
```python
torch.save(the_model, PATH)
```
然后：
```python
the_model = torch.load(PATH)
```
然而，在这种情况下，序列化的数据被绑定到特定的类和固定的目录结构，所以当在其他项目中使用时，或者在一些严重的重构器之后它可能会以各种方式break。
