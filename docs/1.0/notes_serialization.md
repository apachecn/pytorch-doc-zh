# 序列化的相关语义

> 译者：[yuange250](https://github.com/yuange250)

## 最佳方案

### 保存模型的推荐方法

Pytorch主要有两种方法可用于序列化和保存一个模型。

第一种只存取模型的参数(更为推荐）：
保存参数：

```py
torch.save(the_model.state_dict(), PATH)

```

读取参数：

```py
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))

```

第二种方法则将整个模型都保存下来：

```py
torch.save(the_model, PATH)

```

读取的时候也是读取整个模型：

```py
the_model = torch.load(PATH)

```

在第二种方法中, 由于特定的序列化的数据与其特定的类别(class)相绑定，并且在序列化的时候使用了固定的目录结构，所以在很多情况下，如在其他的一些项目中使用，或者代码进行了较大的重构的时候，很容易出现问题。
