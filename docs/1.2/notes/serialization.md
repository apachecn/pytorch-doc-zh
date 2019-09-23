# 序列化语义

## 最佳实践

### 用于保存模型的推荐方法

有序列化和恢复模型的两种主要方法。

第一个（推荐）保存并只加载模型参数：

    
    
    torch.save(the_model.state_dict(), PATH)
    

再后来：

    
    
    the_model = TheModelClass(*args, **kwargs)
    the_model.load_state_dict(torch.load(PATH))
    

第二保存和加载整个模型：

    
    
    torch.save(the_model, PATH)
    

Then later:

    
    
    the_model = torch.load(PATH)
    

然而，在这种情况下，序列化的数据绑定到特定的类和使用的准确的目录结构，所以在其他项目中使用时，它可以通过各种方式突破，或在一些严重的refactors。

[Next ![](../_static/images/chevron-right-orange.svg)](windows.html "Windows
FAQ") [![](../_static/images/chevron-right-orange.svg)
Previous](randomness.html "Reproducibility")

* * *

©版权所有2019年，Torch 贡献者。
