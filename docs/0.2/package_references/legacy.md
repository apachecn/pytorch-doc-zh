# 遗产包 - torch.legacy
此包中包含从Lua Torch移植来的代码。

为了可以使用现有的模型并且方便当前Lua Torch使用者过渡，我们创建了这个包。 可以在`torch.legacy.nn`中找到`nn`代码，并在`torch.legacy.optim`中找到`optim`代码。 API应该完全匹配Lua Torch。
