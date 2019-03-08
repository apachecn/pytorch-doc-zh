# Legacy package - torch.legacy

> 译者：[@那伊抹微笑](https://github.com/wangyangting)
> 
> 校对者：[@smilesboy](https://github.com/smilesboy)

包含从 Lua torch 移植的代码的包.

为了能够与现有的模型一起工作, 并简化当前 Lua torch 用户的过渡, 我们特意创建了这个包. 您可以在 `torch.legacy.nn` 中找到 `nn` 代码, 并在 `torch.legacy.optim` 中进行 `optim` 优化. 该 API 应该完适配 Lua torch.