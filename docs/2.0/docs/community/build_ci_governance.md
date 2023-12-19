# PyTorch 治理 | Build + CI [¶](#pytorch-governance-build-ci "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/community/build_ci_governance>
>
> 原始地址：<https://pytorch.org/docs/stable/community/build_ci_governance.html>


## 如何添加新的维护者 [¶](#how-to-add-a-new-maintainer "永久链接到此标题")


 要成为维护者，一个人需要：



* 将至少六次提交提交到 PyTorch 存储库的相关部分
* 必须在过去六个月内提交这些提交中的至少一项


 要将合格人员添加到维护者列表中，请创建一个 PR，将人员添加到 [感兴趣的人](https://pytorch.org/docs/main/community/persons_of_interest.html) 页面和 [merge_rules ](https://github.com/pytorch/pytorch/blob/main/.github/merge_rules.yaml) 文件。当前的维护者将投票支持。批准 PR 的决策标准：



* 合并前不早于两个工作日(确保大多数贡献者已经看到)
* PR 具有正确的标签(模块：ci )
* 当前维护者没有反对
* 至少有三个来自当前维护人员的净*赞数*（或者当模块的维护人员少于3个时，所有维护人员都投*赞成票*）。
