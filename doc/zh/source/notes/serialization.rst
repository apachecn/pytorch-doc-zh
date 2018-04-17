
序列化语义
=======================

最佳实践
--------------

.. _recommend-saving-models:

保存模型的推荐方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有两种主要的方法可以用来序列化和恢复模型.

第一种方法 (推荐) , 只保存和加载模型的参数::

    torch.save(the_model.state_dict(), PATH)

然后::

    the_model = TheModelClass(*args, **kwargs)
    the_model.load_state_dict(torch.load(PATH))

第二种方法, 保存和加载整个模型::

    torch.save(the_model, PATH)

然后::

    the_model = torch.load(PATH)

但是在这种情况下, 序列化的数据与特定的类和固定的目录结构绑定, 所以当它被用于其他项目中, 或者经过一些重大的重构之后, 可能会以各种各样的方式崩掉.
