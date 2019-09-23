# 与实施例学习PyTorch

**作者** ：[贾斯汀·约翰逊](https://github.com/jcjohnson/pytorch-examples)

这个教程通过自包含的实施例引入了[的基本概念PyTorch ](https://github.com/pytorch/pytorch)。

在其核心，PyTorch提供了两个主要特点：

  * n维张量，类似于numpy的，但可以在GPU上运行
  * 自动分化为建设和培训的神经网络

我们将使用全连接RELU网络我们当前实例。该网络将具有单个隐藏层，并且将与梯度下降来训练通过最小化网络输出和真实输出之间的欧几里得距离，以适应随机数据。

注意

您可以在此页面 的 年底浏览个别的例子。

目录

  * 张量
    * 热身：numpy的
    * PyTorch：张量
  * Autograd 
    * PyTorch：张量和autograd 
    * PyTorch：定义新autograd功能
    * TensorFlow：静态图形
  * NN 模块
    * PyTorch：NN 
    * PyTorch：的Optim 
    * PyTorch：自定义ン模块
    * PyTorch：控制流+重量共享
  * 实施例
    * 张量
    * Autograd 
    * NN 模块

## 张量

### 热身：numpy的

引入PyTorch之前，我们将使用numpy的第一个实施网络。

numpy的提供了一个n维阵列对象，并且许多功能用于操纵这些阵列。
NumPy的是科学计算的通用框架;不知道计算图形，或深学习，或渐变什么。然而，我们可以很容易地使用numpy的通过手动执行前进到适合的两层网络随机数据并通过网络使用numpy的操作向后传递：

    
    
    # -*- coding: utf-8 -*-
    import numpy as np
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # Create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)
    
    # Randomly initialize weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)
    
    learning_rate = 1e-6
    for t in range(500):
        # Forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)
    
        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        print(t, loss)
    
        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)
    
        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    

###  PyTorch：张量

numpy的是一个伟大的框架，但它不能利用GPU来加速其数值计算。对于现代的深层神经网络，图形处理器通常提供的[
50倍以上](https://github.com/jcjohnson/cnn-benchmarks)的加速，所以很遗憾numpy的将是不够的现代深度学习。

在这里，我们介绍的最根本PyTorch概念：
**张量[HTG1。甲PyTorch张量是概念性地等同于numpy的数组：一个张量是n维阵列，并且PyTorch关于这些张量的操作提供了许多功能。在幕后，张量可以跟踪的计算图表和渐变的，但他们也为科学计算的通用工具是有用的。**

也不像numpy的，PyTorch张量可以利用GPU来加速他们的数值计算。要在GPU运行PyTorch张量，只需将它转换为新的数据类型。

这里我们使用PyTorch张量，以适应​​二层网络的随机数据。像numpy的上面的例子，我们需要手动执行向前和向后的通过网络：

    
    
    # -*- coding: utf-8 -*-
    
    import torch
    
    
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    
    # Randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)
    
    learning_rate = 1e-6
    for t in range(500):
        # Forward pass: compute predicted y
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)
    
        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)
    
        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)
    
        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    

##  Autograd 

###  PyTorch：张量和autograd 

在上面的例子中，我们必须手动实现正向和我们的神经网络的落后通行证。手动实现后向通行是不是什么大不了的一个小二层网络，但可以迅速得到大型的复杂网络非常有毛。

幸运的是，我们可以使用[自动分化](https://en.wikipedia.org/wiki/Automatic_differentiation)自动化神经网络的向后传递的计算。在PyTorch的
**autograd** 包提供的正是这种功能。当使用autograd，网络的直传将定义一个 **计算图表**
;图中的节点将是张量，并且边缘将是产生从输入输出张量张量的功能。通过这个图表Backpropagating然后让你轻松计算梯度。

这听起来很复杂，这是很简单的做法是使用。各张量表示在计算图中的节点。如果`× `是一个张量，其具有`x.requires_grad =真 `然后`
x.grad`的另一张量保持的`× `梯度相对于一些标量值。

这里我们使用PyTorch张量和autograd来实现我们的两层网络;现在我们不再需要手动实现通过网络向通行：

    
    
    # -*- coding: utf-8 -*-
    import torch
    
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # Create random Tensors to hold input and outputs.
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Tensors during the backward pass.
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    
    # Create random Tensors for weights.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
    
    learning_rate = 1e-6
    for t in range(500):
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
    
        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())
    
        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()
    
        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
    
            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()
    

###  PyTorch：定义新autograd功能

在内部，每个基元autograd操作者实际上是两个函数的张量进行操作。的 **向前** 函数从输入张量计算输出张量。的 **向后**
功能接收输出张量的梯度相对于一些标量值，并计算输入张量的梯度相对于该相同标量值。

在PyTorch我们可以很容易地通过定义`torch.autograd.Function`一个子类，并实现了`转发
`和定义我们自己autograd操作`向后
`功能。然后，我们可以通过构造一个实例，并调用它像一个函数，传递一个包含输入数据的张量使用我们的新autograd运营商。

在这个例子中，我们定义我们自己的自定义autograd功能进行RELU非线性，并用它来实现我们的两层网络：

    
    
    # -*- coding: utf-8 -*-
    import torch
    
    
    class MyReLU(torch.autograd.Function):
        """
        We can implement our own custom autograd Functions by subclassing
        torch.autograd.Function and implementing the forward and backward passes
        which operate on Tensors.
        """
    
        @staticmethod
        def forward(ctx, input):
            """
            In the forward pass we receive a Tensor containing the input and return
            a Tensor containing the output. ctx is a context object that can be used
            to stash information for backward computation. You can cache arbitrary
            objects for use in the backward pass using the ctx.save_for_backward method.
            """
            ctx.save_for_backward(input)
            return input.clamp(min=0)
    
        @staticmethod
        def backward(ctx, grad_output):
            """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
            """
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            return grad_input
    
    
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # Create random Tensors to hold input and outputs.
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    
    # Create random Tensors for weights.
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
    
    learning_rate = 1e-6
    for t in range(500):
        # To apply our Function, we use Function.apply method. We alias this as 'relu'.
        relu = MyReLU.apply
    
        # Forward pass: compute predicted y using operations; we compute
        # ReLU using our custom autograd operation.
        y_pred = relu(x.mm(w1)).mm(w2)
    
        # Compute and print loss
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())
    
        # Use autograd to compute the backward pass.
        loss.backward()
    
        # Update weights using gradient descent
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
    
            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()
    

###  TensorFlow：静态图形

PyTorch
autograd看起来很像TensorFlow：在两个框架，我们定义的计算图表，并使用自动微分计算梯度。两者之间最大的区别是，TensorFlow的计算图是
**静态** 和PyTorch使用 **动态** 计算图表。

在TensorFlow，我们定义了计算图形一次，然后一遍一遍执行相同的曲线图，可能供给不同的输入数据提供给图。在PyTorch，每个直传定义了一个新的计算曲线图。

静态图是很好的，因为你可以在前面优化图形;例如框架可能决定一些融合图的运算效率，还是拿出了在许多的GPU或者多台机器分布图的策略。如果你一遍又一遍地重复使用同一张图上，那么这种潜在的昂贵的前期优化，可以摊销在同一张图中一遍又一遍地重新运行。

其中的静态和动态的曲线图不同的一个方面是控制流。对于某些型号我们可能希望对于每个数据点执行不同的计算;例如复发性网络可能被展开为不同数量的每个数据点的时间步;这样展开可以作为一个循环来实现。具有静态图中的循环结构需要是图形的一部分;为此TensorFlow为运营商提供如`
tf.scan`嵌入循环到曲线图。随着动态图形的情况比较简单：因为我们建立在即时每个示例图，我们可以用正常的必要的流量控制来执行计算的是不同的每个输入。

为了与PyTorch对比上述autograd例子，这里我们使用TensorFlow，以适应一个简单的两层网：

    
    
    # -*- coding: utf-8 -*-
    import tensorflow as tf
    import numpy as np
    
    # First we set up the computational graph:
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # Create placeholders for the input and target data; these will be filled
    # with real data when we execute the graph.
    x = tf.placeholder(tf.float32, shape=(None, D_in))
    y = tf.placeholder(tf.float32, shape=(None, D_out))
    
    # Create Variables for the weights and initialize them with random data.
    # A TensorFlow Variable persists its value across executions of the graph.
    w1 = tf.Variable(tf.random_normal((D_in, H)))
    w2 = tf.Variable(tf.random_normal((H, D_out)))
    
    # Forward pass: Compute the predicted y using operations on TensorFlow Tensors.
    # Note that this code does not actually perform any numeric operations; it
    # merely sets up the computational graph that we will later execute.
    h = tf.matmul(x, w1)
    h_relu = tf.maximum(h, tf.zeros(1))
    y_pred = tf.matmul(h_relu, w2)
    
    # Compute loss using operations on TensorFlow Tensors
    loss = tf.reduce_sum((y - y_pred) ** 2.0)
    
    # Compute gradient of the loss with respect to w1 and w2.
    grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
    
    # Update the weights using gradient descent. To actually update the weights
    # we need to evaluate new_w1 and new_w2 when executing the graph. Note that
    # in TensorFlow the the act of updating the value of the weights is part of
    # the computational graph; in PyTorch this happens outside the computational
    # graph.
    learning_rate = 1e-6
    new_w1 = w1.assign(w1 - learning_rate * grad_w1)
    new_w2 = w2.assign(w2 - learning_rate * grad_w2)
    
    # Now we have built our computational graph, so we enter a TensorFlow session to
    # actually execute the graph.
    with tf.Session() as sess:
        # Run the graph once to initialize the Variables w1 and w2.
        sess.run(tf.global_variables_initializer())
    
        # Create numpy arrays holding the actual data for the inputs x and targets
        # y
        x_value = np.random.randn(N, D_in)
        y_value = np.random.randn(N, D_out)
        for t in range(500):
            # Execute the graph many times. Each time it executes we want to bind
            # x_value to x and y_value to y, specified with the feed_dict argument.
            # Each time we execute the graph we want to compute the values for loss,
            # new_w1, and new_w2; the values of these Tensors are returned as numpy
            # arrays.
            loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                        feed_dict={x: x_value, y: y_value})
            if t % 100 == 99:
                print(t, loss_value)
    

##  NN 模块

###  PyTorch：NN 

计算图形和autograd是用于定义复杂的操作人员和自动采取衍生物一个非常强大的范例;然而，对于大的神经网络的原始autograd可以有点太级低。

当建立神经网络，我们经常想安排计算分成 **图层** ，其中一些 **学得的参数** 将在学习过程中优化。

在TensorFlow，如[包Keras ](https://github.com/fchollet/keras)，[
TensorFlow修身](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)和[
TFLearn ](http://tflearn.org/)通过原始的计算图表，是构建神经网络的有用提供更高层次的抽象。

在PyTorch时，`NN`包服务于这个相同的目的。的`NN`包定义了一组模块，它大致相当于神经网络层的
**。一个模块接收输入张量，并且计算输出张量，而且还可以保持内部状态，如含有可以学习参数张量。的`NN
`包还定义了一组训练神经网络时，通常使用的有用的损失函数。**

在这个例子中，我们使用`NN`包来实现我们的两层网络：

    
    
    # -*- coding: utf-8 -*-
    import torch
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    
    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    
    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(reduction='sum')
    
    learning_rate = 1e-4
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x)
    
        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())
    
        # Zero the gradients before running the backward pass.
        model.zero_grad()
    
        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()
    
        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
    

###  PyTorch：的Optim 

到现在为止，我们通过人工变异的张量保持可学习参数（`torch.no_grad（） `或`。数据[HTG6更新了模型的重量]
`以避免在autograd跟踪历史）。这不是简单的优化算法，如随机梯度下降一个巨大的负担，但在实践中，我们经常使用更复杂的优化像AdaGrad，RMSProp，亚当等训练神经网络

的`的Optim`包在PyTorch夺取优化算法的思想，并提供了常用的优化算法的实施。

在这个例子中，我们将使用`NN`封装之前定义我们的模型，但是我们将使用由`的Optim提供的亚当算法[HTG6优化模型] `包：

    
    
    # -*- coding: utf-8 -*-
    import torch
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    
    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    
    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)
    
        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())
    
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()
    
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()
    
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    

###  PyTorch：自定义ン模块

有时你会想指定型号是比现有模块的序列更加复杂;对于这些情况，你可以通过继承`nn.Module`和定义[定义你自己的模块HTG4] 向前
接收输入张量和使用的其他模块或其他张量运算autograd产生输出张量。

在这个例子中，我们实现我们的两层网络作为自定义模块的子类：

    
    
    # -*- coding: utf-8 -*-
    import torch
    
    
    class TwoLayerNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            In the constructor we instantiate two nn.Linear modules and assign them as
            member variables.
            """
            super(TwoLayerNet, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)
    
        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred
    
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    
    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)
    
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
    
        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())
    
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

###  PyTorch：控制流+重量共享

动态图形和重量共享的一个例子，我们实现一个很奇怪的模型：即在每个直传选择1和4之间的随机数，并使用该许多隐藏层，多次重复使用相同的权重全连接RELU网络计算隐藏最内层。

对于这个模型，我们可以使用普通的Python流量控制来实现循环，并且我们可以通过定义直传当多次简单地重复使用相同的模块实现最内层之间重量共享。

我们可以很容易地实现这个模型作为一个模块的子类：

    
    
    # -*- coding: utf-8 -*-
    import random
    import torch
    
    
    class DynamicNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            In the constructor we construct three nn.Linear instances that we will use
            in the forward pass.
            """
            super(DynamicNet, self).__init__()
            self.input_linear = torch.nn.Linear(D_in, H)
            self.middle_linear = torch.nn.Linear(H, H)
            self.output_linear = torch.nn.Linear(H, D_out)
    
        def forward(self, x):
            """
            For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
            and reuse the middle_linear Module that many times to compute hidden layer
            representations.
    
            Since each forward pass builds a dynamic computation graph, we can use normal
            Python control-flow operators like loops or conditional statements when
            defining the forward pass of the model.
    
            Here we also see that it is perfectly safe to reuse the same Module many
            times when defining a computational graph. This is a big improvement from Lua
            Torch, where each Module could be used only once.
            """
            h_relu = self.input_linear(x).clamp(min=0)
            for _ in range(random.randint(0, 3)):
                h_relu = self.middle_linear(h_relu).clamp(min=0)
            y_pred = self.output_linear(h_relu)
            return y_pred
    
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    
    # Construct our model by instantiating the class defined above
    model = DynamicNet(D_in, H, D_out)
    
    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
    
        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())
    
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

## 实施例

你可以在这里浏览上面的例子。

### 张量

![../_images/sphx_glr_two_layer_net_numpy_thumb.png](../_images/sphx_glr_two_layer_net_numpy_thumb.png)

[ 热身：numpy的 ](examples_tensor/two_layer_net_numpy.html#sphx-glr-beginner-
examples-tensor-two-layer-net-numpy-py)

![../_images/sphx_glr_two_layer_net_tensor_thumb.png](../_images/sphx_glr_two_layer_net_tensor_thumb.png)

[ PyTorch：张量 ](examples_tensor/two_layer_net_tensor.html#sphx-glr-beginner-
examples-tensor-two-layer-net-tensor-py)

###  Autograd 

![../_images/sphx_glr_two_layer_net_autograd_thumb.png](../_images/sphx_glr_two_layer_net_autograd_thumb.png)

[ PyTorch：张量和autograd  ](examples_autograd/two_layer_net_autograd.html#sphx-
glr-beginner-examples-autograd-two-layer-net-autograd-py)

![../_images/sphx_glr_two_layer_net_custom_function_thumb.png](../_images/sphx_glr_two_layer_net_custom_function_thumb.png)

[ PyTorch：定义新autograd功能
](examples_autograd/two_layer_net_custom_function.html#sphx-glr-beginner-
examples-autograd-two-layer-net-custom-function-py)

![../_images/sphx_glr_tf_two_layer_net_thumb.png](../_images/sphx_glr_tf_two_layer_net_thumb.png)

[ TensorFlow：静态图形 ](examples_autograd/tf_two_layer_net.html#sphx-glr-beginner-
examples-autograd-tf-two-layer-net-py)

###  NN 模块

![../_images/sphx_glr_two_layer_net_nn_thumb.png](../_images/sphx_glr_two_layer_net_nn_thumb.png)

[ PyTorch：NN  ](examples_nn/two_layer_net_nn.html#sphx-glr-beginner-examples-
nn-two-layer-net-nn-py)

![../_images/sphx_glr_two_layer_net_optim_thumb.png](../_images/sphx_glr_two_layer_net_optim_thumb.png)

[ PyTorch：的Optim  ](examples_nn/two_layer_net_optim.html#sphx-glr-beginner-
examples-nn-two-layer-net-optim-py)

![../_images/sphx_glr_two_layer_net_module_thumb.png](../_images/sphx_glr_two_layer_net_module_thumb.png)

[ PyTorch：自定义ン模块 ](examples_nn/two_layer_net_module.html#sphx-glr-beginner-
examples-nn-two-layer-net-module-py)

![../_images/sphx_glr_dynamic_net_thumb.png](../_images/sphx_glr_dynamic_net_thumb.png)

[ PyTorch：控制流+重量共享 ](examples_nn/dynamic_net.html#sphx-glr-beginner-examples-
nn-dynamic-net-py)

[Next ![](../_static/images/chevron-right-
orange.svg)](examples_tensor/two_layer_net_numpy.html "Warm-up: numpy")
[![](../_static/images/chevron-right-orange.svg)
Previous](data_loading_tutorial.html "Data Loading and Processing Tutorial")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 与实施例学习PyTorch 
    * 张量
      * 热身：numpy的
      * PyTorch：张量
    * Autograd 
      * PyTorch：张量和autograd 
      * PyTorch：定义新autograd功能
      * TensorFlow：静态图形
    * NN 模块
      * PyTorch：NN 
      * PyTorch：的Optim 
      * PyTorch：自定义ン模块
      * PyTorch：控制流+重量共享
    * 实施例
      * 张量
      * Autograd 
      * NN 模块

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

