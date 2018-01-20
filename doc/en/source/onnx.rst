torch.onnx
============
.. automodule:: torch.onnx

Example: End-to-end AlexNet from PyTorch to Caffe2
--------------------------------------------------

Here is a simple script which exports a pretrained AlexNet as defined in
torchvision into ONNX.  It runs a single round of inference and then
saves the resulting traced model to ``alexnet.proto``::

    from torch.autograd import Variable
    import torch.onnx
    import torchvision

    dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
    model = torchvision.models.alexnet(pretrained=True).cuda()
    torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True)

The resulting ``alexnet.proto`` is a binary protobuf file which contains both
the network structure and parameters of the model you exported
(in this case, AlexNet).  The keyword argument ``verbose=True`` causes the
exporter to print out a human-readable representation of the network::

    # All parameters are encoded explicitly as inputs.  By convention,
    # learned parameters (ala nn.Module.state_dict) are first, and the
    # actual inputs are last.
    graph(%1 : Float(64, 3, 11, 11)
          %2 : Float(64)
          # The definition sites of all variables are annotated with type
          # information, specifying the type and size of tensors.
          # For example, %3 is a 192 x 64 x 5 x 5 tensor of floats.
          %3 : Float(192, 64, 5, 5)
          %4 : Float(192)
          # ---- omitted for brevity ----
          %15 : Float(1000, 4096)
          %16 : Float(1000)
          %17 : Float(10, 3, 224, 224)) { # the actual input!
      # Every statement consists of some output tensors (and their types),
      # the operator to be run (with its attributes, e.g., kernels, strides,
      # etc.), its input tensors (%17, %1)
      %19 : UNKNOWN_TYPE = Conv[kernels=[11, 11], strides=[4, 4], pads=[2, 2, 2, 2], dilations=[1, 1], group=1](%17, %1), uses = [[%20.i0]];
      # UNKNOWN_TYPE: sometimes type information is not known.  We hope to eliminate
      # all such cases in a later release.
      %20 : Float(10, 64, 55, 55) = Add[broadcast=1, axis=1](%19, %2), uses = [%21.i0];
      %21 : Float(10, 64, 55, 55) = Relu(%20), uses = [%22.i0];
      %22 : Float(10, 64, 27, 27) = MaxPool[kernels=[3, 3], pads=[0, 0, 0, 0], dilations=[1, 1], strides=[2, 2]](%21), uses = [%23.i0];
      # ...
      # Finally, a network returns some tensors
      return (%58);
    }

You can also verify the protobuf using the `onnx <https://github.com/onnx/onnx/>`_ library.
You can install ``onnx`` with conda::

    conda install -c conda-forge onnx

Then, you can run::

    import onnx

    # Load the ONNX model
    model = onnx.load("alexnet.proto")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

To run the exported script with `caffe2 <https://caffe2.ai/>`_, you will need three things:

1. You'll need an install of Caffe2.  If you don't have one already, Please
   `follow the install instructions <https://caffe2.ai/docs/getting-started.html>`_.

2. You'll need `onnx-caffe2 <https://github.com/onnx/onnx-caffe2>`_, a
   pure-Python library which provides a Caffe2 backend for ONNX.  You can install ``onnx-caffe2``
   with pip::

      pip install onnx-caffe2

Once these are installed, you can use the backend for Caffe2::

    # ...continuing from above
    import onnx_caffe2.backend as backend
    import numpy as np

    rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    # For the Caffe2 backend:
    #     rep.predict_net is the Caffe2 protobuf for the network
    #     rep.workspace is the Caffe2 workspace for the network
    #       (see the class onnx_caffe2.backend.Workspace)
    outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
    # To run networks with more than one input, pass a tuple
    # rather than a single numpy ndarray.
    print(outputs[0])

In the future, there will be backends for other frameworks as well.

Limitations
-----------

* The ONNX exporter is a *trace-based* exporter, which means that it
  operates by executing your model once, and exporting the operators which
  were actually run during this run.  This means that if your model is
  dynamic, e.g., changes behavior depending on input data, the export
  won't be accurate.  Similarly, a trace is likely to be valid only
  for a specific input size (which is one reason why we require explicit inputs
  on tracing.)  We recommend examining the model trace and making sure
  the traced operators look reasonable.

* PyTorch and Caffe2 often have implementations of operators with some
  numeric differences.  Depending on model structure, these differences
  may be negligible, but they can also cause major divergences in behavior
  (especially on untrained models.)  In a future release, we plan to
  allow Caffe2 to call directly to Torch implementations of operators, to
  help you smooth over these differences when precision is important,
  and to also document these differences.

Supported operators
-------------------

The following operators are supported:

* add (nonzero alpha not supported)
* sub (nonzero alpha not supported)
* mul
* div
* cat
* mm
* addmm
* neg
* tanh
* sigmoid
* mean
* t
* expand (only when used before a broadcasting ONNX operator; e.g., add)
* transpose
* view
* split
* squeeze
* prelu (single weight shared among input channels not supported)
* threshold (non-zero threshold/non-zero value not supported)
* leaky_relu
* glu
* softmax
* avg_pool2d (ceil_mode not supported)
* log_softmax
* unfold (experimental support with ATen-Caffe2 integration)
* elu
* Conv
* BatchNorm
* MaxPool1d (ceil_mode not supported)
* MaxPool2d (ceil_mode not supported)
* MaxPool3d (ceil_mode not supported)
* Embedding (no optional arguments supported)
* RNN
* ConstantPadNd
* Dropout
* FeatureDropout (training mode not supported)
* Index (constant integer and tuple indices supported)
* Negate

The operator set above is sufficient to export the following models:

* AlexNet
* DCGAN
* DenseNet
* Inception (warning: this model is highly sensitive to changes in operator
  implementation)
* ResNet
* SuperResolution
* VGG
* `word_language_model <https://github.com/pytorch/examples/tree/master/word_language_model>`_

The interface for specifying operator definitions is highly experimental
and undocumented; adventurous users should note that the APIs will probably
change in a future interface.

Functions
--------------------------
.. autofunction:: export
