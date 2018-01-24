torch.cuda
===================================

.. currentmodule:: torch.cuda

.. automodule:: torch.cuda
   :members:

Random Number Generator
-------------------------
.. autofunction:: get_rng_state
.. autofunction:: set_rng_state
.. autofunction:: manual_seed
.. autofunction:: manual_seed_all
.. autofunction:: seed
.. autofunction:: seed_all
.. autofunction:: initial_seed


Communication collectives
-------------------------

.. autofunction:: torch.cuda.comm.broadcast

.. autofunction:: torch.cuda.comm.reduce_add

.. autofunction:: torch.cuda.comm.scatter

.. autofunction:: torch.cuda.comm.gather

Streams and events
------------------

.. autoclass:: Stream
   :members:

.. autoclass:: Event
   :members:

Memory management
-----------------
.. autofunction:: empty_cache

NVIDIA Tools Extension (NVTX)
-----------------------------

.. autofunction:: torch.cuda.nvtx.mark
.. autofunction:: torch.cuda.nvtx.range_push
.. autofunction:: torch.cuda.nvtx.range_pop
