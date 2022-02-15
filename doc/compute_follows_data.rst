Compute Follows Data
====================

Compute follows data means computation on the device where data is placed.
Assume we placed input data for an algorithm we want to run into GPU memory.
The algorithm will be ran on GPU and
resulting data will be placed into GPU memory as well as input data.

Actually data knows which device it is located on and which execution queue associated with the device.
Based on the queue an algorithm is ran on the respective device.

Actually we don't need to know execution queue.
We just need to know device where we want to compute an algorithm.
DPNP provides array constructors like :obj:`dpnp.array`
that have such parameters as ``device``, ``usm_type`` and ``sycl_queue``.
These parameters allow us to specify where to place data.
Also DPNP provides list of functions that don't have such parameters, e.g. :obj:`dpnp.matmul`.
These functions take the parameters from input data.

Сonsider an example where we create two arrays in GPU memory and
compute matrix product of the arrays on GPU.

.. code-block:: python
  :linenos:

  import dpnp

  x = dpnp.array([[1, 1], [1, 1]], device="gpu")
  y = dpnp.array([[1, 1], [1, 1]], device="gpu")

  res = dpnp.matmul(x, y)

Resulting array is placed in GPU memory as well as input arrays.

.. code-block:: python
  :linenos:
  :lineno-start: 7
  
  res_device = res.get_array().sycl_device
  
  res_device.filter_string  # 'opencl:gpu:0'

Compute follows data prevents computation on data located on different devices.
In other words compute follows data prevents implicit cross-device data copying.
Copying makes computation expensive.

Consider an example where we create two arrays on different devices and
try to compute matrix product of the arrays. As a result ``ValueError`` is raised.

.. code-block:: python
  :linenos:
  :lineno-start: 10

  x = dpnp.array([[1, 1], [1, 1]], device="gpu")
  y = dpnp.array([[1, 1], [1, 1]], device="cpu")

  res = dpnp.matmul(x, y)  # ValueError: execution queue could not be determined ...

Execution queue couldn't be determined, because ``x`` and ``y`` are located on different devices.
To avoid ``ValueError`` we can do explicit copying of ``y`` to GPU memory before the function call.

.. code-block:: python
  :linenos:
  :lineno-start: 14

  y = dpnp.asarray(y, device="gpu")

  res = dpnp.matmul(x, y)

For more information about available devices please refer to
`DPCtl's devices documentation <https://intelpython.github.io/dpctl/latest/docfiles/user_guides/manual/dpctl/devices.html>`_.
