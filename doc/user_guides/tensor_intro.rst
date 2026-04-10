.. _user_guide_tensor_intro:

Intro to :py:mod:`dpnp.tensor`
===============================

Supported array data types
--------------------------

The tensor submodule provides an N-dimensional array object for a tensor whose values have the same data type
from the :ref:`following list <dpnp_tensor_data_types>`:

.. currentmodule:: dpnp.tensor

.. list-table::

    * -
      - :attr:`int8`
      - :attr:`int16`
      - :attr:`int32`
      - :attr:`int64`
      -
      - :attr:`float16`
      - :attr:`float32`
      - :attr:`complex64`

    * - :attr:`bool`
      - :attr:`uint8`
      - :attr:`uint16`
      - :attr:`uint32`
      - :attr:`uint64`
      -
      -
      - :attr:`float64`
      - :attr:`complex128`


Creating an array
-----------------

Array :ref:`creation functions <dpnp_tensor_creation_functions>` support keyword arguments that
control the device where the array is allocated as well as aspects of
USM allocation for the array.

These three keywords are:

.. list-table::
    :header-rows: 1

    * - Keyword arguments
      - Default value
      - Description
    * - ``usm_type``
      - ``"device"``
      - type of USM allocation to make
    * - ``device``
      - ``None``
      - :py:class:`dpnp.tensor.Device` instance
    * - ``sycl_queue``
      - ``None``
      - Instance of :class:`dpctl.SyclQueue` associated with array

Arguments ``sycl_queue`` and ``device`` are complementary to each other, and
a user need only provide one of these.

A valid setting for the ``device`` keyword argument is any object that can be passed to :py:meth:`dpnp.tensor.Device.create_device`.
If both ``device`` and ``sycl_queue`` keyword arguments are specified, they must correspond to :class:`dpctl.SyclQueue` instances which
compare equal to one another.

A created instance of :class:`usm_ndarray` has an associated :class:`dpctl.SyclQueue` instance that can be retrieved
using :attr:`dpnp.tensor.usm_ndarray.sycl_queue` property. The underlying USM allocation
is allocated on :class:`dpctl.SyclDevice` and is bound to :class:`dpctl.SyclContext` targeted by this queue.

.. _dpnp_tensor_compute_follows_data:

Execution model
---------------

When one of more instances of ``usm_ndarray`` objects are passed to a function in :py:mod:`dpnp.tensor` other than creation function,
a "compute follows data" execution model is followed.

The model requires that :class:`dpctl.SyclQueue` instances associated with each array compared equal to one another, signifying that
each one corresponds to the same underlying ``sycl::queue`` object. In such a case, the output array is associated with the same
``sycl::queue`` and computations are scheduled for execution using this ``sycl::queue``.

.. note::
    Two instances :class:`dpctl.SyclQueue` may target the same ``sycl::device`` and be using the same ``sycl::context``, but correspond
    to different scheduling entries, and hence be in violation of the compute-follows-data requirement. One common example of this are
    ``SyclQueue`` corresponding to default-selected device and using platform default context but created using different properties, e.g.
    one with `"enable_profiling"` set and another without it.

If input arrays do not conform to the compute-follows-data requirements, :py:exc:`dpctl.utils.ExecutionPlacementError` is raised.
User must explicitly migrate the data to unambiguously control the execution placement.

.. _dpnp_tensor_array_migration:

Migrating arrays
----------------

Array content can be migrated to a different device
using either :meth:`dpnp.tensor.usm_ndarray.to_device` method, or by using :func:`dpnp.tensor.asarray` function.

The ``arr.to_device(device=target_device)`` method will be zero-copy if the ``arr.sycl_queue`` and the :class:`dpctl.SyclQueue`
instance associated with new target device have the same underlying ``sycl::device`` and ``sycl::context`` instances.

Here is an example of migration without a copy using ``.to_device`` method:

.. code-block:: python
    :caption: Example: Use ``.to_device`` to zero-copy migrate array content to be associated with a different ``sycl::queue``

    import dpctl
    from dpnp import tensor

    x = tensor.linspace(0, 1, num=10**8)
    q_prof = dpctl.SyclQueue(x.sycl_context, x.sycl_device, property="enable_profiling")

    timer = dpctl.SyclTimer()
    # no data migration takes place here (zero-copy),
    # but x and x1 arrays do not satisfy compute-follows-data requirements
    x1 = x.to_device(q_prof)

    with timer(q_prof):
        y1 = tensor.sin(2*x1)*tensor.exp(-tensor.square(x1))

    # also a zero copy operation
    y = y1.to_device(x.device)

    host_dt, device_dt = timer.dt
    print(f"Execution on device {x.sycl_device.name} took {device_dt} seconds")
    print(f"Execution on host took {host_dt} seconds")

Data migration when the current and the target SYCL contexts are different is performed via host. That means that data are copied from
the current device to the host, and then from the host to the target device:

.. code-block:: python
    :caption: Example: Using ``.to_device`` to migrate data may involve copy via host

    from dpnp import tensor

    x_cpu = tensor.concat((tensor.ones(10, device="cpu"), tensor.zeros(1000, device="cpu")))

    # data migration is performed via host
    x_gpu = x_cpu.to_device("gpu")

An alternative way to migrate data is to use :py:func:`asarray` and specify device-placement keyword arguments:

.. code-block:: python
    :caption: Example: Using ``asarray`` to migrate data may involve copy via host

    from dpnp import tensor

    x_cpu = tensor.concat((tensor.ones(10, device="cpu"), tensor.zeros(1000, device="cpu")))

    # data migration is performed via host
    x_gpu = tensor.asarray(x_cpu, device="gpu")

An advantage of using the function ``asarray`` is that migration from ``usm_ndarray`` instances allocated on different
devices as well migration from :py:class:`numpy.ndarray` may be accomplished in a single call:

.. code-block:: python
    :caption: Example: ``asarray`` may migrate multiple arrays

    from dpnp import tensor
    import numpy

    x_cpu = tensor.ones((10, 10), device="cpu")
    x_gpu = tensor.zeros((10, 10), device="opencl:gpu")
    x_np = numpy.random.randn(10, 10)

    # Array w has shape (3, 10, 10)
    w = tensor.asarray([x_cpu, x_gpu, x_np], device="level_zero:gpu")

Migration may also occur during calls to other array creation functions, e.g., :py:func:`full` when the ``fill_value`` parameter is an instance
of :py:class:`usm_ndarray`. In such a case default values of device placement keywords are interpreted to avoid data migration, i.e., the
new array is created on the same device where ``fill_value`` array was allocated.

.. code-block:: python
    :caption: Example: Using ``usm_ndarray`` as arguments to array construction

    from dpnp import tensor

    # Zero-dimensional array allocated on CPU device
    pi_on_device = tensor.asarray(tensor.pi, dtype=tensor.float32, device="cpu")

    # x will also be allocated on CPU device
    x = tensor.full(shape=(100, 100), fill_value=pi_on_device)

    # Create array on GPU. Migration of `pi_on_device` to GPU via host
    # takes place under the hood
    y_gpu = tensor.full(shape=(100, 100), fill_value=pi_on_device, device="gpu")


Combining arrays with different USM types
-----------------------------------------

For functions with single argument the returned array has the same ``usm_type`` as the input array.

Functions that combine several ``usm_ndarray`` instances the ``usm_type`` of the output array is determined
using the following coercion rule:

+------------+----------+----------+----------+
|            | "device" | "shared" | "host"   |
+------------+----------+----------+----------+
| "device"   | "device" | "device" | "device" |
+------------+----------+----------+----------+
| "shared"   | "device" | "shared" | "shared" |
+------------+----------+----------+----------+
| "host"     | "device" | "shared" | "host"   |
+------------+----------+----------+----------+

If assigning USM-type "device" a score of 0, USM-type "shared" a score of 1, and USM-type "host" a score of 2,
the USM-type of the output array has the smallest score of all its inputs.

.. currentmodule:: dpctl.utils

The convenience function :py:func:`get_coerced_usm_type` is a convenience function to determine the USM-type
following this convention:

.. code-block:: python

    from dpctl.utils import get_coerced_usm_type

    # r1 has value "device"
    r1 = get_coerced_usm_type(["device", "shared", "host"])

    # r2 has value "shared"
    r2 = get_coerced_usm_type(["shared", "shared", "host"])

    # r3 has value "host"
    r3 = get_coerced_usm_type(["host", "host", "host"])

Sharing data between devices and Python
---------------------------------------

Python objects, such as sequences of :class:`int`, :class:`float`, or :class:`complex` objects,
or NumPy arrays can be converted to :class:`dpnp.tensor.usm_ndarray` using :func:`dpnp.tensor.asarray`
function.

.. code-block:: python

    >>> from dpnp import tensor as dpt
    >>> import numpy as np
    >>> import mkl_random

    >>> # Sample from true random number generator
    >>> rs = mkl_random.RandomState(brng="nondeterm")
    >>> x_np = rs.uniform(-1, 1, size=(6, 512)).astype(np.float32)

    >>> # copy data to USM-device (default) allocated array
    >>> x_usm = dpt.asarray(x_np)
    >>> dpt.max(x_usm, axis=1)
    usm_ndarray([0.9998379 , 0.9963589 , 0.99818915, 0.9975991 , 0.9999802 ,
                0.99851537], dtype=float32)
    >>> np.max(x_np, axis=1)
    array([0.9998379 , 0.9963589 , 0.99818915, 0.9975991 , 0.9999802 ,
          0.99851537], dtype=float32)

The content of :class:`dpnp.tensor.usm_ndarray` may be copied into
a NumPy array using :func:`dpnp.tensor.asnumpy` function:

.. code-block:: python

    from dpnp import tensor as dpt
    import numpy as np

    def sieve_pass(r : dpt.usm_ndarray, v : dpt.usm_ndarray) -> dpt.usm_ndarray:
        "Single pass of sieve of Eratosthenes"
        m = dpt.min(r[r > v])
        r[ (r > m) & (r % m == 0) ] = 0
        return m

    def sieve(n : int) -> dpt.usm_ndarray:
        "Find primes <=n using sieve of Erathosthenes"
        idt = dpt.int32
        s = dpt.concat((
          dpt.arange(2, 3, dtype=idt),
          dpt.arange(3, n + 1, 2, dtype=idt)
        ))
        lb = dpt.zeros(tuple(), dtype=idt)
        while lb * lb < n + 1:
            lb = sieve_pass(s, lb)
        return s[s > 0]

    # get prime numbers <= a million into NumPy array
    # to save to disk
    ps_np = dpt.asnumpy(sieve(10**6))

    np.savetxt("primes.txt", ps_np, fmt="%d")
