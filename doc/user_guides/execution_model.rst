.. _dpnp_execution_model:

########################
oneAPI programming model
########################

oneAPI library and its Python interface
=======================================

Using oneAPI libraries, a user calls functions that take ``sycl::queue`` and a collection of
``sycl::event`` objects among other arguments. For example:

.. code-block:: cpp
    :caption: Prototypical call signature of oneMKL function

    sycl::event
    compute(
        sycl::queue &exec_q,
        ...,
        const std::vector<sycl::event> &dependent_events
    );

The function ``compute`` inserts computational tasks into the queue ``exec_q`` for DPC++ runtime to
execute on the device the queue targets. The execution may begin only after other tasks whose
execution status is represented by ``sycl::event`` objects in the provided ``dependent_events``
vector complete. If the vector is empty, the runtime begins the execution as soon as the device is
ready. The function returns a ``sycl::event`` object representing completion of the set of
computational tasks submitted by the ``compute`` function.

Hence, in the oneAPI programming model, the execution **queue** is used to specify which device the
function will execute on. To create a queue, one must specify a device to target.

In :mod:`dpctl`, the ``sycl::queue`` is represented by :class:`dpctl.SyclQueue` Python type,
and a Python API to call such a function might look like

.. code-block:: python

    def call_compute(
        exec_q : dpctl.SyclQueue,
        ...,
        dependent_events : List[dpctl.SyclEvent] = []
    ) -> dpctl.SyclEvent:
        ...

When building Python API for a SYCL offloading function, and you choose to
map the SYCL API to a different API on the Python side, it must still translate to a
similar call under the hood.

The arguments to the function must be suitable for use in the offloading functions.
Typically these are Python scalars, or objects representing USM allocations, such as
:class:`dpnp.tensor.usm_ndarray`, :class:`dpctl.memory.MemoryUSMDevice` and friends.

.. note::
    The USM allocations these objects represent must not get deallocated before
    offloaded tasks that access them complete.

    This is something authors of DPC++-based Python extensions must take care of,
    and users of such extensions should assume assured.


USM allocations and compute-follows-data
========================================

To make a USM allocation on a device in SYCL, one needs to specify ``sycl::device`` in the
memory of which the allocation is made, and the ``sycl::context`` to which the allocation
is bound.

A ``sycl::queue`` object is often used instead. In such cases ``sycl::context`` and ``sycl::device`` associated
with the queue are used to make the allocation.

.. important::
    :mod:`dpnp.tensor` associates a queue object with every USM allocation.

    The associated queue may be queried using ``.sycl_queue`` property of the
    Python type representing the USM allocation.

This design choice allows :mod:`dpnp.tensor` to have a preferred queue to use when operating on any single
USM allocation. For example:

.. code-block:: python

    def unary_func(x : dpnp.tensor.usm_ndarray):
        code1
        _ = _func_impl(x.sycl_queue, ...)
        code2

When combining several objects representing USM-allocations, the
:ref:`programming model <dpnp_tensor_compute_follows_data>`
adopted in :mod:`dpnp.tensor` insists that queues associated with each object be the same, in which
case it is the execution queue used. Alternatively :exc:`dpctl.utils.ExecutionPlacementError` is raised.

.. code-block:: python

    def binary_func(
        x1 : dpnp.tensor.usm_ndarray,
        x2 : dpnp.tensor.usm_ndarray
    ):
        exec_q = dpctl.utils.get_execution_queue((x1.sycl_queue, x2.sycl_queue))
        if exec_q is None:
            raise dpctl.utils.ExecutionPlacementError
        ...

In order to ensure that compute-follows-data works seamlessly out-of-the-box, :mod:`dpnp.tensor` maintains
a cache with context and device as keys and queues as values used by :class:`dpnp.tensor.Device` class.

.. code-block:: python

    >>> import dpctl
    >>> from dpnp import tensor

    >>> sycl_dev = dpctl.SyclDevice("cpu")
    >>> d1 = tensor.Device.create_device(sycl_dev)
    >>> d2 = tensor.Device.create_device("cpu")
    >>> d3 = tensor.Device.create_device(dpctl.select_cpu_device())

    >>> d1.sycl_queue == d2.sycl_queue, d1.sycl_queue == d3.sycl_queue, d2.sycl_queue == d3.sycl_queue
    (True, True, True)

Since :class:`dpnp.tensor.Device` class is used by all :ref:`array creation functions <dpnp_tensor_creation_functions>`
in :mod:`dpnp.tensor`, the same value used as ``device`` keyword argument results in array instances that
can be combined together in accordance with compute-follows-data programming model.

.. code-block:: python

    >>> from dpnp import tensor
    >>> import dpctl

    >>> # queue for default-constructed device is used
    >>> x1 = tensor.arange(100, dtype="int32")
    >>> x2 = tensor.zeros(100, dtype="int32")
    >>> x12 = tensor.concat((x1, x2))
    >>> x12.sycl_queue == x1.sycl_queue, x12.sycl_queue == x2.sycl_queue
    (True, True)
    >>> # default constructors of SyclQueue class create different instance of the queue
    >>> q1 = dpctl.SyclQueue()
    >>> q2 = dpctl.SyclQueue()
    >>> q1 == q2
    False
    >>> y1 = tensor.arange(100, dtype="int32", sycl_queue=q1)
    >>> y2 = tensor.zeros(100, dtype="int32", sycl_queue=q2)
    >>> # this call raises ExecutionPlacementError since compute-follows-data
    >>> # rules are not met
    >>> tensor.concat((y1, y2))

Please refer to the :ref:`array migration <dpnp_tensor_array_migration>` section of the introduction to
:mod:`dpnp.tensor` for examples on how to resolve ``ExecutionPlacementError`` exceptions.
