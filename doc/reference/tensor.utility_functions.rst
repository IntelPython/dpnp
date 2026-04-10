.. _dpnp_tensor_utility_functions:

Utility functions
=================

.. currentmodule:: dpnp.tensor

.. autosummary::
    :toctree: generated

    all
    any
    allclose
    diff
    asnumpy
    to_numpy

Device object
-------------

.. autoclass:: Device

    .. autosummary::
        ~create_device
        ~sycl_queue
        ~sycl_device
        ~sycl_context
        ~sycl_usm_shared_memory
        ~usm_ndarray_to_device
