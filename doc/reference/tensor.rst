.. _dpnp_tensor_pyapi:

Tensor (``dpnp.tensor``)
========================

.. py:module:: dpnp.tensor

.. currentmodule:: dpnp.tensor

:py:mod:`dpnp.tensor` provides a reference implementation of the
`Python Array API <https://data-apis.org/array-api/latest/>`_ specification. The implementation
uses data-parallel algorithms suitable for execution on accelerators, such as GPUs.

:py:mod:`dpnp.tensor` is written using C++ and `SYCL <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>`_
and oneAPI extensions implemented in `Intel(R) oneAPI DPC++ compiler <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`_.

This module contains:

* Array object :py:class:`usm_ndarray`
* :ref:`Accumulation functions <dpnp_tensor_accumulation_functions>`
* :ref:`Array creation functions <dpnp_tensor_creation_functions>`
* :ref:`Array manipulation functions <dpnp_tensor_manipulation_functions>`
* :ref:`Elementwise functions <dpnp_tensor_elementwise_functions>`
* :ref:`Indexing functions <dpnp_tensor_indexing_functions>`
* :ref:`Introspection functions <dpnp_tensor_inspection>`
* :ref:`Linear algebra functions <dpnp_tensor_linear_algebra>`
* :ref:`Searching functions <dpnp_tensor_searching_functions>`
* :ref:`Set functions <dpnp_tensor_set_functions>`
* :ref:`Sorting functions <dpnp_tensor_sorting_functions>`
* :ref:`Statistical functions <dpnp_tensor_statistical_functions>`
* :ref:`Utility functions <dpnp_tensor_utility_functions>`
* :ref:`Printing functions <dpnp_tensor_print_functions>`
* :ref:`Constants <dpnp_tensor_constants>`


.. toctree::
    :hidden:

    tensor.creation_functions
    tensor.usm_ndarray
    tensor.data_type_functions
    tensor.data_types
    tensor.elementwise_functions
    tensor.accumulation_functions
    tensor.indexing_functions
    tensor.inspection
    tensor.linear_algebra
    tensor.manipulation_functions
    tensor.searching_functions
    tensor.set_functions
    tensor.sorting_functions
    tensor.statistical_functions
    tensor.utility_functions
    tensor.print_functions
    tensor.constants
