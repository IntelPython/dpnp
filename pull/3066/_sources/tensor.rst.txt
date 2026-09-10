.. _tensor:

Tensor (``dpnp.tensor``)
========================

``dpnp.tensor`` provides a reference implementation of the
`Python Array API <https://data-apis.org/array-api/latest/>`_ specification.
The implementation uses data-parallel algorithms suitable for execution on
accelerators, such as GPUs.

It also provides the underlying Array API-compliant implementation
used by ``dpnp``.

``dpnp.tensor`` is written using C++ and
`SYCL <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>`_
and oneAPI extensions implemented in
`Intel(R) oneAPI DPC++ compiler <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`_.

Design and Motivation
---------------------

The tensor implementation was originally developed as a standalone project and
later integrated into the `dpctl <https://intelpython.github.io/dpctl/latest/index.html>`_
library as ``dpctl.tensor``. It has since been migrated into ``dpnp``,
making ``dpnp`` the primary owner and development location of the tensor implementation.

This change simplifies maintenance, reduces cross-project
dependencies, and enables independent development and release cycles.

Relationship to ``dpnp.ndarray``
--------------------------------

:class:`dpnp.ndarray` is a high-level array object built on top of
``dpnp.tensor.usm_ndarray``, storing array data in Unified Shared Memory
(USM) allocated on a SYCL device. Most users interact with
:class:`dpnp.ndarray` directly; ``dpnp.tensor.usm_ndarray`` may appear in error
messages or type signatures when working with device placement or
interoperability.

Relationship to ``dpctl``
-------------------------

The migration of ``dpctl.tensor`` into ``dpnp.tensor`` does not replace
`dpctl <https://intelpython.github.io/dpctl/latest/index.html>`_ itself.
``dpctl`` remains responsible for device and queue management
(:class:`dpctl.SyclDevice`, :class:`dpctl.SyclQueue`) as well as USM memory
allocation. ``dpnp`` builds on top of these capabilities.

Example
-------

.. code-block:: python

    import dpnp
    import dpnp.tensor as dpt

    # Create a tensor array on the default device
    x = dpt.asarray([1.0, 2.0, 3.0])

    # dpnp.ndarray wraps the underlying usm_ndarray
    a = dpnp.asarray([1.0, 2.0, 3.0])
    assert isinstance(a.get_array(), dpt.usm_ndarray)

.. note::

   The ``dpnp.tensor`` API documentation will be added in a future release.

   The current implementation remains compatible with the original
   ``dpctl.tensor`` API. For the complete API reference, see the
   `dpctl 0.21.1 tensor documentation <https://intelpython.github.io/dpctl/0.21.1/api_reference/dpctl/tensor.html>`_.
