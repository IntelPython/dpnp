=========================================================
DPNP -- A NumPy-compatible library for SYCL-based devices
=========================================================

.. module:: dpnp

`DPNP <https://github.com/IntelPython/dpnp>`_ is a NumPy-like library accelerated with SYCL on Intel devices.
It provides Python interfaces for many NumPy functions, and includes a subset of methods of :class:`dpnp.ndarray`.
Under the hood it is based on native C++ and oneMKL based kernels.

Being drop-in replacement for Numpy its usage is very similar to Numpy:

   >>> import dpnp as np

The :class:`dpnp.ndarray` class is a compatible alternative of :class:`numpy.ndarray`.

   >>> x = np.array([1, 2, 3])

``x`` in the above example is an instance of :class:`dpnp.ndarray` that is created identically to ``NumPy``'s one.
The key difference of :class:`dpnp.ndarray` from :class:`numpy.ndarray` is
that the memory is allocated on Intel GPU when setting up ``DPNP_QUEUE_GPU=1`` in the environment.


Most of the array manipulations are also done in the way similar to NumPy such as:

   >>> s = np.sum(x)

Please see the :ref:`API Reference <dpnp_reference>` for the complete list of supported NumPy APIs
along with their limitations.

.. toctree::
   :maxdepth: 2

   install
   reference/index
   dpnp_backend_api
   dpctl
