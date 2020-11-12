==============================================
DPNP -- A NumPy-like API accelerated with SYCL
==============================================

.. module:: dpnp

`DPNP <https://github.com/IntelPython/dpnp>`_ is NumPy-like API accelerated with SYCL on Intel GPU.
DPNP represents Python interface with NumPy-like API that includes a subset of methods of :class:`dpnp.ndarray`
and many functions. Under the interface is C++ library with SYCL based kernels to be executed on Intel GPU.


In the following code, np is an abbreviation of dpnp as is usually done with numpy:

   >>> import dpnp as np

The :class:`dpnp.ndarray` class is in its core, which is a compatible alternative of :class:`numpy.ndarray`.

   >>> x = np.array([1, 2, 3])

``x`` in the above example is an instance of :class:`dpnp.ndarray` that is created identically to ``NumPy``'s one.
The key difference of :class:`cupy.ndarray` from :class:`numpy.ndarray` is
that the memory is allocated on Intel GPU when setting up ``DPNP_QUEUE_GPU=1`` in the environment.


Most of the array manipulations are also done in the way similar to NumPy. Take the sum for example.

   >>> s = np.sum(x)

DPNP implements many functions on :class:`dpnp.ndarray` objects.
See the :ref:`reference <dpnp_reference>` for the supported subset of NumPy API.

.. toctree::
   :maxdepth: 2

   reference/index
   dpnp_backend_api
