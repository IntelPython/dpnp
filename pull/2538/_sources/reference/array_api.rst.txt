.. _array-api-standard-compatibility:

.. https://numpy.org/doc/stable/reference/array_api.html

********************************
Array API standard compatibility
********************************

DPNP's main namespace as well as the :mod:`dpnp.fft` and :mod:`dpnp.linalg`
namespaces are compatible with the
`2024.12 version <https://data-apis.org/array-api/2024.12/index.html>`__
of the Python array API standard.

Inspection
==========

DPNP implements the `array API inspection utilities
<https://data-apis.org/array-api/latest/API_specification/inspection.html>`__.
These functions can be accessed via the ``__array_namespace_info__()``
function, which returns a namespace containing the inspection utilities.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.__array_namespace_info__
