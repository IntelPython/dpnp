.. _dpnp_tensor_inspection:

Inspection API
==============


:mod:`dpnp.tensor` implements a way to introspect implementation- and device-
capabilities of an array library as specified in
`Python Array API standard <array_api_inspection>`_:

.. currentmodule:: dpnp.tensor

.. _array_api_inspection: https://data-apis.org/array-api/latest/API_specification/inspection.html

.. autosummary::
    :toctree: generated
    :template: autosummary/cython_class.rst

    __array_api_version__
    __array_namespace_info__
    _array_api.Info
