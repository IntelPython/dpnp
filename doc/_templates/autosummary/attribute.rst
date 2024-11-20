:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

attribute

.. auto{{ objtype }}:: {{ fullname | replace("dpnp.", "dpnp::") }}

{# In the fullname (e.g. ``dpnp.ndarray.methodname``), the module name is
   ambiguous. Using a ``::`` separator (e.g. ``dpnp::ndarray.methodname``)
   specifies ``dpnp`` as the module name. #}
