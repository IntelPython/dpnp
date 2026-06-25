{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype == "data" %}
.. auto{{ objtype }}:: {{ objname }}
    :no-value:
{% endif %}

{% if objtype == "function" %}
.. auto{{ objtype }}:: {{ objname }}
{% endif %}
