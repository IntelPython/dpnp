{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   ..
      Methods

{% block methods %}

   .. rubric:: Methods

   ..
      Special methods

{% for item in ('__getitem__', '__setitem__', '__len__', '__next__', '__iter__') %}
{% if item in all_methods or item in all_attributes %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}

   ..
      Ordinary methods

{% for item in methods %}
{% if item not in ('__init__',) %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}

   ..
      Special methods

{% for item in ('__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__') %}
{% if item in all_methods %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}
{% endblock %}

   ..
      Atributes

{% block attributes %} {% if attributes %}

   .. rubric:: Attributes

{% for item in attributes %}
   .. autoattribute:: {{ item }}
{%- endfor %}
{% endif %} {% endblock %}
