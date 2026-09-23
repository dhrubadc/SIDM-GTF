{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{% if attributes %}
Variables
--------

{% for item in attributes %}
.. autodata:: {{ fullname }}.{{ item }}

{% endfor %}
{% endif %}

{% if functions %}
Functions
---------

{% for item in functions %}
.. autofunction:: {{ fullname }}.{{ item }}

{% endfor %}
{% endif %}

{% if classes %}
Classes
-------

{% for item in classes %}
.. autoclass:: {{ fullname }}.{{ item }}
   :members:

{% endfor %}
{% endif %}

{% if exceptions %}
Exceptions
----------

{% for item in exceptions %}
.. autoexception:: {{ fullname }}.{{ item }}

{% endfor %}
{% endif %}
