.. _dpnp_tensor_flags_class:

``Flags`` class
===================

.. autoclass:: dpnp.tensor._flags.Flags
    :members:

    Note that dictionary-like access to some members is permitted:

        "C", "C_CONTIGUOUS":
            Equivalent to ``c_contiguous``
        "F", "F_CONTIGUOUS":
            Equivalent to ``f_contiguous``
        "W", "WRITABLE":
            Equivalent to ``writable``
        "FC":
            Equivalent to ``fc``
        "FNC":
            Equivalent to ``fnc``
        "FORC", "CONTIGUOUS":
            Equivalent to ``forc`` and ``contiguous``
