.. currentmodule:: dpnp

Input and output
================

.. hint:: `NumPy API Reference: Input and output <https://numpy.org/doc/stable/reference/routines.io.html>`_

.. NumPy binary files (npy, npz)
.. -----------------------------
.. .. autosummary::
..    :toctree: generated/
..    :nosignatures:

..    load
..    save
..    savez
..    savez_compressed
..    lib.npyio.NpzFile

.. The format of these binary file types is documented in
.. :py:mod:`numpy.lib.format`

Text files
----------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   loadtxt
   savetxt
   genfromtxt
   fromregex
   fromstring
   ndarray.tofile
   ndarray.tolist

Raw binary files
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fromfile
   ndarray.tofile

.. String formatting
.. -----------------
.. .. autosummary::
..    :toctree: generated/
..    :nosignatures:

..    array2string
..    array_repr
..    array_str
..    format_float_positional
..    format_float_scientific

.. Text formatting options
.. -----------------------
.. .. autosummary::
..    :toctree: generated/
..    :nosignatures:

..    set_printoptions
..    get_printoptions
..    printoptions

Base-n representations
----------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   binary_repr
   base_repr
