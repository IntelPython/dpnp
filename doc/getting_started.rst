.. _getting_started:
.. include:: ./ext_links.txt

.. |copy| unicode:: U+000A9

.. |trade| unicode:: U+2122

Getting Started
===============

Requirements
------------

The following Linux distributions are recommended.

* `Ubuntu <https://www.ubuntu.com/>`_ 20.04 LTS (x86_64)

These components must be installed to use DPNP:

* `Python <https://python.org/>`_: v3

Python Dependencies
~~~~~~~~~~~~~~~~~~~

NumPy-compatible API in DPNP is based on `NumPy <https://numpy.org/>`_ 1.18+.


Installing DPNP from conda-forge
--------------------------------

You can install DPNP with Conda/Anaconda from the ``intel`` channel::

    $ conda install -c intel dpnp

.. _install_dpnp_from_source:

Installing DPNP from Source
---------------------------

You can install the latest development version of DPNP from a cloned Git repository on Linux::

  $ git clone --recursive https://github.com/IntelPython/dpnp.git
  $ cd dpnp
  $ python scripts/build_locally.py

.. note::

   To build the source tree downloaded from GitHub, you need to install
   `Intel oneAPI Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html>`_
   and Cython (``pip install cython`` or ``conda install cython``).
