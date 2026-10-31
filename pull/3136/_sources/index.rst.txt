.. _index:
.. include:: ./ext_links.txt

===================================
Data Parallel Extension for NumPy*
===================================

.. module:: dpnp
    :no-index:

:py:mod:`dpnp` is a NumPy-compatible array library for data-parallel
computing. It acts as a drop-in replacement for core `NumPy*`_ functions
and numerical data types and provides implementations of selected
`SciPy*`_ routines for data-parallel devices.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Overview

        Learn about the Data Parallel Extension for NumPy*, its design
        principles, and what it provides.

        +++

        .. button-ref:: overview
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:

            To the overview

    .. grid-item-card:: Quick Start Guide

        Get started with installation, setup, and your first dpnp program.

        +++

        .. button-ref:: quick_start_guide
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:

            To the quick start guide

    .. grid-item-card:: API Reference

        Detailed documentation of all supported NumPy functions and classes
        in :py:mod:`dpnp`.

        +++

        .. button-ref:: dpnp_reference
            :ref-type: ref
            :expand:
            :color: secondary
            :click-parent:

            Access API Reference

    .. grid-item-card:: Tensor (dpnp.tensor)

        The underlying Array API-compliant implementation based on
        data-parallel algorithms for accelerators.

        +++

        .. button-ref:: tensor
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:

            To the tensor documentation

    .. grid-item-card:: Development information

        C++ backend API reference and extension documentation for developers.

        +++

        .. button-ref:: dpnp_backend_api
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:

            To the backend API reference


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   overview
   quick_start_guide
   reference/index
   tensor

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development information

   dpnp_backend_api
