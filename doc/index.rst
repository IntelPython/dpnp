.. _index:
.. include:: ./ext_links.txt

===================================
Data Parallel Extension for NumPy*
===================================

.. module:: dpnp
    :no-index:

Python package :py:mod:`dpnp` implements a subset of `NumPy*`_ that can be
executed on any data parallel device. The subset is a drop-in replacement of
core `NumPy*`_ functions and numerical data types.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Overview

        Learn about the Data Parallel Extension for NumPy*, its design
        principles, and what it provides.

        +++

        .. button-ref:: overview
            :expand:
            :color: secondary
            :click-parent:

            To the overview

    .. grid-item-card:: Quick Start Guide

        Get started with installation, setup, and your first dpnp program.

        +++

        .. button-ref:: quick_start_guide
            :expand:
            :color: secondary
            :click-parent:

            To the quick start guide

    .. grid-item-card:: API Reference

        Detailed documentation of all supported NumPy functions and classes
        in :py:mod:`dpnp`.

        +++

        .. button-ref:: dpnp_reference
            :expand:
            :color: secondary
            :click-parent:

            Access API Reference

    .. grid-item-card:: Tensor (dpnp.tensor)

        The underlying Array API-compliant implementation based on
        data-parallel algorithms for accelerators.

        +++

        .. button-ref:: tensor
            :expand:
            :color: secondary
            :click-parent:

            To the tensor documentation


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   overview
   quick_start_guide
   reference/index
   tensor
   dpnp_backend_api
