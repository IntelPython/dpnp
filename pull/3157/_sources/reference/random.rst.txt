.. currentmodule:: dpnp.random

Random sampling (:mod:`dpnp.random`)
====================================

.. hint:: `NumPy API Reference: Random sampling <https://numpy.org/doc/stable/reference/random/index.html>`_

DPNP draws random numbers on the SYCL device, with the counter-based
Philox4x32x10 generator of the oneMKL device RNG API. It provides the
:class:`Generator` API of NumPy, recommended for new code. The legacy
:class:`RandomState` API and the module-level functions still use the
host-side oneMKL RNG.

.. code-block:: python

    import dpnp

    rng = dpnp.random.default_rng(seed=42)
    x = rng.standard_normal((3, 4))                # on the default device
    y = rng.random(5, usm_type="shared")

    rng = dpnp.random.default_rng(42, device="cpu")

A generator draws all its samples on one SYCL queue, chosen with the
``device`` or ``sycl_queue`` keyword when it is created. All samples are
:class:`dpnp.ndarray`, zero-dimensional when ``size`` is ``None``.

Differences from NumPy
----------------------

* The streams differ from NumPy's. For a given seed, output size and data
  type, the raw bits and the uniform floats are the same on every device,
  while the normal samples agree to a few ULP.
* The values depend on the output size: a draw is not a prefix of a
  larger one, and two draws of ``n`` values differ from one draw of
  ``2 * n``. Code splitting its draws into chunks gets other values.
* The floating-point distributions take a ``dtype`` keyword, as in CuPy,
  which defaults to the default floating-point type of the device, so on a
  device without double precision the same seed gives other values by
  default.
* All methods take a ``usm_type`` keyword. With ``usm_type=None``, the
  default, the samples take the USM type of the array parameters, or
  ``"device"`` when there are none.
* Parameters on the host (Python scalars, lists and NumPy arrays) are
  checked as in NumPy. Parameters in :class:`dpnp.ndarray` are
  not, since that would wait for the device.
  Invalid ones give undefined values, such as NaN.
* Generators and bit generators cannot be pickled or deep-copied, and the
  bit generators have no ``advance`` or ``jumped`` method.

Random Generator
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   default_rng
   Generator

Accessing the bit generator and spawning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Generator.bit_generator
   Generator.spawn
   Generator.sycl_device
   Generator.sycl_queue

Simple random data
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Generator.random

Distributions
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Generator.normal
   Generator.standard_normal
   Generator.uniform

Bit generators
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BitGenerator
   BitGenerator.lock
   BitGenerator.random_raw
   BitGenerator.seed_seq
   BitGenerator.spawn
   BitGenerator.state
   BitGenerator.sycl_device
   BitGenerator.sycl_queue
   Philox4x3210
   SeedSequence

Legacy random generation
------------------------

.. hint:: `NumPy API Reference: Legacy random generation <https://numpy.org/doc/stable/reference/random/legacy.html>`_

Simple random data
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rand
   randn
   randint
   random_integers
   random_sample
   random
   ranf
   sample
   choice
   bytes

Permutations
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   shuffle
   permutation

Distributions
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   beta
   binomial
   chisquare
   dirichlet
   exponential
   f
   gamma
   geometric
   gumbel
   hypergeometric
   laplace
   logistic
   lognormal
   logseries
   multinomial
   multivariate_normal
   negative_binomial
   noncentral_chisquare
   noncentral_f
   normal
   pareto
   poisson
   power
   rayleigh
   standard_cauchy
   standard_exponential
   standard_gamma
   standard_normal
   standard_t
   triangular
   uniform
   vonmises
   wald
   weibull
   zipf

RandomState
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   RandomState
   seed
   get_random_state
   set_random_state
