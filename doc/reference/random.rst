.. currentmodule:: dpnp.random

Random sampling (:mod:`dpnp.random`)
====================================

.. hint:: `NumPy API Reference: Random sampling <https://numpy.org/doc/stable/reference/random/index.html>`_

DPNP draws random numbers on the SYCL device, with the counter-based
Philox4x32x10 generator of the oneMKL device RNG API. It provides both
NumPy random APIs: the :class:`Generator` API recommended for new code,
and the legacy :class:`RandomState` API with the module-level functions
built on it.

.. code-block:: python

    import dpnp

    rng = dpnp.random.default_rng(seed=42)
    x = rng.standard_normal((3, 4))                # on the default device
    y = rng.integers(0, 10, size=5, usm_type="shared")

    rng = dpnp.random.default_rng(42, device="cpu")

A generator draws all its samples on one SYCL queue, chosen with the
``device`` or ``sycl_queue`` keyword when it is created, and the samples
depend only on the seed, not on the device. All samples are
:class:`dpnp.ndarray`, zero-dimensional when ``size`` is ``None``.

Differences from NumPy
----------------------

* The streams differ from NumPy's, since no NumPy bit generator is provided.
  The same seed gives the same stream on every device.
* The floating-point distributions take a ``dtype`` keyword, as in CuPy,
  which defaults to the default floating-point type of the device. All
  methods take a ``usm_type`` keyword.
* Parameters on the host (Python scalars, lists and NumPy arrays) are
  checked as in NumPy. Parameters in :class:`dpnp.ndarray` are not, since
  that would wait for the device, and invalid ones give undefined values.
* The discrete distributions return ``int64`` values. They compute in double
  precision on devices supporting it, and in single precision otherwise,
  where the samples for very large parameters are less accurate.
* ``hypergeometric`` requires ``ngood`` and ``nbad`` below ``10**9``, also in
  :class:`RandomState`.
* :meth:`Generator.shuffle` and :func:`shuffle` only accept a
  :class:`dpnp.ndarray`, as in CuPy.

Random Generator
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   default_rng
   Generator

Simple random data
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Generator.integers
   Generator.random
   Generator.choice
   Generator.bytes

Permutations
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Generator.shuffle
   Generator.permutation
   Generator.permuted

Distributions
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Generator.beta
   Generator.binomial
   Generator.chisquare
   Generator.dirichlet
   Generator.exponential
   Generator.f
   Generator.gamma
   Generator.geometric
   Generator.gumbel
   Generator.hypergeometric
   Generator.laplace
   Generator.logistic
   Generator.lognormal
   Generator.logseries
   Generator.multinomial
   Generator.multivariate_hypergeometric
   Generator.multivariate_normal
   Generator.negative_binomial
   Generator.noncentral_chisquare
   Generator.noncentral_f
   Generator.normal
   Generator.pareto
   Generator.poisson
   Generator.power
   Generator.rayleigh
   Generator.standard_cauchy
   Generator.standard_exponential
   Generator.standard_gamma
   Generator.standard_normal
   Generator.standard_t
   Generator.triangular
   Generator.uniform
   Generator.vonmises
   Generator.wald
   Generator.weibull
   Generator.zipf

Bit generators
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BitGenerator
   Philox4x3210
   SeedSequence

Legacy random generation
------------------------

.. hint:: `NumPy API Reference: Legacy random generation <https://numpy.org/doc/stable/reference/random/legacy.html>`_

:class:`RandomState` keeps the parameter checks of NumPy's legacy one. The
module-level functions use one :class:`RandomState` per SYCL queue: the
queue given by the ``device`` or ``sycl_queue`` keyword, or else the queue of
a :class:`dpnp.ndarray` parameter, or else the default queue.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   RandomState
   seed
   get_state
   set_state

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
