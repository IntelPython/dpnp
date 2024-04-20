.. module:: dpnp.fft

FFT Functions
=============

.. https://docs.scipy.org/doc/numpy/reference/routines.fft.html

Standard FFTs
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.fft.fft
   dpnp.fft.ifft
   dpnp.fft.fft2
   dpnp.fft.ifft2
   dpnp.fft.fftn
   dpnp.fft.ifftn


Real FFTs
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.fft.rfft
   dpnp.fft.irfft
   dpnp.fft.rfft2
   dpnp.fft.irfft2
   dpnp.fft.rfftn
   dpnp.fft.irfftn


Hermitian FFTs
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.fft.hfft
   dpnp.fft.ihfft


Helper routines
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.fft.fftfreq
   dpnp.fft.rfftfreq
   dpnp.fft.fftshift
   dpnp.fft.ifftshift

   .. fft.config module is not implemented yet
   .. dpnp.fft.config.set_cufft_callbacks
   .. dpnp.fft.config.set_cufft_gpus
   .. dpnp.fft.config.get_plan_cache
   .. dpnp.fft.config.show_plan_cache_info
