.. module:: dpnp.fft

FFT Functions
=============

.. https://docs.scipy.org/doc/numpy/reference/routines.fft.html

Standard FFTs
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fft
   ifft
   fft2
   ifft2
   fftn
   ifftn


Real FFTs
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rfft
   irfft
   rfft2
   irfft2
   rfftn
   irfftn


Hermitian FFTs
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   hfft
   ihfft


Helper routines
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fftfreq
   rfftfreq
   fftshift
   ifftshift

   .. fft.config module is not implemented yet
   .. dpnp.fft.config.set_cufft_callbacks
   .. dpnp.fft.config.set_cufft_gpus
   .. dpnp.fft.config.get_plan_cache
   .. dpnp.fft.config.show_plan_cache_info

.. automodule:: dpnp.fft
    :no-index:
