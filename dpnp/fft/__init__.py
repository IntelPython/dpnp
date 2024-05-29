# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""
``dpnp.fft``
===========================
Discrete Fourier Transform.

Fourier analysis is fundamentally a method for expressing a function as a sum
of periodic components, and for recovering the function from those components.
When both the function and its Fourier transform are replaced with discretized
counterparts, it is called the discrete Fourier transform (DFT). The DFT has
become a mainstay of numerical computing in part because of a very fast
algorithm for computing it, called the Fast Fourier Transform (FFT), which was
known to Gauss (1805) and was brought to light in its current form by Cooley
and Tukey.

Because the discrete Fourier transform separates its input into components
that contribute at discrete frequencies, it has a great number of applications
in digital signal processing, e.g., for filtering, and in this context the
discretized input to the transform is customarily referred to as a *signal*,
which exists in the *time domain*. The output is called a *spectrum* or
*transform* and exists in the *frequency domain*.

"""

from dpnp.fft.dpnp_iface_fft import *
from dpnp.fft.dpnp_iface_fft import __all__ as __all__fft

__all__ = __all__fft
