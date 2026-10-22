# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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


import os
import sys
import types
import warnings

__DPNP_ORIGIN__ = int(os.getenv("DPNP_ORIGIN", 0))
"""
Explicitly use original host Python NumPy
"""

__DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK__ = int(
    os.getenv("DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK", 1)
)
"""
Trigger non-implemented exception when DPNP fallbacks on NumPy implementation
"""

# settings that no longer have any effect: attribute name -> env var name
_DEPRECATED_SETTINGS = {
    "__DPNP_QUEUE_GPU__": "DPNP_QUEUE_GPU",
    "__DPNP_OUTPUT_NUMPY__": "DPNP_OUTPUT_NUMPY",
    "__DPNP_OUTPUT_DPCTL_DEFAULT_SHARED__": "DPNP_OUTPUT_DPCTL_DEFAULT_SHARED",
}

# values assigned to deprecated settings after import
_deprecated_values = {}


def _warn_deprecated(name):
    warnings.warn(
        f"`dpnp.config.{name}` is deprecated, and will be removed in a "
        "future release. It has no effect.",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name):
    if name in _DEPRECATED_SETTINGS:
        _warn_deprecated(name)
        if name in _deprecated_values:
            return _deprecated_values[name]
        # parse lazily, so a bad env value can't break `import dpnp`
        return int(os.getenv(_DEPRECATED_SETTINGS[name], 0))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _ConfigModule(types.ModuleType):
    """Warn also on assignment to a deprecated setting."""

    def __setattr__(self, name, value):
        if name in _DEPRECATED_SETTINGS:
            _warn_deprecated(name)
            # keep it out of the module dict, so reads still warn
            _deprecated_values[name] = value
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ConfigModule
