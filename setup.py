# *****************************************************************************
# Copyright (c) 2025, Intel Corporation
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

import skbuild
import versioneer

skbuild.setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=[
        "dpnp",
        "dpnp.dpnp_algo",
        "dpnp.dpnp_utils",
        "dpnp.exceptions",
        "dpnp.fft",
        "dpnp.linalg",
        "dpnp.memory",
        "dpnp.random",
        "dpnp.scipy",
        "dpnp.scipy.linalg",
        "dpnp.scipy.special",
        # TODO: replace with dpctl; dpctl.tensor
        "dpctl_ext",
        "dpctl_ext.tensor",
    ],
    package_data={
        "dpnp": [
            "backend/include/*.hpp",
            "libdpnp_backend_c.so",
            "dpnp_backend_c.lib",
            "dpnp_backend_c.dll",
            "tests/*.*",
            "tests/testing/*.py",
            "tests/third_party/cupy/*.py",
            "tests/third_party/cupy/*/*.py",
            "tests/third_party/cupyx/*.py",
            "tests/third_party/cupyx/*/*.py",
        ]
    },
    include_package_data=False,
)
