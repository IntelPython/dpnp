# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
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
Module to call cmake based procedure by build_cmake_clib command
"""

import os
import sys
import pathlib
from setuptools.command import build_clib
from distutils import log

IS_WIN = False
IS_MAC = False
IS_LIN = False

if 'linux' in sys.platform:
    IS_LIN = True
elif sys.platform == 'darwin':
    IS_MAC = True
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True
else:
    raise EnvironmentError("DPNP cmake builder: " + sys.platform + " not supported")


class custom_build_cmake_clib(build_clib.build_clib):
    def run(self):
        root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        log.info(f"Project directory is: {root_dir}")

        backend_directory = os.path.join(root_dir, "dpnp", "backend")
        install_directory = os.path.join(root_dir, "dpnp")

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        abs_build_temp_path = str(os.path.abspath(build_temp))
        log.info(f"build directory is: {abs_build_temp_path}")

        config = "Debug" if self.debug else "Release"

        cmake_generator = str()
        if IS_WIN:
            cmake_generator = "-GNinja"

        cmake_args = [
            cmake_generator,
            "-S" + backend_directory,
            "-B" + abs_build_temp_path,
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DDPNP_INSTALL_PREFIX=" + install_directory.replace(os.sep, "/" ), # adjust to cmake requirenments
            "-DDPNP_INSTALL_STRUCTURED=OFF",
            # "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + install_directory,
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
        ]

        self.spawn(["cmake"] + cmake_args + [backend_directory])
        if not self.dry_run:
            self.spawn(["cmake", "--build", abs_build_temp_path])
            self.spawn(["cmake", "--install", abs_build_temp_path])
