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

"""Module to customize build_clib command
Originally, 'build_clib' command produce static C library only.
This modification add:
 - build shared C library
 - copy this library to the project tree
 - extra option 'libraries'
 - extra option 'library_dirs'
 - extra option 'runtime_library_dirs'
 - extra option 'extra_preargs'
 - extra option 'extra_link_postargs'
 - extra option 'language'

"""


import os
from setuptools.command import build_clib
from distutils import log
from distutils.file_util import copy_file


class custom_build_clib(build_clib.build_clib):

    def build_libraries(self, libraries):
        """
        This function is overloaded to the original function in build_clib.py file
        """
        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                err_msg = f"in 'libraries' option (library '{lib_name}'),"
                err_msg += f" 'sources' must be present and must be a list of source filenames"
                raise DistutilsSetupError(err_msg)

            sources = list(sources)

            log.info("building '%s' library", lib_name)

            macros = build_info.get('macros')
            include_dirs = build_info.get('include_dirs')
            libraries = build_info.get("libraries")
            library_dirs = build_info.get("library_dirs")
            runtime_library_dirs = build_info.get("runtime_library_dirs")
            extra_preargs = build_info.get("extra_preargs")
            extra_link_postargs = build_info.get("extra_link_postargs")
            language = build_info.get("language")

            objects = self.compiler.compile(sources,
                                            output_dir=self.build_temp,
                                            macros=macros,
                                            include_dirs=include_dirs,
                                            extra_preargs=extra_preargs,
                                            debug=self.debug)

            self.compiler.link_shared_lib(objects,
                                          lib_name,
                                          output_dir=self.build_clib,
                                          libraries=libraries,
                                          library_dirs=library_dirs,
                                          runtime_library_dirs=runtime_library_dirs,
                                          extra_preargs=extra_preargs,
                                          extra_postargs=extra_link_postargs,
                                          debug=self.debug,
                                          build_temp=self.build_temp,
                                          target_lang=language)

            dest_filename = "dpnp"  # TODO need to fix destination directory
            c_library_name = self.compiler.library_filename(lib_name, lib_type='shared')
            c_library_filename = os.path.join(self.build_clib, c_library_name)

            copy_file(c_library_filename, dest_filename, verbose=self.verbose, dry_run=self.dry_run)
