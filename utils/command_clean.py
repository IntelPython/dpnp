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

import os
from setuptools import Command
from fnmatch import fnmatch
from shutil import rmtree


class source_clean(Command):
    """
    Command to clean all generated files in the project

    Usage:
        To run the command: python ./setup.py clean
    """

    description = "Clean up the project source tree"

    CLEAN_ROOTDIRS = ['build', 'build_cython', 'cython_debug', 'Intel_NumPy.egg-info', 'doc/_build']
    CLEAN_DIRS = ['__pycache__']
    CLEAN_FILES = ['*.so', '*.pyc', '*.pyd', '*.dll', '*.lib']

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # removing dirs from root_dir
        for dir_mask in self.CLEAN_ROOTDIRS:
            rdir = os.path.join(root_dir, dir_mask)
            if os.path.isdir(rdir):
                print('CLEAN removing: ', rdir)
                rmtree(rdir)

        for (dirpath, dirnames, filenames) in os.walk(root_dir):
            # removing subdirs
            for dir in dirnames:
                for dir_mask in self.CLEAN_DIRS:
                    if fnmatch(dir, dir_mask):
                        rdir = os.path.join(dirpath, dir)
                        print('CLEAN removing: ', rdir)
                        rmtree(rdir)

            # removing files
            for file in filenames:
                for file_mask in self.CLEAN_FILES:
                    if fnmatch(file, file_mask):
                        rfile = os.path.join(dirpath, file)
                        print('CLEAN removing: ', rfile)
                        os.remove(rfile)

        print('CLEAN done!')
