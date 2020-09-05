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


class source_style(Command):
    """
    Command to check and adjust code style

    Usage:
        To check style: python ./setup.py style
        To fix style: python ./setup.py style -a

    """

    user_options = [
        ('apply', 'a', 'Apply codestyle changes to sources.')
    ]
    description = "Code style check and apply (with -a)"
    boolean_options = []

    _result_marker = "Result:"
    _project_directory_excluded = ['build', '.git']

    _c_formatter = 'clang-format'
    _c_formatter_install_msg = 'pip install clang'
    _c_formatter_command_line = [_c_formatter, '-style=file']
    _c_file_extensions = ['.h', '.c', '.hpp', '.cpp']

    _py_checker = 'pycodestyle'
    _py_formatter = 'autopep8'
    _py_formatter_install_msg = 'pip install --upgrade autopep8\npip install --upgrade pycodestyle'
    _py_checker_command_line = [_py_checker]
    _py_formatter_command_line = [
        _py_formatter,
        '--in-place',
        '--aggressive',
        '--aggressive']
    _py_file_extensions = ['.py', '.pyx', '.pxd', '.pxi']

    def _get_file_list(self, path, search_extentions):
        """ Return file list to be adjusted or checked

        path - is the project base path
        search_extentions - list of strings with files extension to search recurcivly
        """
        files = []
        exluded_directories_full_path = [os.path.join(
            path, excluded_dir) for excluded_dir in self._project_directory_excluded]

        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            # match exclude pattern in current directory
            found = False
            for excluded_dir in exluded_directories_full_path:
                if r.find(excluded_dir) >= 0:
                    found = True

            if found:
                continue

            for file in f:
                filename, extention = os.path.splitext(file)
                if extention in search_extentions:
                    files.append(os.path.join(r, file))

        return files

    def initialize_options(self):
        self.apply = 0

    def finalize_options(self):
        pass

    def run(self):
        root_dir = os.path.join(os.path.dirname(__file__), "..")
        print("Project directory is: %s" % root_dir)

        if self.apply:
            self._c_formatter_command_line += ['-i']
        else:
            self._c_formatter_command_line += ['-output-replacements-xml']

        import subprocess

        bad_style_file_names = []

        # C files handling
        c_files = self._get_file_list(root_dir, self._c_file_extensions)
        try:
            for f in c_files:
                command_output = subprocess.Popen(
                    self._c_formatter_command_line + [f], stdout=subprocess.PIPE)
                command_cout, command_cerr = command_output.communicate()
                if not self.apply:
                    if command_cout.find(b'<replacement ') > 0:
                        bad_style_file_names.append(f)
        except BaseException as original_error:
            print("%s is not installed.\nPlease use: %s" %
                  (self._c_formatter, self._c_formatter_install_msg))
            print("Original error message is:\n", original_error)
            exit(1)

        # Python files handling
        py_files = self._get_file_list(root_dir, self._py_file_extensions)
        try:
            for f in py_files:
                if not self.apply:
                    command_output = subprocess.Popen(
                        self._py_checker_command_line + [f])
                    returncode = command_output.wait()
                    if returncode != 0:
                        bad_style_file_names.append(f)
                else:
                    command_output = subprocess.Popen(
                        self._py_formatter_command_line + [f])
                    command_output.wait()
        except BaseException as original_error:
            print("%s is not installed.\nPlease use: %s" %
                  (self._py_formatter, self._py_formatter_install_msg))
            print("Original error message is:\n", original_error)
            exit(1)

        if bad_style_file_names:
            print("Following files style need to be adjusted:")
            for line in bad_style_file_names:
                print(line)
            print("%s Style check failed" % self._result_marker)
        else:
            print("%s Style check passed" % self._result_marker)
