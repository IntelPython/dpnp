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
import pytest


skip_mark = pytest.mark.skip(reason='Skipping test.')


def pytest_collection_modifyitems(config, items):
    excluded_tests = []
    # global skip file
    test_path = os.path.split(__file__)[0]
    test_exclude_file = os.path.join(test_path, 'skipped_tests.tbl')
    if os.path.exists(test_exclude_file):
        with open(test_exclude_file) as skip_names_file:
            excluded_tests = skip_names_file.read()

    for item in items:
        # some test name contains '\n' in the parameters
        test_name = item.nodeid.replace('\n', '')
#         test_file = test_name.split(':', -1)[0]
#         test_path = os.path.split(test_file)[0]
#         test_exclude_file = os.path.join(test_path, 'skipped_tests.tbl')
#         if os.path.exists(test_exclude_file):
#             with open(test_exclude_file) as skip_names_file:
#                 excluded_tests = skip_names_file.read()

        if test_name in excluded_tests:
            item.add_marker(skip_mark)
