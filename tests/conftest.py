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
import sys
import dpnp
import numpy
import pytest


skip_mark = pytest.mark.skip(reason='Skipping test.')


def get_excluded_tests(test_exclude_file):
    excluded_tests = []
    if os.path.exists(test_exclude_file):
        with open(test_exclude_file) as skip_names_file:
            excluded_tests = skip_names_file.readlines()
    return excluded_tests


def pytest_collection_modifyitems(config, items):
    test_path = os.path.split(__file__)[0]
    excluded_tests = []
    # global skip file
    test_exclude_file = os.path.join(test_path, 'skipped_tests.tbl')

    # global skip file, where gpu device is not supported
    test_exclude_file_gpu = os.path.join(test_path, 'skipped_tests_gpu.tbl')

    current_queue_is_cpu = dpnp.dpnp_queue_is_cpu()
    print("")
    print(f"DPNP current queue is CPU: {current_queue_is_cpu}")
    print(f"DPNP version: {dpnp.__version__}, location: {dpnp}")
    print(f"NumPy version: {numpy.__version__}, location: {numpy}")
    print(f"Python version: {sys.version}")
    print("")
    if not current_queue_is_cpu or os.getenv('DPNP_QUEUE_GPU') == '1':
        excluded_tests.extend(get_excluded_tests(test_exclude_file_gpu))
    else:
        excluded_tests.extend(get_excluded_tests(test_exclude_file))

    for item in items:
        # some test name contains '\n' in the parameters
        test_name = item.nodeid.replace('\n', '').strip()

        for item_tbl in excluded_tests:
            # remove end-of-line character
            item_tbl_str = item_tbl.strip()
            # exact match of the test name with items from excluded_list
            if test_name == item_tbl_str:
                item.add_marker(skip_mark)

@pytest.fixture
def allow_fall_back_on_numpy(monkeypatch):
    monkeypatch.setattr(dpnp.config, '__DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK__', 0)
