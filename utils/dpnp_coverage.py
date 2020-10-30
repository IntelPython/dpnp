#!/usr/bin/env python
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

import inspect
#from pprint import pprint

name_dict = {}
module_names_set = set()
sep = ":"

col0_width = 4
col1_width = 40
col2_width = 60


def fill_data(module_name, module_obj):
    module_names_set.add(module_name)

    for item_name, item_val in inspect.getmembers(module_obj):
        if item_name not in name_dict.keys():
            name_dict[item_name] = dict()

        if module_name not in name_dict[item_name].keys():
            name_dict[item_name][module_name] = "*undefined*"

        if inspect.isfunction(item_val):
            try:
                name_dict[item_name][module_name] = str(inspect.signature(item_val))
            except ValueError:
                name_dict[item_name][module_name] = "*error*"
                # print(f"get signature error with: {item_val}")
                pass


def print_data():
    print("#".center(col0_width), end=sep)
    print("Name".center(col1_width), end=sep)
    for mod_name in module_names_set:
        print(mod_name.center(col2_width), end=sep)
    print()

    print(f"{'='*col0_width}", end=sep)
    print(f"{'='*col1_width}", end=sep)
    for mod_name in module_names_set:
        print(f"{'='*col2_width}", end=sep)
    print()

    symbol_id = 0
    for symbol_name, symbol_values in name_dict.items():
        print(f"{symbol_id:<{col0_width}}", end=sep)
        symbol_id += 1
        print(f"{symbol_name:{col1_width}}", end=sep)

        for mod_name in module_names_set:
            val = symbol_values.get(mod_name, "*empty*")
            val_prn = str(val)[0:col2_width - 1]
            print(f"{val_prn:{col2_width}}", end=sep)

        print()


if __name__ == '__main__':

    try:
        import dpnp
        fill_data("DPNP", dpnp)
    except ImportError:
        print("No DPNP module loaded")

    try:
        import numpy
        fill_data("NumPy", numpy)
    except ImportError:
        print("No NumPy module loaded")

    try:
        import cupy
        fill_data("cuPy", cupy)
    except ImportError:
        print("No cuPy module loaded")

    print_data()
