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

import os
import inspect

name_dict = {}
module_names_set = dict()
extra_modules = ["fft", "linalg", "random", "char"]
sep = ":"

col0_width = 4
col1_width = 40
col2_width = 60


def print_header_line():
    print(f"{'='*col0_width}", end=sep)
    print(f"{'='*col1_width}", end=sep)
    for mod_name in module_names_set.keys():
        print(f"{'='*col2_width}", end=sep)
    print()


def print_header():
    print_header_line()

    print("#".center(col0_width), end=sep)
    print("Name".center(col1_width), end=sep)
    for mod_name in module_names_set.keys():
        print(mod_name.center(col2_width), end=sep)
    print()

    print_header_line()


def print_footer():
    print_header_line()

    print("".center(col0_width), end=sep)
    print("".center(col1_width), end=sep)
    for mod_name, mod_sym_count in module_names_set.items():
        count_str = mod_name + " total " + str(mod_sym_count)
        print(count_str.rjust(col2_width), end=sep)
    print()

    print_header_line()


def add_symbol(item_name, module_name, item_val):
    if item_name not in name_dict.keys():
        name_dict[item_name] = dict()
    if not name_dict[item_name].get(module_name, False):
        name_dict[item_name][module_name] = str(item_val)

        if module_name not in module_names_set.keys():
            module_names_set[module_name] = 0
        else:
            module_names_set[module_name] += 1
#     else:
#         print(f"item_name={item_name}, {name_dict[item_name][module_name]} replaced with {str(item_val)}")


def fill_data(module_name, module_obj, parent_module_name=""):
    for item_name_raw, item_val in inspect.getmembers(module_obj):
        if (item_name_raw[0] == "_"):
            continue

        item_name = os.path.join(parent_module_name, item_name_raw)
        if getattr(item_val, '__call__', False):
            str_item = item_val
            try:
                str_item = inspect.signature(item_val)
            except ValueError:
                pass
            add_symbol(item_name, module_name, str_item)
        elif inspect.ismodule(item_val):
            if item_name in extra_modules:
                fill_data(module_name, item_val, parent_module_name=item_name)
            else:
                print(f"IGNORED: {module_name}: module: {item_name}")
#         elif isinstance(item_val, (tuple, list, float, int)):
#             add_symbol(item_name, module_name, item_val)
#         elif isinstance(item_val, str):
#             add_symbol(item_name, module_name, item_val.replace('\n', '').strip())
#         else:
#             add_symbol(item_name, module_name, type(item_val))
#             print(f"Symbol {item_name} unrecognized. Symbol: {item_val}, type: {type(item_val)}")


def print_data():
    print_header()

    symbol_id = 0
    for symbol_name, symbol_values in sorted(name_dict.items()):
        print(f"{symbol_id:<{col0_width}}", end=sep)
        symbol_id += 1
        print(f"{symbol_name:{col1_width}}", end=sep)

        for mod_name in module_names_set.keys():
            val = symbol_values.get(mod_name, "")
            val_prn = str(val)[0:col2_width - 1]
            print(f"{val_prn:{col2_width}}", end=sep)

        print()

    print_footer()


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
