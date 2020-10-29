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

from types import FunctionType, MethodType

try:
    import cupy
except (ModuleNotFoundError, OSError):
    pass

import numpy
import dpnp


def get_object_funcs(obj):
    """Get all functions of specified object"""
    return [item for item in dir(obj) if isinstance(getattr(obj, item), FunctionType)]


def get_object_methods(obj):
    """Get all methods of specified object"""
    return [item for item in dir(obj) if isinstance(getattr(obj, item), MethodType)]


def get_object_properties(obj):
    """Get all properties of specified object"""
    return [item for item in dir(obj) if isinstance(getattr(obj, item), property)]


def get_module_classes(py_module):
    """Get all classes of specified module"""
    return [item for item in dir(py_module) if isinstance(getattr(py_module, item), type)]


def get_module_methods(py_module):
    """
    Get all methods of specified module in the following structure:
        {'class_name': ['first_method_name', 'second_method_name', 'third_method_name']}
    """
    all_methods = {}
    for class_name in get_module_classes(py_module):
        attr = getattr(py_module, class_name)
        methods = get_object_funcs(attr) + get_object_methods(attr)
        if methods:
            all_methods[class_name] = methods
    return all_methods


def get_module_properties(py_module):
    """
    Get all properties of specified module in the following structure:
        {'class_name': ['first_property_name', 'second_property_name', 'third_property_name']}
    """
    all_props = {}
    for class_name in get_module_classes(py_module):
        props = get_object_properties(getattr(py_module, class_name))
        if props:
            all_props[class_name] = props
    return all_props

def get_module_items(py_module):
    """
    Get all items of specified module in the following structure:
        {'class_name': ['first_item_name', 'second_item_name', 'third_item_name']}
    """
    all_items = {}
    for class_name in get_module_classes(py_module):
        attr = getattr(py_module, class_name)
        items = get_object_funcs(attr) + get_object_methods(attr) + get_object_properties(attr)
        if items:
            all_items[class_name] = items
    return all_items


def get_public(items):
    return [i for i in items if not i.startswith('_')]


class LibAPI:
    def __init__(self, lib):
        self.lib = lib

        self.all_funcs = get_object_funcs(self.lib)
        self.all_classes = get_module_classes(self.lib)
        self.all_methods = get_module_methods(self.lib)
        self.all_props = get_module_properties(self.lib)
        self.all_items = get_module_items(self.lib)

        self.public_funcs = [name for name in self.all_funcs if not name.startswith('_')]
        self.public_methods = {cl: get_public(methods) for cl, methods in self.all_methods.items()}
        self.public_props = {cl: get_public(props) for cl, props in self.all_props.items()}
        self.public_items = {cl: get_public(items) for cl, items in self.all_items.items()}

    @property
    def total_funcs(self):
        return len(self.public_funcs)

    @property
    def total_methods(self):
        return sum(len(methods) for methods in self.public_methods.values())

    @property
    def total_props(self):
        return sum(len(props) for props in self.public_props.values())

    @property
    def total_items(self):
        return sum(len(items) for items in self.public_items.values())

    @property
    def total(self):
        return self.total_funcs + self.total_items

    @property
    def public_funcs_as_str(self, separator=', '):
        return separator.join(self.public_funcs)

    @property
    def public_methods_as_str(self, separator=', '):
        return separator.join(f'{cl}.{m}' for cl, methods in self.public_methods.items() for m in methods)

    @property
    def public_props_as_str(self, separator=', '):
        return separator.join(f'{cl}.{p}' for cl, props in self.public_props.items() for p in props)

    def show(self):
        tmpl = """{lib_name} public API:
    functions: {public_funcs}
    methods: {public_methods}
    properties: {public_props}

        """
        data = {
            'lib_name': self.lib.__name__,
            'public_funcs': self.public_funcs_as_str,
            'public_methods': self.public_methods_as_str,
            'public_props': self.public_props_as_str,
        }
        print(tmpl.format(**data))


if __name__ == '__main__':
    LibAPI(numpy).show()

    try:
        LibAPI(cupy).show()
    except NameError:
        pass

    LibAPI(dpnp).show()
