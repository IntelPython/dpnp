# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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
Script to run numpy tests under dpnp.
>>> python -m tests_external.numpy.runtests
to run specific test suite:
>>> python -m tests_external.numpy.runtests core/tests/test_umath.py
to run specific test case:
>>> python -m tests_external.numpy.runtests core/tests/test_umath.py::TestHypot::test_simple
"""

import numpy.conftest
import numpy.core._rational_tests
import numpy
import argparse
import unittest
import site
import sys
import types

from pathlib import Path

import pytest
import dpnp

from dpnp.dparray import dparray


class dummymodule:
    pass


class DummyClass:
    def __init__(self, *args, **kwargs):
        pass


def dummy_func(*args, **kwargs):
    pass


class dummy_multiarray_tests(dummymodule):
    run_byteorder_converter = dummy_func
    run_casting_converter = dummy_func
    run_clipmode_converter = dummy_func
    run_intp_converter = dummy_func
    run_order_converter = dummy_func
    run_searchside_converter = dummy_func
    run_selectkind_converter = dummy_func
    run_sortkind_converter = dummy_func


dummy_sctypes = {"uint": [], "int": [], "float": []}


def define_func_types(mod, func_names, types_, default=""):
    """Define attribute types to specified functions of specified module"""
    for obj in mod.__dict__.values():
        if not isinstance(obj, types.FunctionType):
            continue

        if obj.__name__ in func_names:
            obj.types = types_
        else:
            obj.types = default


def redefine_strides(f):
    """Redefine attribute strides in dparray returned by specified function"""

    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        if not isinstance(res, dparray):
            return res

        strides = dpnp.asnumpy(res).strides
        res._dparray_strides = strides

        return res

    return wrapper


def replace_arg_value(f, arg_pos, in_values, out_value):
    """Replace value of positional argument of specified function"""

    def wrapper(*args, **kwargs):
        if len(args) <= arg_pos:
            return f(*args, **kwargs)

        args = list(args)
        arg_value = args[arg_pos]
        for in_value in in_values:
            if arg_value == in_value or arg_value is in_value:
                args[arg_pos] = out_value

        return f(*args, **kwargs)

    return wrapper


def replace_kwarg_value(f, arg_name, in_values, out_value):
    """Replace value of keyword argument of specified function"""

    def wrapper(*args, **kwargs):
        arg_value = kwargs.get(arg_name)
        for in_value in in_values:
            if arg_value == in_value or arg_value is in_value:
                kwargs[arg_name] = out_value

        return f(*args, **kwargs)

    return wrapper


# setting some dummy attrubutes to dpnp
unsupported_classes = [
    "byte",
    "bytes_",
    "cdouble",
    "character",
    "clongdouble",
    "complex_",
    "complexfloating",
    "datetime64",
    "flexible",
    "floating",
    "generic",
    "half",
    "inexact",
    "int_",
    "int16",
    "int8",
    "intc",
    "integer",
    "longlong",
    "matrix",
    "memmap",
    "nditer",
    "nextafter",
    "number",
    "object_",
    "short",
    "signedinteger",
    "single",
    "stack",
    "timedelta64",
    "ubyte",
    "uint",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "uintc",
    "ulonglong",
    "unsignedinteger",
    "ushort",
    "vectorize",
    "VisibleDeprecationWarning",
]
for klass in unsupported_classes:
    setattr(dpnp, klass, DummyClass)

"""
Some replacements because of incomplete support:

dpnp.array(dpnp.nan) -> dpnp.array([dpnp.nan])
dpnp.array([None])   -> dpnp.array([dpnp.nan])

dpnp.array([2. + 1j, 1. + 2j]) -> dpnp.array([])
dpnp.array([2. + 1j, 1. + 2j, 3. - 3j]) -> dpnp.array([])

dpnp.array([['one', 'two'], ['three', 'four']]) -> dpnp.array([[], []])
dpnp.array([[1., 2 + 3j], [2 - 3j, 1]]) -> dpnp.array([[], []])
dpnp.array([[1. + 2j, 2 + 3j], [3 + 4j, 4 + 5j]]) -> dpnp.array([[], []])
dpnp.array([[2. + 1j, 1. + 2j], [1 - 1j, 2 - 2j]]) -> dpnp.array([[], []])
dpnp.array([[2. + 1j, 1. + 2j, 1 + 3j], [1 - 2j, 1 - 3j, 1 - 6j]]) -> dpnp.array([[], []])
dpnp.array([[1. + 1j, 2. + 2j, 3. - 3j], [3. - 5j, 4. + 9j, 6. + 2j]]) -> dpnp.array([[], []])

dpnp.array([[2. + 1j, 1. + 2j], [1 - 1j, 2 - 2j], [1 - 1j, 2 - 2j]]) -> dpnp.array([[2., 1.], [1., 2.], [1., 2.]])
dpnp.array([[1. + 1j, 2. + 2j], [3. - 3j, 4. - 9j], [5. - 4j, 6. + 8j]]) -> dpnp.array([[1., 2.], [3., 4.], [5., 6.]])

dpnp.array(object, dtype='m8')       -> dpnp.array(object, dtype=None)
dpnp.array(object, dtype=dpnp.uint8) -> dpnp.array(object, dtype=None)
dpnp.array(object, dtype='i4,i4')    -> dpnp.array(object, dtype=None)
dpnp.array(object, dtype=object)     -> dpnp.array(object, dtype=None)
dpnp.array(object, dtype=rational)   -> dpnp.array(object, dtype=None)
dpnp.array(object, 'i,i')            -> dpnp.array(object, None)

dpnp.full(shape, -2**64+1) -> dpnp.full(shape, 0)
dpnp.full(shape, fill_value, dtype=object) -> dpnp.full(shape, fill_value, dtype=None)

a = dpnp.ones(shape) -> a.strides = numpy.ones(shape).strides
dpnp.ones(shape, dtype='i,i') -> dpnp.ones(shape, dtype=None)

dpnp.zeros(shape, dtype='m8') -> dpnp.zeros(shape, dtype=None)
dpnp.zeros(shape, dtype=dpnp.dtype(dict(
    formats=['<i4', '<i4'],
    names=['a', 'b'],
    offsets=[0, 2],
    itemsize=6
))) -> dpnp.zeros(shape, dtype=None)
"""
array_input_replace_map = [
    (dpnp.nan, [dpnp.nan]),
    ([None], [dpnp.nan]),
    ([2.0 + 1j, 1.0 + 2j], []),
    ([2.0 + 1j, 1.0 + 2j, 3.0 - 3j], []),
    ([["one", "two"], ["three", "four"]], [[], []]),
    ([[1.0, 2 + 3j], [2 - 3j, 1]], [[], []]),
    ([[1.0 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], [[], []]),
    ([[2.0 + 1j, 1.0 + 2j], [1 - 1j, 2 - 2j]], [[], []]),
    ([[2.0 + 1j, 1.0 + 2j, 1 + 3j], [1 - 2j, 1 - 3j, 1 - 6j]], [[], []]),
    (
        [[1.0 + 1j, 2.0 + 2j, 3.0 - 3j], [3.0 - 5j, 4.0 + 9j, 6.0 + 2j]],
        [[], []],
    ),
    (
        [[2.0 + 1j, 1.0 + 2j], [1 - 1j, 2 - 2j], [1 - 1j, 2 - 2j]],
        [[2.0, 1.0], [1.0, 2.0], [1.0, 2.0]],
    ),
    (
        [[1.0 + 1j, 2.0 + 2j], [3.0 - 3j, 4.0 - 9j], [5.0 - 4j, 6.0 + 8j]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ),
]
for in_value, out_value in array_input_replace_map:
    dpnp.array = replace_arg_value(dpnp.array, 0, [in_value], out_value)

rational = numpy.core._rational_tests.rational
dpnp.array = replace_kwarg_value(
    dpnp.array, "dtype", ["m8", dpnp.uint8, "i4,i4", object, rational], None
)
dpnp.array = replace_arg_value(dpnp.array, 1, ["i,i"], None)

dpnp.full = replace_arg_value(dpnp.full, 1, [-(2**64) + 1], 0)
dpnp.full = replace_kwarg_value(dpnp.full, "dtype", [object], None)
dpnp.ones = redefine_strides(dpnp.ones)
dpnp.ones = replace_kwarg_value(dpnp.ones, "dtype", ["i,i"], None)
dpnp.zeros = replace_kwarg_value(
    dpnp.zeros,
    "dtype",
    [
        "m8",
        dpnp.dtype(
            dict(
                formats=["<i4", "<i4"],
                names=["a", "b"],
                offsets=[0, 2],
                itemsize=6,
            )
        ),
    ],
    None,
)

# setting some dummy attrubutes to dpnp
dpnp.add.reduce = dummy_func
dpnp.allclose = dummy_func
dpnp.csingle = dpnp.complex64
dpnp.double = dpnp.float64
dpnp.identity = dummy_func
dpnp.minimum.reduce = dummy_func
dpnp.product = dummy_func
dpnp.sctypes = dummy_sctypes
dpnp.str_ = DummyClass
dpnp.unicode = str
dpnp.unicode_ = dpnp.str_

dpnp.compat = dummymodule
dpnp.compat.unicode = dpnp.unicode

dpnp.core = dpnp.core.umath = dpnp

dpnp.core._exceptions = dummymodule
dpnp.core._exceptions._ArrayMemoryError = DummyClass
dpnp.core._exceptions._UFuncNoLoopError = DummyClass

dpnp.core._multiarray_tests = dummy_multiarray_tests
dpnp.core._multiarray_umath = dummymodule

dpnp.core._operand_flag_tests = dummymodule

dpnp.core._rational_tests = dummymodule
dpnp.core._rational_tests.rational = DummyClass

dpnp.core._umath_tests = dummymodule

dpnp.core.numerictypes = dummymodule
dpnp.core.numerictypes.sctypes = dummy_sctypes

dpnp.linalg._umath_linalg = dummymodule

dpnp.ufunc = types.FunctionType


# setting some numpy attrubutes to dpnp
NUMPY_ONLY_ATTRS = [
    "BUFSIZE",
    "_NoValue",
    "errstate",
    "finfo",
    "iinfo",
    "inf",
    "intp",
    "longdouble",
    "NZERO",
    "pi",
    "testing",
    "typecodes",
]
for attr in NUMPY_ONLY_ATTRS:
    setattr(dpnp, attr, getattr(numpy, attr))

# to be able to import core/tests/test_longdouble.py::TestFileBased
# dpnp.array([ldbl]*5) -> dpnp.array([dbl]*5)
LD_INFO = numpy.finfo(numpy.longdouble)
ldbl = 1 + LD_INFO.eps
D_INFO = numpy.finfo(numpy.float64)
dbl = 1 + D_INFO.eps
dpnp.array = replace_arg_value(dpnp.array, 0, [[ldbl] * 5], [dbl] * 5)

# to be able to import core/tests/test_ufunc.py
unary_ufuncs = [
    obj
    for obj in numpy.core.umath.__dict__.values()
    if isinstance(obj, numpy.ufunc)
]
unary_object_ufuncs_names = {
    uf.__name__ for uf in unary_ufuncs if "O->O" in uf.types
}
define_func_types(dpnp, unary_object_ufuncs_names, "O->O")

dpnp.conftest = numpy.conftest

del numpy
sys.modules["numpy"] = dpnp  # next import of numpy will be replaced with dpnp


NUMPY_TESTS = [
    "core",
    "fft",
    "linalg/tests/test_build.py",
    "linalg/tests/test_deprecations.py",
    # disabled due to __setitem__ limitation:
    # https://github.com/numpy/numpy/blob/d7a75e8e8fefc433cf6e5305807d5f3180954273/numpy/linalg/tests/test_linalg.py#L293
    # 'linalg/tests/test_linalg.py',
    "linalg/tests/test_regression.py",
    "random",
]
NUMPY_NOT_FOUND = 3
TESTS_EXT_PATH = Path(__file__).parents[1]
ABORTED_TESTS_FILE = TESTS_EXT_PATH / "skipped_tests_numpy_aborted.tbl"
SKIPPED_TESTS_FILE = TESTS_EXT_PATH / "skipped_tests_numpy.tbl"
FAILED_TESTS_FILE = TESTS_EXT_PATH / "failed_tests_numpy.tbl"


def get_excluded_tests():
    for skipped_tests_fpath in [ABORTED_TESTS_FILE, SKIPPED_TESTS_FILE]:
        if not skipped_tests_fpath.exists():
            continue

        with skipped_tests_fpath.open() as fd:
            for line in fd:
                yield line.strip()


def pytest_collection_modifyitems(config, items):
    skip_mark = pytest.mark.skip(reason="Skipping test.")

    for item in items:
        test_name = item.nodeid.strip()

        for skipped_test in get_excluded_tests():
            if test_name == skipped_test:
                item.add_marker(skip_mark)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()

    if not rep.failed:
        return None

    mode = "a" if FAILED_TESTS_FILE.exists() else "w"
    with FAILED_TESTS_FILE.open(mode) as f:
        f.write(rep.nodeid.strip() + "\n")


dpnp.conftest.pytest_collection_modifyitems = pytest_collection_modifyitems
dpnp.conftest.pytest_runtest_makereport = pytest_runtest_makereport


def find_pkg(name):
    """Find package in site-packages"""
    for p in site.getsitepackages():
        pkg_path = Path(p) / name
        if pkg_path.exists():
            return pkg_path

    return None


def tests_from_cmdline():
    """Get relative paths to tests from command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("tests", nargs="*", help="list of tests to run")
    args = parser.parse_args()

    return args.tests


def get_tests(base_path):
    """Get tests paths from command line or NUMPY_TESTS"""
    tests_relpaths = tests_from_cmdline()
    if tests_relpaths:
        for test_relpath in tests_relpaths:
            yield base_path / test_relpath
        return None

    for test_relpath in NUMPY_TESTS:
        yield base_path / test_relpath

    return None


def run():
    numpy_path = find_pkg("numpy")
    if numpy_path is None:
        print("Numpy not found in the environment.")
        return NUMPY_NOT_FOUND

    if FAILED_TESTS_FILE.exists():
        FAILED_TESTS_FILE.unlink()

    test_suites = [str(tests_path) for tests_path in get_tests(numpy_path)]

    try:
        for test_suite in test_suites:
            code = pytest.main([test_suite])
            if code:
                break
    except SystemExit as exc:
        code = exc.code

    return code


if __name__ == "__main__":
    exit(run())
