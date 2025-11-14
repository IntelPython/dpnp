# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

mypath = os.path.dirname(os.path.realpath(__file__))

# workaround against hanging in OneMKL calls and in DPCTL
os.environ.setdefault("SYCL_QUEUE_THREAD_POOL_SIZE", "6")

import dpctl

dpctlpath = os.path.dirname(dpctl.__file__)

# For Windows OS with Python >= 3.7, it is required to explicitly define a path
# where to search for DLLs towards both DPNP backend and DPCTL Sycl interface,
# otherwise DPNP import will be failing. This is because the libraries
# are not installed under any of default paths where Python is searching.

if sys.platform == "win32":  # pragma: no cover
    os.add_dll_directory(mypath)
    os.add_dll_directory(dpctlpath)
    os.environ["PATH"] = os.pathsep.join(
        [os.getenv("PATH", ""), mypath, dpctlpath]
    )
    # For virtual environments on Windows, add folder with DPC++ libraries
    # to the DLL search path
    if sys.base_exec_prefix != sys.exec_prefix and os.path.isfile(
        os.path.join(sys.exec_prefix, "pyvenv.cfg")
    ):
        dll_path = os.path.join(sys.exec_prefix, "Library", "bin")
        if os.path.isdir(dll_path):
            os.environ["PATH"] = os.pathsep.join(
                [os.getenv("PATH", ""), dll_path]
            )

# Borrowed from DPCTL
from dpctl.tensor import __array_api_version__, DLDeviceType

from .dpnp_array import dpnp_array as ndarray
from .dpnp_array_api_info import __array_namespace_info__
from .dpnp_flatiter import flatiter as flatiter
from ._version import get_versions
from . import exceptions as exceptions
from . import fft as fft
from . import linalg as linalg
from . import random as random
from . import scipy as scipy


# =============================================================================
# Data types, constants and type-related helpers
# =============================================================================

# -----------------------------------------------------------------------------
# Data types (borrowed from NumPy)
#
# The order of these declarations are borrowed from the NumPy document:
# https://numpy.org/doc/stable/reference/arrays.scalars.html
# -----------------------------------------------------------------------------
from .dpnp_iface_types import (
    bool,
    bool_,
    byte,
    cdouble,
    complex128,
    complex64,
    complexfloating,
    csingle,
    double,
    dtype,
    float16,
    float32,
    float64,
    floating,
    inexact,
    int_,
    int8,
    int16,
    int32,
    int64,
    integer,
    intc,
    intp,
    longlong,
    number,
    short,
    signedinteger,
    single,
    ubyte,
    uint8,
    uint16,
    uint32,
    uint64,
    uintc,
    uintp,
    unsignedinteger,
    ushort,
    ulonglong,
)

# -----------------------------------------------------------------------------
# Constants (borrowed from NumPy)
# -----------------------------------------------------------------------------
from .dpnp_iface_types import (
    e,
    euler_gamma,
    inf,
    nan,
    newaxis,
    pi,
)

# -----------------------------------------------------------------------------
# Type-related helper functions
# -----------------------------------------------------------------------------
from .dpnp_iface_types import (
    common_type,
    finfo,
    iinfo,
    isdtype,
    issubdtype,
    is_type_supported,
)

# =============================================================================
# Routines
#
# The order of these declarations are borrowed from the NumPy document:
# https://numpy.org/doc/stable/reference/routines.html
# =============================================================================

# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------
from .dpnp_iface_arraycreation import (
    arange,
    array,
    asanyarray,
    asarray,
    ascontiguousarray,
    asfortranarray,
    astype,
    copy,
    diag,
    diagflat,
    empty,
    empty_like,
    eye,
    frombuffer,
    fromfile,
    fromfunction,
    fromiter,
    fromstring,
    from_dlpack,
    full,
    full_like,
    geomspace,
    identity,
    linspace,
    loadtxt,
    logspace,
    meshgrid,
    mgrid,
    ogrid,
    ones,
    ones_like,
    trace,
    tri,
    tril,
    triu,
    vander,
    zeros,
    zeros_like,
)

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------
from .dpnp_iface_manipulation import (
    append,
    array_split,
    asarray_chkfinite,
    asfarray,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_arrays,
    broadcast_shapes,
    broadcast_to,
    can_cast,
    column_stack,
    concat,
    concatenate,
    copyto,
    delete,
    dsplit,
    dstack,
    expand_dims,
    flip,
    fliplr,
    flipud,
    hsplit,
    hstack,
    insert,
    matrix_transpose,
    moveaxis,
    ndim,
    pad,
    permute_dims,
    ravel,
    repeat,
    require,
    reshape,
    resize,
    result_type,
    roll,
    rollaxis,
    rot90,
    row_stack,
    shape,
    size,
    split,
    squeeze,
    stack,
    swapaxes,
    tile,
    transpose,
    trim_zeros,
    unique,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
    unstack,
    vsplit,
    vstack,
)

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
from .dpnp_iface_bitwise import (
    binary_repr,
    bitwise_and,
    bitwise_count,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_not,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    invert,
    left_shift,
    right_shift,
)

# -----------------------------------------------------------------------------
# Functional routines
# -----------------------------------------------------------------------------
from .dpnp_iface_functional import (
    apply_along_axis,
    apply_over_axes,
    piecewise,
)

# -----------------------------------------------------------------------------
# Indexing routines
# -----------------------------------------------------------------------------
from .dpnp_iface_indexing import (
    choose,
    compress,
    diag_indices,
    diag_indices_from,
    diagonal,
    extract,
    fill_diagonal,
    flatnonzero,
    indices,
    iterable,
    ix_,
    mask_indices,
    ndindex,
    nonzero,
    place,
    put,
    put_along_axis,
    putmask,
    ravel_multi_index,
    select,
    take,
    take_along_axis,
    tril_indices,
    tril_indices_from,
    triu_indices,
    triu_indices_from,
    unravel_index,
)

# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
from .dpnp_iface_linearalgebra import (
    dot,
    einsum,
    einsum_path,
    inner,
    kron,
    matmul,
    matvec,
    outer,
    tensordot,
    vdot,
    vecdot,
    vecmat,
)

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------
from .dpnp_iface_logic import (
    all,
    allclose,
    any,
    array_equal,
    array_equiv,
    equal,
    greater,
    greater_equal,
    isclose,
    iscomplex,
    iscomplexobj,
    isfinite,
    isfortran,
    isinf,
    isnan,
    isneginf,
    isposinf,
    isreal,
    isrealobj,
    isscalar,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    not_equal,
)

# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------
from .dpnp_iface_mathematical import (
    abs,
    absolute,
    add,
    angle,
    around,
    ceil,
    clip,
    conj,
    conjugate,
    copysign,
    cross,
    cumprod,
    cumsum,
    cumulative_prod,
    cumulative_sum,
    diff,
    divide,
    ediff1d,
    fabs,
    fix,
    float_power,
    floor,
    floor_divide,
    fmax,
    fmin,
    fmod,
    frexp,
    gcd,
    gradient,
    heaviside,
    i0,
    imag,
    interp,
    lcm,
    ldexp,
    maximum,
    minimum,
    mod,
    modf,
    multiply,
    nan_to_num,
    negative,
    nextafter,
    positive,
    pow,
    power,
    prod,
    proj,
    real,
    real_if_close,
    remainder,
    rint,
    round,
    sign,
    signbit,
    sinc,
    spacing,
    subtract,
    sum,
    trapezoid,
    true_divide,
    trunc,
)
from .dpnp_iface_nanfunctions import (
    nancumprod,
    nancumsum,
    nanprod,
    nansum,
)

from .dpnp_iface_statistics import convolve

from .dpnp_iface_trigonometric import (
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    asin,
    asinh,
    acos,
    acosh,
    atan,
    atan2,
    atanh,
    cbrt,
    cos,
    cosh,
    cumlogsumexp,
    deg2rad,
    degrees,
    exp,
    exp2,
    expm1,
    hypot,
    log,
    log10,
    log1p,
    log2,
    logaddexp,
    logaddexp2,
    logsumexp,
    rad2deg,
    radians,
    reciprocal,
    reduce_hypot,
    rsqrt,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
    unwrap,
)

# -----------------------------------------------------------------------------
# Miscellaneous routines
# -----------------------------------------------------------------------------
from .dpnp_iface_utils import byte_bounds

# -----------------------------------------------------------------------------
# Sorting, searching, and counting
# -----------------------------------------------------------------------------
from .dpnp_iface_counting import count_nonzero

from .dpnp_iface_nanfunctions import nanargmax, nanargmin

from .dpnp_iface_searching import (
    argmax,
    argmin,
    argwhere,
    searchsorted,
    where,
)

from .dpnp_iface_sorting import (
    argsort,
    partition,
    sort,
    sort_complex,
)

from .dpnp_iface_window import (
    bartlett,
    blackman,
    hamming,
    hanning,
    kaiser,
)

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
from .dpnp_iface_histograms import (
    bincount,
    digitize,
    histogram,
    histogram_bin_edges,
    histogram2d,
    histogramdd,
)

from .dpnp_iface_nanfunctions import (
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanstd,
    nanvar,
)

from .dpnp_iface_statistics import (
    amax,
    amin,
    average,
    corrcoef,
    correlate,
    cov,
    max,
    mean,
    median,
    min,
    ptp,
    std,
    var,
)

# -----------------------------------------------------------------------------
# DPNP iface functions
# -----------------------------------------------------------------------------
from .dpnp_iface import (
    are_same_logical_tensors,
    asnumpy,
    as_usm_ndarray,
    check_limitations,
    check_supported_arrays_type,
    default_float_type,
    get_dpnp_descriptor,
    get_include,
    get_normalized_queue_device,
    get_result_array,
    get_usm_ndarray,
    get_usm_ndarray_or_scalar,
    is_cuda_backend,
    is_supported_array_or_scalar,
    is_supported_array_type,
    synchronize_array_data,
)


# add submodules
__all__ = ["exceptions", "fft", "linalg", "random", "scipy"]


__version__ = get_versions()["version"]
del get_versions
