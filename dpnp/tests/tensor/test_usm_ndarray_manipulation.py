# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

import itertools

import dpctl

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
import numpy as np
import pytest
from dpctl.utils import ExecutionPlacementError
from dpctl_ext.tensor._numpy_helper import AxisError
from numpy.testing import assert_, assert_array_equal, assert_raises_regex

from .helper import get_queue_or_skip


def test_permute_dims_incorrect_type():
    X_list = list([[1, 2, 3], [4, 5, 6]])
    X_tuple = tuple(X_list)
    Xnp = np.array(X_list)

    pytest.raises(TypeError, dpt.permute_dims, X_list, (1, 0))
    pytest.raises(TypeError, dpt.permute_dims, X_tuple, (1, 0))
    pytest.raises(TypeError, dpt.permute_dims, Xnp, (1, 0))


def test_permute_dims_empty_array():
    q = get_queue_or_skip()

    Xnp = np.empty((10, 0))
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.permute_dims(X, (1, 0))
    Ynp = np.transpose(Xnp, (1, 0))
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_permute_dims_0d_1d():
    q = get_queue_or_skip()

    Xnp_0d = np.array(1, dtype="int64")
    X_0d = dpt.asarray(Xnp_0d, sycl_queue=q)
    Y_0d = dpt.permute_dims(X_0d, ())
    assert_array_equal(dpt.asnumpy(Y_0d), dpt.asnumpy(X_0d))

    Xnp_1d = np.random.randint(0, 2, size=6, dtype="int64")
    X_1d = dpt.asarray(Xnp_1d, sycl_queue=q)
    Y_1d = dpt.permute_dims(X_1d, (0))
    assert_array_equal(dpt.asnumpy(Y_1d), dpt.asnumpy(X_1d))

    pytest.raises(ValueError, dpt.permute_dims, X_1d, ())
    pytest.raises(AxisError, dpt.permute_dims, X_1d, (1))
    pytest.raises(ValueError, dpt.permute_dims, X_1d, (1, 0))
    pytest.raises(
        ValueError, dpt.permute_dims, dpt.reshape(X_1d, (2, 3)), (1, 1)
    )


@pytest.mark.parametrize("shapes", [(2, 2), (1, 4), (3, 3, 3), (4, 1, 3)])
def test_permute_dims_2d_3d(shapes):
    q = get_queue_or_skip()

    Xnp_size = np.prod(shapes)

    Xnp = np.random.randint(0, 2, size=Xnp_size, dtype="int64").reshape(shapes)
    X = dpt.asarray(Xnp, sycl_queue=q)
    X_ndim = X.ndim
    if X_ndim == 2:
        Y = dpt.permute_dims(X, (1, 0))
        Ynp = np.transpose(Xnp, (1, 0))
    elif X_ndim == 3:
        X = dpt.asarray(Xnp, sycl_queue=q)
        Y = dpt.permute_dims(X, (2, 0, 1))
        Ynp = np.transpose(Xnp, (2, 0, 1))
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_expand_dims_incorrect_type():
    X_list = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        dpt.permute_dims(X_list, axis=1)


def test_expand_dims_0d():
    q = get_queue_or_skip()

    Xnp = np.array(1, dtype="int64")
    X = dpt.asarray(Xnp, sycl_queue=q)

    Y = dpt.expand_dims(X, axis=0)
    Ynp = np.expand_dims(Xnp, axis=0)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.expand_dims(X, axis=-1)
    Ynp = np.expand_dims(Xnp, axis=-1)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    pytest.raises(AxisError, dpt.expand_dims, X, axis=1)
    pytest.raises(AxisError, dpt.expand_dims, X, axis=-2)


@pytest.mark.parametrize("shapes", [(3,), (3, 3), (3, 3, 3)])
def test_expand_dims_1d_3d(shapes):
    q = get_queue_or_skip()

    Xnp_size = np.prod(shapes)

    Xnp = np.random.randint(0, 2, size=Xnp_size, dtype="int64").reshape(shapes)
    X = dpt.asarray(Xnp, sycl_queue=q)
    shape_len = len(shapes)
    for axis in range(-shape_len - 1, shape_len):
        Y = dpt.expand_dims(X, axis=axis)
        Ynp = np.expand_dims(Xnp, axis=axis)
        assert_array_equal(Ynp, dpt.asnumpy(Y))

    pytest.raises(AxisError, dpt.expand_dims, X, axis=shape_len + 1)
    pytest.raises(AxisError, dpt.expand_dims, X, axis=-shape_len - 2)


@pytest.mark.parametrize(
    "axes", [(0, 1, 2), (0, -1, -2), (0, 3, 5), (0, -3, -5)]
)
def test_expand_dims_tuple(axes):
    q = get_queue_or_skip()

    Xnp = np.empty((3, 3, 3), dtype="u1")
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.expand_dims(X, axis=axes)
    Ynp = np.expand_dims(Xnp, axis=axes)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_expand_dims_incorrect_tuple():
    try:
        X = dpt.empty((3, 3, 3), dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(AxisError):
        dpt.expand_dims(X, axis=(0, -6))
    with pytest.raises(AxisError):
        dpt.expand_dims(X, axis=(0, 5))

    with pytest.raises(ValueError):
        dpt.expand_dims(X, axis=(1, 1))


def test_squeeze_incorrect_type():
    X_list = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        dpt.permute_dims(X_list, 1)


def test_squeeze_0d():
    q = get_queue_or_skip()

    Xnp = np.array(1)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.squeeze(X)
    Ynp = Xnp.squeeze()
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.squeeze(X, 0)
    Ynp = Xnp.squeeze(0)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.squeeze(X, (0))
    Ynp = Xnp.squeeze(0)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.squeeze(X, -1)
    Ynp = Xnp.squeeze(-1)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    pytest.raises(AxisError, dpt.squeeze, X, 1)
    pytest.raises(AxisError, dpt.squeeze, X, -2)
    pytest.raises(AxisError, dpt.squeeze, X, (1))
    pytest.raises(AxisError, dpt.squeeze, X, (-2))
    pytest.raises(ValueError, dpt.squeeze, X, (0, 0))


@pytest.mark.parametrize(
    "shapes",
    [
        (0),
        (1),
        (1, 2),
        (2, 1),
        (1, 1),
        (2, 2),
        (1, 0),
        (0, 1),
        (1, 2, 1),
        (2, 1, 2),
        (2, 2, 2),
        (1, 1, 1),
        (1, 0, 1),
        (0, 1, 0),
    ],
)
def test_squeeze_without_axes(shapes):
    q = get_queue_or_skip()

    Xnp = np.empty(shapes, dtype="u1")
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.squeeze(X)
    Ynp = Xnp.squeeze()
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize("axes", [0, 2, (0), (2), (0, 2)])
def test_squeeze_axes_arg(axes):
    q = get_queue_or_skip()

    Xnp = np.array([[[1], [2], [3]]], dtype="u1")
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.squeeze(X, axes)
    Ynp = Xnp.squeeze(axes)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize("axes", [1, -2, (1), (-2), (0, 0), (1, 1)])
def test_squeeze_axes_arg_error(axes):
    q = get_queue_or_skip()

    Xnp = np.array([[[1], [2], [3]]], dtype="u1")
    X = dpt.asarray(Xnp, sycl_queue=q)
    pytest.raises(ValueError, dpt.squeeze, X, axes)


@pytest.mark.parametrize(
    "data",
    [
        [np.array(0, dtype="u1"), (0,)],
        [np.array(0, dtype="u1"), (1,)],
        [np.array(0, dtype="u1"), (3,)],
        [np.ones(1, dtype="u1"), (1,)],
        [np.ones(1, dtype="u1"), (2,)],
        [np.ones(1, dtype="u1"), (1, 2, 3)],
        [np.arange(3, dtype="u1"), (3,)],
        [np.arange(3, dtype="u1"), (1, 3)],
        [np.arange(3, dtype="u1"), (2, 3)],
        [np.ones(0, dtype="u1"), 0],
        [np.ones(1, dtype="u1"), 1],
        [np.ones(1, dtype="u1"), 2],
        [np.ones(1, dtype="u1"), (0,)],
        [np.ones((1, 2), dtype="u1"), (0, 2)],
        [np.ones((2, 1), dtype="u1"), (2, 0)],
    ],
)
def test_broadcast_to_succeeds(data):
    q = get_queue_or_skip()

    Xnp, target_shape = data
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.broadcast_to(X, target_shape)
    Ynp = np.broadcast_to(Xnp, target_shape)
    assert_array_equal(dpt.asnumpy(Y), Ynp)


@pytest.mark.parametrize(
    "data",
    [
        [(0,), ()],
        [(1,), ()],
        [(3,), ()],
        [(3,), (1,)],
        [(3,), (2,)],
        [(3,), (4,)],
        [(1, 2), (2, 1)],
        [(1, 1), (1,)],
        [(1,), -1],
        [(1,), (-1,)],
        [(1, 2), (-1, 2)],
    ],
)
def test_broadcast_to_raises(data):
    q = get_queue_or_skip()

    orig_shape, target_shape = data
    Xnp = np.zeros(orig_shape, dtype="i1")
    X = dpt.asarray(Xnp, sycl_queue=q)
    pytest.raises(ValueError, dpt.broadcast_to, X, target_shape)


def assert_broadcast_correct(input_shapes):
    q = get_queue_or_skip()
    np_arrays = [np.zeros(s, dtype="i1") for s in input_shapes]
    out_np_arrays = np.broadcast_arrays(*np_arrays)
    usm_arrays = [dpt.asarray(Xnp, sycl_queue=q) for Xnp in np_arrays]
    out_usm_arrays = dpt.broadcast_arrays(*usm_arrays)
    for Xnp, X in zip(out_np_arrays, out_usm_arrays):
        assert_array_equal(
            Xnp, dpt.asnumpy(X), err_msg=f"Failed for {input_shapes})"
        )


def assert_broadcast_arrays_raise(input_shapes):
    q = get_queue_or_skip()
    usm_arrays = [dpt.asarray(np.zeros(s), sycl_queue=q) for s in input_shapes]
    pytest.raises(ValueError, dpt.broadcast_arrays, *usm_arrays)


def test_broadcast_arrays_same():
    q = get_queue_or_skip()
    Xnp = np.arange(10)
    Ynp = np.arange(10)
    res_Xnp, res_Ynp = np.broadcast_arrays(Xnp, Ynp)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.asarray(Ynp, sycl_queue=q)
    res_X, res_Y = dpt.broadcast_arrays(X, Y)
    assert_array_equal(res_Xnp, dpt.asnumpy(res_X))
    assert_array_equal(res_Ynp, dpt.asnumpy(res_Y))


def test_broadcast_arrays_one_off():
    q = get_queue_or_skip()
    Xnp = np.array([[1, 2, 3]])
    Ynp = np.array([[1], [2], [3]])
    res_Xnp, res_Ynp = np.broadcast_arrays(Xnp, Ynp)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.asarray(Ynp, sycl_queue=q)
    res_X, res_Y = dpt.broadcast_arrays(X, Y)
    assert_array_equal(res_Xnp, dpt.asnumpy(res_X))
    assert_array_equal(res_Ynp, dpt.asnumpy(res_Y))


@pytest.mark.parametrize(
    "shapes",
    [
        (),
        (1,),
        (3,),
        (0, 1),
        (0, 3),
        (1, 0),
        (3, 0),
        (1, 3),
        (3, 1),
        (3, 3),
    ],
)
def test_broadcast_arrays_same_shapes(shapes):
    for shape in shapes:
        single_input_shapes = [shape]
        assert_broadcast_correct(single_input_shapes)
        double_input_shapes = [shape, shape]
        assert_broadcast_correct(double_input_shapes)
        triple_input_shapes = [shape, shape, shape]
        assert_broadcast_correct(triple_input_shapes)


@pytest.mark.parametrize(
    "shapes",
    [
        [[(1,), (3,)]],
        [[(1, 3), (3, 3)]],
        [[(3, 1), (3, 3)]],
        [[(1, 3), (3, 1)]],
        [[(1, 1), (3, 3)]],
        [[(1, 1), (1, 3)]],
        [[(1, 1), (3, 1)]],
        [[(1, 0), (0, 0)]],
        [[(0, 1), (0, 0)]],
        [[(1, 0), (0, 1)]],
        [[(1, 1), (0, 0)]],
        [[(1, 1), (1, 0)]],
        [[(1, 1), (0, 1)]],
    ],
)
def test_broadcast_arrays_same_len_shapes(shapes):
    # Check that two different input shapes of the same length, but some have
    # ones, broadcast to the correct shape.

    for input_shapes in shapes:
        assert_broadcast_correct(input_shapes)
        assert_broadcast_correct(input_shapes[::-1])


@pytest.mark.parametrize(
    "shapes",
    [
        [[(), (3,)]],
        [[(3,), (3, 3)]],
        [[(3,), (3, 1)]],
        [[(1,), (3, 3)]],
        [[(), (3, 3)]],
        [[(1, 1), (3,)]],
        [[(1,), (3, 1)]],
        [[(1,), (1, 3)]],
        [[(), (1, 3)]],
        [[(), (3, 1)]],
        [[(), (0,)]],
        [[(0,), (0, 0)]],
        [[(0,), (0, 1)]],
        [[(1,), (0, 0)]],
        [[(), (0, 0)]],
        [[(1, 1), (0,)]],
        [[(1,), (0, 1)]],
        [[(1,), (1, 0)]],
        [[(), (1, 0)]],
        [[(), (0, 1)]],
    ],
)
def test_broadcast_arrays_different_len_shapes(shapes):
    # Check that two different input shapes (of different lengths) broadcast
    # to the correct shape.

    for input_shapes in shapes:
        assert_broadcast_correct(input_shapes)
        assert_broadcast_correct(input_shapes[::-1])


@pytest.mark.parametrize(
    "shapes",
    [
        [[(3,), (4,)]],
        [[(2, 3), (2,)]],
        [[(3,), (3,), (4,)]],
        [[(1, 3, 4), (2, 3, 3)]],
    ],
)
def test_incompatible_shapes_raise_valueerror(shapes):
    for input_shapes in shapes:
        assert_broadcast_arrays_raise(input_shapes)
        assert_broadcast_arrays_raise(input_shapes[::-1])


def test_broadcast_arrays_no_args():
    with pytest.raises(ValueError):
        dpt.broadcast_arrays()


def test_flip_axis_incorrect():
    q = get_queue_or_skip()

    X_np = np.ones((4, 4))
    X = dpt.asarray(X_np, sycl_queue=q)

    pytest.raises(AxisError, dpt.flip, dpt.asarray(np.ones(4)), axis=1)
    pytest.raises(AxisError, dpt.flip, X, axis=2)
    pytest.raises(AxisError, dpt.flip, X, axis=-3)
    pytest.raises(AxisError, dpt.flip, X, axis=(0, 3))


def test_flip_0d():
    q = get_queue_or_skip()

    Xnp = np.array(1, dtype="int64")
    X = dpt.asarray(Xnp, sycl_queue=q)
    Ynp = np.flip(Xnp)
    Y = dpt.flip(X)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    pytest.raises(AxisError, dpt.flip, X, axis=0)
    pytest.raises(AxisError, dpt.flip, X, axis=1)
    pytest.raises(AxisError, dpt.flip, X, axis=-1)


def test_flip_1d():
    q = get_queue_or_skip()

    Xnp = np.arange(6)
    X = dpt.asarray(Xnp, sycl_queue=q)

    for ax in range(-X.ndim, X.ndim):
        Ynp = np.flip(Xnp, axis=ax)
        Y = dpt.flip(X, axis=ax)
        assert_array_equal(Ynp, dpt.asnumpy(Y))

    Ynp = np.flip(Xnp, axis=0)
    Y = dpt.flip(X, axis=0)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "shapes",
    [
        (3, 2),
        (2, 3),
        (2, 2),
        (3, 3),
        (3, 2, 3),
        (2, 3, 2),
        (2, 2, 2),
        (3, 3, 3),
    ],
)
def test_flip_2d_3d(shapes):
    q = get_queue_or_skip()

    Xnp_size = np.prod(shapes)
    Xnp = np.arange(Xnp_size).reshape(shapes)
    X = dpt.asarray(Xnp, sycl_queue=q)
    for ax in range(-X.ndim, X.ndim):
        Y = dpt.flip(X, axis=ax)
        Ynp = np.flip(Xnp, axis=ax)
        assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "shapes",
    [
        (1,),
        (3,),
        (2, 3),
        (3, 2),
        (2, 2),
        (1, 2, 3),
        (2, 1, 3),
        (2, 3, 1),
        (3, 2, 1),
        (3, 3, 3),
    ],
)
def test_flip_default_axes(shapes):
    q = get_queue_or_skip()

    Xnp_size = np.prod(shapes)
    Xnp = np.arange(Xnp_size).reshape(shapes)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.flip(X)
    Ynp = np.flip(Xnp)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "shapes",
    [
        (0),
        (1),
        (1, 1),
        (1, 0),
        (0, 1),
        (1, 1, 1),
        (1, 0, 1),
        (0, 1, 0),
    ],
)
def test_flip_empty_0_size_dim(shapes):
    q = get_queue_or_skip()

    X = dpt.empty(shapes, sycl_queue=q)
    dpt.flip(X)


@pytest.mark.parametrize(
    "data",
    [
        [(2, 3), (0, 1)],
        [(2, 3), (1, 0)],
        [(2, 3), ()],
        [(2, 1, 3), (0, 2)],
        [(3, 1, 2), (2, 0)],
        [(3, 3, 3), (2,)],
        [(1, 2, 3), [0, -2]],
        [(3, 1, 2), [-1, 0]],
        [(3, 3, 3), [-2, -1]],
    ],
)
def test_flip_multiple_axes(data):
    q = get_queue_or_skip()

    shape, axes = data
    Xnp_size = np.prod(shape)
    Xnp = np.arange(Xnp_size).reshape(shape)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.flip(X, axis=axes)
    Ynp = np.flip(Xnp, axis=axes)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_roll_scalar():
    q = get_queue_or_skip()

    Xnp = np.ones([], dtype="f4")
    X = dpt.asarray(Xnp, sycl_queue=q)

    Y = dpt.roll(X, 1)
    Ynp = np.roll(Xnp, 1)
    assert_array_equal(Ynp, dpt.asnumpy(Y))
    with pytest.raises(AxisError):
        dpt.roll(X, 1, axis=0)
    with pytest.raises(AxisError):
        dpt.roll(X, 1, axis=1)


@pytest.mark.parametrize(
    "data",
    [
        [2, None],
        [-2, None],
        [2, 0],
        [-2, 0],
        [2, ()],
        [11, 0],
    ],
)
def test_roll_1d(data):
    q = get_queue_or_skip()

    Xnp = np.arange(10)
    X = dpt.asarray(Xnp, sycl_queue=q)
    sh, ax = data

    Y = dpt.roll(X, sh, axis=ax)
    Ynp = np.roll(Xnp, sh, axis=ax)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.roll(X, sh, axis=ax)
    Ynp = np.roll(Xnp, sh, axis=ax)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "data",
    [
        [1, None],
        [1, 0],
        [1, 1],
        [1, ()],
        # Roll multiple axes at once
        [1, (0, 1)],
        [(1, 0), (0, 1)],
        [(-1, 0), (1, 0)],
        [(0, 1), (0, 1)],
        [(0, -1), (0, 1)],
        [(1, 1), (0, 1)],
        [(-1, -1), (0, 1)],
        # Roll the same axis multiple times.
        [1, (0, 0)],
        [1, (1, 1)],
        # Roll more than one turn in either direction.
        [6, 1],
        [-4, 1],
    ],
)
def test_roll_2d(data):
    q = get_queue_or_skip()

    Xnp = np.arange(10).reshape(2, 5)
    X = dpt.asarray(Xnp, sycl_queue=q)
    sh, ax = data

    Y = dpt.roll(X, sh, axis=ax)
    Ynp = np.roll(Xnp, sh, axis=ax)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_roll_out_bounds_shifts():
    "See gh-1857"
    get_queue_or_skip()

    x = dpt.arange(4)
    y = dpt.roll(x, np.uint64(2**63 + 2))
    expected = dpt.roll(x, 2)
    assert dpt.all(y == expected)

    x_empty = x[1:1]
    y = dpt.roll(x_empty, 11)
    assert y.size == 0

    x_2d = dpt.reshape(x, (2, 2))
    y = dpt.roll(x_2d, np.uint64(2**63 + 1), axis=1)
    expected = dpt.roll(x_2d, 1, axis=1)
    assert dpt.all(y == expected)

    x_2d_empty = x_2d[:, 1:1]
    y = dpt.roll(x_2d_empty, 3, axis=1)
    expected = dpt.empty_like(x_2d_empty)
    assert dpt.all(y == expected)


def test_roll_validation():
    get_queue_or_skip()

    X = {}
    with pytest.raises(TypeError):
        dpt.roll(X)

    X = dpt.empty((1, 2, 3))
    shift = ((2, 3, 1), (1, 2, 3))
    with pytest.raises(ValueError):
        dpt.roll(X, shift=shift, axis=(0, 1, 2))


def test_concat_incorrect_type():
    Xnp = np.ones((2, 2))
    with pytest.raises(TypeError):
        dpt.concat()
    with pytest.raises(TypeError):
        dpt.concat([])
    with pytest.raises(TypeError):
        dpt.concat(Xnp)
    with pytest.raises(TypeError):
        dpt.concat([Xnp, Xnp])


def test_concat_incorrect_queue():
    q1 = get_queue_or_skip()
    q2 = get_queue_or_skip()

    X = dpt.ones((2, 2), sycl_queue=q1)
    Y = dpt.ones((2, 2), sycl_queue=q2)

    pytest.raises(ValueError, dpt.concat, [X, Y])


def test_concat_different_dtype():
    q = get_queue_or_skip()

    X = dpt.ones((2, 2), dtype=np.int64, sycl_queue=q)
    Y = dpt.ones((3, 2), dtype=np.uint32, sycl_queue=q)

    XY = dpt.concat([X, Y])

    assert XY.dtype is X.dtype
    assert XY.shape == (5, 2)
    assert XY.sycl_queue == q

    X1 = dpt.arange(10, dtype="i2", sycl_queue=q)
    Y1 = dpt.arange(5, dtype="i4", sycl_queue=q)

    XY1 = dpt.concat([X1[::2], Y1[::-1]], axis=None)
    assert XY1.shape == (10,)
    assert XY1.sycl_queue == q
    assert XY1.dtype == Y1.dtype


def test_concat_incorrect_ndim():
    q = get_queue_or_skip()

    X = dpt.ones((2, 2), sycl_queue=q)
    Y = dpt.ones((2, 2, 2), sycl_queue=q)

    pytest.raises(ValueError, dpt.concat, [X, Y])


@pytest.mark.parametrize(
    "data",
    [
        [(2, 2), (3, 3), 0],
        [(2, 2), (3, 3), 1],
        [(3, 2), (3, 3), 0],
        [(2, 3), (3, 3), 1],
    ],
)
def test_concat_incorrect_shape(data):
    q = get_queue_or_skip()

    Xshape, Yshape, axis = data

    X = dpt.ones(Xshape, sycl_queue=q)
    Y = dpt.ones(Yshape, sycl_queue=q)

    pytest.raises(ValueError, dpt.concat, [X, Y], axis=axis)


@pytest.mark.parametrize(
    "data",
    [
        [(6,), 0],
        [(2, 3), 1],
        [(3, 2), -1],
        [(1, 6), 0],
        [(2, 1, 3), 2],
    ],
)
def test_concat_1array(data):
    q = get_queue_or_skip()

    Xshape, axis = data

    Xnp = np.arange(6).reshape(Xshape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.concatenate([Xnp], axis=axis)
    Y = dpt.concat([X], axis=axis)

    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Ynp = np.concatenate((Xnp,), axis=axis)
    Y = dpt.concat((X,), axis=axis)

    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "data",
    [
        [(1,), (1,), 0],
        [(0, 2), (0, 2), 1],
        [(0, 2), (2, 2), 0],
        [(2, 1), (2, 2), -1],
        [(2, 2, 2), (2, 1, 2), 1],
        [(3, 3, 3), (2, 2), None],
    ],
)
def test_concat_2arrays(data):
    q = get_queue_or_skip()

    Xshape, Yshape, axis = data

    Xnp = np.ones(Xshape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.zeros(Yshape)
    Y = dpt.asarray(Ynp, sycl_queue=q)

    Znp = np.concatenate([Xnp, Ynp], axis=axis)
    Z = dpt.concat([X, Y], axis=axis)

    assert_array_equal(Znp, dpt.asnumpy(Z))


@pytest.mark.parametrize(
    "data",
    [
        [(1,), (1,), (1,), 0],
        [(0, 2), (2, 2), (1, 2), 0],
        [(2, 1, 2), (2, 2, 2), (2, 4, 2), 1],
    ],
)
def test_concat_3arrays(data):
    q = get_queue_or_skip()

    Xshape, Yshape, Zshape, axis = data

    Xnp = np.ones(Xshape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.zeros(Yshape)
    Y = dpt.asarray(Ynp, sycl_queue=q)

    Znp = np.full(Zshape, 2.0)
    Z = dpt.asarray(Znp, sycl_queue=q)

    Rnp = np.concatenate([Xnp, Ynp, Znp], axis=axis)
    R = dpt.concat([X, Y, Z], axis=axis)

    assert_array_equal(Rnp, dpt.asnumpy(R))


def test_concat_axis_none_strides():
    q = get_queue_or_skip()
    Xnp = np.arange(0, 18).reshape((6, 3))
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.arange(20, 36).reshape((4, 2, 2))
    Y = dpt.asarray(Ynp, sycl_queue=q)

    Znp = np.concatenate([Xnp[::2], Ynp[::2]], axis=None)
    Z = dpt.concat([X[::2], Y[::2]], axis=None)

    assert_array_equal(Znp, dpt.asnumpy(Z))


def test_stack_incorrect_shape():
    q = get_queue_or_skip()

    X = dpt.ones((1,), sycl_queue=q)
    Y = dpt.ones((2,), sycl_queue=q)

    with pytest.raises(ValueError):
        dpt.stack([X, Y], axis=0)


@pytest.mark.parametrize(
    "data",
    [
        [(6,), 0],
        [(2, 3), 1],
        [(3, 2), -1],
        [(1, 6), 2],
        [(2, 1, 3), 2],
    ],
)
def test_stack_1array(data):
    q = get_queue_or_skip()

    shape, axis = data

    Xnp = np.arange(6).reshape(shape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.stack([Xnp], axis=axis)
    Y = dpt.stack([X], axis=axis)

    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Ynp = np.stack((Xnp,), axis=axis)
    Y = dpt.stack((X,), axis=axis)

    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "data",
    [
        [(1,), 0],
        [(0, 2), 0],
        [(2, 0), 0],
        [(2, 3), 0],
        [(2, 3), 1],
        [(2, 3), 2],
        [(2, 3), -1],
        [(2, 3), -2],
        [(2, 2, 2), 1],
    ],
)
def test_stack_2arrays(data):
    q = get_queue_or_skip()

    shape, axis = data

    Xnp = np.ones(shape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.zeros(shape)
    Y = dpt.asarray(Ynp, sycl_queue=q)

    Znp = np.stack([Xnp, Ynp], axis=axis)
    Z = dpt.stack([X, Y], axis=axis)

    assert_array_equal(Znp, dpt.asnumpy(Z))


@pytest.mark.parametrize(
    "data",
    [
        [(1,), 0],
        [(0, 2), 0],
        [(2, 1, 2), 1],
    ],
)
def test_stack_3arrays(data):
    q = get_queue_or_skip()

    shape, axis = data

    Xnp = np.ones(shape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.zeros(shape)
    Y = dpt.asarray(Ynp, sycl_queue=q)

    Znp = np.full(shape, 2.0)
    Z = dpt.asarray(Znp, sycl_queue=q)

    Rnp = np.stack([Xnp, Ynp, Znp], axis=axis)
    R = dpt.stack([X, Y, Z], axis=axis)

    assert_array_equal(Rnp, dpt.asnumpy(R))


def test_can_cast():
    q = get_queue_or_skip()

    # incorrect input
    X = dpt.ones((2, 2), dtype=dpt.int16, sycl_queue=q)
    pytest.raises(TypeError, dpt.can_cast, X, 1)
    pytest.raises(TypeError, dpt.can_cast, X, X)
    X_np = np.ones((2, 2), dtype=np.int16)

    assert dpt.can_cast(X, "float32") == np.can_cast(X_np, "float32")
    assert dpt.can_cast(X, dpt.int32) == np.can_cast(X_np, np.int32)
    assert dpt.can_cast(X, dpt.int64) == np.can_cast(X_np, np.int64)


def test_result_type():
    q = get_queue_or_skip()

    usm_ar = dpt.ones((2), dtype=dpt.int16, sycl_queue=q)
    np_ar = dpt.asnumpy(usm_ar)

    X = [usm_ar, dpt.int32, "int64", usm_ar]
    X_np = [np_ar, np.int32, "int64", np_ar]

    assert dpt.result_type(*X) == np.result_type(*X_np)

    X = [usm_ar, dpt.int32, "int64", True]
    X_np = [np_ar, np.int32, "int64", True]

    assert dpt.result_type(*X) == np.result_type(*X_np)

    X = [usm_ar, dpt.int32, "int64", 2]
    X_np = [np_ar, np.int32, "int64", 2]

    assert dpt.result_type(*X) == np.result_type(*X_np)

    X = [dpt.int32, "int64", 2]
    X_np = [np.int32, "int64", 2]

    assert dpt.result_type(*X) == np.result_type(*X_np)

    X = [usm_ar, dpt.int32, "int64", 2.0]
    X_np = [np_ar, np.int32, "int64", 2.0]

    assert dpt.result_type(*X).kind == np.result_type(*X_np).kind

    X = [usm_ar, dpt.int32, "int64", 2.0 + 1j]
    X_np = [np_ar, np.int32, "int64", 2.0 + 1j]

    assert dpt.result_type(*X).kind == np.result_type(*X_np).kind


def test_swapaxes_1d():
    get_queue_or_skip()
    x = np.array([[1, 2, 3]])
    exp = np.swapaxes(x, 0, 1)

    y = dpt.asarray([[1, 2, 3]])
    res = dpt.swapaxes(y, 0, 1)

    assert_array_equal(exp, dpt.asnumpy(res))


def test_swapaxes_2d():
    get_queue_or_skip()
    x = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    exp = np.swapaxes(x, 0, 2)

    y = dpt.asarray([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    res = dpt.swapaxes(y, 0, 2)

    assert_array_equal(exp, dpt.asnumpy(res))


@pytest.mark.parametrize(
    "source, expected",
    [
        (0, (6, 7, 5)),
        (1, (5, 7, 6)),
        (2, (5, 6, 7)),
        (-1, (5, 6, 7)),
    ],
)
def test_moveaxis_move_to_end(source, expected):
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(5 * 6 * 7), (5, 6, 7))
    actual = dpt.moveaxis(x, source, -1).shape
    assert_(actual, expected)


@pytest.mark.parametrize(
    "source, destination, expected",
    [
        (0, 1, (2, 1, 3, 4)),
        (1, 2, (1, 3, 2, 4)),
        (1, -1, (1, 3, 4, 2)),
    ],
)
def test_moveaxis_new_position(source, destination, expected):
    get_queue_or_skip()
    x = dpt.reshape(dpt.arange(24), (1, 2, 3, 4))
    actual = dpt.moveaxis(x, source, destination).shape
    assert_(actual, expected)


@pytest.mark.parametrize(
    "source, destination",
    [
        (0, 0),
        (3, -1),
        (-1, 3),
        ([0, -1], [0, -1]),
        ([2, 0], [2, 0]),
    ],
)
def test_moveaxis_preserve_order(source, destination):
    get_queue_or_skip()
    x = dpt.zeros((1, 2, 3, 4))
    actual = dpt.moveaxis(x, source, destination).shape
    assert_(actual, (1, 2, 3, 4))


@pytest.mark.parametrize(
    "shape, source, destination, expected",
    [
        ((0, 1, 2, 3), [0, 1], [2, 3], (2, 3, 0, 1)),
        ((0, 1, 2, 3), [2, 3], [0, 1], (2, 3, 0, 1)),
        ((0, 1, 2, 3), [0, 1, 2], [2, 3, 0], (2, 3, 0, 1)),
        ((0, 1, 2, 3), [3, 0], [1, 0], (0, 3, 1, 2)),
        ((0, 1, 2, 3), [0, 3], [0, 1], (0, 3, 1, 2)),
        ((1, 2, 3, 4), range(4), range(4), (1, 2, 3, 4)),
    ],
)
def test_moveaxis_move_multiples(shape, source, destination, expected):
    get_queue_or_skip()
    x = dpt.zeros(shape)
    y = dpt.moveaxis(x, source, destination)
    actual = y.shape
    assert_(actual, expected)
    assert y._pointer == x._pointer


def test_moveaxis_errors():
    try:
        x_flat = dpt.arange(6)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    x = dpt.reshape(x_flat, (1, 2, 3))
    assert_raises_regex(
        AxisError, "source.*out of bounds", dpt.moveaxis, x, 3, 0
    )
    assert_raises_regex(
        AxisError, "source.*out of bounds", dpt.moveaxis, x, -4, 0
    )
    assert_raises_regex(
        AxisError, "destination.*out of bounds", dpt.moveaxis, x, 0, 5
    )
    assert_raises_regex(
        ValueError, "repeated axis in `source`", dpt.moveaxis, x, [0, 0], [0, 1]
    )
    assert_raises_regex(
        ValueError,
        "repeated axis in `destination`",
        dpt.moveaxis,
        x,
        [0, 1],
        [1, 1],
    )
    assert_raises_regex(
        ValueError, "must have the same number", dpt.moveaxis, x, 0, [0, 1]
    )
    assert_raises_regex(
        ValueError, "must have the same number", dpt.moveaxis, x, [0, 1], [0]
    )


def test_unstack_axis0():
    try:
        x_flat = dpt.arange(6)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    y = dpt.reshape(x_flat, (2, 3))
    res = dpt.unstack(y)

    assert_array_equal(dpt.asnumpy(y[0, ...]), dpt.asnumpy(res[0]))
    assert_array_equal(dpt.asnumpy(y[1, ...]), dpt.asnumpy(res[1]))


def test_unstack_axis1():
    try:
        x_flat = dpt.arange(6)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    y = dpt.reshape(x_flat, (2, 3))
    res = dpt.unstack(y, axis=1)

    assert_array_equal(dpt.asnumpy(y[:, 0, ...]), dpt.asnumpy(res[0]))
    assert_array_equal(dpt.asnumpy(y[:, 1, ...]), dpt.asnumpy(res[1]))
    assert_array_equal(dpt.asnumpy(y[:, 2, ...]), dpt.asnumpy(res[2]))


def test_unstack_axis2():
    try:
        x_flat = dpt.arange(60)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    y = dpt.reshape(x_flat, (4, 5, 3))
    res = dpt.unstack(y, axis=2)

    assert_array_equal(dpt.asnumpy(y[:, :, 0, ...]), dpt.asnumpy(res[0]))
    assert_array_equal(dpt.asnumpy(y[:, :, 1, ...]), dpt.asnumpy(res[1]))
    assert_array_equal(dpt.asnumpy(y[:, :, 2, ...]), dpt.asnumpy(res[2]))


def test_finfo_object():
    fi = dpt.finfo(dpt.float32)
    assert isinstance(fi.bits, int)
    assert isinstance(fi.max, float)
    assert isinstance(fi.min, float)
    assert isinstance(fi.eps, float)
    assert isinstance(fi.epsneg, float)
    assert isinstance(fi.smallest_normal, float)
    assert isinstance(fi.tiny, float)
    assert isinstance(fi.precision, float)
    assert isinstance(fi.resolution, float)
    assert isinstance(fi.dtype, dpt.dtype)
    assert isinstance(str(fi), str)
    assert isinstance(repr(fi), str)


def test_repeat_scalar_sequence_agreement():
    get_queue_or_skip()

    x = dpt.arange(5, dtype="i4")
    expected_res = dpt.empty(10, dtype="i4")
    expected_res[1::2], expected_res[::2] = x, x

    # scalar case
    reps = 2
    res = dpt.repeat(x, reps)
    assert dpt.all(res == expected_res)

    # tuple
    reps = (2, 2, 2, 2, 2)
    res = dpt.repeat(x, reps)
    assert dpt.all(res == expected_res)


def test_repeat_as_broadcasting():
    get_queue_or_skip()

    reps = 5
    x = dpt.arange(reps, dtype="i4")
    x1 = x[:, dpt.newaxis]
    expected_res = dpt.broadcast_to(x1, (reps, reps))

    res = dpt.repeat(x1, reps, axis=1)
    assert dpt.all(res == expected_res)

    x2 = x[dpt.newaxis, :]
    expected_res = dpt.broadcast_to(x2, (reps, reps))

    res = dpt.repeat(x2, reps, axis=0)
    assert dpt.all(res == expected_res)


def test_repeat_axes():
    get_queue_or_skip()

    reps = 2
    x = dpt.reshape(dpt.arange(5 * 10, dtype="i4"), (5, 10))
    expected_res = dpt.empty((x.shape[0] * 2, x.shape[1]), dtype=x.dtype)
    expected_res[::2, :], expected_res[1::2] = x, x
    res = dpt.repeat(x, reps, axis=0)
    assert dpt.all(res == expected_res)

    expected_res = dpt.empty((x.shape[0], x.shape[1] * 2), dtype=x.dtype)
    expected_res[:, ::2], expected_res[:, 1::2] = x, x
    res = dpt.repeat(x, reps, axis=1)
    assert dpt.all(res == expected_res)

    x = dpt.arange(10, dtype="i4")
    expected_res = dpt.empty(x.shape[0] * reps, dtype=x.dtype)
    expected_res[::2], expected_res[1::2] = x, x
    res = dpt.repeat(x, reps, axis=0)
    assert dpt.all(res == expected_res)


def test_repeat_size_0_outputs():
    get_queue_or_skip()

    x = dpt.ones((3, 0, 5), dtype="i4")
    reps = 10
    res = dpt.repeat(x, reps, axis=0)
    assert res.size == 0
    assert res.shape == (30, 0, 5)

    res = dpt.repeat(x, reps, axis=1)
    assert res.size == 0
    assert res.shape == (3, 0, 5)

    res = dpt.repeat(x, (2, 2, 2), axis=0)
    assert res.size == 0
    assert res.shape == (6, 0, 5)

    x = dpt.ones((3, 2, 5))
    res = dpt.repeat(x, 0, axis=1)
    assert res.size == 0
    assert res.shape == (3, 0, 5)

    res = dpt.repeat(x, (0, 0), axis=1)
    assert res.size == 0
    assert res.shape == (3, 0, 5)

    # axis=None cases
    res = dpt.repeat(x, 0)
    assert res.size == 0

    res = dpt.repeat(x, (0,) * x.size)
    assert res.size == 0


def test_repeat_strides():
    get_queue_or_skip()

    reps = 2
    x = dpt.reshape(dpt.arange(10 * 10, dtype="i4"), (10, 10))
    x1 = x[:, ::-2]
    expected_res = dpt.empty((10, 10), dtype="i4")
    expected_res[:, ::2], expected_res[:, 1::2] = x1, x1
    res = dpt.repeat(x1, reps, axis=1)
    assert dpt.all(res == expected_res)
    res = dpt.repeat(x1, (reps,) * x1.shape[1], axis=1)
    assert dpt.all(res == expected_res)

    x1 = x[::-2, :]
    expected_res = dpt.empty((10, 10), dtype="i4")
    expected_res[::2, :], expected_res[1::2, :] = x1, x1
    res = dpt.repeat(x1, reps, axis=0)
    assert dpt.all(res == expected_res)
    res = dpt.repeat(x1, (reps,) * x1.shape[0], axis=0)
    assert dpt.all(res == expected_res)

    # axis=None
    x = dpt.reshape(dpt.arange(10 * 10), (10, 10))
    x1 = dpt.reshape(x[::-2, :], -1)
    x2 = x[::-2, :]
    expected_res = dpt.empty(10 * 10, dtype="i4")
    expected_res[::2], expected_res[1::2] = x1, x1
    res = dpt.repeat(x2, reps)
    assert dpt.all(res == expected_res)
    res = dpt.repeat(x2, (reps,) * x1.size)
    assert dpt.all(res == expected_res)


def test_repeat_casting():
    get_queue_or_skip()

    x = dpt.arange(5, dtype="i4")
    # i4 is cast to i8
    reps = dpt.ones(5, dtype="i4")
    res = dpt.repeat(x, reps)
    assert res.shape == x.shape
    assert dpt.all(res == x)


def test_repeat_strided_repeats():
    get_queue_or_skip()

    x = dpt.arange(5, dtype="i4")
    reps = dpt.ones(10, dtype="i8")
    reps[::2] = 0
    reps = reps[::-2]
    res = dpt.repeat(x, reps)
    assert res.shape == x.shape
    assert dpt.all(res == x)


def test_repeat_size1_repeats():
    get_queue_or_skip()

    x = dpt.arange(5, dtype="i4")
    expected_res = dpt.repeat(x, 2)
    # 0D repeats
    reps_0d = dpt.asarray(2, dtype="i8")
    res = dpt.repeat(x, reps_0d)
    assert dpt.all(res == expected_res)
    # 1D repeats
    reps_1d = dpt.asarray([2], dtype="i8")
    res = dpt.repeat(x, reps_1d)
    assert dpt.all(res == expected_res)


def test_repeat_arg_validation():
    get_queue_or_skip()

    x = {}
    with pytest.raises(TypeError):
        dpt.repeat(x, 2)

    # axis must be 0 for scalar
    x = dpt.empty(())
    with pytest.raises(ValueError):
        dpt.repeat(x, 2, axis=1)

    # repeats must be positive
    x = dpt.empty(5)
    with pytest.raises(ValueError):
        dpt.repeat(x, -2)

    # repeats must be integers
    with pytest.raises(TypeError):
        dpt.repeat(x, 2.0)

    # repeats tuple must be the same length as axis
    with pytest.raises(ValueError):
        dpt.repeat(x, (1, 2))

    # repeats tuple elements must be positive
    with pytest.raises(ValueError):
        dpt.repeat(x, (-1,))

    # repeats must be int or tuple
    with pytest.raises(TypeError):
        dpt.repeat(x, dict())

    # repeats array must be 0d or 1d
    with pytest.raises(ValueError):
        dpt.repeat(x, dpt.ones((1, 1), dtype="i8"))

    # repeats must be castable to i8
    with pytest.raises(TypeError):
        dpt.repeat(x, dpt.asarray(2.0, dtype="f4"))

    # compute follows data
    q2 = dpctl.SyclQueue()
    reps = dpt.asarray(1, dtype="i8", sycl_queue=q2)
    with pytest.raises(ExecutionPlacementError):
        dpt.repeat(x, reps)

    # repeats array must not contain negative elements
    reps = dpt.asarray(-1, dtype="i8")
    with pytest.raises(ValueError):
        dpt.repeat(x, reps)
    reps = dpt.asarray([1, 1, 1, 1, -1], dtype="i8")
    with pytest.raises(ValueError):
        dpt.repeat(x, reps)

    # repeats must broadcastable to axis size
    reps = dpt.arange(10, dtype="i8")
    with pytest.raises(ValueError):
        dpt.repeat(x, reps)


def test_tile_basic():
    get_queue_or_skip()

    reps = 2
    x = dpt.arange(5, dtype="i4")
    res = dpt.tile(x, reps)
    assert res.shape == (x.shape[0] * reps,)
    assert dpt.all(res[: x.size] == res[x.size :])

    reps = (2, 1)
    expected_sh = (2, x.shape[0])
    expected_res = dpt.broadcast_to(x, expected_sh)
    res = dpt.tile(x, reps)
    assert res.shape == expected_sh
    assert dpt.all(expected_res == res)


def test_tile_size_1():
    get_queue_or_skip()

    reps = 5
    # test for 0d array
    x1 = dpt.asarray(2, dtype="i4")
    res = dpt.tile(x1, reps)
    assert dpt.all(res == dpt.full(reps, 2, dtype="i4"))

    # test for 1d array with single element
    x2 = dpt.asarray([2], dtype="i4")
    res = dpt.tile(x2, reps)
    assert dpt.all(res == dpt.full(reps, 2, dtype="i4"))

    reps = ()
    # test for gh-1627 behavior
    res = dpt.tile(x1, reps)
    assert x1.shape == res.shape
    assert_array_equal(dpt.asnumpy(x1), dpt.asnumpy(res))

    res = dpt.tile(x2, reps)
    assert x2.shape == res.shape
    assert_array_equal(dpt.asnumpy(x2), dpt.asnumpy(res))


def test_tile_prepends_axes():
    get_queue_or_skip()

    reps = (2,)
    x = dpt.ones((5, 10), dtype="i4")
    expected_res = dpt.ones((5, 20), dtype="i4")
    res = dpt.tile(x, reps)
    assert dpt.all(res == expected_res)

    reps = (3, 2, 2)
    expected_res = dpt.ones((3, 10, 20), dtype="i4")
    res = dpt.tile(x, reps)
    assert dpt.all(res == expected_res)


def test_tile_empty_outputs():
    get_queue_or_skip()

    x = dpt.asarray((), dtype="i4")
    reps = 10
    res = dpt.tile(x, reps)
    assert res.size == 0
    assert res.shape == (0,)

    x = dpt.ones((3, 0, 5), dtype="i4")
    res = dpt.tile(x, reps)
    assert res.size == 0
    assert res.shape == (3, 0, 50)

    reps = (2, 1, 2)
    res = dpt.tile(x, reps)
    assert res.size == 0
    assert res.shape == (6, 0, 10)

    x = dpt.ones((2, 3, 4), dtype="i4")
    reps = (0, 1, 1)
    res = dpt.tile(x, reps)
    assert res.size == 0
    assert res.shape == (0, 3, 4)


def test_tile_strides():
    get_queue_or_skip()

    reps = (1, 2)
    x = dpt.reshape(dpt.arange(10 * 10, dtype="i4"), (10, 10))
    x1 = x[:, ::-2]
    expected_res = dpt.empty((10, 10), dtype="i4")
    expected_res[:, : x1.shape[1]], expected_res[:, x1.shape[1] :] = x1, x1
    res = dpt.tile(x1, reps)
    assert dpt.all(res == expected_res)

    reps = (2, 1)
    x1 = x[::-2, :]
    expected_res = dpt.empty((10, 10), dtype="i4")
    expected_res[: x1.shape[0], :], expected_res[x1.shape[0] :, :] = x1, x1
    res = dpt.tile(x1, reps)
    assert dpt.all(res == expected_res)


def test_tile_size_1_axes():
    get_queue_or_skip()

    reps = (1, 2, 1)
    x = dpt.ones((2, 1, 3), dtype="i4")
    res = dpt.tile(x, reps)
    expected_res = dpt.broadcast_to(x, (2, 2, 3))
    assert dpt.all(res == expected_res)


def test_tile_arg_validation():
    get_queue_or_skip()

    with pytest.raises(TypeError):
        dpt.tile(dict(), 2)

    # repetitions must be int or tuple
    x = dpt.empty(())
    with pytest.raises(TypeError):
        dpt.tile(x, dict())


def test_repeat_0_size():
    get_queue_or_skip()

    x = dpt.ones((0, 10, 0), dtype="i4")
    repetitions = 2
    res = dpt.repeat(x, repetitions)
    assert res.shape == (0,)
    res = dpt.repeat(x, repetitions, axis=2)
    assert res.shape == x.shape
    res = dpt.repeat(x, repetitions, axis=1)
    axis_sz = x.shape[1] * repetitions
    assert res.shape == (0, 20, 0)

    repetitions = dpt.asarray(2, dtype="i4")
    res = dpt.repeat(x, repetitions)
    assert res.shape == (0,)
    res = dpt.repeat(x, repetitions, axis=2)
    assert res.shape == x.shape
    res = dpt.repeat(x, repetitions, axis=1)
    assert res.shape == (0, 20, 0)

    repetitions = dpt.arange(10, dtype="i4")
    res = dpt.repeat(x, repetitions, axis=1)
    axis_sz = dpt.sum(repetitions)
    assert res.shape == (0, axis_sz, 0)

    repetitions = (2,) * 10
    res = dpt.repeat(x, repetitions, axis=1)
    axis_sz = 2 * x.shape[1]
    assert res.shape == (0, axis_sz, 0)


def test_result_type_bug_1874():
    py_sc = True
    np_sc = np.asarray([py_sc])[0]
    dts_bool = [py_sc, np_sc]
    py_sc = int(1)
    np_sc = np.asarray([py_sc])[0]
    dts_ints = [py_sc, np_sc]
    dts_floats = [float(1), np.float64(1)]
    dts_complexes = [complex(1), np.complex128(1)]

    # iterate over two categories
    for dts1, dts2 in itertools.product(
        [dts_bool, dts_ints, dts_floats, dts_complexes], repeat=2
    ):
        res_dts = []
        # iterate over Python scalar/NumPy scalar choices within categories
        for dt1, dt2 in itertools.product(dts1, dts2):
            res_dt = dpt.result_type(dt1, dt2)
            res_dts.append(res_dt)
        # check that all results are the same
        assert res_dts and all(res_dts[0] == el for el in res_dts[1:])
