import warnings

import dpctl
import numpy
import pytest

import dpnp as cupy
from tests.third_party.cupy import testing


class TestBasic:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_different_contiguity(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 2), xp, dtype).T
        b = xp.empty((2, 3, 2), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_dtype(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype="?")
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_broadcast(self, xp, dtype):
        a = testing.shaped_arange((3, 1), xp, dtype)
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @pytest.mark.parametrize(
        ("dst_shape", "src_shape"),
        [
            ((), (2,)),
            ((2, 0, 5, 4), (2, 0, 3, 4)),
            ((6,), (2, 3)),
            ((2, 3), (6,)),
        ],
    )
    def test_copyto_raises_shape(self, dst_shape, src_shape):
        for xp in (numpy, cupy):
            dst = xp.zeros(dst_shape, dtype=xp.int64)
            src = xp.zeros(src_shape, dtype=xp.int64)
            with pytest.raises(ValueError):
                xp.copyto(dst, src)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_squeeze(self, xp, dtype):
        a = testing.shaped_arange((1, 1, 3, 4), xp, dtype)
        b = xp.empty((3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_squeeze_different_contiguity(self, xp, dtype):
        a = testing.shaped_arange((1, 1, 3, 4), xp, dtype)
        b = xp.empty((4, 3), dtype=dtype).T
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_squeeze_broadcast(self, xp, dtype):
        a = testing.shaped_arange((1, 2, 1, 4), xp, dtype)
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_where(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xp, dtype)
        c = testing.shaped_arange((2, 3, 4), xp, "?")
        xp.copyto(a, b, where=c)
        return a

    @pytest.mark.parametrize("shape", [(2, 3, 4), (0,)])
    @testing.for_all_dtypes(no_bool=True)
    def test_copyto_where_raises(self, dtype, shape):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(shape, xp, "i")
            b = testing.shaped_reverse_arange(shape, xp, "i")
            c = testing.shaped_arange(shape, xp, dtype)
            with pytest.raises(TypeError):
                xp.copyto(a, b, where=c)

    @testing.for_all_dtypes()
    def test_copyto_where_multidevice_raises(self, dtype):
        a = testing.shaped_arange(
            (2, 3, 4), cupy, dtype, device=dpctl.SyclQueue()
        )
        b = testing.shaped_reverse_arange(
            (2, 3, 4), cupy, dtype, device=dpctl.SyclQueue()
        )
        c = testing.shaped_arange(
            (2, 3, 4), cupy, "?", device=dpctl.SyclQueue()
        )
        with pytest.raises(
            dpctl.utils.ExecutionPlacementError,
            match="arrays have different associated queues",
        ):
            cupy.copyto(a, b, where=c)

    @testing.for_all_dtypes()
    def test_copyto_noncontinguous(self, dtype):
        src = testing.shaped_arange((2, 3, 4), cupy, dtype)
        src = src.swapaxes(0, 1)

        dst = cupy.empty_like(src)
        cupy.copyto(dst, src)

        expected = testing.shaped_arange((2, 3, 4), numpy, dtype)
        expected = expected.swapaxes(0, 1)

        testing.assert_array_equal(expected, src)
        testing.assert_array_equal(expected, dst)


@testing.parameterize(
    *testing.product(
        {
            "src": [float(3.2), int(0), int(4), int(-4), True, False, 1 + 1j],
            "dst_shape": [(), (0,), (1,), (1, 1), (2, 2)],
        }
    )
)
class TestCopytoFromScalar:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto(self, xp, dtype):
        dst = xp.ones(self.dst_shape, dtype=dtype)
        xp.copyto(dst, self.src)
        return dst

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto_where(self, xp, dtype):
        dst = xp.ones(self.dst_shape, dtype=dtype)
        mask = (testing.shaped_arange(self.dst_shape, xp, dtype) % 2).astype(
            xp.bool_
        )
        xp.copyto(dst, self.src, where=mask)
        return dst


@pytest.mark.parametrize(
    "casting", ["no", "equiv", "safe", "same_kind", "unsafe"]
)
class TestCopytoFromNumpyScalar:
    @testing.for_all_dtypes_combination(("dtype1", "dtype2"))
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto(self, xp, dtype1, dtype2, casting):
        if casting == "safe":
            pytest.skip(
                "NEP50 doesn't work properly in numpy with casting='safe'"
            )

        dst = xp.zeros((2, 3, 4), dtype=dtype1)
        src = numpy.array(1, dtype=dtype2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", numpy.ComplexWarning)
            xp.copyto(dst, src, casting)
        return dst

    @testing.for_all_dtypes()
    @pytest.mark.parametrize(
        "make_src",
        [lambda dtype: numpy.array([1], dtype=dtype), lambda dtype: dtype(1)],
    )
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto2(self, xp, make_src, dtype, casting):
        dst = xp.zeros((2, 3, 4), dtype=dtype)
        src = make_src(dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", numpy.ComplexWarning)
            xp.copyto(dst, src, casting)
        return dst

    @testing.for_all_dtypes_combination(("dtype1", "dtype2"))
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto_where(self, xp, dtype1, dtype2, casting):
        if casting == "safe":
            pytest.skip(
                "NEP50 doesn't work properly in numpy with casting='safe'"
            )

        shape = (2, 3, 4)
        dst = xp.ones(shape, dtype=dtype1)
        src = numpy.array(1, dtype=dtype2)
        mask = (testing.shaped_arange(shape, xp, dtype1) % 2).astype(xp.bool_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", numpy.ComplexWarning)
            xp.copyto(dst, src, casting=casting, where=mask)
        return dst


@pytest.mark.parametrize("shape", [(3, 2), (0,)])
@pytest.mark.parametrize(
    "where", [float(3.2), int(0), int(4), int(-4), True, False, 1 + 1j]
)
@testing.for_all_dtypes()
@testing.numpy_cupy_allclose()
def test_copyto_scalarwhere(xp, dtype, where, shape):
    dst = xp.zeros(shape, dtype=dtype)
    src = xp.ones(shape, dtype=dtype)
    xp.copyto(dst, src, where=where)
    return dst
