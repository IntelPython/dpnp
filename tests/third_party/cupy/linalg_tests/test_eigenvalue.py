import unittest

import numpy
import pytest

import dpnp as cupy
from tests.third_party.cupy import testing


def _get_hermitian(xp, a, UPLO):
    # TODO: fix this, currently dpnp.transpose() doesn't support complex types
    # and no dpnp_array.swapaxes()
    a = _wrap_as_numpy_array(xp, a)
    _xp = numpy

    if UPLO == 'U':
        _a = _xp.triu(a) + _xp.triu(a, k=1).swapaxes(-2, -1).conj()
    else:
        _a = _xp.tril(a) + _xp.tril(a, k=-1).swapaxes(-2, -1).conj()
    return xp.array(_a)

# TODO: remove once all required functionality is supported
def _wrap_as_numpy_array(xp, a):
    return a.asnumpy() if xp is cupy else a

@testing.parameterize(*testing.product({
    'UPLO': ['U', 'L'],
}))
class TestEigenvalue(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eigh(self, xp, dtype):
        if xp == numpy and dtype == numpy.float16:
            # NumPy's eigh does not support float16
            _dtype = 'f'
        else:
            _dtype = dtype
        if numpy.dtype(_dtype).kind == 'c':
            a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], _dtype)
        else:
            a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], _dtype)
        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)

        # Changed the verification method to check if Av and vw match, since
        # the eigenvectors of eigh() with CUDA 11.6 are mathematically correct
        # but may not match NumPy.
        A = _get_hermitian(xp, a, self.UPLO)
        if _dtype == numpy.float16:
            tol = 1e-3
        else:
            tol = 1e-5

        # TODO: remove _wrap_as_numpy_array() once @ support complex types
        testing.assert_allclose(_wrap_as_numpy_array(xp, A) @ _wrap_as_numpy_array(xp, v),
                                _wrap_as_numpy_array(xp, v) @ numpy.diag(_wrap_as_numpy_array(xp, w)),
                                atol=tol, rtol=tol)

        # Check if v @ vt is an identity matrix
        testing.assert_allclose(_wrap_as_numpy_array(xp, v) @ _wrap_as_numpy_array(xp, v).swapaxes(-2, -1).conj(),
                                numpy.identity(_wrap_as_numpy_array(xp, A).shape[-1], _dtype), atol=tol,
                                rtol=tol)
        if xp == numpy and dtype == numpy.float16:
            w = w.astype('e')
        return w

    @testing.for_all_dtypes(no_bool=True, no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eigh_batched(self, xp, dtype):
        a = xp.array([[[1, 0, 3], [0, 5, 0], [7, 0, 9]],
                      [[3, 0, 3], [0, 7, 0], [7, 0, 11]]], dtype)
        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)

        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so w's should be directly comparable. However, both cuSOLVER
        # and rocSOLVER pick a different convention for constructing
        # eigenvectors, so v's are not directly comparible and we verify
        # them through the eigen equation A*v=w*v.
        A = _get_hermitian(xp, a, self.UPLO)
        for i in range(a.shape[0]):
            testing.assert_allclose(
                A[i].dot(v[i]), w[i]*v[i], rtol=1e-5, atol=1e-5)
        return w

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eigh_complex_batched(self, xp, dtype):
        print()
        a = xp.array([[[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]],
                      [[0, 2j, 3], [4j, 4, 6j], [7, 8j, 8]]], dtype)
        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)

        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so w's should be directly comparable. However, both cuSOLVER
        # and rocSOLVER pick a different convention for constructing
        # eigenvectors, so v's are not directly comparible and we verify
        # them through the eigen equation A*v=w*v.
        A = _get_hermitian(xp, a, self.UPLO)

        # TODO: remove _wrap_as_numpy_array() once dpnp.dot() support complex types
        A = _wrap_as_numpy_array(xp, A)
        v = _wrap_as_numpy_array(xp, v)
        w = _wrap_as_numpy_array(xp, w)

        for i in range(a.shape[0]):
            testing.assert_allclose(
                A[i].dot(v[i]), w[i]*v[i], rtol=1e-5, atol=1e-5)
        return w

    @pytest.mark.skip("No support of dpnp.eigvalsh()")
    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvalsh(self, xp, dtype):
        a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so they should be directly comparable
        return w

    @pytest.mark.skip("No support of dpnp.eigvalsh()")
    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvalsh_batched(self, xp, dtype):
        a = xp.array([[[1, 0, 3], [0, 5, 0], [7, 0, 9]],
                      [[3, 0, 3], [0, 7, 0], [7, 0, 11]]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so they should be directly comparable
        return w

    @pytest.mark.skip("No support of dpnp.eigvalsh()")
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvalsh_complex(self, xp, dtype):
        a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so they should be directly comparable
        return w

    @pytest.mark.skip("No support of dpnp.eigvalsh()")
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvalsh_complex_batched(self, xp, dtype):
        a = xp.array([[[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]],
                      [[0, 2j, 3], [4j, 4, 6j], [7, 8j, 8]]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so they should be directly comparable
        return w


@testing.parameterize(*testing.product({
    'UPLO': ['U', 'L'],
    'shape': [(0, 0),
              (2, 0, 0),
              (0, 3, 3)]
}))
class TestEigenvalueEmpty(unittest.TestCase):

    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose()
    def test_eigh(self, xp, dtype):
        a = xp.empty(self.shape, dtype=dtype)
        assert a.size == 0
        return xp.linalg.eigh(a, UPLO=self.UPLO)

    @pytest.mark.skip("No support of dpnp.eigvalsh()")
    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose()
    def test_eigvalsh(self, xp, dtype):
        a = xp.empty(self.shape, dtype)
        assert a.size == 0
        return xp.linalg.eigvalsh(a, UPLO=self.UPLO)


@testing.parameterize(*testing.product({
    'UPLO': ['U', 'L'],
    'shape': [(),
              (3,),
              (2, 3),
              (4, 0),
              (2, 2, 3),
              (0, 2, 3)]
}))
class TestEigenvalueInvalid(unittest.TestCase):

    def test_eigh_shape_error(self):
        for xp in (numpy, cupy):
            a = xp.zeros(self.shape)
            with pytest.raises((numpy.linalg.LinAlgError, ValueError)):
                xp.linalg.eigh(a, self.UPLO)

    @pytest.mark.skip("No support of dpnp.eigvalsh()")
    def test_eigvalsh_shape_error(self):
        for xp in (numpy, cupy):
            a = xp.zeros(self.shape)
            with pytest.raises((numpy.linalg.LinAlgError, ValueError)):
                xp.linalg.eigvalsh(a, self.UPLO)
