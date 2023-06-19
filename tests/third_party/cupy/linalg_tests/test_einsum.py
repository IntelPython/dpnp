import unittest
import warnings

import numpy
import pytest

import dpnp as cupy
from tests.third_party.cupy import testing


def _dec_shape(shape, dec):
    # Test smaller shape
    return tuple(1 if s == 1 else max(0, s - dec) for s in shape)


@pytest.mark.usefixtures('allow_fall_back_on_numpy')
class TestEinSumError(unittest.TestCase):

    def test_irregular_ellipsis1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('..', xp.zeros((2, 2, 2)))

    def test_irregular_ellipsis2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('...i...', xp.zeros((2, 2, 2)))

    def test_irregular_ellipsis3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i...->...i...', xp.zeros((2, 2, 2)))

    def test_irregular_ellipsis4(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('...->', xp.zeros((2, 2, 2)))

    def test_no_arguments(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum()

    def test_one_argument(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('')

    def test_not_string_subject(self):
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.einsum(0, 0)

    def test_bad_argument(self):
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.einsum('', 0, bad_arg=0)

    def test_too_many_operands1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('', 0, 0)

    def test_too_many_operands2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(
                    'i,j',
                    xp.array([0, 0]),
                    xp.array([0, 0]),
                    xp.array([0, 0]))

    def test_too_few_operands1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(',', 0)

    def test_too_many_dimension1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i', 0)

    def test_too_many_dimension2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ij', xp.array([0, 0]))

    def test_too_many_dimension3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ijk...->...', xp.arange(6).reshape(2, 3))

    def test_too_few_dimension(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i->i', xp.arange(6).reshape(2, 3))

    def test_invalid_char1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i%', xp.array([0, 0]))

    def test_invalid_char2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('j$', xp.array([0, 0]))

    def test_invalid_char3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i->&', xp.array([0, 0]))

    # output subscripts must appear in inumpy.t
    def test_invalid_output_subscripts1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i->ij', xp.array([0, 0]))

    # output subscripts may only be specified once
    def test_invalid_output_subscripts2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ij->jij', xp.array([[0, 0], [0, 0]]))

    # output subscripts must not incrudes comma
    def test_invalid_output_subscripts3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ij->i,j', xp.array([[0, 0], [0, 0]]))

    # dimensions much match when being collapsed
    def test_invalid_diagonal1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ii', xp.arange(6).reshape(2, 3))

    def test_invalid_diagonal2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ii->', xp.arange(6).reshape(2, 3))

    def test_invalid_diagonal3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ii', xp.arange(3).reshape(1, 3))

    def test_dim_mismatch_char1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i,i', xp.arange(2), xp.arange(3))

    def test_dim_mismatch_ellipsis1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('...,...', xp.arange(2), xp.arange(3))

    def test_dim_mismatch_ellipsis2(self):
        for xp in (numpy, cupy):
            a = xp.arange(12).reshape(2, 3, 2)
            with pytest.raises(ValueError):
                xp.einsum('i...,...i', a, a)

    def test_dim_mismatch_ellipsis3(self):
        for xp in (numpy, cupy):
            a = xp.arange(12).reshape(2, 3, 2)
            with pytest.raises(ValueError):
                xp.einsum('...,...', a, a[:, :2])

    # invalid -> operator
    def test_invalid_arrow1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i-i', xp.array([0, 0]))

    def test_invalid_arrow2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i>i', xp.array([0, 0]))

    def test_invalid_arrow3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i->->i', xp.array([0, 0]))

    def test_invalid_arrow4(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i-', xp.array([0, 0]))


class TestListArgEinSumError(unittest.TestCase):

    def test_invalid_sub1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [None])

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_invalid_sub2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [0], [1])

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_invalid_sub3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [Ellipsis, 0, Ellipsis])

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_dim_mismatch1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [0], xp.arange(3), [0])

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_dim_mismatch2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [0], xp.arange(3), [0], [0])

    def test_dim_mismatch3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(6).reshape(2, 3), [0, 0])

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_too_many_dims1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(3, [0])

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_too_many_dims2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [0, 1])

    def test_too_many_dims3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(6).reshape(2, 3), [Ellipsis, 0, 1, 2])


class TestEinSumUnaryOperationWithScalar(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_scalar_int(self, xp, dtype):
        return xp.asarray(xp.einsum('->', 2, dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_scalar_float(self, xp, dtype):
        return xp.asarray(xp.einsum('', 2.0, dtype=dtype))


class TestEinSumBinaryOperationWithScalar(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_scalar_1(self, xp, dtype):
        shape_a = (2,)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.asarray(xp.einsum(',i->', 3, a))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_scalar_2(self, xp, dtype):
        shape_a = (2,)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.asarray(xp.einsum('i,->', a, 4))


@testing.parameterize(*([
    # memory constraint
    {'subscript': 'a,b,c->abc', 'opt': ('greedy', 0)},
    {'subscript': 'acdf,jbje,gihb,hfac', 'opt': ('greedy', 0)},
] + testing.product({'subscript': [
    # long paths
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac',
    'chd,bde,agbc,hiad,bdi,cgh,agdb',
    # edge cases
    'eb,cb,fb->cef',
    'dd,fb,be,cdb->cef',
    'bca,cdb,dbf,afc->',
    'dcc,fce,ea,dbf->ab',
    'a,ac,ab,ad,cd,bd,bc->',
], 'opt': ['greedy', 'optimal'],
})))
class TestEinSumLarge(unittest.TestCase):

    def setUp(self):
        chars = 'abcdefghij'
        sizes = numpy.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3])
        size_dict = {}
        for size, char in zip(sizes, chars):
            size_dict[char] = size

        # Builds views based off initial operands
        string = self.subscript
        operands = [string]
        terms = string.split('->')[0].split(',')
        for term in terms:
            dims = [size_dict[x] for x in term]
            operands.append(numpy.ones(*dims))

        self.operands = operands

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum(self, xp):
        # TODO(kataoka): support memory efficient cupy.einsum
        with warnings.catch_warnings(record=True) as ws:
            # I hope there's no problem with np.einsum for these cases...
            out = xp.einsum(*self.operands, optimize=self.opt)
            if xp is not numpy and \
                    isinstance(self.opt, tuple):  # with memory limit
                for w in ws:
                    self.assertIn('memory', str(w.message))
            else:
                self.assertEqual(len(ws), 0)
        return out
