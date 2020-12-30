import pytest

import dpnp as inp

import numpy


@pytest.mark.parametrize("array",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],
                          [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
                          [[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]
                          ],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]',
                              '[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]',
                              '[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]'
                              ])
def test_diff(array):
    a = numpy.array(array)
    ia = inp.array(a)
    expected = numpy.diff(a)
    result = inp.diff(ia)
    numpy.testing.assert_allclose(expected, result)


@pytest.mark.parametrize("array", [[1, 2, 3, 4, 5],
                                   [1, 2, numpy.nan, 4, 5],
                                   [[1, 2, numpy.nan], [3, -4, -5]]])
def test_nancumprod(array):
    a = numpy.array(array)
    ia = inp.array(a)

    result = inp.nancumsum(ia)
    expected = numpy.nancumsum(a)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("array", [[1, 2, 3, 4, 5],
                                   [1, 2, numpy.nan, 4, 5],
                                   [[1, 2, numpy.nan], [3, -4, -5]]])
def test_nancumsum(array):
    a = numpy.array(array)
    ia = inp.array(a)

    result = inp.nancumsum(ia)
    expected = numpy.nancumsum(a)
    numpy.testing.assert_array_equal(expected, result)


class TestEdiff1d:

    def test_ediff1d_int(self):
        a = numpy.array([1, 2, 4, 7, 0])
        ia = inp.array(a)

        result = inp.ediff1d(ia)
        expected = numpy.ediff1d(a)
        numpy.testing.assert_array_equal(expected, result)

    def test_ediff1d_args(self):
        a = numpy.array([1, 2, 4, 7, 0])
        ia = inp.array(a)

        to_begin = numpy.array([-20, -30])
        i_to_begin = inp.array(to_begin)

        to_end = numpy.array([20, 15])
        i_to_end = inp.array(to_end)

        result = inp.ediff1d(ia, to_end=i_to_end, to_begin=i_to_begin)
        expected = numpy.ediff1d(a, to_end=to_end, to_begin=to_begin)
        numpy.testing.assert_array_equal(expected, result)

    def test_ediff1d_float(self):
        a = numpy.array([1., 2.5, 6., 7., 3.])
        ia = inp.array(a)

        result = inp.ediff1d(ia)
        expected = numpy.ediff1d(a)
        numpy.testing.assert_array_equal(expected, result)


class TestTrapz:

    @pytest.mark.parametrize("array", [[1, 2, 4, 5],
                                       [1., 2.5, 6., 7., 3.],
                                       [2, 4, 6, 8]])
    def test_trapz_without_params(self, array):
        a = numpy.array(array)
        ia = inp.array(a)

        result = inp.trapz(ia)
        expected = numpy.trapz(a)
        numpy.testing.assert_array_equal(expected, result)

    @pytest.mark.parametrize("y_array", [[1, 2, 4, 5],
                                         [1., 2.5, 6., 7., ],
                                         [2, 4, 6, 8]])
    @pytest.mark.parametrize("x_array", [[1, 2, 3, 4],
                                         [2, 4, 6, 8]])
    def test_trapz_without_params(self, y_array, x_array):
        y = numpy.array(y_array)
        iy = inp.array(y)

        x = numpy.array(x_array)
        ix = inp.array(x)

        result = inp.trapz(iy, x=ix)
        expected = numpy.trapz(y, x=x)
        numpy.testing.assert_array_equal(expected, result)

    @pytest.mark.parametrize("y_array", [[1, 2, 4, 5],
                                         [1., 2.5, 6., 7., ],
                                         [2, 4, 6, 8]])
    @pytest.mark.parametrize("dx", [1, 2, 3, 4])
    def test_trapz_without_params(self, y_array, dx):
        y = numpy.array(y_array)
        iy = inp.array(y)

        result = inp.trapz(iy, dx=dx)
        expected = numpy.trapz(y, dx=dx)
        numpy.testing.assert_array_equal(expected, result)
