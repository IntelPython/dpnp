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


@pytest.mark.parametrize("val_type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['numpy.float64', 'numpy.float32', 'numpy.int64', 'numpy.int32'])
@pytest.mark.parametrize("data_type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['numpy.float64', 'numpy.float32', 'numpy.int64', 'numpy.int32'])
@pytest.mark.parametrize("val",
                         [0, 1, 5],
                         ids=['0', '1', '5'])
@pytest.mark.parametrize("array",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],
                          [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
                          [[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]',
                              '[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]',
                              '[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]'])
def test_power(array, val, data_type, val_type):
    a = numpy.array(array, dtype=data_type)
    ia = inp.array(a)
    val_ = val_type(val)
    result = inp.power(ia, val_)
    expected = numpy.power(ia, val_)
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


class TestCross:

    @pytest.mark.parametrize("axis", [None, 0],
                             ids=['None', '0'])
    @pytest.mark.parametrize("axisc", [-1, 0],
                             ids=['-1', '0'])
    @pytest.mark.parametrize("axisb", [-1, 0],
                             ids=['-1', '0'])
    @pytest.mark.parametrize("axisa", [-1, 0],
                             ids=['-1', '0'])
    @pytest.mark.parametrize("x1", [[1, 2, 3],
                                    [1., 2.5, 6.],
                                    [2, 4, 6]],
                             ids=['[1, 2, 3]',
                                  '[1., 2.5, 6.]',
                                  '[2, 4, 6]'])
    @pytest.mark.parametrize("x2", [[4, 5, 6],
                                    [1., 5., 2.],
                                    [6, 4, 3]],
                             ids=['[4, 5, 6]',
                                  '[1., 5., 2.]',
                                  '[6, 4, 3]'])
    def test_cross_3x3(self, x1, x2, axisa, axisb, axisc, axis):
        x1_ = numpy.array(x1)
        ix1_ = inp.array(x1_)

        x2_ = numpy.array(x2)
        ix2_ = inp.array(x2_)

        result = inp.cross(ix1_, ix2_, axisa, axisb, axisc, axis)
        expected = numpy.cross(x1_, x2_, axisa, axisb, axisc, axis)
        numpy.testing.assert_array_equal(expected, result)
