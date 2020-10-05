import pytest

import dpnp.random
import numpy


@pytest.mark.parametrize("func",
                         [dpnp.random.rand,
                          dpnp.random.randn],
                         ids=['rand', 'randn'])
def test_random_input_size(func):
    output_shape = (10,)
    size = 10
    res = func(size)
    assert output_shape == res.shape


@pytest.mark.parametrize("func",
                         [dpnp.random.random,
                          dpnp.random.random_sample,
                          dpnp.random.randf],
                         ids=['random', 'random_sample',
                              'randf'])
def test_random_input_shape(func):
    shape = (10, 5)
    res = func(shape)
    assert shape == res.shape


@pytest.mark.parametrize("func",
                         [dpnp.random.random,
                          dpnp.random.random_sample,
                          dpnp.random.randf,
                          dpnp.random.rand],
                         ids=['random', 'random_sample',
                              'randf', 'rand'])
def test_random_check_otput(func):
    shape = (10, 5)
    size = 10 * 5
    if func == dpnp.random.rand:
        res = func(size)
    else:
        res = func(shape)
#    assert numpy.all(res >= 0)
#    assert numpy.all(res < 1)
    for i in range(res.size):
        assert res[i] >= 0.0
        assert res[i] < 1.0

