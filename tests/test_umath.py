import pytest

import numpy
import dpnp

# full list of umaths
umaths = [i for i in dir(numpy) if isinstance(getattr(numpy, i), numpy.ufunc)]
# print(umaths)
umaths = ['equal']
# trigonometric
umaths.extend(['arccos', 'arcsin', 'arctan', 'cos', 'deg2rad', 'degrees',
               'rad2deg', 'radians', 'sin', 'tan', 'arctan2', 'hypot'])
# 'unwrap'

types = {
    'd': numpy.float64,
    'f': numpy.float32,
    'l': numpy.int64,
    'i': numpy.int32,
}

supported_types = 'dfli'


def check_types(args_str):
    for s in args_str:
        if s not in supported_types:
            return False
    return True


def shaped_arange(shape, xp=numpy, dtype=numpy.float32):
    size = 1
    for i in shape:
        size = size * i
    array_data = numpy.arange(1, size + 1, 1).tolist()
    return xp.array(array_data, dtype=dtype).reshape(shape)


def get_args(args_str, xp=numpy):
    args = []
    for s in args_str:
        args.append(shaped_arange(shape=(3, 4), xp=xp, dtype=types[s]))
    return tuple(args)


test_cases = []
for umath in umaths:
    np_umath = getattr(numpy, umath)
    _types = np_umath.types
    for type in _types:
        args_str = type[:type.find('->')]
        if check_types(args_str):
            test_cases.append((umath, args_str))


def get_id(val):
    return val.__str__()


@pytest.mark.parametrize('test_cases', test_cases, ids=get_id)
def test_umaths(test_cases):
    umath, args_str = test_cases
    args = get_args(args_str, xp=numpy)
    iargs = get_args(args_str, xp=dpnp)

    # original NumPy
    expected = getattr(numpy, umath)(*args)

    # Intel NumPy
    result = getattr(dpnp, umath)(*iargs)

    numpy.testing.assert_allclose(result, expected, rtol=1e-6)
