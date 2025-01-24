import dpctl
import dpctl.tensor as dpt
import numpy
import pytest

import dpnp


class TestIsSupportedArrayOrScalar:
    @pytest.mark.parametrize(
        "array",
        [
            dpnp.array([1, 2, 3]),
            dpnp.array(1),
            dpt.asarray([1, 2, 3]),
        ],
    )
    def test_valid_arrays(self, array):
        assert dpnp.is_supported_array_or_scalar(array) is True

    @pytest.mark.parametrize(
        "value",
        [
            42,
            True,
            "1",
        ],
    )
    def test_valid_scalars(self, value):
        assert dpnp.is_supported_array_or_scalar(value) is True

    @pytest.mark.parametrize(
        "array",
        [
            [1, 2, 3],
            (1, 2, 3),
            None,
            numpy.array([1, 2, 3]),
        ],
    )
    def test_invalid_arrays(self, array):
        assert not dpnp.is_supported_array_or_scalar(array) is True


class TestSynchronizeArrayData:
    @pytest.mark.parametrize(
        "array",
        [
            dpnp.array([1, 2, 3]),
            dpt.asarray([1, 2, 3]),
        ],
    )
    def test_synchronize_array_data(self, array):
        a_copy = dpnp.copy(array, sycl_queue=array.sycl_queue)
        try:
            dpnp.synchronize_array_data(a_copy)
        except Exception as e:
            pytest.fail(f"synchronize_array_data failed: {e}")

    @pytest.mark.parametrize(
        "input",
        [
            [1, 2, 3],
            numpy.array([1, 2, 3]),
        ],
    )
    def test_unsupported_type(self, input):
        with pytest.raises(TypeError):
            dpnp.synchronize_array_data(input)
