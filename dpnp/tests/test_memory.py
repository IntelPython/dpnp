import dpctl.tensor as dpt
import numpy
import pytest

import dpnp
import dpnp.memory as dpm


class IntUsmData(dpt.usm_ndarray):
    """Class that overrides `usm_data` property in `dpt.usm_ndarray`."""

    @property
    def usm_data(self):
        return 1


class TestCreateData:
    @pytest.mark.parametrize("x", [numpy.ones(4), dpnp.zeros(2)])
    def test_wrong_input_type(self, x):
        with pytest.raises(TypeError):
            dpm.create_data(x)

    def test_wrong_usm_data(self):
        a = dpt.ones(10)
        d = IntUsmData(a.shape, buffer=a)

        with pytest.raises(TypeError):
            dpm.create_data(d)

    def test_dpctl_view(self):
        a = dpt.arange(10)
        view = a[3:]

        data = dpm.create_data(view)
        assert data.ptr == view._pointer

    def test_dpctl_different_views(self):
        a = dpt.reshape(dpt.arange(12), (3, 4))

        data0 = dpm.create_data(a[0])
        data1 = dpm.create_data(a[1])

        # Verify independent wrapper objects
        assert data0 is not data1

        # Verify correct pointers
        assert data0.ptr == a[0]._pointer
        assert data1.ptr == a[1]._pointer
        assert data0.ptr != data1.ptr

    def test_repeated_calls(self):
        a = dpt.arange(20)
        view = a[5:15]

        # Multiple calls should return independent objects with same ptr
        data1 = dpm.create_data(view)
        data2 = dpm.create_data(view)

        assert data1 is not data2, "Should create independent wrapper objects"
        assert data1.ptr == data2.ptr, "Both should point to same location"
        assert data1.ptr == view._pointer


class TestNdarray:
    def test_ndarray_from_data(self):
        a = dpnp.empty(5)
        b = dpnp.ndarray(a.shape, buffer=a.data)
        assert b.data.ptr == a.data.ptr

    def test_view_non_zero_offset(self):
        n, m = 2, 8
        plane = n * m

        a = dpnp.empty(4 * plane)
        sl = a[plane:]  # non-zero offset view

        pl = dpnp.ndarray((n, m), dtype=a.dtype, buffer=sl)
        assert pl.data.ptr == sl.data.ptr
        assert a.data.ptr != sl.data.ptr

    def test_slices_2d(self):
        # Create 2D array and verify slices have different pointers
        a = dpnp.arange(12, dtype=dpnp.float32).reshape(3, 4)

        # Each row should have a different pointer
        row0_ptr = a[0].data.ptr
        row1_ptr = a[1].data.ptr
        row2_ptr = a[2].data.ptr

        assert (
            row0_ptr != row1_ptr
        ), "a[0] and a[1] should have different pointers"
        assert (
            row1_ptr != row2_ptr
        ), "a[1] and a[2] should have different pointers"

        # Check byte offsets match expected stride
        stride = a.strides[0]  # stride between rows in bytes
        assert row1_ptr - row0_ptr == stride
        assert row2_ptr - row1_ptr == stride

    def test_slices_multidimensional(self):
        # 3D array
        a = dpnp.zeros((5, 10, 20), dtype=dpnp.int32)

        # Different slices along first axis should have different pointers
        slice0_ptr = a[0].data.ptr
        slice1_ptr = a[1].data.ptr

        assert slice0_ptr != slice1_ptr
        assert slice1_ptr - slice0_ptr == a.strides[0]

    def test_repeated_access(self):
        a = dpnp.arange(20).reshape(4, 5)

        # Multiple accesses to same slice should give same ptr value
        ptr1 = a[2].data.ptr
        ptr2 = a[2].data.ptr

        assert ptr1 == ptr2, "Same slice should have consistent ptr value"

        # But different slices should have different ptrs
        assert a[0].data.ptr != a[2].data.ptr

    def test_array_on_view_with_slicing(self):
        # Original array
        a = dpnp.arange(24, dtype=dpnp.float32).reshape(6, 4)

        # Create view using slicing
        view = a[2:5]

        # Construct new array from view
        new_arr = dpnp.ndarray(view.shape, dtype=view.dtype, buffer=view)

        # Pointers should match
        assert new_arr.data.ptr == view.data.ptr
        # And should be different from base array
        assert new_arr.data.ptr != a.data.ptr
