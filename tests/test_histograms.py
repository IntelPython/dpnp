import numpy
import pytest

import dpnp


class TestHistogram:
    def setup(self):
        pass

    def teardown(self):
        pass

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_simple(self):
        n = 100
        v = dpnp.random.rand(n)
        (a, b) = dpnp.histogram(v)
        # check if the sum of the bins equals the number of samples
        numpy.testing.assert_equal(dpnp.sum(a, axis=0), n)
        # check that the bin counts are evenly spaced when the data is from
        # a linear function
        (a, b) = dpnp.histogram(numpy.linspace(0, 10, 100))
        numpy.testing.assert_array_equal(a, 10)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_one_bin(self):
        # Ticket 632
        hist, edges = dpnp.histogram([1, 2, 3, 4], [1, 2])
        numpy.testing.assert_array_equal(
            hist,
            [
                2,
            ],
        )
        numpy.testing.assert_array_equal(edges, [1, 2])
        numpy.testing.assert_raises(ValueError, dpnp.histogram, [1, 2], bins=0)
        h, e = dpnp.histogram([1, 2], bins=1)
        numpy.testing.assert_equal(h, dpnp.array([2]))
        numpy.testing.assert_allclose(e, dpnp.array([1.0, 2.0]))

    def test_density(self):
        # Check that the integral of the density equals 1.
        n = 100
        v = dpnp.random.rand(n)
        a, b = dpnp.histogram(v, density=True)
        area = dpnp.sum(a * dpnp.diff(b)[0])[0]
        numpy.testing.assert_almost_equal(area, 1)

        # Check with non-constant bin widths
        v = dpnp.arange(10)
        bins = [0, 1, 3, 6, 10]
        a, b = dpnp.histogram(v, bins, density=True)
        numpy.testing.assert_array_equal(a, 0.1)
        numpy.testing.assert_equal(dpnp.sum(a * dpnp.diff(b))[0], 1)

        # Test that passing False works too
        a, b = dpnp.histogram(v, bins, density=False)
        numpy.testing.assert_array_equal(a, [1, 2, 3, 4])

        # Variable bin widths are especially useful to deal with
        # infinities.
        v = dpnp.arange(10)
        bins = [0, 1, 3, 6, numpy.inf]
        a, b = dpnp.histogram(v, bins, density=True)
        numpy.testing.assert_array_equal(a, [0.1, 0.1, 0.1, 0.0])

        # Taken from a bug report from N. Becker on the numpy-discussion
        # mailing list Aug. 6, 2010.
        counts, dmy = dpnp.histogram(
            [1, 2, 3, 4], [0.5, 1.5, numpy.inf], density=True
        )
        numpy.testing.assert_equal(counts, [0.25, 0])

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_arr_weights_mismatch(self):
        a = dpnp.arange(10) + 0.5
        w = dpnp.arange(11) + 0.5
        with numpy.testing.assert_raises_regex(ValueError, "same shape as"):
            h, b = dpnp.histogram(a, range=[1, 9], weights=w, density=True)
