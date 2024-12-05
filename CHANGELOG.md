# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.16.1] - 12/XX/2024

This is a bug-fix release.

### Changed

* Changed to use `Miniforge` installer in GutHub actions [#2057](https://github.com/IntelPython/dpnp/pull/2057)
* Updated `README.md` to reflect current installation requirements and available options [#2166](https://github.com/IntelPython/dpnp/pull/2166)

### Fixed

* Resolved a compilation error when building with DPC++ 2025.1 compiler [#2211](https://github.com/IntelPython/dpnp/pull/2211)


## [0.16.0] - 10/14/2024

This release reaches an important milestone by making offloading fully asynchronous. Calls to `dpnp` submit tasks for execution to DPC++ runtime and return without waiting for execution of these tasks to finish. The sequential semantics a user comes to expect from execution of Python script is preserved though.
In addition, this release completes implementation of `dpnp.fft` module and adds several new array manipulation, indexing and elementwise routines. Moreover, it adds support to build `dpnp` for Nvidia GPUs.

### Added

* Added implementation of `dpnp.gradient` function [#1859](https://github.com/IntelPython/dpnp/pull/1859)
* Added implementation of `dpnp.sort_complex` function [#1864](https://github.com/IntelPython/dpnp/pull/1864)
* Added implementation of `dpnp.fft.fft` and `dpnp.fft.ifft` functions [#1879](https://github.com/IntelPython/dpnp/pull/1879)
* Added implementation of `dpnp.isneginf` and `dpnp.isposinf` functions [#1888](https://github.com/IntelPython/dpnp/pull/1888)
* Added implementation of `dpnp.fft.fftfreq` and `dpnp.fft.rfftfreq` functions [#1898](https://github.com/IntelPython/dpnp/pull/1898)
* Added implementation of `dpnp.fft.fftshift` and `dpnp.fft.ifftshift` functions [#1900](https://github.com/IntelPython/dpnp/pull/1900)
* Added implementation of `dpnp.isreal`, `dpnp.isrealobj`, `dpnp.iscomplex`, and `dpnp.iscomplexobj` functions [#1916](https://github.com/IntelPython/dpnp/pull/1916)
* Added support to build `dpnp` for Nvidia GPU [#1926](https://github.com/IntelPython/dpnp/pull/1926)
* Added implementation of `dpnp.fft.rfft` and `dpnp.fft.irfft` functions [#1928](https://github.com/IntelPython/dpnp/pull/1928)
* Added implementation of `dpnp.nextafter` function [#1938](https://github.com/IntelPython/dpnp/pull/1938)
* Added implementation of `dpnp.trim_zero` function [#1941](https://github.com/IntelPython/dpnp/pull/1941)
* Added implementation of `dpnp.fft.hfft` and `dpnp.fft.ihfft` functions [#1954](https://github.com/IntelPython/dpnp/pull/1954)
* Added implementation of `dpnp.logaddexp2` function [#1955](https://github.com/IntelPython/dpnp/pull/1955)
* Added implementation of `dpnp.flatnonzero` function [#1956](https://github.com/IntelPython/dpnp/pull/1956)
* Added implementation of `dpnp.float_power` function [#1957](https://github.com/IntelPython/dpnp/pull/1957)
* Added implementation of `dpnp.fft.fft2`, `dpnp.fft.ifft2`, `dpnp.fft.fftn`, and `dpnp.fft.ifftn` functions [#1961](https://github.com/IntelPython/dpnp/pull/1961)
* Added implementation of `dpnp.array_equal` and `dpnp.array_equiv` functions [#1965](https://github.com/IntelPython/dpnp/pull/1965)
* Added implementation of `dpnp.nan_to_num` function [#1966](https://github.com/IntelPython/dpnp/pull/1966)
* Added implementation of `dpnp.fix` function [#1971](https://github.com/IntelPython/dpnp/pull/1971)
* Added implementation of `dpnp.fft.rfft2`, `dpnp.fft.irfft2`, `dpnp.fft.rfftn`, and `dpnp.fft.irfftn` functions [#1982](https://github.com/IntelPython/dpnp/pull/1982)
* Added implementation of `dpnp.argwhere` function [#2000](https://github.com/IntelPython/dpnp/pull/2000)
* Added implementation of `dpnp.real_if_close` function [#2002](https://github.com/IntelPython/dpnp/pull/2002)
* Added implementation of `dpnp.ndim` and `dpnp.size` functions [#2014](https://github.com/IntelPython/dpnp/pull/2014)
* Added implementation of `dpnp.append` and `dpnp.asarray_chkfinite` functions [#2015](https://github.com/IntelPython/dpnp/pull/2015)
* Added implementation of `dpnp.array_split`, `dpnp.split`, `dpnp.hsplit`, `dpnp.vsplit`, and `dpnp.dsplit` functions [#2017](https://github.com/IntelPython/dpnp/pull/2017)
* Added runtime dependency on `intel-gpu-ocl-icd-system` package [#2023](https://github.com/IntelPython/dpnp/pull/2023)
* Added implementation of `dpnp.ravel_multi_index` and `dpnp.unravel_index` functions [#2022](https://github.com/IntelPython/dpnp/pull/2022)
* Added implementation of `dpnp.resize` and `dpnp.rot90` functions [#2030](https://github.com/IntelPython/dpnp/pull/2030)
* Added implementation of `dpnp.require` function [#2036](https://github.com/IntelPython/dpnp/pull/2036)

### Changed

* Extended pre-commit pylint check to `dpnp.fft` module [#1860](https://github.com/IntelPython/dpnp/pull/1860)
* Reworked `vm` vector math backend to reuse `dpctl.tensor` functions around unary and binary functions [#1868](https://github.com/IntelPython/dpnp/pull/1868)
* Extended `dpnp.ndarray.astype` method to support `device` keyword argument [#1870](https://github.com/IntelPython/dpnp/pull/1870)
* Improved performance of `dpnp.linalg.solve` by implementing a dedicated kernel for its batch implementation [#1877](https://github.com/IntelPython/dpnp/pull/1877)
* Extended `dpnp.fabs` to support `order` and `out` keyword arguments by writing a dedicated kernel for it [#1878](https://github.com/IntelPython/dpnp/pull/1878)
* Extended `dpnp.linalg` module to support `usm_ndarray` as input [#1880](https://github.com/IntelPython/dpnp/pull/1880)
* Reworked `dpnp.mod` implementation to be an alias for `dpnp.remainder` [#1882](https://github.com/IntelPython/dpnp/pull/1882)
* Removed the legacy implementation of linear algebra functions from the backend [#1887](https://github.com/IntelPython/dpnp/pull/1887)
* Removed the legacy implementation of elementwise functions from the backend [#1890](https://github.com/IntelPython/dpnp/pull/1890)
* Extended `dpnp.all` and `dpnp.any` to support `out` keyword argument [#1893](https://github.com/IntelPython/dpnp/pull/1893)
* Reworked `dpnp.repeat` to add a explicit type check of input array [#1894](https://github.com/IntelPython/dpnp/pull/1894)
* Improved performance of different functions by adopting asynchronous implementation of `dpctl` [#1897](https://github.com/IntelPython/dpnp/pull/1897)
* Extended `dpnp.fmax` and `dpnp.fmin` to support `order` and `out` keyword arguments by writing dedicated kernels for them [#1905](https://github.com/IntelPython/dpnp/pull/1905)
* Removed the legacy implementation of array creation and manipulation functions from the backend [#1903](https://github.com/IntelPython/dpnp/pull/1903)
* Extended `dpnp.extract` implementation to align with NumPy [#1906](https://github.com/IntelPython/dpnp/pull/1906)
* Reworked backend implementation to align with non-backward compatible changes in DPC++ 2025.0 [#1907](https://github.com/IntelPython/dpnp/pull/1907)
* Removed the legacy implementation of indexing functions from the backend [#1908](https://github.com/IntelPython/dpnp/pull/1908)
* Extended `dpnp.take` implementation to align with NumPy [#1909](https://github.com/IntelPython/dpnp/pull/1909)
* Extended `dpnp.place` implementation to align with NumPy [#1912](https://github.com/IntelPython/dpnp/pull/1912)
* Reworked the implementation of indexing functions to avoid unnecessary casting to `dpnp_array` when input is `usm_ndarray` [#1913](https://github.com/IntelPython/dpnp/pull/1913)
* Reduced code duplication in the implementation of sorting functions [#1914](https://github.com/IntelPython/dpnp/pull/1914)
* Removed the obsolete dparray interface [#1915](https://github.com/IntelPython/dpnp/pull/1915)
* Improved performance of `dpnp.linalg` module for BLAS routines by adopting asynchronous implementation of `dpctl` [#1919](https://github.com/IntelPython/dpnp/pull/1919)
* Relocated `dpnp.einsum` utility functions to a separate file [#1920](https://github.com/IntelPython/dpnp/pull/1920)
* Improved performance of `dpnp.linalg` module for LAPACK routines by adopting asynchronous implementation of `dpctl` [#1922](https://github.com/IntelPython/dpnp/pull/1922)
* Reworked `dpnp.matmul` to allow larger batch size to be used [#1927](https://github.com/IntelPython/dpnp/pull/1927)
* Removed data synchronization where it is not needed [#1930](https://github.com/IntelPython/dpnp/pull/1930)
* Leveraged `dpctl.tensor` implementation for `dpnp.where` to support scalar as input [#1932](https://github.com/IntelPython/dpnp/pull/1932)
* Improved performance of `dpnp.linalg.eigh` by implementing a dedicated kernel for its batch implementation [#1936](https://github.com/IntelPython/dpnp/pull/1936)
* Reworked `dpnp.isclose` and `dpnp.allclose` to comply with compute follows data approach [#1937](https://github.com/IntelPython/dpnp/pull/1937)
* Extended `dpnp.deg2rad` and `dpnp.radians` to support `order` and `out` keyword arguments by writing dedicated kernels for them [#1943](https://github.com/IntelPython/dpnp/pull/1943)
* `dpnp` uses pybind11 2.13.1 [#1944](https://github.com/IntelPython/dpnp/pull/1944)
* Extended `dpnp.degrees` and `dpnp.rad2deg` to support `order` and `out` keyword arguments by writing dedicated kernels for them [#1949](https://github.com/IntelPython/dpnp/pull/1949)
* Extended `dpnp.unwrap` to support all keyword arguments provided by NumPy [#1950](https://github.com/IntelPython/dpnp/pull/1950)
* Leveraged `dpctl.tensor` implementation for `dpnp.count_nonzero` function [#1962](https://github.com/IntelPython/dpnp/pull/1962)
* Leveraged `dpctl.tensor` implementation for `dpnp.diff` function [#1963](https://github.com/IntelPython/dpnp/pull/1963)
* Leveraged `dpctl.tensor` implementation for `dpnp.take_along_axis` function [#1969](https://github.com/IntelPython/dpnp/pull/1969)
* Reworked `dpnp.ediff1d` implementation through existing functions instead of a separate kernel [#1970](https://github.com/IntelPython/dpnp/pull/1970)
* Reworked `dpnp.unique` implementation through existing functions when `axis` is given otherwise through leveraging `dpctl.tensor` implementation [#1972](https://github.com/IntelPython/dpnp/pull/1972)
* Improved performance of `dpnp.linalg.svd` by implementing a dedicated kernel for its batch implementation [#1936](https://github.com/IntelPython/dpnp/pull/1936)
* Leveraged `dpctl.tensor` implementation for `shape.setter` method [#1975](https://github.com/IntelPython/dpnp/pull/1975)
* Extended `dpnp.ndarray.copy` to support compute follow data keyword arguments [#1976](https://github.com/IntelPython/dpnp/pull/1976)
* Reworked `dpnp.select` implementation through existing functions instead of a separate kernel [#1977](https://github.com/IntelPython/dpnp/pull/1977)
* Leveraged `dpctl.tensor` implementation for `dpnp.from_dlpack` and `dpnp.ndarray.__dlpack__` functions [#1980](https://github.com/IntelPython/dpnp/pull/1980)
* Reworked `dpnp.linalg` module backend implementation for BLAS rouitnes to work with OneMKL interfaces [#1981](https://github.com/IntelPython/dpnp/pull/1981)
* Reworked `dpnp.ediff1d` implementation to reduce code duplication [#1983](https://github.com/IntelPython/dpnp/pull/1983)
* `dpnp` can be used with any NumPy from 1.23 to 2.0 [#1985](https://github.com/IntelPython/dpnp/pull/1985)
* Reworked `dpnp.unique` implementation to properly handle NaNs values [#1972](https://github.com/IntelPython/dpnp/pull/1972)
* Removed `dpnp.issubcdtype` per NumPy 2.0 recommendation [#1996](https://github.com/IntelPython/dpnp/pull/1996)
* Reworked `dpnp.unique` implementation to align with NumPy 2.0 [#1999](https://github.com/IntelPython/dpnp/pull/1999)
* Reworked `dpnp.linalg.solve` backend implementation to work with OneMKL Interfaces [#2001](https://github.com/IntelPython/dpnp/pull/2001)
* Reworked `dpnp.trapezoid` implementation through existing functions instead of falling back on NumPy [#2003](https://github.com/IntelPython/dpnp/pull/2003)
* Added `copy` keyword to `dpnp.array` to align with NumPy 2.0  [#2006](https://github.com/IntelPython/dpnp/pull/2006)
* Extended `dpnp.heaviside` to support `order` and `out` keyword arguments by writing dedicated kernel for it [#2008](https://github.com/IntelPython/dpnp/pull/2008)
* `dpnp` uses pybind11 2.13.5 [#2010](https://github.com/IntelPython/dpnp/pull/2010)
* Added `COMPILER_VERSION_2025_OR_LATER` flag to be able to run `dpnp.fft` module with both 2024.2 and 2025.0 versions of the compiler [#2025](https://github.com/IntelPython/dpnp/pull/2025)
* Cleaned up an implementation of `dpnp.gradient` by removing obsolete TODO which is not going to be done [#2032](https://github.com/IntelPython/dpnp/pull/2032)
* Updated `Array Manipulation Routines` page in documentation to add missing functions and to remove duplicate entries [#2033](https://github.com/IntelPython/dpnp/pull/2033)
* `dpnp` uses pybind11 2.13.6 [#2041](https://github.com/IntelPython/dpnp/pull/2041)
* Updated `dpnp.fft` backend to depend on `INTEL_MKL_VERSION` flag to ensures that the appropriate code segment is executed based on the version of OneMKL [#2035](https://github.com/IntelPython/dpnp/pull/2035)
* Use `dpctl::tensor::alloc_utils::sycl_free_noexcept` instead of `sycl::free` in `host_task` tasks associated with life-time management of temporary USM allocations [#2058](https://github.com/IntelPython/dpnp/pull/2058)
* Improved implementation of `dpnp.kron` to avoid unnecessary copy for non-contiguous arrays [#2059](https://github.com/IntelPython/dpnp/pull/2059)
* Updated the test suit for `dpnp.fft` module [#2071](https://github.com/IntelPython/dpnp/pull/2071)
* Reworked `dpnp.clip` implementation to align with Python Array API 2023.12 specification [#2048](https://github.com/IntelPython/dpnp/pull/2048)
* Skipped outdated tests for `dpnp.linalg.solve` due to compatibility issues with NumPy 2.0 [#2074](https://github.com/IntelPython/dpnp/pull/2074)
* Updated installation instructions [#2098](https://github.com/IntelPython/dpnp/pull/2098)

### Fixed

* Resolved an issue with `dpnp.matmul` when an f_contiguous `out` keyword is passed to the the function [#1872](https://github.com/IntelPython/dpnp/pull/1872)
* Resolved a possible race condition in `dpnp.inv` [#1940](https://github.com/IntelPython/dpnp/pull/1940)
* Resolved an issue with failing tests for `dpnp.append` when running on a device without fp64 support [#2034](https://github.com/IntelPython/dpnp/pull/2034)
* Resolved an issue with input array of `usm_ndarray` passed into `dpnp.ix_` [#2047](https://github.com/IntelPython/dpnp/pull/2047)
* Added a workaround to prevent crash in tests on Windows in internal CI/CD (when running on either Lunar Lake or Arrow Lake) [#2062](https://github.com/IntelPython/dpnp/pull/2062)
* Fixed a crash in `dpnp.choose` caused by missing control of releasing temporary allocated device memory [#2063](https://github.com/IntelPython/dpnp/pull/2063)
* Resolved compilation warning and error while building in debug mode [#2066](https://github.com/IntelPython/dpnp/pull/2066)
* Fixed an issue with asynchronous execution in `dpnp.fft` module [#2067](https://github.com/IntelPython/dpnp/pull/2067)

## [0.15.0] - 05/25/2024

This release completes implementation of `dpnp.linalg` module and array creation routine, adds cumulative reductions and histogram functions.

### Added

* Implemented `dpnp.frombuffer`, `dpnp.fromfile` and `dpnp.fromstring` functions [#1727](https://github.com/IntelPython/dpnp/pull/1727)
* Implemented `dpnp.fromfunction`, `dpnp.fromiter` and `dpnp.loadtxt` functions [#1728](https://github.com/IntelPython/dpnp/pull/1728)
* Added implementation of `dpnp.linalg.pinv` function [#1704](https://github.com/IntelPython/dpnp/pull/1704)
* Added implementation of `dpnp.linalg.eigvalsh` function [#1714](https://github.com/IntelPython/dpnp/pull/1714)
* Added implementation of `dpnp.linalg.tensorinv` function [#1752](https://github.com/IntelPython/dpnp/pull/1752)
* Added implementation of `dpnp.linalg.tensorsolve` function [#1753](https://github.com/IntelPython/dpnp/pull/1753)
* Added implementation of `dpnp.linalg.lstsq` function [#1792](https://github.com/IntelPython/dpnp/pull/1792)
* Added implementation of `dpnp.einsum` and `dpnp.einsum_path` functions [#1779](https://github.com/IntelPython/dpnp/pull/1779)
* Added implementation of `dpnp.histogram` function [#1785](https://github.com/IntelPython/dpnp/pull/1785)
* Added implementation of `dpnp.histogram_bin_edges` function [#1823](https://github.com/IntelPython/dpnp/pull/1823)
* Added implementation of `dpnp.digitize` function [#1847](https://github.com/IntelPython/dpnp/pull/1847)
* Extended pre-commit hooks with `pylint` configuration [#1718](https://github.com/IntelPython/dpnp/pull/1718)
* Extended pre-commit hooks with `codespell` configuration [#1798](https://github.com/IntelPython/dpnp/pull/1798)
* Added a Security policy page [#1730](https://github.com/IntelPython/dpnp/pull/1730)
* Implemented `nin` and `nout` properties for `dpnp` elementwise functions [#1712](https://github.com/IntelPython/dpnp/pull/1712)
* Implemented `outer` method for `dpnp` elementwise functions [#1813](https://github.com/IntelPython/dpnp/pull/1813)

### Changed

* Added support of more number of data types and dimensions for input arrays, and all keyword arguments in `dpnp.cross` function [#1715](https://github.com/IntelPython/dpnp/pull/1715)
* Added support of more number of data types and dimensions for input array, and all keyword arguments in `dpnp.linalg.matrix_rank` function [#1717](https://github.com/IntelPython/dpnp/pull/1717)
* Added support of more number of data types and dimensions for input arrays in `dpnp.inner` function [#1726](https://github.com/IntelPython/dpnp/pull/1726)
* Added support of more number of data types and dimensions for input arrays in `dpnp.linalg.multi_dot` function [#1729](https://github.com/IntelPython/dpnp/pull/1729)
* Added support of more number of data types and dimensions for input arrays in `dpnp.kron` function [#1732](https://github.com/IntelPython/dpnp/pull/1732)
* Added support of more number of data types and dimensions for input arrays in `dpnp.linalg.matrix_power` function [#1748](https://github.com/IntelPython/dpnp/pull/1748)
* Added support of more number of data types and dimensions for input array, and all keyword arguments in `dpnp.norm` function [#1746](https://github.com/IntelPython/dpnp/pull/1746)
* Added support of more number of data types and dimensions for input array in `dpnp.cond` function [#1773](https://github.com/IntelPython/dpnp/pull/1773)
* Extended `dpnp.matmul` function to support `axes` keyword argument [#1705](https://github.com/IntelPython/dpnp/pull/1705)
* Extended `dpnp.searchsorted` function to support `side` and `sorter` keyword arguments [#1751](https://github.com/IntelPython/dpnp/pull/1751)
* Extended `dpnp.where` function to support scalar type by `x` and `y` arrays [#1760](https://github.com/IntelPython/dpnp/pull/1760)
* Extended `dpnp.ndarray.transpose` method to support `axes` keyword as a list [#1770](https://github.com/IntelPython/dpnp/pull/1770)
* Extended `dpnp.nancumsum` function to support `axis`, `dtype` and `out` keyword arguments [#1781](https://github.com/IntelPython/dpnp/pull/1781)
* Extended `dpnp.nancumprod` function to support `axis`, `dtype` and `out` keyword arguments [#1812](https://github.com/IntelPython/dpnp/pull/1812)
* Extended `dpnp.put` function to support more number of data types and dimensions for input arrays [#1838](https://github.com/IntelPython/dpnp/pull/1838)
* Extended `dpnp.trace` function to support `axis1`, `axis2`, `dtype` and `out` keyword arguments [#1842](https://github.com/IntelPython/dpnp/pull/1842)
* Corrected `dpnp.ndarray.real`and `dpnp.ndarray.imag` methods to return a view of the array [#1719](https://github.com/IntelPython/dpnp/pull/1719)
* Corrected `dpnp.nonzero` function to raise `TypeError` exception for input array of unexpected type [#1764](https://github.com/IntelPython/dpnp/pull/1764)
* Corrected `dpnp.diagonal` function to return a view of the array [#1817](https://github.com/IntelPython/dpnp/pull/1817)
* Removed `dpnp.find_common_type` function as it was deprecated since NumPy 1.25.0 [#1742](https://github.com/IntelPython/dpnp/pull/1742)
* Removed use of `dpctl` queue manager API [#1735](https://github.com/IntelPython/dpnp/pull/1735)
* Leveraged `dpctl.tensor` implementation for `dpnp.cumsum` function [#1772](https://github.com/IntelPython/dpnp/pull/1772)
* Leveraged `dpctl.tensor` implementation for `dpnp.cumprod` function [#1811](https://github.com/IntelPython/dpnp/pull/1811)
* Leveraged `dpctl.tensor` implementation for `dpnp.cumlogsumexp` function [#1816](https://github.com/IntelPython/dpnp/pull/1816)
* Leveraged `dpctl.tensor` support of `out` keyword argument in reduction and `dpnp.where` functions [#1808](https://github.com/IntelPython/dpnp/pull/1808)
* Aligned with `dpctl` interface changes per Python Array API 2023.12 specification [#1774](https://github.com/IntelPython/dpnp/pull/1774)
* Reworked `dpnp.linalg.eig` and `dpnp.linalg.eigvals` implementations to fall back on on NumPy calculation due to a lack of required functionality in OneMKL LAPACK [#1780](https://github.com/IntelPython/dpnp/pull/1780)
* `dpnp` uses pybind11 2.12.0 [#1783](https://github.com/IntelPython/dpnp/pull/1783)
* Improved `dpnp.matmul` implementation to use column major `gemm` layout for F-contiguous input arrays [#1793](https://github.com/IntelPython/dpnp/pull/1793)
* Improved performance of `dpnp.matmul` function by call of `dpnp.kron` and `dpnp.dot` for special cases [#1815](https://github.com/IntelPython/dpnp/pull/1815)
* Improved performance of `dpnp.diag` function by use of `dpnp.diagonal` which returns a view of the array [#1822](https://github.com/IntelPython/dpnp/pull/1822)
* Removed limitations from `diag_indices`, `diag_indices_from`, `fill_diagonal`, `tril_indices`, `tril_indices_from`, `triu_indices`, `triu_indices_from` functions
and added implementation of `dpnp.mask_indices` function [#1814](https://github.com/IntelPython/dpnp/pull/1814)

### Fixed

* Changed `dpnp.linalg.solve` to use a pair of `getrf` and `getrs` calls from OneMKL library instead of `gesv` one to mitigate an unexpected `RuntimeError` exception [#1763](https://github.com/IntelPython/dpnp/pull/1763)
* Resolved a hang in batch implementation of `dpnp.linalg.solve` when computes on CPU device [#1778](https://github.com/IntelPython/dpnp/pull/1778)
* Resolved an unexpected `TypeError` exception raised from `dpnp.random.vonmises` when used with a scalar `kappa` argument [#1799](https://github.com/IntelPython/dpnp/pull/1799)
* Changed `dpnp.flatten` to comply with compute follows data approach [#1825](https://github.com/IntelPython/dpnp/pull/1825)
* Resolved a hang in batch implementation of `dpnp.linalg.eigh` when computes on CPU device [#1832](https://github.com/IntelPython/dpnp/pull/1832)
* Resolved an unexpected `ValueError` exception raised from `dpnp.linalg.pinv` due to a shape issue in `dpnp.matmul` [#1843](https://github.com/IntelPython/dpnp/pull/1843)


## [0.14.0] - 02/16/2024

This release will require DPC++ `2024.1.0`, which no longer supports Intel Gen9 integrated GPUs found in Intel CPUs of 10th generation and older.

### Added

* Added implementation of `dpnp.tensordot` function [#1699](https://github.com/IntelPython/dpnp/pull/1699)
* Added implementation of `dpnp.nanmean` and `dpnp.nanstd` functions [#1654](https://github.com/IntelPython/dpnp/pull/1654)
* Added implementation of `dpnp.angle` function [#1650](https://github.com/IntelPython/dpnp/pull/1650)
* Added implementation of `dpnp.logsumexp` and `dpnp.reduce_hypot` functions [#1648](https://github.com/IntelPython/dpnp/pull/1648)
* Added implementation of `dpnp.column_stack`, `dpnp.dstack` and `dpnp.row_stack` functions [#1647](https://github.com/IntelPython/dpnp/pull/1647)
* Added implementation of `dpnp.nanargmax`, `dpnp.nanargmin`, `dpnp.nanmax` and `dpnp.nanmin` functions [#1646](https://github.com/IntelPython/dpnp/pull/1646)
* Added implementation of `dpnp.clip` function, available as well as a method of dpnp array [#1645](https://github.com/IntelPython/dpnp/pull/1645)
* Added implementation of `dpnp.copysign` and `dpnp.rsqrt` functions [#1624](https://github.com/IntelPython/dpnp/pull/1624)
* Added implementation of `dpnp.linalg.slogdet` function [#1607](https://github.com/IntelPython/dpnp/pull/1607)
* Added implementation of `dpnp.can_cast` function [#1600](https://github.com/IntelPython/dpnp/pull/1600)
* Added implementation of `dpnp.linalg.solve` function [#1598](https://github.com/IntelPython/dpnp/pull/1598)
* Added implementation of `dpnp.broadcast_arrays` function [#1594](https://github.com/IntelPython/dpnp/pull/1594)
* Added implementation of `dpnp.tile` function [#1586](https://github.com/IntelPython/dpnp/pull/1586)
* Added implementation of `dpnp.iinfo` and `dpnp.finfo` functions [#1582](https://github.com/IntelPython/dpnp/pull/1582)
* Added implementation of `dpnp.logaddexp` function [#1561](https://github.com/IntelPython/dpnp/pull/1561)
* Added implementation of `dpnp.positive` function [#1559](https://github.com/IntelPython/dpnp/pull/1559)

### Changed

* Changed exception type from `ValueError` to `NotImplementedError` for unsupporting keyword arguments in array creation functions [#1695](https://github.com/IntelPython/dpnp/pull/1695)
* Enabled compatibility support against numpy `1.26.4` [#1690](https://github.com/IntelPython/dpnp/pull/1690)
* Implemented `dpnp.true_divide` as an alias on `dpnp.divide` function [#1641](https://github.com/IntelPython/dpnp/pull/1641)
* Added support of more number of data types and dimensions for input array in `dpnp.vdot` function [#1692](https://github.com/IntelPython/dpnp/pull/1692)
* Added support of more number of data types and dimensions for input array in `dpnp.linalg.qr` function [#1673](https://github.com/IntelPython/dpnp/pull/1673)
* Added support of more number of data types and dimensions for input array in `dpnp.dot` function [#1669](https://github.com/IntelPython/dpnp/pull/1669)
* Added support of more number of data types and dimensions for input array in `dpnp.linalg.inv` function [#1665](https://github.com/IntelPython/dpnp/pull/1665)
* Added support of more number of data types for input array in `dpnp.sort` and `dpnp.argsort` functions, as well as implementing support of `axis` keyword [#1660](https://github.com/IntelPython/dpnp/pull/1660)
* Added support of more number of data types and dimensions for input array in `dpnp.linalg.cholesky` function, as well as implementing support of `upper` keyword [#1638](https://github.com/IntelPython/dpnp/pull/1638)
* Added support of more number of data types and dimensions for input array in `dpnp.diff`, as well as implementing support of `prepend` and `append` keywords [#1637](https://github.com/IntelPython/dpnp/pull/1637)
* Added support of more number of data types and dimensions for input array in `dpnp.matmul` function [#1616](https://github.com/IntelPython/dpnp/pull/1616)
* Added support of more number of data types and dimensions for input array in `dpnp.linalg.det` function [#1607](https://github.com/IntelPython/dpnp/pull/1607)
* Added support of more number of data types and dimensions for input array in `dpnp.linalg.svd` function, as well as implementing support of `full_matrices`, `compute_uv` and `hermitian` keywords [#1604](https://github.com/IntelPython/dpnp/pull/1604)
* Accepted different data types and dimensions of input arrays in `dpnp.put_along_axis` and `dpnp.take_along_axis` functions, as well as available values of `axis` keyword [#1636](https://github.com/IntelPython/dpnp/pull/1636)
* Added `keepdims`, `initial` and `where` keywords to `dpnp.amax` and `dpnp.amin` functions [#1639](https://github.com/IntelPython/dpnp/pull/1639)
* Extended `dpnp.meshgrid` function to support `sparse` and `copy` keyword arguments [#1675](https://github.com/IntelPython/dpnp/pull/1675)
* Extended `dpnp.average` function to support `axis`, `weights`, `returned` and `keepdims` keywords and `dpnp.nansum` function with `axis`, `dtype`, `keepdims` and `out` keyword arguments [#1654](https://github.com/IntelPython/dpnp/pull/1654)
* Extended `dpnp.std`, `dpnp.var` and `nanvar` functions to support `axis`, `dtype`, `out` and `keepdims` keyword arguments [#1635](https://github.com/IntelPython/dpnp/pull/1635)
* Extended `dpnp.ogrid` and `dpnp.mgrid` functions with support of device-aware keywords of compute follows data paradigm [#1622](https://github.com/IntelPython/dpnp/pull/1622)
* Extended `dpnp.indices` function to support `dtype` and `sparse` keyword arguments, as well as device-aware keywords of compute follows data paradigm [#1622](https://github.com/IntelPython/dpnp/pull/1622)
* Extended `dpnp.count_nonzero` function to support `axis` and `keepdims` keyword arguments [#1615](https://github.com/IntelPython/dpnp/pull/1615)
* Extended `dpnp.put_along_axis` and `dpnp.take_along_axis` functions to support `out`, `dtype` and `casting` keyword arguments [#1608](https://github.com/IntelPython/dpnp/pull/1608)
* Extended `dpnp.stack` and `dpnp.concatenate` functions to support `out`, `dtype` and `casting` keyword arguments [#1608](https://github.com/IntelPython/dpnp/pull/1608)
* Extended `dpnp.vstack` function to support `dtype` and `casting` keyword arguments [#1595](https://github.com/IntelPython/dpnp/pull/1595)
* Extended `dpnp.diag`, `dpnp.diagflat`, `dpnp.ptp` and `dpnp.vander` functions with support of extra keywords to align with compute follows data paradigm [#1579](https://github.com/IntelPython/dpnp/pull/1579)
* Extended `dpnp.tri` and `dpnp.identity` functions with support of device-aware keywords of compute follows data paradigm [#1577](https://github.com/IntelPython/dpnp/pull/1577)
* Added dedicated in-place kernels to `dpnp.divide` and `dpnp.floor_divide` functions [#1587](https://github.com/IntelPython/dpnp/pull/1587)
* Redesigned `dpnp.cbrt` and `dpnp.exp2` functions through pybind11 extension of OneMKL call where possible or leveraging on `dpctl.tensor` implementation [#1624](https://github.com/IntelPython/dpnp/pull/1624)
* Redesigned `dpnp.exp`, `dpnp.expm1`, `dpnp.log10`, `dpnp.log1p` and `dpnp.log2` functions through pybind11 extension of OneMKL call where possible or leveraging on `dpctl.tensor` implementation [#1576](https://github.com/IntelPython/dpnp/pull/1576)
* Redesigned `dpnp.abs` function through pybind11 extension of OneMKL call where possible or leveraging on `dpctl.tensor` implementation [#1575](https://github.com/IntelPython/dpnp/pull/1575)
* Redesigned `dpnp.hypot` function through pybind11 extension of OneMKL call where possible or leveraging on `dpctl.tensor` implementation [#1560](https://github.com/IntelPython/dpnp/pull/1560)
* Leveraged `dpctl.tensor` implementation for `dpnp.reciprocal` function [#1650](https://github.com/IntelPython/dpnp/pull/1650)
* Leveraged `dpctl.tensor` implementation for `dpnp.mean` function [#1632](https://github.com/IntelPython/dpnp/pull/1632)
* Leveraged `dpctl.tensor` implementation for `dpnp.repeat` function [#1614](https://github.com/IntelPython/dpnp/pull/1614)
* Leveraged `dpctl.tensor` implementation for `dpnp.argmax` and `dpnp.argmin` functions [#1610](https://github.com/IntelPython/dpnp/pull/1610)
* Leveraged `dpctl.tensor` implementation for `dpnp.geomspace` and `dpnp.logspace` functions [#1603](https://github.com/IntelPython/dpnp/pull/1603)
* Leveraged `dpctl.tensor` implementation for `dpnp.max` and `dpnp.min` functions [#1602](https://github.com/IntelPython/dpnp/pull/1602)
* Leveraged `dpctl.tensor` implementation for `dpnp.astype` function [#1597](https://github.com/IntelPython/dpnp/pull/1597)
* Leveraged `dpctl.tensor` implementation for `dpnp.maximum` and `dpnp.minimum` functions [#1558](https://github.com/IntelPython/dpnp/pull/1558)

### Fixed

* Resolved potential raising of execution placement error from `dpnp.take_along_axis` and `dpnp.put_along_axis` functions [#1702](https://github.com/IntelPython/dpnp/pull/1702)
* Improved performance of `dpnp.matmul` and `dpnp.dot` function when `out` keyword is passed [#1694](https://github.com/IntelPython/dpnp/pull/1694)
* Completed documentation for each array creation functions [#1674](https://github.com/IntelPython/dpnp/pull/1674)
* Aligned `dpnp.clip` where both `min` and `max` keywords have `None` value with NumPy implementation [#1670](https://github.com/IntelPython/dpnp/pull/1670)
* Fixed a bug related to `out` keyword in elementwise functions [#1656](https://github.com/IntelPython/dpnp/pull/1656)
* Resolved compilation warnings due to `-Wvla-extension` option enabled by default [#1651](https://github.com/IntelPython/dpnp/pull/1651)
* Replaced deprecated `IntelDPCPPConfig.cmake` script with vendored `IntelSYCLConfig.cmake` [#1611](https://github.com/IntelPython/dpnp/pull/1611)
* Improved coverage report to include code of pybind11 extensions [#1609](https://github.com/IntelPython/dpnp/pull/1609)
* Improved performance of `dpnp.atleast_2d` and `dpnp.atleast_3d` functions and fixed to return a correct shape of resulting array [#1560](https://github.com/IntelPython/dpnp/pull/1560)


## [0.13.0] - 09/29/2023

### Added

* Added implementation of `dpnp.imag` and `dpnp.real` functions, as well as the corresponding properties and setters of dpnp array [#1557](https://github.com/IntelPython/dpnp/pull/1557)
* Added implementation of flipping functions: `dpnp.flip`, `dpnp.fliplr` and `dpnp.flipud` [#1543](https://github.com/IntelPython/dpnp/pull/1543)
* Added implementation of `dpnp.rint` function through `dpnp.round` call [#1537](https://github.com/IntelPython/dpnp/pull/1537)
* Added in-place support for arithmetic operators [#1530](https://github.com/IntelPython/dpnp/pull/1530)
* Dropped build and uploading the package with `python=3.8` to `dppy/label/dev` channel of Anaconda [#1534](https://github.com/IntelPython/dpnp/pull/1534)
* Implemented build and uploading the package with `python=3.11` to `dppy/label/dev` channel of Anaconda [#1501](https://github.com/IntelPython/dpnp/pull/1501)
* Added the versioneer to compute a product version number [#1497](https://github.com/IntelPython/dpnp/pull/1497)
* Added `cython` support of `3.0.0` or above version [#1495](https://github.com/IntelPython/dpnp/pull/1495)

### Changed

* Updated `Build from source` section in `README.md` to state all the required prerequisite packages [#1553](https://github.com/IntelPython/dpnp/pull/1553)
* Reworked `dpnp.hstack` and `dpnp.atleast_1d` through existing functions to get rid of falling back on NumPy [#1544](https://github.com/IntelPython/dpnp/pull/1544)
* Reworked `dpnp.asfarray` through existing functions to get rid of falling back on NumPy [#1542](https://github.com/IntelPython/dpnp/pull/1542)
* Converted from `raw` to `multi_ptr` with `address_space_cast` to adopt towards changes introduced in `SYCL 2020` [#1538](https://github.com/IntelPython/dpnp/pull/1538)
* Updated install instruction via `pip` [#1531](https://github.com/IntelPython/dpnp/pull/1531)
* Reworked `dpnp.copyto` through existing functions instead of a separate kernel [#1516](https://github.com/IntelPython/dpnp/pull/1516)
* Aligned default order value with NumPy in asarray-like functions [#1526](https://github.com/IntelPython/dpnp/pull/1526)
* Created unary and binary elementwise functions at module import [#1522](https://github.com/IntelPython/dpnp/pull/1522)
* Redesigned trigonometric and hyperbolic functions through pybind11 extension of OneMKL calls where possible or leveraging on `dpctl.tensor` implementation [#1545](https://github.com/IntelPython/dpnp/pull/1545)
* Added `dpnp.signbit` and `dpnp.proj` functions implemented through pybind11 extension of OneMKL calls where possible or leveraging on `dpctl.tensor` implementation [#1535](https://github.com/IntelPython/dpnp/pull/1535)
* Redesigned `dpnp.round` and `dpnp.around` functions through pybind11 extension of OneMKL calls where possible or leveraging on `dpctl.tensor` implementation [#1520](https://github.com/IntelPython/dpnp/pull/1520)
* Redesigned `dpnp.sign` and `dpnp.negative` functions through pybind11 extension of OneMKL calls where possible or leveraging on `dpctl.tensor` implementation [#1523](https://github.com/IntelPython/dpnp/pull/1523)
* Redesigned `dpnp.conjugate` and `dpnp.conj` functions through pybind11 extension of OneMKL calls where possible or leveraging on `dpctl.tensor` implementation [#1519](https://github.com/IntelPython/dpnp/pull/1519)
* Redesigned `dpnp.ceil`, `dpnp.floor` and `dpnp.trunc` functions through pybind11 extension of OneMKL calls where possible or leveraging on `dpctl.tensor` implementation [#1518](https://github.com/IntelPython/dpnp/pull/1518)
* Redesigned `dpnp.remainder` and `dpnp.mod` functions through pybind11 extension of OneMKL calls where possible or leveraging on `dpctl.tensor` implementation [#1515](https://github.com/IntelPython/dpnp/pull/1515)
* Redesigned `dpnp.power` function through pybind11 extension of OneMKL calls where possible or leveraging on `dpctl.tensor` implementation [#1476](https://github.com/IntelPython/dpnp/pull/1476)
* Leveraged `dpctl.tensor` implementation for `dpnp.put` function [#1529](https://github.com/IntelPython/dpnp/pull/1529)
* Leveraged `dpctl.tensor` implementation for `dpnp.roll` and `dpnp.rollaxis` functions [#1517](https://github.com/IntelPython/dpnp/pull/1517)
* Leveraged `dpctl.tensor` implementation for `dpnp.copy` function [#1540](https://github.com/IntelPython/dpnp/pull/1540)
* Leveraged `dpctl.tensor` implementation for `dpnp.expand_dims` and `dpnp.swapaxes` functions [#1532](https://github.com/IntelPython/dpnp/pull/1532)
* Leveraged `dpctl.tensor` implementation for bitwise operations [#1508](https://github.com/IntelPython/dpnp/pull/1508)
* Leveraged `dpctl.tensor` implementation for `dpnp.all` and `dpnp.any` functions [#1512](https://github.com/IntelPython/dpnp/pull/1512)
* Leveraged `dpctl.tensor` implementation for `dpnp.stack` function [#1509](https://github.com/IntelPython/dpnp/pull/1509)
* Leveraged `dpctl.tensor` implementation for `dpnp.concatenate` function [#1507](https://github.com/IntelPython/dpnp/pull/1507)
* Leveraged `dpctl.tensor` implementation for `dpnp.isnan`, `dpnp.isinf` and `dpnp.isfinite` functions [#1504](https://github.com/IntelPython/dpnp/pull/1504)
* Leveraged `dpctl.tensor` implementation for `dpnp.take` function [#1492](https://github.com/IntelPython/dpnp/pull/1492)
* Refreshed API References block in the documentation [#1490](https://github.com/IntelPython/dpnp/pull/1490)
* Refreshed documentation to reflect an actual product behavior [#1485](https://github.com/IntelPython/dpnp/pull/1485)
* Upgraded the build flow to use newer `pybind11=2.11.1` version [#1510](https://github.com/IntelPython/dpnp/pull/1510)
* Updated pre-commit hooks to run with `flake8=6.1.0` and `black=23.7.0` [#1505](https://github.com/IntelPython/dpnp/pull/1505)
* Pinned DPC++ and OneMKL versions to `2023.2`` release [#1496](https://github.com/IntelPython/dpnp/pull/1496)
* Added a specialized kernel for F-contiguous arrays to `dpnp.sum` with `axis=1` [#1489](https://github.com/IntelPython/dpnp/pull/1489)
* Removed a workaround to Klockwork since it is not used anymore due to transition to Coverity tool [#1493](https://github.com/IntelPython/dpnp/pull/1493)

### Fixed

* Resolved `Logically dead code` issue addressed by Coverity scan [#1541](https://github.com/IntelPython/dpnp/pull/1541)
* Resolved `Arguments in wrong order` issue addressed by Coverity scan [#1513](https://github.com/IntelPython/dpnp/pull/1513)
* Resolved `Pointer to local outside scope` issue addressed by Coverity scan [#1514](https://github.com/IntelPython/dpnp/pull/1514)
* Fixed assigning a value to potentially none-valued dictionary coverage generation script [#1511](https://github.com/IntelPython/dpnp/pull/1511)
* Resolved issues with running `dpnp.allclose` function on a device without fp64 support [#1536](https://github.com/IntelPython/dpnp/pull/1536)
* Resolved issues with running FFT functions on a device without fp64 support [#1524](https://github.com/IntelPython/dpnp/pull/1524)
* Resolved issues with running mathematical functions on a device without fp64 support [#1502](https://github.com/IntelPython/dpnp/pull/1502)
* Resolved issues with running random functions on a device without fp64 support [#1498](https://github.com/IntelPython/dpnp/pull/1498)
* Resolved issues with running statistics functions on a device without fp64 support [#1494](https://github.com/IntelPython/dpnp/pull/1494)


## [0.12.1] - 07/18/2023

### Added

* Added `classifiers metadata` to a description of dpnp package [#1460](https://github.com/IntelPython/dpnp/pull/1460)
* Added `pygrep-hooks` to pre-commit config [#1454](https://github.com/IntelPython/dpnp/pull/1454)
* Added `flake8` to pre-commit config [#1453](https://github.com/IntelPython/dpnp/pull/1453)
* Added `isort` to pre-commit config [#1451](https://github.com/IntelPython/dpnp/pull/1451)
* Added `clang` format to pre-commit config [#1450](https://github.com/IntelPython/dpnp/pull/1450)
* Added `black` to pre-commit config [#1449](https://github.com/IntelPython/dpnp/pull/1449)
* Added `pre-commit` hooks [#1448](https://github.com/IntelPython/dpnp/pull/1448)

### Changed

* Pinned to `dpctl>=0.14.5` as host and run dependencies [#1481](https://github.com/IntelPython/dpnp/pull/1481)
* Pinned dependent `cython` package to a version less than `3.0.0` [#1480](https://github.com/IntelPython/dpnp/pull/1480)
* Added a specialized kernel for `dpnp.sum` with `axis=0` as a pybind11 extension [#1479](https://github.com/IntelPython/dpnp/pull/1479)
* Redesigned `dpnp.square` function through pybind11 extension of OneMKL call where possible or leveraging on `dpctl.tensor` implementation [#1473](https://github.com/IntelPython/dpnp/pull/1473)
* Redesigned `dpnp.cos` and `dpnp.sin` functions through pybind11 extension of OneMKL calls where possible or leveraging on `dpctl.tensor` implementation [#1471](https://github.com/IntelPython/dpnp/pull/1471)
* Redesigned `dpnp.sqrt` function through pybind11 extension of OneMKL call where possible or leveraging on `dpctl.tensor` implementation [#1470](https://github.com/IntelPython/dpnp/pull/1470)
* Redesigned `dpnp.log` function through pybind11 extension of OneMKL call where possible or leveraging on `dpctl.tensor` implementation [#1469](https://github.com/IntelPython/dpnp/pull/1469)
* Leveraged `dpctl.tensor` implementation for logical operations [#1464](https://github.com/IntelPython/dpnp/pull/1464)
* Leveraged `dpctl.tensor` implementation for `dpnp.floor_divide` function [#1462](https://github.com/IntelPython/dpnp/pull/1462)
* Leveraged `dpctl.tensor` implementation for comparison functions [#1458](https://github.com/IntelPython/dpnp/pull/1458)

### Fixed

* Improved `dpnp.dot` function to support OneMKL calls for input and output arrays with strides [#1477](https://github.com/IntelPython/dpnp/pull/1477)
* Resolved issues with running `dpnp.linalg` functions on a device without fp64 support [#1474](https://github.com/IntelPython/dpnp/pull/1474)
* Added `dtype` check of fp64 support by the resulting array in `call_origin` function [#1457](https://github.com/IntelPython/dpnp/pull/1457)
* Resolved a compilation warning with `std::getenv()` call on Windows [#1452](https://github.com/IntelPython/dpnp/pull/1452)
* Corrected a link to OneAPI Toolkit in Installation Guide [#1445](https://github.com/IntelPython/dpnp/pull/1445)

## [0.12.0] - 06/15/2023

### Added

* Implemented `dpnp.broadcast_to` function [#1333](https://github.com/IntelPython/dpnp/pull/1333)
* Implemented `dpnp.extract` function [#1340](https://github.com/IntelPython/dpnp/pull/1340)
* Implemented `dpnp.linalg.eigh` function through pybind11 extension of OneMKL call [#1383](https://github.com/IntelPython/dpnp/pull/1383)
* Implemented `dpnp.mean` function [#1431](https://github.com/IntelPython/dpnp/pull/1431)
* Added support of bool types in bitwise operations [#1334](https://github.com/IntelPython/dpnp/pull/1334)
* Added `out` parameter in `dpnp.add` function [#1329](https://github.com/IntelPython/dpnp/pull/1329)
* Added `out` parameter in `dpnp.multiply` function [#1365](https://github.com/IntelPython/dpnp/pull/1365)
* Added `out` parameter in `dpnp.sqrt` function [#1332](https://github.com/IntelPython/dpnp/pull/1332)
* Added `rowvar` parameter in `dpnp.cov` function [#1371](https://github.com/IntelPython/dpnp/pull/1371)
* Added `nbytes` property to dpnp array [#1359](https://github.com/IntelPython/dpnp/pull/1359)
* Introduced a new github Action to control code coverage [#1373](https://github.com/IntelPython/dpnp/pull/1373)
* Added change log [#1439](https://github.com/IntelPython/dpnp/pull/1439)


### Changed

* Leveraged `dpctl.tensor` implementation for `dpnp.place` function [#1337](https://github.com/IntelPython/dpnp/pull/1337)
* Leveraged `dpctl.tensor` implementation for `dpnp.moveaxis` function [#1382](https://github.com/IntelPython/dpnp/pull/1382)
* Leveraged `dpctl.tensor` implementation for `dpnp.squeeze` function [#1381](https://github.com/IntelPython/dpnp/pull/1381)
* Leveraged `dpctl.tensor` implementation for `dpnp.where` function [#1380](https://github.com/IntelPython/dpnp/pull/1380)
* Leveraged `dpctl.tensor` implementation for `dpnp.transpose` function [#1389](https://github.com/IntelPython/dpnp/pull/1389)
* Leveraged `dpctl.tensor` implementation for `dpnp.reshape` function [#1391](https://github.com/IntelPython/dpnp/pull/1391)
* Leveraged `dpctl.tensor` implementation for `dpnp.add`, `dpnp.multiply` and `dpnp.subtract` functions [#1430](https://github.com/IntelPython/dpnp/pull/1430)
* Leveraged `dpctl.tensor` implementation for `dpnp.sum` function [#1426](https://github.com/IntelPython/dpnp/pull/1426)
* Leveraged `dpctl.tensor` implementation for `dpnp.result_type` function [#1435](https://github.com/IntelPython/dpnp/pull/1435)
* Reused OneDPL `std::nth_element` function in `dpnp.partition` with 1d array [#1406](https://github.com/IntelPython/dpnp/pull/1406)
* Transitioned dpnp build system to use scikit-build [#1349](https://github.com/IntelPython/dpnp/pull/1349)
* Renamed included dpnp_algo_*.pyx files to *.pxi [#1356](https://github.com/IntelPython/dpnp/pull/1356)
* Implemented support of key as a tuple in `dpnp.__getitem__()` and `dpnp.__setitem__()` functions [#1362](https://github.com/IntelPython/dpnp/pull/1362)
* Selected dpnp own kernels for elementwise functions instead of OneMKL VM calls on a device without fp64 aspect [#1386](https://github.com/IntelPython/dpnp/pull/1386)
* Pinned to `sysroot>=2.28` and transitioned to `conda-forge` channel [#1408](https://github.com/IntelPython/dpnp/pull/1408)
* Redesigned `dpnp.divide` implementation to call `div` from OneMKL for C-contiguous data or to use `dpctl.tensor` library otherwise [#1418](https://github.com/IntelPython/dpnp/pull/1418)
* Changed an engine used for random generated array on GPU device from MT19937 to MCG59 [#1423](https://github.com/IntelPython/dpnp/pull/1423)
* Implemented in-place support of `dpnp.divide` [#1434](https://github.com/IntelPython/dpnp/pull/1434)
* Redesigned `dpnp.outer` implementation through `dpnp.multiply` with broadcasted arrays [#1436](https://github.com/IntelPython/dpnp/pull/1436)
* Pinned to `dpctl>=0.14.3` as host and run dependencies [#1437](https://github.com/IntelPython/dpnp/pull/1437)
* Reimplemented `dpnp.cov` through existing dpnp function instead of a separate kernel [#1396](https://github.com/IntelPython/dpnp/pull/1396)


### Fixed

* Fixed `dpnp.asarray` function to accept a sequence of dpnp arrays [#1355](https://github.com/IntelPython/dpnp/pull/1355)
* Fixed crash in `dpnp.sum` with an empty array [#1369](https://github.com/IntelPython/dpnp/pull/1369)
* Fixed compilation error around `sycl::abs` with DPC++ 2023.2.0 [#1393](https://github.com/IntelPython/dpnp/pull/1393)
* Fixed Klockwork run and enabled cmake verbose mode for conda build [#1433](https://github.com/IntelPython/dpnp/pull/1433)
