# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
