# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.13.0] - TBA

### Added

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
