REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
SET "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

REM Since the 60.0.0 release, setuptools includes a local, vendored copy
REM of distutils (from late copies of CPython) that is enabled by default.
REM It breaks build for Windows, so use distutils from "stdlib" as before.
REM @TODO: remove the setting, once transition to build backend on Windows
REM to cmake is complete.
SET "SETUPTOOLS_USE_DISTUTILS=stdlib"

IF DEFINED DPLROOT (
    ECHO "Sourcing DPLROOT"
    SET "INCLUDE=%DPLROOT%\include;%INCLUDE%"
)

%PYTHON% setup.py build_clib
%PYTHON% setup.py build_ext install
