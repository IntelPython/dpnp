REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
set "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

"%PYTHON%" setup.py clean --all

set "MKLROOT=%PREFIX%/Library"
set "TBB_ROOT_HINT=%PREFIX%/Library"
set "DPL_ROOT_HINT=%PREFIX%/Library"

REM Overriding IPO is useful for building in resources constrained VMs (public CI)
if DEFINED OVERRIDE_INTEL_IPO (
  set "CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=FALSE"
)

FOR %%V IN (17.0.0 17 18.0.0 18 19.0.0 19) DO @(
  REM set DIR_HINT if directory exists
  IF EXIST "%BUILD_PREFIX%\Library\lib\clang\%%V\" (
    SET "SYCL_INCLUDE_DIR_HINT=%BUILD_PREFIX%\Library\lib\clang\%%V"
  )
)

set "PATCHED_CMAKE_VERSION=3.26"
set "PLATFORM_DIR=%PREFIX%\Library\share\cmake-%PATCHED_CMAKE_VERSION%\Modules\Platform"
set "FN=Windows-IntelLLVM.cmake"

rem Save the original file, and copy patched file to
rem fix the issue with IntelLLVM integration with cmake on Windows
if EXIST "%PLATFORM_DIR%" (
  dir "%PLATFORM_DIR%\%FN%"
  copy /Y "%PLATFORM_DIR%\%FN%" .
  if errorlevel 1 exit 1
  copy /Y ".github\workflows\Windows-IntelLLVM_%PATCHED_CMAKE_VERSION%.cmake" "%PLATFORM_DIR%\%FN%"
  if errorlevel 1 exit 1
)

set "CC=icx"
set "CXX=icx"

set "CMAKE_GENERATOR=Ninja"
:: Make CMake verbose
set "VERBOSE=1"

%PYTHON% -m build -w -n -x
if %ERRORLEVEL% neq 0 exit 1
