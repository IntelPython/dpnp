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

FOR %%V IN (17.0.0 17 18.0.0 18 19.0.0 19 20.0.0 20 21.0.0 21 22.0.0 22) DO @(
  REM set DIR_HINT if directory exists
  IF EXIST "%BUILD_PREFIX%\Library\lib\clang\%%V\" (
    set "SYCL_INCLUDE_DIR_HINT=%BUILD_PREFIX%\Library\lib\clang\%%V"
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

:: `pip install dist\numpy*.whl` does not work on windows,
:: so use a loop; there's only one wheel in dist/ anyway
for /f %%f in ('dir /b /S .\dist') do (
  %PYTHON% -m wheel tags --remove --build %GIT_DESCRIBE_NUMBER% %%f
  if %ERRORLEVEL% neq 0 exit 1
)

:: wheel file was renamed
for /f %%f in ('dir /b /S .\dist') do (
  %PYTHON% -m pip install %%f ^
    --no-build-isolation ^
    --no-deps ^
    --only-binary :all: ^
    --no-index ^
    --prefix %PREFIX% ^
    -vv
  if %ERRORLEVEL% neq 0 exit 1
)

:: Copy wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
  copy dist\dpnp*.whl %WHEELS_OUTPUT_FOLDER%
  if %ERRORLEVEL% neq 0 exit 1
)
