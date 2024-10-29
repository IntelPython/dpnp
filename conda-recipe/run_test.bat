@echo on

REM if ONEAPI_ROOT is specified (use all from it)
if defined ONEAPI_ROOT (
    set "DPCPPROOT=%ONEAPI_ROOT%\compiler\latest"
    set "MKLROOT=%ONEAPI_ROOT%\mkl\latest"
    set "TBBROOT=%ONEAPI_ROOT%\tbb\latest"
    set "DPLROOT=%ONEAPI_ROOT%\dpl\latest"
)

REM if DPCPPROOT is specified (work with custom DPCPP)
if defined DPCPPROOT (
    call "%DPCPPROOT%\env\vars.bat"
)

REM if MKLROOT is specified (work with custom math library)
if defined MKLROOT (
    call "%MKLROOT%\env\vars.bat"
)

REM have to activate while SYCL CPU device/driver needs paths
REM if TBBROOT is specified
if defined TBBROOT (
    call "%TBBROOT%\env\vars.bat"
)

REM If PYTHON is not set
REM assign it to the Python interpreter from the testing environment
if not defined PYTHON (
    for %%I in (python.exe) do set PYTHON=%%~$PATH:I
)


"%PYTHON%" -c "import dpctl; print(dpctl.__version__)"
if errorlevel 1 exit 1

"%PYTHON%" -m dpctl -f
if errorlevel 1 exit 1

"%PYTHON%" -m pytest -q -ra --disable-warnings --pyargs dpnp -vv
if errorlevel 1 exit 1
