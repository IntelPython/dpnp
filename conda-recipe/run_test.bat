@echo on

REM if ONEAPI_ROOT is specified (use all from it)
if defined ONEAPI_ROOT (
    set "DPCPPROOT=%ONEAPI_ROOT%\compiler\latest"
    set "MKLROOT=%ONEAPI_ROOT%\mkl\latest"
    set "TBBROOT=%ONEAPI_ROOT%\tbb\latest"
    set "DPLROOT=%ONEAPI_ROOT%\dpl\latest"
)

echo "DPCPPROOT=%DPCPPROOT%"
echo "MKLROOT=%MKLROOT%"
echo "TBBROOT=%TBBROOT%"
echo "DPLROOT=%DPLROOT%"

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


"%PYTHON%" -c "import dpnp; print(dpnp.__version__)"
if %errorlevel% neq 0 exit 1

"%PYTHON%" -m dpctl -f
if %errorlevel% neq 0 exit 1

echo "Test only dpnp.tests.test_usm_type::TestFft::test_rfftn"

REM https://cje-fm-owrp-prod04.devtools.intel.com/satg-dap-intelpython/job/intel-packages/job/dpnp/job/dev-windows-py3.11/job/test-wheel-conda-stable/236
"%PYTHON%" -m pytest --count 300 -ra -v --pyargs dpnp.tests.test_usm_type::TestFft::test_rfftn
if %errorlevel% neq 0 exit 1

"%PYTHON%" -m pytest --count 100 -ra -v --pyargs dpnp.tests.test_usm_type::TestFft::test_fft
if %errorlevel% neq 0 exit 1

echo "Test only dpnp.tests.test_usm_type::test_norm"

REM https://cje-fm-owrp-prod04.devtools.intel.com/satg-dap-intelpython/job/intel-packages/job/dpnp/job/dev-windows-py3.12/job/test-stable/137/
"%PYTHON%" -m pytest --count 300 -ra -v --pyargs dpnp.tests.test_usm_type::test_norm
if %errorlevel% neq 0 exit 1

echo "Test only dpnp.tests.test_usm_type::TestFft::test_fftfreq"

REM https://cje-fm-owrp-prod04.devtools.intel.com/satg-dap-intelpython/job/intel-packages/job/dpnp/job/dev-windows-py3.11/job/test-stable/120/
"%PYTHON%" -m pytest --count 100 -ra -v --pyargs dpnp.tests.test_usm_type::TestFft::test_fftfreq
if %errorlevel% neq 0 exit 1

echo "Test only dpnp.tests.test_usm_type::TestFft"

REM https://cje-fm-owrp-prod04.devtools.intel.com/satg-dap-intelpython/job/intel-packages/job/dpnp/job/dev-windows-py3.11/job/test/1713/
REM https://cje-fm-owrp-prod04.devtools.intel.com/satg-dap-intelpython/job/intel-packages/job/dpnp/job/dev-windows-py3.11/job/test-wheel-conda-stable/232/
REM https://cje-fm-owrp-prod04.devtools.intel.com/satg-dap-intelpython/job/intel-packages/job/dpnp/job/dev-windows-py3.12/job/test-wheel-conda-stable/234/
"%PYTHON%" -m pytest --count 100 -ra -v --pyargs dpnp.tests.test_usm_type::TestFft
if %errorlevel% neq 0 exit 1

echo "Test only dpnp.tests.test_usm_type"

"%PYTHON%" -m pytest --count 100 -ra -v --pyargs dpnp.tests.test_usm_type
if %errorlevel% neq 0 exit 1

echo "Test only dpnp.tests.test_indexing"

REM https://cje-fm-owrp-prod04.devtools.intel.com/satg-dap-intelpython/job/intel-packages/job/dpnp/job/dev-windows-py3.11/job/test-wheel-conda/1733/
"%PYTHON%" -m pytest --count 100 -ra -v --pyargs dpnp.tests.test_indexing
if %errorlevel% neq 0 exit 1

echo "Test everything"

"%PYTHON%" -m pytest -ra --pyargs dpnp
if %errorlevel% neq 0 exit 1
