@echo on

:: Skip tensor tests by default to avoid OOM in conda builds.
:: Set SKIP_TENSOR_TESTS=0 to run them on machines with enough memory.
if not defined SKIP_TENSOR_TESTS (
    set "SKIP_TENSOR_TESTS=1"
)

"%PYTHON%" -c "import dpnp; print(dpnp.__version__)"
if %errorlevel% neq 0 exit 1

"%PYTHON%" -m dpctl -f
if %errorlevel% neq 0 exit 1

"%PYTHON%" -m pytest -ra --pyargs dpnp
if %errorlevel% neq 0 exit 1
