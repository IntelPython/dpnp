
SET "ONEAPI_ROOT=C:\oneapi"
CALL "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
CALL "%ONEAPI_ROOT%\mkl\latest\env\vars.bat"
CALL "%ONEAPI_ROOT%\tbb\latest\env\vars.bat"
CALL "%ONEAPI_ROOT%\dpl\latest\env\vars.bat"

SET "DPCPPROOT=%ONEAPI_ROOT%\compiler\latest"
