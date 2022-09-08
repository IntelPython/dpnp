REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
SET "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

IF DEFINED DPLROOT (
    ECHO "Sourcing DPLROOT"
    SET "INCLUDE=%DPLROOT%\include;%INCLUDE%"
)

%PYTHON% setup.py build_clib
%PYTHON% setup.py build_ext install
