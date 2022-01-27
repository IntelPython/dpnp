REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
SET "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

IF DEFINED DPLROOT (
    ECHO "Sourcing DPLROOT"
    SET "INCLUDE=%DPLROOT%\include;%INCLUDE%"
)

%PYTHON% setup.py build_clib
%PYTHON% setup.py build_ext install
