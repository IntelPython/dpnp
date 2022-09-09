rem git clone --branch 0.5.0rc2 https://github.com/IntelPython/dpctl.git
rem for /f "tokens=* delims=" %%a in ('git tag -l') do git tag -d %%a
rem git tag 0.5.0rc2

echo +++++++++++++++++++++++++ Python version +++++++++++++++++++++++++++
call python --version

rem need to keep package file exists because it will be used as a channel for conda build later
set DPCTL_DIST=%CD%\dist_dpctl

call conda uninstall -y dpctl

echo +++++++++++++++++++++++++ Downlowd DPCTL +++++++++++++++++++++++++++
call git clone https://github.com/IntelPython/dpctl.git
cd dpctl

set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI\"
echo +++++++++++++++++++++++++ Build DPCTL +++++++++++++++++++++++++++
call conda build --croot=C:/tmp conda-recipe --no-test -c "%ONEAPI_ROOT%\conda_channel" --output-folder %DPCTL_DIST%

echo +++++++++++++++++++++++++ get DPCTL package name +++++++++++++++++++++++++++
rem this garbage here is because
rem I didn't find a method how to get path and filename for the package built by conda build
FOR /F "tokens=* USEBACKQ" %%F IN (`dir /B %DPCTL_DIST%\win-64\dpctl*.bz2`) DO SET DPCTL_PACKAGE_NAME=%%F
echo %DPCTL_DIST%
echo %DPCTL_PACKAGE_NAME%

echo +++++++++++++++++++++++++ install DPCTL +++++++++++++++++++++++++++
call conda install -y %DPCTL_DIST%/win-64/%DPCTL_PACKAGE_NAME%

cd ..
echo +++++++++++++++++++++++++ cleanup DPCTL sources +++++++++++++++++++++++++++
del /F/Q/S dpctl
rmdir /Q/S dpctl

dir /s/b "%ONEAPI_ROOT%\libDPCTLSyclInterface.so"

call conda list
