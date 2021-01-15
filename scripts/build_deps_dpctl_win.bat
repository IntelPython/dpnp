echo +++++++++++++++++++++++++ Build DPCTL 0.5.0rc2 +++++++++++++++++++++++++++
git clone --branch 0.5.0rc2 https://github.com/IntelPython/dpctl.git 

cd dpctl

:: didn't find better way to set required version
for /f "tokens=* delims=" %%a in ('git tag -l') do git tag -d %%a
git tag 0.5.0rc2

conda build --croot=C:/tmp conda-recipe -c "%ONEAPI_ROOT%\conda_channel"

dir /s/b "%ONEAPI_ROOT%\libDPCTLSyclInterface.so"
