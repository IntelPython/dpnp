rem git clone --branch 0.5.0rc2 https://github.com/IntelPython/dpctl.git 
rem for /f "tokens=* delims=" %%a in ('git tag -l') do git tag -d %%a
rem git tag 0.5.0rc2

echo +++++++++++++++++++++++++ Python version +++++++++++++++++++++++++++
call python --version
echo +++++++++++++++++++++++++ Downlowd DPCTL +++++++++++++++++++++++++++
call git clone https://github.com/IntelPython/dpctl.git 
cd dpctl

set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI\"
echo +++++++++++++++++++++++++ Build DPCTL +++++++++++++++++++++++++++
call conda build --croot=C:/tmp conda-recipe --no-test -c "%ONEAPI_ROOT%\conda_channel"

echo +++++++++++++++++++++++++ install DPCTL +++++++++++++++++++++++++++
call conda install -y dpctl --strict-channel-priority -c local -c intel

cd ..
echo +++++++++++++++++++++++++ cleanup DPCTL sources +++++++++++++++++++++++++++
del /F/Q/S dpctl

dir /s/b "%ONEAPI_ROOT%\libDPCTLSyclInterface.so"

call conda list
