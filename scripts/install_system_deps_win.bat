echo =============== update setuptools required by VS procedure ===============
pip install --upgrade setuptools

echo ========================= Install VS components ==========================
curl -o webimage.exe ^
  --retry 5 --retry-delay 5 ^
  -L https://download.visualstudio.microsoft.com/download/pr/9b3476ff-6d0a-4ff8-956d-270147f21cd4/ccfb9355f4f753315455542f966025f96de734292d3908c8c3717e9685b709f0/vs_BuildTools.exe

dir

start /b /wait webimage.exe ^
  --add Microsoft.VisualStudio.Workload.VCTools ^
  --includeOptional --includeRecommended --nocache --wait --passive --quiet

del webimage.exe

:: dir "c:\Program Files (x86)\Microsoft Visual Studio\Installer"

:: start /b /wait "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"      ^
::   update --add Microsoft.VisualStudio.Workload.VCTools                           ^
::   --installpath "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise" ^
::   --includeOptional --includeRecommended --passive --norestart

:: --add Microsoft.VisualStudio.Workload.NativeDesktop
:: --add Microsoft.VisualStudio.Workload.VCTools
:: "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe" ^
::     modify ^
::     --add Microsoft.VisualStudio.Workload.NativeDesktop ^
::     --installpath "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise" ^
::     --passive --norestart

:: echo ========================= configure VS ===================================
:: dir /s/b "C:\Program Files (x86)\Microsoft Visual Studio\*.bat"
:: call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

echo ========================= download oneapi ================================
curl.exe --output webimage.exe                                                                                  ^
  --url https://registrationcenter-download.intel.com/akdlm/irc_nas/17453/w_BaseKit_p_2021.1.0.2664_offline.exe ^
  --retry 5 --retry-delay 5

start /b /wait webimage.exe -s -x -f webimage_extracted
del webimage.exe

echo ========================= install onepai =================================
:: it is expected that multy-line end-line symbol will be diffrent on MS :-)
webimage_extracted\bootstrapper.exe -s --action install      ^
  --eula=accept --continue-with-optional-error=yes           ^
  -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0

echo ========================= copy OpenCL ====================================
dir "C:\Program Files (x86)\Intel\oneAPI\intelpython\python3.7\Library"
copy "C:\Program Files (x86)\Intel\oneAPI\intelpython\python3.7\Library\OpenCL.dll" C:\Windows\System32\
