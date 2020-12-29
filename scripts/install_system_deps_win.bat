:: echo ========================= configure VS ================================
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
