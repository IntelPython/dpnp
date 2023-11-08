:: Keep commented lines for sometime to illustrate possible variants of VS installation

:: - powershell: |
::     Start-Process -Wait -FilePath  "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe" -ArgumentList "modify --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.Component.MSBuild --add Microsoft.VisualStudio.Component.Windows10SDK --add Microsoft.VisualStudio.Component.VC.CoreBuildTools --passive --norestart --installpath ""C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"""

:: echo =============== update setuptools required by VS procedure ===============
:: pip install --upgrade setuptools

:: echo =============== VS config ===============
:: "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe" ^
::   export ^
::   --installPath "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise" ^
::   --config "%CD%\vs_config.txt" ^
::   --passive
:: type "%CD%\vs_config.txt"
:: del "%CD%\vs_config.txt"

:: echo ========================= Install VS components ==========================
:: curl -o webimage.exe ^
::   --retry 5 --retry-delay 5 ^
::   -L https://download.visualstudio.microsoft.com/download/pr/9b3476ff-6d0a-4ff8-956d-270147f21cd4/ccfb9355f4f753315455542f966025f96de734292d3908c8c3717e9685b709f0/vs_BuildTools.exe

:: start /b /wait webimage.exe ^
::   --add Microsoft.VisualStudio.Component.Roslyn.Compiler ^
::   --add Microsoft.Component.MSBuild ^
::   --add Microsoft.VisualStudio.Component.CoreBuildTools ^
::   --add Microsoft.VisualStudio.Workload.MSBuildTools ^
::   --add Microsoft.VisualStudio.Component.Windows10SDK ^
::   --add Microsoft.VisualStudio.Component.VC.CoreBuildTools ^
::   --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
::   --add Microsoft.VisualStudio.Component.VC.Redist.14.Latest ^
::   --add Microsoft.VisualStudio.Component.Windows10SDK.18362 ^
::   --add Microsoft.VisualStudio.Component.VC.CMake.Project ^
::   --add Microsoft.VisualStudio.Component.TestTools.BuildTools ^
::   --add Microsoft.VisualStudio.Component.VC.ATL ^
::   --add Microsoft.VisualStudio.Component.VC.ATLMFC ^
::   --add Microsoft.Net.Component.4.8.SDK ^
::   --add Microsoft.Net.Component.4.6.1.TargetingPack ^
::   --add Microsoft.VisualStudio.Component.VC.CLI.Support ^
::   --add Microsoft.VisualStudio.Component.VC.ASAN ^
::   --add Microsoft.VisualStudio.Component.VC.Modules.x86.x64 ^
::   --add Microsoft.VisualStudio.Component.TextTemplating ^
::   --add Microsoft.VisualStudio.Component.VC.CoreIde ^
::   --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core ^
::   --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset ^
::   --add Microsoft.VisualStudio.Component.VC.Llvm.Clang ^
::   --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Llvm.Clang ^
::   --add Microsoft.VisualStudio.Component.Windows10SDK.17763 ^
::   --add Microsoft.VisualStudio.Component.Windows10SDK.17134 ^
::   --add Microsoft.VisualStudio.Component.Windows10SDK.16299 ^
::   --add Microsoft.VisualStudio.Component.VC.v141.x86.x64 ^
::   --add Microsoft.Component.VC.Runtime.UCRTSDK ^
::   --add Microsoft.VisualStudio.Component.VC.140 ^
::   --add Microsoft.VisualStudio.Workload.VCTools ^
::   --includeOptional --includeRecommended --nocache --wait --passive --quiet ^
::   --installpath "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"

:: del webimage.exe

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
set "ONEAPI_WEB_URL=https://registrationcenter-download.intel.com/akdlm/irc_nas/17453/w_BaseKit_p_2021.1.0.2664_offline.exe"
call curl.exe --output webimage.exe  ^
              --url %ONEAPI_WEB_URL% ^
              --retry 5              ^
              --retry-delay 5

start /b /wait webimage.exe -s -x -f webimage_extracted
del webimage.exe

echo ========================= install onepai =================================
:: it is expected that multy-line end-line symbol will be different on MS :-)
call webimage_extracted\bootstrapper.exe -s --action install                   ^
                                            --eula=accept                      ^
                                            --continue-with-optional-error=yes ^
                                            -p=NEED_VS2017_INTEGRATION=0       ^
                                            -p=NEED_VS2019_INTEGRATION=0

echo ========================= copy OpenCL ====================================
dir "C:\Program Files (x86)\Intel\oneAPI\intelpython\python3.7\Library"
copy /Y "C:\Program Files (x86)\Intel\oneAPI\intelpython\python3.7\Library\OpenCL.dll" C:\Windows\System32\
echo ========================= end install_system_deps_win.bat ====================================
