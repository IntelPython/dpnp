echo ========================= install cmake ==================================
:: curl.exe --output cmake_webimage.msi ^
::  --url https://cmake.org/files/v3.19/cmake-3.19.2-win64-x64.msi --retry 5 --retry-delay 5
:: msiexec /i cmake_webimage.msi /quiet /qn /norestart /log install.log
:: dir "C:\Program Files\"
:: dir "C:\Program Files\CMake"
:: dir "C:\Program Files\CMake\bin"
:: set PATH="C:\Program Files\CMake\bin";%PATH%

call curl.exe --output cmake_webimage.zip                                    ^
              --url https://cmake.org/files/v3.19/cmake-3.19.2-win64-x64.zip ^
              --retry 5 --retry-delay 5

call  tar -xf cmake_webimage.zip
del cmake_webimage.zip
set PATH=%CD%\cmake-3.19.2-win64-x64\bin;%PATH%

call cmake --version
