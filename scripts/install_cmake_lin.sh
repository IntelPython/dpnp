#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

echo ========================= install cmake ==================================
curl --output cmake_webimage.tar.gz \
  --url https://cmake.org/files/v3.19/cmake-3.19.2-Linux-x86_64.tar.gz \
  --retry 5 --retry-delay 5

tar -xzf cmake_webimage.tar.gz
rm -f cmake_webimage.tar.gz

export PATH=`pwd`/cmake-3.19.2-Linux-x86_64/bin:$PATH

which cmake
cmake --version
