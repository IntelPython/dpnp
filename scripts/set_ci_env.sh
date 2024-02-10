#!/bin/bash

THEDIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")

echo
echo ========================= Set DPNP environment ===========================
echo SHELL="${SHELL}"
echo PWD="${PWD}"
echo HOME="${HOME}"
ls -l
echo ========================= current machine kernel =========================
uname -a

"${THEDIR}"/install_system_deps.sh
. ./scripts/install_cmake_lin.sh

echo ========================= setup Intel OneAPI python changed to Intel OneAPI ====
. /opt/intel/oneapi/setvars.sh

"${THEDIR}"/install_python_deps.sh

echo ========================= SW versions ===============================
g++ --version
command -v g++

clang++ --version
command -v clang++

dpcpp --version
command -v dpcpp
