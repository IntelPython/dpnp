#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

echo
echo ========================= Set DPNP environment ===========================
echo SHELL=${SHELL}
echo PWD=${PWD}
ls -l
echo ========================= current machine kernel =========================
uname -a

echo ========================= install Intel repository =======================
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
sudo apt-get update

echo ========================= install Python 3.8 ===============================
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 10
echo ========================= Python version ===============================
python --version
python3 --version

echo ========================= install valgrind ===============================
sudo apt-get install valgrind pytest-valgrind

echo ========================= install Numpy cython pytest ===============================
pip3 install numpy cython pytest

echo ========================= install/delete libstdc++-dev ===============================
sudo apt remove -y gcc-7 g++-7 gcc-8 g++-8 gcc-10 g++-10
# oneapi beta 10 can not work with libstdc++-10-dev 
sudo apt remove -y libstdc++-10-dev
sudo apt autoremove
sudo apt install --reinstall -y gcc-9 g++-9 libstdc++-9-dev

echo ========================= install Intel OneAPI ===============================
sudo apt-get install intel-oneapi-mkl       \
                     intel-oneapi-mkl-devel \
                     intel-oneapi-dpcpp-cpp-compiler

echo ========================= setup Intel OneAPI ===============================
ls -l /opt/intel/oneapi/
. /opt/intel/oneapi/setvars.sh
g++ --version
sudo apt list --installed

echo ========================= Current clang version ===============================
valgrind --version
which valgrind
clang++ --version
which clang++
dpcpp --version
which dpcpp
