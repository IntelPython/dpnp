#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

echo +++++++++++++++++++++++++ System prerequisites +++++++++++++++++++++++++++

echo ========================= install Intel repository =======================
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
sudo apt-get update

echo ========================= make same libstdc++ ===========================
echo ========================= remove GCC ===========================
sudo apt remove -y gcc g++ gcc-6 g++-6 gcc-7 g++-7 gcc-8 g++-8 gcc-10 g++-10 gcc-11 g++-11 libstdc++6 libstdc++-7-dev libstdc++-8-dev libstdc++-10-dev libstdc++-11-dev
echo ========================= auto-remove GCC ===========================
sudo apt autoremove
echo ========================= install GCC ===========================
sudo apt install --reinstall -y gcc-9 g++-9 libstdc++-9-dev

echo ========================= install Intel OneAPI ===========================
sudo apt-get install intel-oneapi-mkl                \
                     intel-oneapi-mkl-devel          \
                     intel-oneapi-dpcpp-cpp-compiler \
                     intel-oneapi-python

echo ========================= list /opt/intel/oneapi/ ========================
ls -l /opt/intel/oneapi/

echo ========================= Change python3 to python executanble name ======
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 10
echo ========================= Python version =================================
python --version
python3 --version

echo ========================= install extra packages ==========================
sudo apt-get install cmake valgrind libgtest-dev

#echo ========================= install/delete libstdc++-dev ===================
#sudo apt remove -y gcc-7 g++-7 gcc-8 g++-8 gcc-10 g++-10
# oneapi beta 10 can not work with libstdc++-10-dev 
#sudo apt remove -y libstdc++-10-dev
#sudo apt autoremove
#sudo apt install --reinstall -y gcc-9 g++-9 libstdc++-9-dev


# echo ========================= SW versions ====================================
# sudo apt list --installed

# gcc --version
# which gcc

# clang++ --version
# which clang++

# dpcpp --version
# which dpcpp

# valgrind --version
# which valgrind
