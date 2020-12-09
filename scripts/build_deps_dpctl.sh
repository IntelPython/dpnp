#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

echo +++++++++++++++++++++++++ Build DPCTL +++++++++++++++++++++++++++
git clone https://github.com/IntelPython/dpctl.git

cd dpctl

python ./setup.py develop
python ./setup.py install

cd ..
