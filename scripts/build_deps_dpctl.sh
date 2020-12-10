#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

echo +++++++++++++++++++++++++ Build DPCTL +++++++++++++++++++++++++++
git clone https://github.com/IntelPython/dpctl.git

cd dpctl

# didn't find better way to set required version
git tag 5.0

python ./setup.py develop
python ./setup.py install

conda-build conda-recipe/ --no-test

cd ..
