#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

echo +++++++++++++++++++++++++ Build DPCTL 0.5.0rc2 +++++++++++++++++++++++++++
git clone --branch 0.5.0rc2 https://github.com/IntelPython/dpctl.git 

cd dpctl

# didn't find better way to set required version
git tag 5.0

# python ./setup.py develop
# python ./setup.py install

conda build conda-recipe/ --no-test

# ls -lR /opt/intel/oneapi/intelpython/latest/conda-bld

conda install /opt/intel/oneapi/intelpython/latest/conda-bld/linux-64/dpctl*

echo ========================= DPCTL version ==================================
python -c "import dpctl as sw; print(f\"sw.version={sw.version}\nsw.version.version={sw.version.version}\nsw.get_include={sw.get_include()}\")"

echo ========================= where DPCTL ===============================
find /opt/intel -name libDPCTLSyclInterface.so
echo ========================= where mkl_sycl ===============================
find /opt/intel -name libmkl_sycl.so

cd ..
