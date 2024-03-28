#!/bin/bash

DPCTL_TARGET_VERSION=0.5.0rc2
echo ++++++++++++++++++ Build DPCTL ${DPCTL_TARGET_VERSION} +++++++++++++++++++
git clone --branch ${DPCTL_TARGET_VERSION} https://github.com/IntelPython/dpctl.git

cd dpctl || exit 1

# didn't find better way to set required version
git tag -d "$(git tag -l)"
git tag ${DPCTL_TARGET_VERSION}

# python ./setup.py develop
# python ./setup.py install

conda build conda-recipe/ --no-test -c "${ONEAPI_ROOT}"/conda_channel

# ls -lR /opt/intel/oneapi/intelpython/latest/conda-bld

conda install /opt/intel/oneapi/intelpython/latest/conda-bld/linux-64/dpctl*

echo ==================== delete DPCTL build tree==============================
cd ..
rm -rf dpctl

echo ========================= DPCTL version ==================================
python -c "import dpctl as sw; print(f\"sw.__version__={sw.__version__}\nsw.get_include={sw.get_include()}\")"

echo ========================= where DPCTL ====================================
find /opt/intel -name libDPCTLSyclInterface.so
