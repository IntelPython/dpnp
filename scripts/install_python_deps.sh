#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

echo +++++++++++++++++++++++++ Python prerequisites +++++++++++++++++++++++++++

echo ========================= PIP3: install prerequisites ===============================
pip3 install --yes numpy cython pytest pytest-valgrind

echo ========================= Conda: install prerequisites ===============================
conda install -y conda-build

echo ========================= SW versions ===============================
conda list

python --version
which python

python -c "import numpy as sw; print(f\"sw.version={sw.version}\nsw.version.version={sw.version.version}\nsw.get_include={sw.get_include()}\")"
python -c "import dpctl as sw; print(f\"sw.version={sw.version}\nsw.version.version={sw.version.version}\nsw.get_include={sw.get_include()}\")"
python -c "import dpnp as sw; print(f\"sw.version={sw.version}\nsw.version.version={sw.version.version}\nsw.get_include={sw.get_include()}\")"
