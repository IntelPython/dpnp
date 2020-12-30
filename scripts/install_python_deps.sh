#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

echo +++++++++++++++++++++++++ Python prerequisites +++++++++++++++++++++++++++

echo ========================= PIP3: install prerequisites ===============================
pip3 install numpy cython pytest pytest-valgrind hypothesis

echo ========================= Conda: install prerequisites ===============================
conda install -y conda-build numpy cython pytest hypothesis

echo ========================= SW versions ====================================
conda list

python --version
which python

python -c "import numpy as sw; print(f\"sw.__version__={sw.__version__}\nsw.get_include={sw.get_include()}\")"
python -c "import dpctl as sw; print(f\"sw.__version__={sw.__version__}\nsw.get_include={sw.get_include()}\")"
python -c "import dpnp as sw; print(f\"sw.__version__={sw.__version__}\nsw.get_include={sw.get_include()}\")"
