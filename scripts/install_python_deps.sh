#!/bin/bash

echo +++++++++++++++++++++++++ Python prerequisites +++++++++++++++++++++++++++++++++

echo ========================= Conda: install prerequisites =========================
# explicitly install mkl blas instead of openblas
# because numpy is installed with openblas for Python 3.9 by default
conda install -y conda-build numpy=1.20.1 blas=*=mkl cython pytest hypothesis

echo ========================= Conda: remove mkl ====================================
conda remove mkl --force -y || true

echo ========================= PIP3: install prerequisites ==========================
pip3 install pytest-valgrind

echo ========================= SW versions ==========================================
conda list

python --version
command -v python

python -c "import numpy as sw; print(f\"sw.__version__={sw.__version__}\nsw.get_include={sw.get_include()}\")"
python -c "import dpctl as sw; print(f\"sw.__version__={sw.__version__}\nsw.get_include={sw.get_include()}\")"
python -c "import dpnp as sw; print(f\"sw.__version__={sw.__version__}\nsw.get_include={sw.get_include()}\")"
