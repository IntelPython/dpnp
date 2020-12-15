#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

echo +++++++++++++++++++++++++ Intel OneAPI Python ++++++++++++++++++++++++++++

sudo apt-get install intel-oneapi-python
sudo chmod -R a+w /opt/intel/oneapi/intelpython
sudo chmod -R a+w /opt/intel/oneapi/conda_channel
