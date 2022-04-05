#!/bin/bash

PYTHON_VERSION=$1
DPLROOT=$2

export DPLROOT

CHANNELS="-c dppy/label/dev -c intel -c defaults --override-channels"
VERSIONS="--python $PYTHON_VERSION"
TEST="--no-test"

conda build \
  $TEST \
  $VERSIONS \
  $CHANNELS \
  conda-recipe
