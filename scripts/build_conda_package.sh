#!/bin/bash

PYTHON_VERSION=$1
DPLROOT=$2

export DPLROOT

VERSIONS="--python $PYTHON_VERSION"
TEST="--no-test"

conda build \
  $TEST \
  $VERSIONS \
  $CHANNELS \
  conda-recipe
