#!/bin/bash

BUILDDOCDIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")
ROOTDIR=$BUILDDOCDIR/..

cd "$ROOTDIR" || exit 1
python setup.py develop

cd "$BUILDDOCDIR" || exit 2
make clean
make html
