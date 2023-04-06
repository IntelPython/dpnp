#!/bin/bash

BUILDDOCDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))
ROOTDIR=$BUILDDOCDIR/..

cd $ROOTDIR
python setup.py develop

cd $BUILDDOCDIR
make clean
make html
