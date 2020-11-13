#!/bin/bash

BUILDDOCDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))
ROOTDIR=$BUILDDOCDIR/..

cd $ROOTDIR
. 0.env.sh
python setup.py develop

cd $BUILDDOCDIR
make clean
make html
