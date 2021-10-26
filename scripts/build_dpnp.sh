#!/bin/bash

#THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))

#cd ${THEDIR}

while getopts "t:p:" opt; do
  case $opt in
    p) py_ver="$OPTARG";;
    t) test_skip="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

run_test=""

if [ $test_skip -ge 1 ]; 
then 
  run_test="--no-test"
fi

printf "Argument py_ver is %s\n" "$py_ver"
printf "Argument run_test is %s\n" "$run_test"

echo ========================= create Conda ENV ===========================
conda create -q -y -n dpnp$py_ver conda-build --override-channels -c intel -c conda-forge
. /usr/share/miniconda/etc/profile.d/conda.sh
conda activate dpnp$py_ver
echo ========================= build DPNP and run tests ==========================
export OCL_ICD_FILENAMES=libintelocl.so
conda-build --python $py_ver --override-channels -c intel -c conda-forge -c dppy/label/dev ./conda-recipe/
