#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
PYTHONPATH="$PWD":$PYTHONPATH
# check the enviroment info
DATA_ROOT="$(dirname "$PWD")/dataset"
DATA_DIR="${DATA_ROOT}/cityscapes"
SAVE_DIR="${DATA_ROOT}/seg_result/cityscapes/"

echo $DATA_ROOT
echo $DATA_DIR
echo $SAVE_DIR
echo $PYTHONPATH


