#!/bin/bash

## USER INPUT
VERSION="1.1.0_sextractor"
MAX_PARALLEL_JOBS=125   # max cc LAM : 125
NB_CHECK_IMAGES=10      # number of images segmentation/residual/fitting/background

## DIRECTORIES
PIX2SHEAR_DIR="/net/GECO/pisco/PIX2SHEAR"
PISCO_DATASETS_DIR="$PIX2SHEAR_DIR/DATASETS/pisco_euclid"
DATATEST_DIR="$PISCO_DATASETS_DIR/$VERSION"
PSF_DIR="$PISCO_DATASETS_DIR/PSF"
RESULTS_DIR="$PIX2SHEAR_DIR/RESULTS_SEXTRACTOR/$VERSION"
LOGS_DIR="$RESULTS_DIR/logs"

## CHECKS
# Sextractor version already done
if [ -d "$RESULTS_DIR" ]; then
    echo -e "⚠️  \033[1;31m$VERSION\033[0m has been done ⚠️"
    read -n1 -p "Do you want to remove it? (y/n): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        rm -rf $RESULTS_DIR
        mkdir -p $RESULTS_DIR
    else
        exit 1
    fi
fi
# Database version exists
if [ ! -d "$DATATEST_DIR" ]; then
    echo "ERROR: '$VERSION' does not exist"
    exit 1
fi
# Logs directory exists
if [[ -d "$LOGS_DIR" ]]; then
    rm -f "$LOGS_DIR"/*.log
    echo "Successfully deleted all log files"
else
    echo "ERROR: Logs directory does not exist: $LOGS_DIR"
fi

## CREATE RESULTS DIRECTORIES
create_dir() {
    mkdir -p "$1" && echo "$1"  # return the directory path
}
IMAGES_DIR=$(create_dir "$RESULTS_DIR/IMAGES")
CATALOG_DIR=$(create_dir "$RESULTS_DIR/CATALOG")
LOGS_DIR=$(create_dir "$RESULTS_DIR/log")

## RUN IN PARALLEL
NUM_FILES=$(find "$DATATEST_DIR" -name "*.fits" | wc -l)
ARRAY_RANGE="0-$((NUM_FILES - 1))" # slurm arrays are zero-indexed
echo "Array range: $ARRAY_RANGE and $NUM_FILES files"
sbatch --array=${ARRAY_RANGE}%${MAX_PARALLEL_JOBS} \
       --error=${LOGS_DIR}/EO_%a.log \
       --output=${LOGS_DIR}/EO_%a.log \
       parallel_sextractor.sh $DATATEST_DIR $PSF_DIR $RESULTS_DIR $IMAGES_DIR $CATALOG_DIR $NB_CHECK_IMAGES