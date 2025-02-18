#!/bin/bash

## USER INPUT
VERSION="1.1.0_sextractor"
MAX_PARALLEL_JOBS=125   # max cc LAM : 125
NB_CHECK_IMAGES=5      # number of images seg/fit/residual

## DIRECTORIES
PIX2SHEAR_DIR="/net/GECO/pisco/PIX2SHEAR"
PISCO_DATASETS_DIR="$PIX2SHEAR_DIR/DATASETS/pisco_euclid"
DATATEST_DIR="$PISCO_DATASETS_DIR/$VERSION"
PSF_DIR="$PISCO_DATASETS_DIR/PSF"
RESULTS_DIR="$PIX2SHEAR_DIR/RESULTS_SEXTRACTOR/$VERSION"
LOGS_DIR="$PIX2SHEAR_DIR/RESULTS_SEXTRACTOR/logs"

## CHECKS
# Sextractor version already done
if [ -d "$RESULTS_DIR" ]; then
    echo "ERROR: '$VERSION' already done"
    exit 1
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
    exit 1
fi

## CREATE RESULTS DIRECTORIES
create_dir() {
    mkdir -p "$1" && echo "$1"  # return the directory path
}
RESULTS_DIR=$(create_dir "$RESULTS_DIR")
IMAGES_DIR=$(create_dir "$RESULTS_DIR/IMAGES")
CATALOG_DIR=$(create_dir "$RESULTS_DIR/CATALOG")

## RUN IN PARALLEL
# NUM_FILES=$(find "$DATATEST_DIR" -name "*.fits" | wc -l)
NUM_FILES=125
ARRAY_RANGE="0-$((NUM_FILES - 1))" # slurm arrays are zero-indexed
echo "Array range: $ARRAY_RANGE and $NUM_FILES files"
sbatch --array=${ARRAY_RANGE}%${MAX_PARALLEL_JOBS} \
       --error=${LOGS_DIR}/EO_%A_%a.log \
       --output=${LOGS_DIR}/EO_%A_%a.log \
       parallel_sextractor.sh $DATATEST_DIR $PSF_DIR $RESULTS_DIR $IMAGES_DIR $CATALOG_DIR $NB_CHECK_IMAGES