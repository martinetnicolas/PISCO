#!/bin/bash
PIX2SHEAR_DIR=/net/GECO/pisco/PIX2SHEAR

## USER INPUT
VERSION=0.0.1
MAX_PARALLEL_JOBS=115   # max cc LAM : 125
NB_CHECK_IMAGES=100      # number of images segmentation/residual/fitting/background
ASSOC=1                 # with associated locations (0 or 1)
ENTIRE_DATASET=True

## DIRECTORIES
DATATEST_DIR=$PIX2SHEAR_DIR/DATASETS/pisco_euclid_se/$VERSION
RESULTS_DIR=$PIX2SHEAR_DIR/RESULTS_SEXTRACTOR/$VERSION++
if [ "$ASSOC" -eq 1 ]; then
    RESULTS_DIR="${RESULTS_DIR}_assoc"
fi

## CHECKS
# Sextractor version already done
if [ -d $RESULTS_DIR ]; then
    echo -e "⚠️  \033[1;31m$(basename $RESULTS_DIR)\033[0m has been done ⚠️"
    read -n1 -p "Do you want to remove it? (y/n) :" choice
    echo
    if [[ $choice =~ ^[Yy]$ ]]; then
        rm -rf $RESULTS_DIR
        mkdir -p $RESULTS_DIR
    else
        exit 1
    fi
fi
# Database version exists
if [ ! -d $DATATEST_DIR ]; then
    echo ERROR: $VERSION does not exist
    exit 1
fi

## CREATE RESULTS DIRECTORIES
create_dir() {
    mkdir -p $1 && echo $1  # return the directory path
}
IMAGES_DIR=$(create_dir $RESULTS_DIR/IMAGES)
CATALOG_DIR=$(create_dir $RESULTS_DIR/CATALOG)
LOG_DIR=$(create_dir $RESULTS_DIR/logs)

## NUMBER OF FILES COMPUTED
if [ $ENTIRE_DATASET = "True" ]; then
  NUM_FILES=$(find "$DATATEST_DIR" -name "*.fits" | wc -l)
else
  NUM_FILES=5
fi

## RUN IN PARALLEL
ARRAY_RANGE=0-$((NUM_FILES - 1)) # slurm arrays are zero-indexed
sbatch --array=${ARRAY_RANGE}%${MAX_PARALLEL_JOBS} \
       --error=$LOG_DIR/Err_%a.log \
       --output=$LOG_DIR/Out_%a.log \
       --time 00:10:00  \
       --mem 5G \
       --ntasks-per-core 1 \
       --ntasks 1 \
       --cpus-per-task 1 \
       --nodes 1 \
       -p short \
       --job-name=SE++_PISCO \
       --mail-type=FAIL \
       --mail-user=romain.brunet@lam.fr \
       parallel_se++.sh $DATATEST_DIR $RESULTS_DIR $IMAGES_DIR $CATALOG_DIR $NB_CHECK_IMAGES $ASSOC