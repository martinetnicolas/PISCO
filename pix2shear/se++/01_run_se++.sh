#!/bin/bash

## USER INPUT
VERSION=0.0.1
MAX_PARALLEL_JOBS=120   # max cc LAM : 125
NB_CHECK_IMAGES=100     # number of images segmentation/residual/fitting/background
ASSOC=0                 # with associated locations (0 or 1)
ENTIRE_DATASET=True
FIRST_PART_DATASET=False
STEP_NB_FILES=50000
PSF="psf_style-airy_size-41_pixelscale-0.1.fits"
SAMPLING_FACTOR=1
THRESHOLD=1.5
MIN_NB_PIXEL=6
DATATEST_DIR=/net/GECO/pisco/PIX2SHEAR/DATASETS/pisco_euclid_se/$VERSION
NB_FILES=$(find "$DATATEST_DIR" -name "*.fits" | wc -l)

# RANGE
if [ $ENTIRE_DATASET = "True" ]; then
    ARRAY_RANGE=0-$((NB_FILES - 1))
else
    if [ $FIRST_PART_DATASET = "True" ]; then
        ARRAY_RANGE=0-$((STEP_NB_FILES - 1))
    else
        ARRAY_RANGE=$STEP_NB_FILES-$((NB_FILES - 1))
    fi
fi
echo $ARRAY_RANGE


## CHECKS
# Database version exists
if [ ! -d $DATATEST_DIR ]; then
    echo ERROR: $VERSION does not exist
    exit 1
fi
# SE++ version already done
RESULTS_DIR=/net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR/$VERSION++
if [ $ASSOC -eq 1 ]; then
    RESULTS_DIR="${RESULTS_DIR}_assoc"
fi
if [ $ENTIRE_DATASET = "True" ]; then
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
fi


## CREATE RESULTS DIRECTORIES
create_dir() {
    mkdir -p $1 && echo $1
}
PSF_DIR=/net/GECO/pisco/PIX2SHEAR/DATASETS/psf_se

IMAGES_DIR=$(create_dir $RESULTS_DIR/IMAGES)
CATALOG_DIR=$(create_dir $RESULTS_DIR/CATALOG)
LOG_DIR=$(create_dir $RESULTS_DIR/logs)


## RUN IN PARALLEL
sbatch  -p short \
        --array=${ARRAY_RANGE}%${MAX_PARALLEL_JOBS} \
        --error=$LOG_DIR/Err_%a.log \
        --output=$LOG_DIR/Out_%a.log \
        --time 00:10:00  \
        --mem 5G \
        --ntasks-per-core 1 \
        --ntasks 1 \
        --cpus-per-task 1 \
        --nodes 1 \
        --job-name=SE++_PISCO \
        --mail-type=FAIL \
        --mail-user=romain.brunet@lam.fr \
        parallel_se++.sh $DATATEST_DIR $RESULTS_DIR $IMAGES_DIR $CATALOG_DIR $NB_CHECK_IMAGES $ASSOC $PSF_DIR/$PSF $SAMPLING_FACTOR $THRESHOLD $MIN_NB_PIXEL