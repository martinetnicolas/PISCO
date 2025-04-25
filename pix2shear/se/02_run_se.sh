#!/bin/bash
module load mkl/2024.2
PIX2SHEAR_DIR=/net/GECO/pisco/PIX2SHEAR

## USER INPUT
VERSION=2.1.1
MAX_PARALLEL_JOBS=125       # max cc LAM => 125
NB_CHECK_IMAGES=20          # number of images segmentation/residual/fitting/background
ASSOC=1                     # with associated locations (0 or 1)
CONV=$PIX2SHEAR_DIR/DATASETS/psf_se/gauss_3.0_7x7.conv
PSF_NAME=psf_style-airy_size-41_pixelscale-0.1
# PSF_NAME=psf_style-airy_size-81_pixelscale-0.06


## DIRECTORIES & FILENAMES
PSF=$PIX2SHEAR_DIR/RESULTS_SEXTRACTOR/$PSF_NAME/$PSF_NAME.psf
DATATEST_DIR=$PIX2SHEAR_DIR/DATASETS/pisco_euclid_se/$VERSION
RESULTS_DIR=$PIX2SHEAR_DIR/RESULTS_SEXTRACTOR/$VERSION


## CHECKS
if [ -d $RESULTS_DIR ]; then
    echo -e "⚠️  \033[1;31m$VERSION\033[0m has been done ⚠️"
    read -n1 -p "Do you want to remove it? (y/n) :" choice
    echo
    if [[ $choice =~ ^[Yy]$ ]]; then
        rm -rf $RESULTS_DIR
        mkdir -p $RESULTS_DIR
    else
        exit 1
    fi
fi
if [ ! -d $DATATEST_DIR ]; then
    echo ERROR: $VERSION does not exist
    exit 1
fi


## CREATE RESULTS DIRECTORIES
create_dir() {
    mkdir -p $1 && echo $1
}
IMAGES_DIR=$(create_dir "$RESULTS_DIR/IMAGES")
CATALOG_DIR=$(create_dir "$RESULTS_DIR/CATALOG")
LOG_DIR=$(create_dir "$RESULTS_DIR/logs")


## RUN IN PARALLEL
# NUM_FILES=10
NUM_FILES=$(find "$DATATEST_DIR" -name "*.fits" | wc -l)
ARRAY_RANGE=0-$((NUM_FILES - 1)) # slurm arrays are zero-indexed
sbatch --array=${ARRAY_RANGE}%${MAX_PARALLEL_JOBS} \
       --output=$LOG_DIR/Out_%a.log \
       --error=$LOG_DIR/Err_%a.log \
       --time 00:10:00  \
       --mem 5G \
       --ntasks-per-core 1 \
       --ntasks 1 \
       --cpus-per-task 1 \
       --nodes 1 \
       --job-name SE_PARALLEL \
       --mail-type=FAIL \
       --mail-user=romain.brunet@lam.fr \
       run_se_parallel.sh $DATATEST_DIR $IMAGES_DIR $CATALOG_DIR $NB_CHECK_IMAGES $ASSOC $CONV $PSF