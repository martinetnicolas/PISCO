#!/bin/bash
module load mkl/2024.2

## USER INPUT
# PSF_NAME=psf_style-airy_size-41_pixelscale-0.1 
PSF_NAME=psf_style-airy_size-81_pixelscale-0.06 
PSF_SAMPLING=0.6

## DIRECTORIES
PSF_DIR=/net/GECO/pisco/PIX2SHEAR/DATASETS/psf
SEXTRACTOR_DIR=/net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR
PSF_DIR_RESULTS=$SEXTRACTOR_DIR/$PSF_NAME

## CHECKS
if [ -d $PSF_DIR_RESULTS ]; then
    echo -e "⚠️  \033[1;31m$PSF_NAME\033[0m has been done ⚠️"
    read -n1 -p "Do you want to remove it? (y/n) :" choice
    echo
    if [[ $choice =~ ^[Yy]$ ]]; then
        rm -rf $PSF_DIR_RESULTS
        mkdir -p $PSF_DIR_RESULTS
    else
        exit 1
    fi
else
    mkdir -p $PSF_DIR_RESULTS
fi
LOG_FILE=$PSF_DIR_RESULTS/ErrorOutput_psf.log
if [ -d $LOG_FILE ]; then
    rm -f $LOG_FILE
fi

echo $PSF_NAME >> $LOG_FILE 2>&1 && echo "" >> $LOG_FILE 2>&1

## SOURCEXTRACTOR RUN ON PSF FILE
sex $PSF_DIR/$PSF_NAME.fits \
    -c get_psf.sex \
    -CATALOG_TYPE FITS_LDAC \
    -CATALOG_NAME $PSF_DIR_RESULTS/$PSF_NAME.ldac \
    -FILTER Y \
    -FILTER_NAME $PSF_DIR/gauss_3.0_7x7.conv \
    -CHECKIMAGE_TYPE OBJECTS \
    -CHECKIMAGE_NAME $PSF_DIR_RESULTS/$PSF_NAME.obj.fits \
    >> $LOG_FILE 2>&1

## PSFEX RUN TO CREATE .psf FILE
psfex $PSF_DIR_RESULTS/$PSF_NAME.ldac \
      -c get_psf.psfex \
      -WRITE_XML Y \
      -XML_NAME $PSF_DIR_RESULTS/${PSF_NAME}_psfex.xml \
      -CHECKIMAGE_TYPE PROTOTYPES,SAMPLES,RESIDUALS,SNAPSHOTS \
      -CHECKIMAGE_NAME $PSF_DIR_RESULTS/psfex.prototypes.fits,$PSF_DIR_RESULTS/psfex.samples.fits,$PSF_DIR_RESULTS/psfex.residuals.fits,$PSF_DIR_RESULTS/psfex.snapshots.fits \
      -VERBOSE_TYPE FULL \
      -PSF_SAMPLING $PSF_SAMPLING \
      -PSF_PIXELSIZE 1.0 \
      -PSF_ACCURACY 0.00 \
      -PSF_RECENTER Y \
      -PSF_SIZE 41,41 \
      -PSF_DIR $PSF_DIR_RESULTS \
      -PSF_SUFFIX .psf \
      >> $LOG_FILE 2>&1