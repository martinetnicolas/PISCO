#!/bin/bash

## Arguments
DATATEST_DIR=$1
IMAGES_DIR=$2
CATALOG_DIR=$3
NB_CHECK_IMAGES=$4
ASSOC=$5
CONV=$6
PSF=$7

## SLURM VARIABLES (OVERLOAD)
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE

## ACTIVATE ENV
eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO 

## GET EACH SIMU TO RUN
index=$SLURM_ARRAY_TASK_ID
file_list=($(ls $DATATEST_DIR/*_simu.fits | sort -V))
fits_file=${file_list[$index]}
echo $fits_file
assoc_list=($(ls $DATATEST_DIR/*_assoc.txt | sort -V))
assoc_file=${assoc_list[$index]}
simu_name=$(basename $fits_file)
base_simu_name=${simu_name%.fits}

## COMMAND TO RUN
# Main outputs
command=(
    sex $fits_file 
    -c run_sextractor.sex 
    -FILTER Y                                                   # apply filter for detection (Y or N)?
    -FILTER_NAME $CONV                                          # name of the file containing the filter
    -CATALOG_TYPE FITS_LDAC                                     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT, ASCII_VOTABLE, FITS_1.0 or FITS_LDAC
    -CATALOG_NAME $CATALOG_DIR/$base_simu_name.cat              # name of the output catalog
    -PSF_NAME $PSF                                              # File containing the PSF model
    -VERBOSE_TYPE FULL                                          # can be QUIET, NORMAL or FULL
)
# Adding assoc
if [[ $ASSOC -eq 1 ]]; then
    echo $assoc_file
    command+=(
        -ASSOC_NAME $assoc_file 
        -ASSOC_DATA 1,2                 # columns of the data to replicate (0=all)
        -ASSOC_PARAMS 1,2               # columns of xpos,ypos[,mag]
        -ASSOCCOORD_TYPE PIXEL          # ASSOC coordinates: PIXEL or WORLD
        -ASSOC_RADIUS 10                # cross-matching radius (pixels)
        -ASSOC_TYPE NEAREST             # ASSOCiation method: FIRST, NEAREST, MEAN, MAG_MEAN, SUM, MAG_SUM, MIN or MAX
        -ASSOCSELEC_TYPE ALL            # ASSOC selection type: ALL, MATCHED or -MATCHED
    )
fi
# Images checks for the first NB_CHECK_IMAGES simulations
if [[ $index -lt $NB_CHECK_IMAGES ]]; then
    command+=(
        -CHECKIMAGE_TYPE BACKGROUND,SEGMENTATION,MODELS,-MODELS,OBJECTS,-OBJECTS \
        -CHECKIMAGE_NAME $IMAGES_DIR/back_$simu_name,$IMAGES_DIR/seg_$simu_name,$IMAGES_DIR/model_$simu_name,$IMAGES_DIR/submodel_$simu_name,$IMAGES_DIR/obj_$simu_name,$IMAGES_DIR/subobj_$simu_name \
    )
fi

## RUN
${command[@]}
echo "Job running on $HOSTNAME => Finished processing $simu_name"

# SAVE OUTPUTS
python run_sextractor_outputs.py $CATALOG_DIR $base_simu_name