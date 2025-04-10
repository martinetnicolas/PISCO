#!/bin/bash

## Arguments
DATATEST_DIR=$1
RESULTS_DIR=$2
IMAGES_DIR=$3
CATALOG_DIR=$4
NB_CHECK_IMAGES=$5
ASSOC=$6

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

## COMMAND TO RUN
# Main outputs
command=(
    sourcextractor++
    --detection-image $fits_file
    --python-config-file $(pwd)/config.py $fits_file
    --config-file $(pwd)/config.config
    --output-catalog-filename $CATALOG_DIR/$simu_name
    --detection-threshold 2                 # detection threshold above the background
    --detection-minimum-area 3              # min. # of pixels above threshold
    --detection-image-saturation 0          # detection image saturation level (0 = no saturation)
    --detection-image-interpolation 0       # interpolate bad pixels in detection image
    --detection-image-interpolation-gap 5   # Maximum number if pixels to interpolate over
    --smoothing-box-size 3                  # background median filter size
    --model-fitting-iterations 300          # maximum number of iterations allowed for model fitting
    --model-fitting-engine levmar           # Least squares engine
    --psf-pixel-sampling 0.6                # Generate a gaussian PSF with the given pixel sampling step size
    --sampling-scale-factor 1               # Scaling factor for the rendering of models (e.g. 2 = twice the resolution)
)
# Adding assoc
if [[ $ASSOC -eq 1 ]]; then
    echo $assoc_file
    command+=(
        --output-properties PixelBoundaries,FlexibleModelFitting,Vignet,AutoPhotometry,FluxRadius,ShapeParameters,AssocMode
        --assoc-catalog $assoc_file     # Assoc catalog file
        --assoc-columns 1,2             # Assoc columns to specify
        --assoc-mode NEAREST            # Assoc mode [FIRST, NEAREST, MEAN, MAG_MEAN, SUM, MAG_SUM, MIN, MAX]
        --assoc-radius 10               # Assoc radius (Always in pixels)
        --assoc-coord-type PIXEL        # Assoc coordinates type
        --assoc-filter ALL              # Assoc catalog filter setting: ALL, MATCHED, UNMATCHED
    )
else
    command+=(
        --output-properties PixelBoundaries,FlexibleModelFitting,Vignet,AutoPhotometry,FluxRadius,ShapeParameters
    )
fi
# Images checks for the first NB_CHECK_IMAGES simulations
if [[ $index -lt $NB_CHECK_IMAGES ]]; then
    command+=(
        --check-image-segmentation $IMAGES_DIR/segmentation_$simu_name
        --check-image-background $IMAGES_DIR/background_$simu_name
        --check-image-model-fitting $IMAGES_DIR/fitting.fits
        --check-image-residual $IMAGES_DIR/residual.fits
        --check-image-variance $IMAGES_DIR/variance_$simu_name
        --check-image-psf $IMAGES_DIR/psf$simu_name
        --check-image-measurement-background $IMAGES_DIR/semeasurement-background_gmentation_$simu_name
        --check-image-measurement-variance $IMAGES_DIR/measurement-variance_$simu_name
    )
fi

## RUN
${command[@]}
echo "Job running on $HOSTNAME => Finished processing $simu_name"

## SAVE OUTPUTS
python extract_outputs.py $CATALOG_DIR $IMAGES_DIR $simu_name $index $NB_CHECK_IMAGES $ASSOC