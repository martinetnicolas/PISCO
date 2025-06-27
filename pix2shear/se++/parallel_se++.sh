#!/bin/bash

## SLURM VARIABLES (OVERLOAD)
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE


## Arguments
DATATEST_DIR=$1
RESULTS_DIR=$2
IMAGES_DIR=$3
CATALOG_DIR=$4
NB_CHECK_IMAGES=$5
ASSOC=$6
PSF=$7
SAMPLING_FACTOR=$8
THRESHOLD=$9
MIN_NB_PIXEL=${10}


## ACTIVATE ENV
eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO


## GET EACH SIMU NUMBER TO RUN
index=$SLURM_ARRAY_TASK_ID
echo $index
index_file=$(printf '%06d' $index)
fits_file="${DATATEST_DIR}/${index_file}_simu.fits"
assoc_file="${DATATEST_DIR}/${index_file}_simu_assoc.txt"
simu_name=$(basename $fits_file)


## RUN
command=(
    sourcextractor++
    --detection-image $fits_file
    --python-config-file $(pwd)/config.py $fits_file $PSF
    --config-file $(pwd)/config.config
    --output-catalog-filename $CATALOG_DIR/$simu_name
    --detection-threshold $THRESHOLD                # detection threshold above the background
    --detection-minimum-area $MIN_NB_PIXEL          # min. # of pixels above threshold
    --sampling-scale-factor $SAMPLING_FACTOR        # Scaling factor for the rendering of models (e.g. 2 = twice the resolution)
    --model-fitting-iterations 300                  # maximum number of iterations allowed for model fitting
    --partition-threshold-count 32
    --partition-minimum-contrast 0.001
    --partition-minimum-area 2
)
# Adding assoc
if [[ $ASSOC -eq 1 ]]; then
    echo "with $assoc_file"
    command+=(
        --output-properties PixelBoundaries,FlexibleModelFitting,Vignet,AutoPhotometry,FluxRadius,SourceFlags,AssocMode
        --assoc-catalog $assoc_file     # Assoc catalog file
        --assoc-columns 1,2             # Assoc columns to specify
        --assoc-mode NEAREST            # Assoc mode [FIRST, NEAREST, MEAN, MAG_MEAN, SUM, MAG_SUM, MIN, MAX]
        --assoc-radius 5                # Assoc radius (Always in pixels)
        --assoc-coord-type PIXEL        # Assoc coordinates type
        --assoc-filter ALL              # Assoc catalog filter setting: ALL, MATCHED, UNMATCHED
    )
else
    command+=(
        --output-properties PixelBoundaries,FlexibleModelFitting,Vignet,AutoPhotometry,FluxRadius,SourceFlags
        --check-image-residual $IMAGES_DIR/residual.fits
    )
fi
# Images checks for the first NB_CHECK_IMAGES simulations
if [[ $index -lt $NB_CHECK_IMAGES ]]; then
    command+=(
        --check-image-segmentation $IMAGES_DIR/segmentation_$simu_name
        --check-image-background $IMAGES_DIR/background_$simu_name
        --check-image-model-fitting $IMAGES_DIR/fitting.fits
        --check-image-variance $IMAGES_DIR/variance_$simu_name
        --check-image-psf $IMAGES_DIR/psf$simu_name
        --check-image-measurement-background $IMAGES_DIR/measurement-background_$simu_name
        --check-image-measurement-variance $IMAGES_DIR/measurement-variance_$simu_name
    )
fi


## RUN
${command[@]}
echo "Job running on $HOSTNAME => Finished processing $simu_name"


## SAVE OUTPUTS
python extract_outputs.py $CATALOG_DIR $IMAGES_DIR $simu_name $index $NB_CHECK_IMAGES $ASSOC