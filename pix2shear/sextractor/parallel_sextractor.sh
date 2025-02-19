#!/bin/bash
#SBATCH --time 00:05:00
#SBATCH --mem 5G
#SBATCH --ntasks-per-core 1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --job-name S++_PARALLEL
#SBATCH --mail-type=FAIL   
#SBATCH --mail-user=romain.brunet@lam.fr

## SLURM
# Arguments
DATATEST_DIR=$1
PSF_DIR=$2
RESULTS_DIR=$3
IMAGES_DIR=$4
CATALOG_DIR=$5
NB_CHECK_IMAGES=$6
# Environment variables to avoid overload
export MKL_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"

## ACTIVATE ENV
eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO 

## GET EACH SIMU TO RUN
index=$SLURM_ARRAY_TASK_ID
file_list=($(ls "$DATATEST_DIR"/*.fits | sort -V))
fits_file=${file_list[$index]}
simu_name=$(basename "$fits_file")

## COMMAND TO RUN
# Main outputs
command=(
    sourcextractor++
    --detection-image $fits_file
    --python-config-file $(pwd)/config.py $PSF_DIR $fits_file
    --output-catalog-filename "$CATALOG_DIR/catalog_$simu_name"
    --output-properties PixelBoundaries,FlexibleModelFitting,ShapeParameters,Vignet
    --background-cell-size 176   # background mesh cell size to determine a value
    --smoothing-box-size 3  # background median filter size
    --detection-threshold 2   # detection threshold above the background
    --detection-image-saturation 0  # detection image saturation level (0 = no saturation)
    --detection-image-interpolation 0   # interpolate bad pixels in detection image
    --detection-minimum-area 2   # min. # of pixels above threshold
    --use-attractors-partition 0    # enables the use of attractors for partitioning
    --thread-count 0    # Number of worker threads (-1=automatic, 0=disable all multithreading)
    --model-fitting-iterations 300  # maximum number of iterations allowed for model fitting
    --output-catalog-format FITS    # the format of the output catalog, one of ASCII or FITS
    --vignet-default-pixval 0   # Default pixel value for the vignet data
    --vignet-size 71,71 # X and Y size of the vignet
    --sampling-scale-factor 2   # Scaling factor for the rendering of models (e.g. 2 = twice the resolution)
    --log-level WARN    # FATAL, ERROR, WARN, INFO (default), DEBUG
)
# Images checks for the first NB_CHECK_IMAGES simulations
if [[ $index -lt $NB_CHECK_IMAGES ]]; then
    command+=(
        --check-image-segmentation "$IMAGES_DIR/segmentation_$simu_name"\
        --check-image-background "$IMAGES_DIR/background_$simu_name"\
        --check-image-model-fitting "$IMAGES_DIR/fitting.fits"\
        --check-image-residual "$IMAGES_DIR/residual.fits"
    )
fi
# Run the command
"${command[@]}"
echo -e "Job running on $HOSTNAME => Finished processing $simu_name"

## SAVE VIGNETTES & GALAXIES PARAMETERS
python extract_outputs.py $CATALOG_DIR $IMAGES_DIR $simu_name $index $NB_CHECK_IMAGES