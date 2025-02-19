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
    --config-file "$(pwd)/config.config"
    --detection-image "$fits_file"
    --python-config-file "$(pwd)/config.py" "$PSF_DIR" "$fits_file"
    --output-catalog-filename "$CATALOG_DIR/catalog_$simu_name"
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