#!/bin/bash

#SBATCH --time 02:00:00
#SBATCH --mem 20G
#SBATCH --ntasks-per-core 1
#SBATCH --nodes 1
#SBATCH --ntasks 8
#SBATCH --cpus-per-task 1
#SBATCH --job-name DynamicArray
#SBATCH --array=0-9%10  # Run 100 tasks in parallel, 10 at a time
#SBATCH --mail-type=END,FAIL   
#SBATCH --mail-user=romain.brunet@lam.fr
#SBATCH --error /net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR/logs/E_%A_%a.log 
#SBATCH --output /net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR/logs/O_%A_%a.log
##SBATCH --error /dev/null
##SBATCH --output /dev/null


## USER INPUTS
DATA_PARENT="1.0.4_sextractor"
echo "DATA_PARENT: $DATA_PARENT"
ALL_DATASETS_DIR="/net/GECO/pisco/PIX2SHEAR/DATASETS/pisco_euclid"
PSF_DIR="$ALL_DATASETS_DIR/PSF"
RESULTS_DIR="/net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR"


## SLURM
export MKL_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"


## ACTIVATE ENV
eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO 
echo -e "Job $0 running on $HOSTNAME and version $(sourcextractor++ --version)"


## DIRECTORIES
WORKING_DIR=$(pwd)
if [ -d "$ALL_DATASETS_DIR/$DATA_PARENT" ]; then
    DATASET_DIR="$ALL_DATASETS_DIR/$DATA_PARENT"
    echo "Using dataset: $DATASET_DIR"
else
    echo "Error: Directory '$ALL_DATASETS_DIR/$DATA_PARENT' does not exist!"
    exit 1
fi
RESULTS_PATH="$RESULTS_DIR/$DATA_PARENT"
create_dir() {
    local dir=$1
    local full_path="$dir"
    if [ ! -d "$full_path" ]; then
        mkdir -p "$full_path"
    fi
    echo $full_path
}
declare -A dir
dir["RESULTS_PARENT"]=$(create_dir "$RESULTS_PATH")
dir["IMAGES"]=$(create_dir "${dir["RESULTS_PARENT"]}/IMAGES")
dir["CATALOG"]=$(create_dir "${dir["RESULTS_PARENT"]}/CATALOG")


# Get the list of FITS files and select one for this array job
file_list=($(ls "$DATASET_DIR"/*.fits | sort -V))
NB_FILES=${#file_list[@]}
echo "Found $NB_FILES FITS files in $DATASET_DIR"
index=$SLURM_ARRAY_TASK_ID
if [[ $index -ge $NB_FILES ]]; then
    echo "Task ID $index is out of range. Skipping."
    exit 1
fi
fits_file=${file_list[$index]}
simu_name=$(basename "$fits_file" .fits)
echo "Processing $simu_name (Task ID: $index)"


## MAIN OUTPUTS WANTED
command=(
    sourcextractor++
    --config-file my_sepp.config
    --detection-image "$DATASET_DIR/$simu_name.fits"
    --python-config-file "$WORKING_DIR/config_file.py" "$PSF_DIR" "$DATASET_DIR/$simu_name.fits"
    --output-catalog-filename "${dir["CATALOG"]}/catalog_$simu_name.fits"
)
# Extra checks for the first "NB_EXTRA" simulations
NB_EXTRA=5
if [[ $SLURM_ARRAY_TASK_ID -lt $NB_EXTRA ]]; then
    command+=(
        --check-image-segmentation "${dir["IMAGES"]}/segmentation_$simu_name.fits"\
        --check-image-background "${dir["IMAGES"]}/background_$simu_name.fits"\
        --check-image-model-fitting "${dir["IMAGES"]}/fitting.fits"\
        --check-image-residual "${dir["IMAGES"]}/residual.fits"
    )
fi
# Run the command
"${command[@]}"
echo "Finished processing $simu_name"


## SAVE VIGNETTES & GALAXIES PARAMETERS
python extract_catalog.py ${dir["CATALOG"]} ${dir["IMAGES"]} $simu_name $SLURM_ARRAY_TASK_ID $NB_EXTRA