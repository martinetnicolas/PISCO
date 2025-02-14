#!/bin/bash
#SBATCH --time 00:10:00
#SBATCH --mem 5G
#SBATCH --ntasks-per-core 1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --job-name S++_PARALLEL
#SBATCH --array=0-4999%125  # Run 2000 tasks in parallel, 125 at a time (max cc LAM)
#SBATCH --mail-type=END,FAIL   
#SBATCH --mail-user=romain.brunet@lam.fr
#SBATCH --error /net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR/logs/EO_%A_%a.log 
#SBATCH --output /net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR/logs/EO_%A_%a.log
##SBATCH --error /dev/null # To disable error logs
##SBATCH --output /dev/null # To disable output logs


## USER INPUTS
DATA_PARENT="1.0.4_sextractor" # Directory containing the FITS files
NB_EXTRA=5 # Number of simulations to save check images
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


# GET EACH SIMU TO RUN
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
    --config-file "$(pwd)/my_sepp.config"
    --detection-image "$DATASET_DIR/$simu_name.fits"
    --python-config-file "$WORKING_DIR/config_file.py" "$PSF_DIR" "$DATASET_DIR/$simu_name.fits"
    --output-catalog-filename "${dir["CATALOG"]}/catalog_$simu_name.fits"
)
# Images checks for the first "NB_EXTRA" simulations
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