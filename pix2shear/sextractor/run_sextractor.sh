#!/bin/bash
#SBATCH --time 02:00:00
#SBATCH --mem 20G
#SBATCH --ntasks-per-core 1
#SBATCH --nodes 1
#SBATCH --ntasks 8
#SBATCH --cpus-per-task 1
#SBATCH --job-name sbatch_SEXTRACTOR++
#SBATCH --error /net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR/logs/E_%A_%a.log 
#SBATCH --output /net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR/logs/O_%A_%a.log
#SBATCH --array=0-99%10  # Run 100 tasks in parallel, 10 at a time
#SBATCH --mail-type=END,FAIL   
#SBATCH --mail-user=romain.brunet@lam.fr


## USER INPUTS
DATA_PARENT="1.0.4_sextractor"
echo "DATA_PARENT: $DATA_PARENT"
ALL_DATASETS_DIR="/net/GECO/pisco/PIX2SHEAR/DATASETS/pisco_euclid"
PSF_DIR="$ALL_DATASETS_DIR/PSF"
RESULTS_DIR="/net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR"


# SLURM
export MKL_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"


## ACTIVATE ENV
eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO 
RED='\033[0;31m'
NC='\033[0m'
echo -e "${RED}Job $0 running on $HOSTNAME and version $(sourcextractor++ --version) ${NC}"


## DIRECTORIES & CHECKS

WORKING_DIR=$(pwd)
if [ -d "$ALL_DATASETS_DIR/$DATA_PARENT" ]; then
    DATASET_DIR="$ALL_DATASETS_DIR/$DATA_PARENT"
    NB_FILES=$(find "$DATASET_DIR" -maxdepth 1 -type f -name "*.fits" | wc -l)
    echo "Using dataset: $DATASET_DIR with $NB_FILES files"
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
echo "Found ${#file_list[@]} FITS files in $DATASET_DIR"

index=$SLURM_ARRAY_TASK_ID


# Select FITS file
fits_file=${file_list[$index]}
simu_name=$(basename "$fits_file" .fits)

echo "Processing $simu_name (Task ID: $index)"

sourcextractor++ --config-file my_sepp.config \
    --detection-image "$DATASET_DIR/$simu_name.fits"\
    --python-config-file "$WORKING_DIR/config_file.py" "$PSF_DIR" "$DATASET_DIR/$simu_name.fits"\
    --output-catalog-filename "${dir["CATALOG"]}/catalog_$simu_name.fits"\
    --check-image-segmentation "${dir["IMAGES"]}/segmentation_$simu_name.fits"\
    --check-image-background "${dir["IMAGES"]}/background_$simu_name.fits"\
    --check-image-model-fitting "${dir["IMAGES"]}/fitting.fits"\
    --check-image-residual "${dir["IMAGES"]}/residual.fits"


echo "Finished processing $simu_name"


# ## LOOP DATASET
# counter=0
# for file_path in $(ls "$DATASET_DIR"/*.fits | sort -V); do
#     simu_name=$(basename "$file_path" | sed 's/\.[^.]*$//')
#     echo "Processing $simu_name ($((counter+1))/$NB_FILES)"

# 	# algo detection et predictions
#     cmd=(
#         sourcextractor++
#         --config-file my_sepp.config
#         # --thread-count "$Nth"
#         --detection-image "$DATASET_DIR/$simu_name.fits"
#         --python-config-file "$WORKING_DIR/config_file.py" "$PSF_DIR" "$DATASET_DIR/$simu_name.fits"
#         --output-catalog-filename "${dir["CATALOG"]}/catalog_$simu_name.fits"
#         --log-file "${dir["RESULTS_PARENT"]}/log_file.log"
#         --log-level ERROR # FATAL, ERROR, WARN, INFO, DEBUG
#     )

# 	# add several stuff
#     if [[ $counter -lt $NB_PRINT ]]; then
#         cmd+=(
#             --check-image-segmentation "${dir["IMAGES"]}/segmentation_$simu_name.fits"
#             --check-image-background "${dir["IMAGES"]}/background_$simu_name.fits"
#             --check-image-model-fitting "${dir["IMAGES"]}/fitting.fits"
#             --check-image-residual "${dir["IMAGES"]}/residual.fits"
#         )
#     fi

#     # run the command
#     "${cmd[@]}"

# 	# saving vignettes & galaxies parameters
# 	python extract_catalog.py ${dir["CATALOG"]} ${dir["IMAGES"]} $simu_name $counter $NB_PRINT
# 	wait

#     # increment counter
#     ((counter++))
#     if [[ $counter -ge $STOP_IT ]]; then
#         break
#     fi
# done

# erase the .fits files
# rm -f "${dir["CATALOG"]}"/*.fits
# rm -f "${dir["CATALOG"]}"/*.fits
