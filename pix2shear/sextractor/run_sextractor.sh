#!/bin/bash
#SBATCH --time 1-
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-core 1
#SBATCH --mem 10GB
#SBATCH -J sbatch_SEXTRACTOR++_1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --mail-type=END,FAIL   
#SBATCH --mail-user=romain.brunet@lam.fr


# SLURM
export MKL_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"
Nth=$SLURM_NTASKS
test -z $SLURM_NTASKS && Nth=30


## ACTIVATE ENV
eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO 
GREEN='\033[0;31m'
NC='\033[0m'
echo -e "${GREEN}Job $0 running on $HOSTNAME with $Nth cores and version $(sourcextractor++ --version) ${NC}"


## USER INPUTS
JSON_FILE="dataset.jsonc"
if [ -f "$JSON_FILE" ]; then
    DATA_PARENT=$(jq -r '.DATA_PARENT' "$JSON_FILE")
    NB_PRINT=$(jq -r '.NB_PRINT' "$JSON_FILE")
    STOP_IT=$(jq -r '.STOP_IT' "$JSON_FILE")
else
    echo "Error: JSON config file '$JSON_FILE' not found!"
    exit 1
fi
echo "DATA_PARENT: $DATA_PARENT"
echo "NB_PRINT: $NB_PRINT"
echo "STOP_IT: $STOP_IT"


## DIRECTORIES & CHECKS
ALL_DATASETS_DIR="/net/GECO/pisco/PIX2SHEAR/DATASETS/pisco_euclid"
PSF_DIR="$ALL_DATASETS_DIR/PSF"
RESULTS_DIR="/net/GECO/pisco/PIX2SHEAR/RESULTS_SEXTRACTOR"
WORKING_DIR=$(pwd)
if [ -d "$ALL_DATASETS_DIR/$DATA_PARENT" ]; then
    DATASET_DIR="$ALL_DATASETS_DIR/$DATA_PARENT"
    echo "Using dataset: $DATASET_DIR"
else
    echo "Error: Directory '$ALL_DATASETS_DIR/$DATA_PARENT' does not exist!"
    exit 1
fi
RESULTS_PATH="$RESULTS_DIR/$DATA_PARENT"
if [ -d "$RESULTS_PATH" ]; then
    echo "Error: Run for '$RESULTS_PATH' already done!"
    exit 1
fi
rm -f $RESULTS_DIR/log_file.log
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
dir["VIGNET"]=$(create_dir "${dir["RESULTS_PARENT"]}/VIGNET")
dir["IMAGES"]=$(create_dir "${dir["RESULTS_PARENT"]}/IMAGES")
dir["CATALOG"]=$(create_dir "${dir["RESULTS_PARENT"]}/CATALOG")


## LOOP DATASET
counter=0
for file_path in $(ls "$DATASET_DIR"/*.fits | sort -V); do
    # names
    general_name=$(basename "$file_path" | sed 's/\.[^.]*$//')
    file_name="${general_name}.fits"

	# algo detection et predictions
    cmd=(
        sourcextractor++
        --config-file my_sepp.config
        --thread-count "$Nth"
        --detection-image "$DATASET_DIR/$file_name"
        --python-config-file "$WORKING_DIR/config_file.py" "$PSF_DIR" "$DATASET_DIR/$file_name"
        --output-catalog-filename "${dir["CATALOG"]}/catalog_$file_name"
        --log-file "${dir["RESULTS_PARENT"]}/log_file.log"
        --log-level ERROR # FATAL, ERROR, WARN, INFO, DEBUG
    )

	# add several stuff
    if [[ $counter -lt $NB_PRINT ]]; then
        cmd+=(
            --check-image-segmentation "${dir["IMAGES"]}/segmentation_$file_name"
            --check-image-background "${dir["IMAGES"]}/background_$file_name"
            --check-image-model-fitting "${dir["IMAGES"]}/fitting.fits"
            --check-image-residual "${dir["IMAGES"]}/residual.fits"
        )
    fi

    # run the command
    "${cmd[@]}"

	# saving vignettes for the first 5
	if [[ $counter -lt $NB_PRINT ]]; then
		python save_vignet.py ${dir["CATALOG"]} ${dir["VIGNET"]} $general_name
	fi
	wait

	# saving galaxies parameters
	python extract_catalog.py ${dir["CATALOG"]}/catalog_$file_name ${dir["CATALOG"]}/catalog_$general_name.csv
	wait

    # increment counter
    ((counter++))
    if [[ $counter -ge $STOP_IT ]]; then
        break
    fi

done