#!/bin/bash
#SBATCH -J sbatch_create_database1
#SBATCH -p batch
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH --mem 30GB
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --mail-type=END,FAIL   
#SBATCH --mail-user=romain.brunet@lam.fr

eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO
DIR_RESULTS="/net/GECO/pisco/PIX2SHEAR/DATASETS"

echo "START COMPUTING DATABASE" >> ${DIR_RESULTS}/outputs_and_errors_create_database1.txt
python -u create_database.py >> ${DIR_RESULTS}/outputs_and_errors_create_database1.txt