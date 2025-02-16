#!/bin/bash
#SBATCH -J create_database_PISCO
#SBATCH -p batch
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH --mem 30GB
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --error /net/GECO/pisco/PIX2SHEAR/DATASETS/EO_PISCO_create_database_euclid.log 
#SBATCH --output /net/GECO/pisco/PIX2SHEAR/DATASETS/EO_PISCO_create_database_euclid.log
#SBATCH --mail-type=END,FAIL   
#SBATCH --mail-user=romain.brunet@lam.fr

eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO

DIR_RESULTS="/net/GECO/pisco/PIX2SHEAR/DATASETS"
ERROR_FILE="EO_PISCO_create_database_euclid"
echo "START COMPUTING DATABASE" >> ${DIR_RESULTS}/${ERROR_FILE}.log

python -u create_database.py >> ${DIR_RESULTS}/${ERROR_FILE}.log
chmod 770 ${DIR_RESULTS}/${ERROR_FILE}.log