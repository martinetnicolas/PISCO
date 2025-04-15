#!/bin/bash
#SBATCH -J PISCO_create_database_tfds

#SBATCH -p batch
#SBATCH -N 1 
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH --mem 80GB
#SBATCH --error=/dev/null 
#SBATCH --output=/dev/null
#SBATCH --mail-type=FAIL   
#SBATCH --mail-user=romain.brunet@lam.fr

eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO

ERROR_FILE="/net/GECO/pisco/PIX2SHEAR/DATASETS/EO_PISCO_create_database_tfds.log"
echo "START COMPUTING DATABASE" >> $ERROR_FILE 2>&1

python create_database_tfds.py >> $ERROR_FILE 2>&1

chmod 770 $ERROR_FILE