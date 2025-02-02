#!/bin/bash
#SBATCH -p batch
#SBATCH -J camb_test
#SBATCH -N 1 
#SBATCH -c 10
#SBATCH --mem=30GB
#SBATCH -t 4:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.buron@lam.fr

DIR_RESULTS="/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/"

eval "$(micromamba shell hook --shell bash)"
micromamba activate GLASS

echo "START GLASS GENERATOR" >> ${DIR_RESULTS}/outputs_and_errors.txt
python glass_map_camb_test.py >> ${DIR_RESULTS}/outputs_and_errors.txt
