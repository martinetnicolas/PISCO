#!/bin/bash
#SBATCH -p batch
#SBATCH -J batch12
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH --mem=80GB
#SBATCH -t 16:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.buron@lam.fr

DIR_RESULTS="/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/"

eval "$(micromamba shell hook --shell bash)"
micromamba activate GLASS

echo "START GLASS GENERATOR" >> ${DIR_RESULTS}/outputs_and_errors_train.txt
jupyter TrainCosmoSlics.ipynb >> ${DIR_RESULTS}/outputs_and_errors_train.txt
