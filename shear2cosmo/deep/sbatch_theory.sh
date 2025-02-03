#!/bin/bash
#SBATCH -p batch
#SBATCH -J theory2000restricted
#SBATCH -N 1 
#SBATCH -c 15
#SBATCH --mem=30GB
#SBATCH -t 5:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.buron@lam.fr

DIR_RESULTS="/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/LHS_comparison"

eval "$(micromamba shell hook --shell bash)"
micromamba activate SHEAR2COSMO

echo "START GLASS GENERATOR" >> ${DIR_RESULTS}/outputs_and_errors_train.txt
python compute_theory_k2pcf.py >> ${DIR_RESULTS}/outputs_and_errors_train.txt
