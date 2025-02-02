#!/bin/bash
#SBATCH -p batch
#SBATCH -J cart_seed3
#SBATCH -N 1 
#SBATCH -c 10
#SBATCH --mem=50GB
#SBATCH -t 0:30:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.buron@lam.fr

DIR_RESULTS="/net/GECO/pisco/SHEAR2COSMO/DBB_OPT_COSMOSLICS/"

eval "$(micromamba shell hook --shell bash)"
micromamba activate GLASS

echo "START " >> ${DIR_RESULTS}/outputs_and_errors_spher2cart.txt
python spheric2cart.py >> ${DIR_RESULTS}/outputs_and_errors_spher2cart.txt
