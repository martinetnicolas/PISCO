#!/bin/bash
#SBATCH -p batch
#SBATCH -J GLASS_seed3
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH --mem=80GB
#SBATCH -t 2:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.buron@lam.fr

DIR_RESULTS="/net/GECO/pisco/SHEAR2COSMO/Leo_COSMO-SLICS/"

eval "$(micromamba shell hook --shell bash)"
micromamba activate GLASS

echo "START GLASS GENERATOR" >> ${DIR_RESULTS}/outputs_and_errors.txt
python glass_map_generator_loop.py >> ${DIR_RESULTS}/outputs_and_errors.txt
