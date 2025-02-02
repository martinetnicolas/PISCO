#!/bin/bash
#SBATCH -p batch
#SBATCH -J sbatch_matter_cls
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH --mem=80GB
#SBATCH -t 1-
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=romain.brunet@lam.fr

eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO
DIR_RESULTS="/net/GECO/pisco/SHEAR2COSMO"

echo "START GENERATING" >> ${DIR_RESULTS}/outputs_and_errors_matter_cls.txt
python -u 3_generate_matter_cls.py >> ${DIR_RESULTS}/outputs_and_errors_matter_cls.txt