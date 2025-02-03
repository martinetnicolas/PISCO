#!/bin/bash
#SBATCH -J sbatch_training_networks1
#SBATCH -t 1-
#SBATCH -w node29
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem 80GB
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --mail-type=END,FAIL   
#SBATCH --mail-user=romain.brunet@lam.fr 

eval "$(micromamba shell hook --shell bash)"
micromamba activate PISCO
DIR_RESULTS="/net/GECO/pisco/PIX2SHEAR/RESULTS"

echo "START TRAINING" >> ${DIR_RESULTS}/outputs_and_errors_training_networks1.txt
python -u train_network.py >> ${DIR_RESULTS}/outputs_and_errors_training_networks1.txt