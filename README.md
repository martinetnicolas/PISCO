# PISCO

## Generating datasets

Because the environment requirements for the various tools needed
to generate training data are quite complex, we are providing a
conda environment file featuring all the necessary dependencies.

We further recommend to use the much faster and lighter micromamba instead of conda.

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba config set channel_priority flexible
micromamba env create -f /path-to-PISCO/env_PISCO.yml
```

```bash
$ micromamba env create -f sims-environment.yml -y
$ micromamba activate pisco-sims
(pisco-sims)$ cd pisco/datasets/pisco_lsst
(pisco-sims)$ tfds build
```

## Glass

alias cpuGLASS='srun -c32 -p batch -t 03:00:00 --mem 80GB --pty bash -c "zsh"'

## Pix2shear

nico : datasets & notebooks folders

database => creation + plots databases  
networks => training + inference + plots neural networks  
sextractor =>  run + plots sextractor


