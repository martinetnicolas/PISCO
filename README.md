# PISCO

## Generating datasets

Because the environment requirements for the various tools needed
to generate training data are quite complex, we are providing a
conda environment file featuring all the necessary dependencies.

We further recommend to use the much faster and lighter micromamba instead of conda.

```bash
$ micromamba env create -f sims-environment.yml -y
OR
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba env create -f /path-to-PISCO/env_PISCO.yml
```

```bash
$ micromamba activate pisco-sims
(pisco-sims)$ cd pisco/datasets/pisco_lsst
(pisco-sims)$ tfds build
```

Will run the simulations and assemble the dataset.

ml use
```bash
ssh -L8080:localhost:8080 rbrunet@cc

alias visu_lab='ssh -L8080:localhost:8080 rbrunet@cc'

function mlflow_ui() {
  if [ -z "$1" ]; then
    echo "Usage: mlflow_ui <folder_name>"
  else
    mlflow ui --port 8080 --backend-store-uri "file:///net/GECO/pisco/PIX2SHEAR/RESULTS/$1"
  fi
```

```bash
alias node29='srun -p gpu --gres=gpu:a40:1 --mem 80GB -t 3:00:00 -w node29 --pty bash -c "zsh"'
alias cpu32='srun -c32 -p short -t 02:00:00 --mem 20GB --pty bash -c "zsh"'
```


## Glass

alias cpuGLASS='srun -c32 -p batch -t 03:00:00 --mem 80GB --pty bash -c "zsh"'