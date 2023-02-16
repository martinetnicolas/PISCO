# PISCO

## Generating datasets

Because the environment requirements for the various tools needed
to generate training data are quite complex, we are providing a
conda environment file featuring all the necessary dependencies.

We further recommend to use the much faster and lighter micromamba instead of conda.

```bash
$ micromamba env create -f sims-environment.yml -y
```

```bash
$ micromamba activate pisco-sims
(pisco-sims)$ tfds build pisco
```

Will run the simulations and assemble the dataset.
