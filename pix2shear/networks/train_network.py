import resource
from functools import partial
from itertools import product
from time import time

import orbax.checkpoint
import tensorflow_datasets
from architecture import (
    LIST_EVAL_FUNCTION,
    LIST_UPDATE_FUNCTION,
    create_train_state,
    preprocessing_ellips,
    preprocessing_shear,
)
from flax.training import early_stopping, orbax_utils
from jax import default_backend, numpy
from pandas import read_csv
from tqdm import tqdm
from utils import (
    architecture_params,
    erase_test,
    general_params,
    get_name,
    network_params,
    slurm_mode,
)
from utils_plot import loss_optimiser

resource.setrlimit(resource.RLIMIT_NOFILE, (50000, 50000))
print(f"JAX => {default_backend()}")
BAR = slurm_mode()


# Datasets folder
FOLDER_DB = general_params()
FOLDER_DATASETS = FOLDER_DB.parent
FOLDER_RESULTS = FOLDER_DATASETS.parent / "RESULTS"


# User inputs
styles, batches, STEPS, versions, predicts = network_params()
kernel_sizes_and_pads, pre_layers, layers, nb_cpts, nb_feats = architecture_params()


# Loop over dataset
for (
    VERSION,
    BATCH,
    STYLE,
    KERNEL_SIZE_AND_PAD,
    PRE_LAYER,
    LAYER,
    NB_CPT,
    NB_FEAT,
    PREDICT,
) in product(
    versions,
    batches,
    styles,
    kernel_sizes_and_pads,
    pre_layers,
    layers,
    nb_cpts,
    nb_feats,
    predicts,
):  # iteration for all cases wanted
    model_config = {}
    model_config["style"] = STYLE
    model_config["batch"] = BATCH
    model_config["steps"] = STEPS
    KERNEL_SIZE, PAD = KERNEL_SIZE_AND_PAD[0], KERNEL_SIZE_AND_PAD[1]
    model_config["kernel_size"] = KERNEL_SIZE
    model_config["pad"] = PAD
    model_config["pre_layer"] = PRE_LAYER
    model_config["layer"] = LAYER
    model_config["nb_cpt"] = NB_CPT
    model_config["nb_feat"] = NB_FEAT
    print(CASE := get_name(model_config, VERSION, PREDICT))

    # Create or check if case already done
    FOLDER_CASE = FOLDER_RESULTS / CASE
    erase_test(FOLDER_CASE)

    # train state
    schedule, state = create_train_state(model_config=model_config)

    # stats of the dataset
    stats = read_csv(FOLDER_DB / VERSION / "statistics.csv")

    # normalisation of inputs/outputs & create processing function for dataloader
    std_images = numpy.float32(stats["std"].iloc[0])
    if PREDICT == "shear":
        min = numpy.float32(stats["min_shears"].iloc[0])
        max = numpy.float32(stats["max_shears"].iloc[0])
        PROCESS_FUNCTION = partial(
            preprocessing_shear, std_images=std_images, old_min=min, old_max=max
        )
    elif PREDICT == "ellips":
        min = numpy.float32(stats["min_ellips"].iloc[0])
        max = numpy.float32(stats["max_ellips"].iloc[0])
        PROCESS_FUNCTION = partial(
            preprocessing_ellips, std_images=std_images, old_min=min, old_max=max
        )

    # update & eval functions wrt if univariate/multivariate/scalar outputs and the same style
    UPDATE_FUNCTION = LIST_UPDATE_FUNCTION[STYLE]
    EVAL_FUNCTION = LIST_EVAL_FUNCTION[STYLE]

    # training dataloader
    train_loader = (
        tensorflow_datasets.load(
            f"pisco_euclid:{VERSION}",
            split="train[:80%]",  # first 80% taken
            data_dir=FOLDER_DATASETS,
            shuffle_files=True,
        )
        .repeat()  # to not stop after the network has seen the entire dataset
        .shuffle(40000, reshuffle_each_iteration=True)
        .map(PROCESS_FUNCTION)  # wanted normalisation
        .batch(BATCH, drop_remainder=True)  # size of batch and drop if the last one is not complete
        .prefetch(-1)  # pre-load next batch while running
        .as_numpy_iterator()
    )

    # validation dataloader
    val_loader = (
        tensorflow_datasets.load(
            f"pisco_euclid:{VERSION}",
            split="train[80%:90%]",  # from 80% to 90% taken
            data_dir=FOLDER_DATASETS,
            shuffle_files=False,  # no shuffle
        )
        .repeat()
        .map(PROCESS_FUNCTION)
        .batch(BATCH, drop_remainder=True)
        .prefetch(-1)
        .as_numpy_iterator()
    )

    list_train_loss, list_val_loss, list_steps = [], [], []
    early_stop = early_stopping.EarlyStopping(min_delta=1e-10, patience=100)
    start_time = time()
    for step in tqdm(range(STEPS), disable=BAR):
        batch = next(train_loader)  # take a batch
        state, loss = UPDATE_FUNCTION(
            state, x=numpy.nan_to_num(batch[0]), y=numpy.array(batch[1])
        )  # batch[0] is the image and batch[1] is the shear or ellips

        if step % 100 == 0:
            list_train_loss.append(loss)
            list_steps.append(schedule(step))
            batch = next(val_loader)
            val_loss = EVAL_FUNCTION(state, x=numpy.nan_to_num(batch[0]), y=numpy.array(batch[1]))
            list_val_loss.append(val_loss)

            early_stop = early_stop.update(metric=val_loss)
            if early_stop.should_stop:
                print(f"EARLY STOPPING at steps {step + 1}")
                break
            if early_stop.has_improved:
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                ckpt = {"model": state, "config": {"step": step}}
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(FOLDER_CASE, ckpt, save_args=save_args, force=True)

    total = time() - start_time
    fmt_time = f"{int(total // 3600):02}:{int((total % 3600) // 60):02}"
    name = FOLDER_CASE / "training_data.npz"
    numpy.savez(
        name, train_loss=list_train_loss, val_loss=list_val_loss, steps=list_steps, time=fmt_time
    )

    # pre print of losses
    data = numpy.load(name)
    loss_optimiser(
        data["train_loss"], data["val_loss"], FOLDER_CASE, steps=data["steps"], temps=data["time"]
    )

print("FINITO")
