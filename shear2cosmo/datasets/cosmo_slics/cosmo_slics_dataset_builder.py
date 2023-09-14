"""cosmo_slics dataset."""

import tensorflow_datasets as tfds

def _get_Map(seed):
    
    
    
    
    
    
    return seed, {'Map':full_image.array, 
        'S8':S8.astype(np.float32)}

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cosmo_slics dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(cosmo_slics): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            #'e12': tfds.features.Tensor(shape=(512, 512, 2), dtype=np.float32),
            'Map': tfds.features.Tensor(shape=(512, 512), dtype=np.float32),
            #'CP': tfds.features.Tensor(shape=(4,), dtype=np.float32),
            'S8': tfds.features.Tensor(shape=(1,), dtype=np.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('Map', 'S8'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(cosmo_slics): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(cosmo_slics): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        #'train': self._generate_examples(path / 'train_imgs'),
        'train': self._generate_examples(),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    from multiprocessing import Pool, cpu_count
    # get the number of logical cpu cores
    n_cores = cpu_count()
    pool = Pool(processes=n_cores)
    ntrial = 100

    # Generate all images at once 
    results = pool.map(_get_Map, np.arange(ntrial))

    # Done, closing pool
    pool.close()

    for trial, result in results:
      yield int(trial), result
