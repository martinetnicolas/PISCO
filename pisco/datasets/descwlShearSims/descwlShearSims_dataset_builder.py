"""descwlShearSims dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for descwlShearSims dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(descwlShearSims): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Download the source catalog
    path = dl_manager.download_and_extract('https://www.cosmo.bnl.gov/www/esheldon/data/catsim.tar.gz')

    # TODO(descwlShearSims): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    import numpy as np
    from descwl_shear_sims.sim import make_sim, get_se_dim
    from descwl_shear_sims.galaxies import FixedGalaxyCatalog, WLDeblendGalaxyCatalog
    from descwl_shear_sims.stars import StarCatalog
    from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf

    seed = 8312
    rng = np.random.RandomState(seed)

    ntrial = 100_000
    coadd_dim = 351
    buff = 50


    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }
