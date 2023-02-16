"""pisco_lsst dataset."""
import numpy as np
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for pisco_lsst dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(pisco_lsst): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Tensor(shape=(224, 224, 3), dtype=np.float32),
            'g': tfds.features.Tensor(shape=(2,), dtype=np.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'g'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(pisco_lsst): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://www.cosmo.bnl.gov/www/esheldon/data/catsim.tar.gz')

    # TODO(pisco_lsst): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    import os
    from multiprocessing import Pool, cpu_count
    os.environ['CATSIM_DIR'] = str(path)+'/catsim'
    from descwl_shear_sims.sim import make_sim, get_se_dim
    from descwl_shear_sims.galaxies import FixedGalaxyCatalog, WLDeblendGalaxyCatalog
    from descwl_shear_sims.stars import StarCatalog
    from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf

    seed = 9137
    # get the number of logical cpu cores
    n_cores = cpu_count()
    pool = Pool(processes=n_cores)
    ntrial = 10_000
    coadd_dim = 214
    buff = 0

    def _get_trial(trial):
      rng = np.random.RandomState(trial+seed)
      # Randomly generate shear data
      g1 = rng.uniform(low=-0.1, high=0.1)
      g2 = rng.uniform(low=-0.1, high=0.1)
      
      galaxy_catalog = WLDeblendGalaxyCatalog(
          rng=rng,
          coadd_dim=coadd_dim,
          buff=buff,
      )

      star_catalog = StarCatalog(
          rng=rng,
          coadd_dim=coadd_dim,
          buff=buff,
      )

      # make a fixed psf, this assumes for instance a metacalibrated image
      psf = make_fixed_psf(psf_type='gauss')

      # generate some simulation data, with a particular shear,
      # and dithering, rotation, cosmic rays, bad columns, star bleeds
      # turned on.  By sending the star catalog we generate stars and
      # some can be saturated and bleed

      sim_data = make_sim(
          rng=rng,
          galaxy_catalog=galaxy_catalog,
          star_catalog=star_catalog,
          coadd_dim=coadd_dim,
          g1=g1,
          g2=g2,
          psf=psf,
          psf_dim=51,
          bands=['r', 'i', 'z'],
          noise_factor=0.58,
          cosmic_rays=True,
          bad_columns=True,
          star_bleeds=True,
      )

      # Extract the center part of the image 
      image = np.stack([sim_data['band_data'][b][0].image.array for b in ['r','i','z']], axis=-1)

      return trial, {'image':image, 
                    'g':np.array([g1,g2]).astype(np.float32)}

    for batch in range(ntrial // n_cores):
      trials = np.arange(batch*n_cores, (batch+1)*n_cores)
      results = pool.map(_get_trial, trials)
      for trial, result in results:
        yield trial, result