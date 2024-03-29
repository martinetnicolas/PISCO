"""cosmo_slics dataset."""

import tensorflow_datasets as tfds
import numpy as np

def _get_Map(trial):
    
    N_CP = 25
    N_f = 2
    N_LOS = 5#5
    N_shape = 5#10
    N_Q = 4
    
    CP = int(trial // (N_f * N_LOS * N_shape * N_Q))
    remainder = int(trial % (N_f * N_LOS * N_shape * N_Q))
    IC_int = int(remainder // (N_LOS * N_shape * N_Q))
    remainder = int(remainder % (N_LOS * N_shape * N_Q))
    if IC_int == 0:
        IC = 'a'
    elif IC_int == 1:
        IC= 'f'
    LOS = int(remainder // (N_shape * N_Q)) + 1
    remainder = int(remainder % (N_shape * N_Q))
    shape = int(remainder // (N_Q)) + 1001
    remainder = int(remainder % (N_Q))
    Q = remainder + 1
    
    #print(trial,CP,IC,LOS,shape,Q)
    
    cosmovalues = np.loadtxt('/net/GECO/nas12c/users/nmartinet/PISCO/SHEAR2COSMO/NBODYMOCKS/COSMOSLICS/cosmoslics_cosmologies.dat')
    filename='/net/GECO/nas12c/users/nmartinet/PISCO/SHEAR2COSMO/MASSMAPS/COSMOSLICS/'+str(CP).zfill(2)+'_'+str(IC)+'/GalCatalog_LOS_cone'+str(LOS)+'.fits_s1_zmin0.0_zmax3.0.fits_s'+str(shape)+'_spec0_p2.34_rout10.0_rin0.4_xc0.15_mapmap_4Q'+str(Q)+'.res'
    
    Map = np.loadtxt(filename)
    
    #cosmo = cosmovalues[CP]
    #S8 = np.array([cosmo[1]])
    
    cosmo = np.array(cosmovalues[CP])

    return trial, {'Map':Map.astype(np.float32), 
        #'S8':S8.astype(np.float32)}
        'CP':cosmo.astype(np.float32)}

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cosmo_slics dataset."""

  VERSION = tfds.core.Version('1.1.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release. Only S8',
      '1.1.0': 'Om, S8, w, h',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(cosmo_slics): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            #'e12': tfds.features.Tensor(shape=(512, 512, 2), dtype=np.float32),
            'Map': tfds.features.Tensor(shape=(128, 128), dtype=np.float32),
            'CP': tfds.features.Tensor(shape=(4,), dtype=np.float32),
            #'S8': tfds.features.Tensor(shape=(1,), dtype=np.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        #supervised_keys=('Map', 'S8'),  # Set to `None` to disable
        supervised_keys=('Map', 'CP'),  # Set to `None` to disable
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

  def _generate_examples(self):
    """Yields examples."""
    from multiprocessing import Pool, cpu_count
    # get the number of logical cpu cores
    n_cores = cpu_count()
    pool = Pool(processes=n_cores)

    ntrial = 5_000
    
    # Generate all images at once 
    results = pool.map(_get_Map, np.arange(ntrial))

    # Done, closing pool
    pool.close()

    for trial, result in results:
      yield int(trial), result
