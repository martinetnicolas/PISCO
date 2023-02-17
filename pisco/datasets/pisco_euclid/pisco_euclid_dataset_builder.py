"""pisco_euclid dataset."""
import numpy as np
import tensorflow_datasets as tfds


def _get_image(seed, catalog=None):
  import galsim
  import math

  pixel_scale = 0.1    # pix size in arcsec (sizes in input catalog in pixels)
  xsize = 64           # gal patch size in pixels (6.4'')
  ysize = 64           # gal patch size in pixels (6.4'')
  image_size = 352     # image size in pixels (0.59')

  nobj = 10

  t_exp = 3*565 #s
  gain = 3.1 #e-/ADU
  readoutnoise = 4.2 #e-
  sky_bkg = 22.35 #mag/arcsec2
  
  ZP=24.0 #mag

  F_sky = pixel_scale**(2)*t_exp*10**(-(sky_bkg-ZP)/2.5) #e-/pixel
  noise_variance = ( np.sqrt( ( (readoutnoise)**2 + F_sky ) ) *1/gain )**2 #e- -> ADU by dividing sigma by gain ; sigma = 4.9ADU
######
  

###CREATE OUTPUT IMAGES###
  full_image = galsim.ImageF(image_size, image_size)
  full_image.setOrigin(1,1)
######

###MAKE THE WCS COORDINATES (test11)###
  # Make a slightly non-trivial WCS.  We'll use a slightly rotated coordinate system
  # and center it at the image center.
  theta = 0.17 * galsim.degrees
  dudx = np.cos(theta) * pixel_scale
  dudy = -np.sin(theta) * pixel_scale
  dvdx = np.sin(theta) * pixel_scale
  dvdy = np.cos(theta) * pixel_scale
  image_center = full_image.true_center
  affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)

  # We can also put it on the celestial sphere to give it a bit more realism.
  # The TAN projection takes a (u,v) coordinate system on a tangent plane and projects
  # that plane onto the sky using a given point as the tangent point.  The tangent 
  # point should be given as a CelestialCoord.
  sky_center = galsim.CelestialCoord(ra=3.544151*galsim.hours, dec=-27.791371*galsim.degrees)
  # The third parameter, units, defaults to arcsec, but we make it explicit here.
  # It sets the angular units of the (u,v) intermediate coordinate system.

  wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
  full_image.wcs = wcs
  #logger.info('Image %r and %r created',file_name,file_name_noise)
######


###TUNE THE SPEED OF FFT###
  #slightly decrease the precision on fourrier and convolution to speed up.
  #Taken from Jarvis discussion https://github.com/GalSim-developers/GalSim/issues/566
  gsparams = galsim.GSParams(xvalue_accuracy=2.e-4, kvalue_accuracy=2.e-4,
                          maxk_threshold=5.e-3, folding_threshold=1.e-2)
######

###BUILD PSF###
  psf = galsim.Airy(lam=800, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec,flux=1./3) + galsim.Airy(lam=700, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec,flux=1./3) + galsim.Airy(lam=600, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec,flux=1./3)

###PAINT GALAXIES###
  #draw constant shear
  uds = galsim.UniformDeviate(seed)
  g1 = (uds()-0.5)/0.5*0.06
  g2 = (uds()-0.5)/0.5*0.06

  #initiate mean ell and g
  ave_ell1 = 0
  ave_ell2 = 0
  ave_g1 = 0
  ave_g2 = 0

  #loop on gals
  for i in range(nobj):
    #Draw a random entry in the catalog
    k = np.random.randint(0,len(catalog))
    
    #Read galaxy parameters from catalog
    x = catalog[k, 0] 
    y = catalog[k, 1] 
    mag = catalog[k, 4] 
    half_light_radius = catalog[k, 6] 
    nsersic = catalog[k, 5] 
    ells1 = catalog[k, 8] 
    ells2 = catalog[k, 9] 
    
    #Get position on sky
    image_pos = galsim.PositionD(x,y)
    world_pos = affine.toWorld(image_pos)

    #Calculate gal flux
    fluxflux = t_exp/gain*10**(-(mag-ZP)/2.5)
    
    #Compute Sersic profile
    gal = galsim.Sersic(n=nsersic, half_light_radius=half_light_radius, flux=fluxflux, gsparams=gsparams, trunc=half_light_radius*4.5)
    gal = gal.shear(e1=ells1, e2=ells2)

    #Rotate galaxy
    #gal = gal.rotate(theta=ang*galsim.degrees)

    #Apply shear
    gal = gal.shear(g1=g1, g2=g2)
                
    #Calculate mean gal ellipticity
    ave_ell1 = ave_ell1 + (ells1 + g1) / nobj #refine calculation
    ave_ell2 = ave_ell2 + (ells2 + g2) / nobj
    ave_g1 = ave_g1 + g1 / nobj
    ave_g2 = ave_g2 + g2 / nobj

    #convolve galaxy with PSF
    final = galsim.Convolve([psf, gal])

    #offset the center for pixelization (of random fraction of half a pixel)
    ud = galsim.UniformDeviate(seed+k)
    x_nominal = image_pos.x+0.5
    y_nominal = image_pos.y+0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = (x_nominal - ix_nominal)*(2*ud()-1)
    dy = (y_nominal - iy_nominal)*(2*ud()-1)
    offset = galsim.PositionD(dx,dy)

    #draw galaxy
    image = galsim.ImageF(xsize,ysize,scale=pixel_scale)
    final.drawImage(image=image,wcs=wcs.local(image_pos), offset=offset)
    image.setCenter(ix_nominal,iy_nominal)

    #add stamps to single image
    bounds = image.bounds & full_image.bounds
    full_image[bounds] += image[bounds]

  e = np.array([ave_ell1,ave_ell2])
  g = np.array([ave_g1,ave_g2])

###ADD NOISE###
  #add Gaussian noise
  rng = galsim.BaseDeviate(seed)
  noise = galsim.GaussianNoise(rng, sigma=math.sqrt(noise_variance))
  full_image.addNoise(noise)
######

  return seed, {'image':full_image.array, 
                'g':g.astype(np.float32),
                'e':e.astype(np.float32)}


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for pisco_euclid dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(pisco_euclid): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=(352, 352), dtype=np.float32),
            'g': tfds.features.Tensor(shape=(2,), dtype=np.float32),
            'e': tfds.features.Tensor(shape=(2,), dtype=np.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'g'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(pisco_euclid): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(pisco_euclid): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples('ingal_1_b24.5_1000000.npy'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    from multiprocessing import Pool, cpu_count
    # get the number of logical cpu cores
    n_cores = cpu_count()
    pool = Pool(processes=n_cores)
    ntrial = 100

    catalog = np.load(path)

    # Generate all images at once 
    results = pool.map(lambda x: _get_image(x, catalog=catalog), np.arange(ntrial))

    # Done, closing pool
    pool.close()

    for trial, result in results:
      yield int(trial), result
