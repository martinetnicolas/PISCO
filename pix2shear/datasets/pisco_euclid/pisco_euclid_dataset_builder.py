"""pisco_euclid dataset."""
import numpy as np
import tensorflow_datasets as tfds

# TODO: Remove the need to load a dataset
#catalog = np.load('ingal_1_b24.5_1000000.npy')
#catalog = np.load('/net/GECO/nas12c/users/nmartinet/PISCO/PIX2SHEAR/PROGRAMS_GIT/PISCO/pisco/datasets/pisco_euclid/ingal_1_b24.5_1000000.npy')

def _get_image(seed):
  import galsim
  import math
  np.random.seed(seed)

  ######function to generate gal properties##########
  def generateaUDFgal(seed,image_size,sigma_eps):
    #read CDF computed from UDF data
    PDFmag = np.array([[20.5,0.04508],[21.0,0.09016],[21.5,0.14754],[22.0,0.20902],[22.5,0.31967],[23.0,0.43033],[23.5,0.67213],[24.0,1.0]])    
    PDFn_cmag = np.array([[20.5,0, 0],[20.5,0.5,0.2727],[20.5,0.75,0.5454],[20.5,1.0,0.63636],[20.5,1.25,0.7272],[20.5,1.75,0.81818],[20.5,3.0,1.0],[21.0,0.0,0.0],[21.0,0.75,0.54545],[21.0,1.25,0.63636],[21.0,1.75,0.72727],[21.0,4.0,0.909090],[21.0,4.5,1.0],[21.5,0.0,0.0],[21.5,0.25,0.14285],[21.5,0.5,0.357142],[21.5,1.0,0.428571],[21.5,1.25,0.57142],[21.5,2.0,0.642857],[21.5,2.25,0.71428],[21.5,4.0,0.785714],[21.5,4.5,0.857142],[21.5,5.0,0.928571],[21.5,5.25,1.0],[22.0,0.0,0.0],[22.0,0.25,0.13333],[22.0,0.5,0.266666],[22.0,0.75,0.46666],[22.0,1.25,0.53333],[22.0,1.5,0.599999],[22.0,1.75,0.66666],[22.0,2.5,0.799999],[22.0,3.0,0.866666],[22.0,3.75,0.93333],[22.0,4.25,1.0],[22.5,0.0,0.0],[22.5,0.25,0.11111],[22.5,0.50,0.37037],[22.5,0.750,0.48140],[22.5,1.0,0.740740],[22.5,1.25,0.81481],[22.5,1.75,0.85185],[22.5,2.0,0.888888],[22.5,2.25,0.92592],[22.5,2.75,0.96296],[22.5,4.0,1.0],[23.0,0.0,0.0],[23.0,0.25,0.22222],[23.0,0.50,0.44444],[23.0,0.75,0.70370],[23.0,1.00,0.88888],[23.0,1.25,0.92592],[23.0,2.00,0.96296],[23.0,2.50,1.0],[23.5,0.00,0.0],[23.5,0.25,0.1],[23.5,0.50,0.28333],[23.5,0.75,0.44999],[23.5,1.00,0.61666],[23.5,1.25,0.76666],[23.5,1.50,0.83333],[23.5,2.25,0.9],[23.5,2.50,0.95],[23.5,2.75,0.96666],[23.5,6.00,0.98333],[23.5,6.75,1.0],[24.0,0.00,0.0],[24.0,0.25,0.05000],[24.0,0.50,0.11250],[24.0,0.75,0.26250],[24.0,1.00,0.41249],[24.0,1.25,0.57499],[24.0,1.50,0.66249],[24.0,1.75,0.75000],[24.0,2.00,0.80000],[24.0,2.25,0.83750],[24.0,2.50,0.87500],[24.0,2.75,0.88749],[24.0,3.00,0.92499],[24.0,3.25,0.93749],[24.0,3.50,0.94999],[24.0,3.75,0.96249],[24.0,5.00,0.97499],[24.0,7.50,0.98749],[24.0,8.75,1.0]])
    PDFhlr_cmag = np.array([[20.5,0.,0.],[20.5,0.2,0.0909],[20.5,0.3,0.1818],[20.5,0.4,0.4545],[20.5,0.7,0.5454],[20.5,0.9,0.6363],[20.5,1.,0.8181],[20.5,1.1,1.],[21.0,0.,0.],[21.0,0.2,0.1818],[21.0,0.3,0.5454],[21.0,0.5,0.6363],[21.0,0.6,0.7272],[21.0,0.8,1.],[21.5,0.,0.],[21.5,0.1,0.0714],[21.5,0.2,0.3571],[21.5,0.3,0.5],[21.5,0.4,0.6428],[21.5,0.5,0.7857],[21.5,0.6,0.9285],[21.5,0.7,1.],[22.0,0.,0.],[22.0,0.1,0.0666],[22.0,0.2,0.1333],[22.0,0.3,0.4666],[22.0,0.4,0.6],[22.0,0.5,0.7333],[22.0,0.6,0.9333],[22.0,0.8,1.],[22.5,0.,0.],[22.5,0.1,0.0370],[22.5,0.2,0.1111],[22.5,0.3,0.3703],[22.5,0.4,0.7037],[22.5,0.5,0.7777],[22.5,0.6,0.8148],[22.5,0.7,0.9259],[22.5,0.8,1.],[23.0,0.,0.],[23.0,0.1,0.0740],[23.0,0.2,0.2962],[23.0,0.3,0.4814],[23.0,0.4,0.7777],[23.0,0.5,0.8888],[23.0,0.6,0.9629],[23.0,0.8,1.],[23.5,0.,0.0166],[23.5,0.1,0.1833],[23.5,0.2,0.45],[23.5,0.3,0.7166],[23.5,0.4,0.9166],[23.5,0.5,1.],[24.0,0.,0.],[24.0,0.1,0.3125],[24.0,0.2,0.6],[24.0,0.3,0.8625],[24.0,0.4,0.9625],[24.0,0.5,1.]])
    
    #define granularity of CDfs
    dmag = 0.5
    dhlr=0.1
  
    #initialize random generators
    uds_gal = galsim.UniformDeviate(seed)
    gds_gal = galsim.GaussianDeviate(seed=seed, mean=0.0, sigma=sigma_eps)
  
    #generate mag
    magi = np.min(PDFmag[PDFmag[:,1] >= uds_gal()][:,0])
    mag = magi + dmag * uds_gal()
  
    #generate ns and hlr from mag bin
    PDFn_cmagi = PDFn_cmag[PDFn_cmag[:,0] == magi]
    ns = np.min(PDFn_cmagi[PDFn_cmagi[:,2] >= uds_gal()][:,1])
  
    PDFhlr_cmagi = PDFhlr_cmag[PDFhlr_cmag[:,0] == magi]
    hlr = np.min(PDFhlr_cmagi[PDFhlr_cmagi[:,2] >= uds_gal()][:,1]) + dhlr * uds_gal()
  
    #put ns in galsim range
    if ns < 0.3:
      ns = 0.3
    if ns > 6.2:
      ns = 6.2
  
    #generate x,y
    x = uds_gal()*image_size
    y = uds_gal()*image_size
  
    #generate ells1,ells2
    ells1 = gds_gal()
    ells2 = gds_gal()
  
    #remove large ellipticity modulus
    while np.sqrt(ells1**2 + ells2**2) > 0.7:
      ells1 = gds_gal()
      ells2 = gds_gal()
  
    return x,y,mag,hlr,ns,ells1,ells2
  ##################################""
  

  pixel_scale = 0.1    # pix size in arcsec (sizes in input catalog in pixels)
  xsize = 64           # gal patch size in pixels (6.4'')
  ysize = 64           # gal patch size in pixels (6.4'')
  image_size = 352     # image size in pixels (0.59')

  nobj = 10
  sigma_eps = 0.26

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
    #k = np.random.randint(0,len(catalog))
    
    #Read galaxy parameters from catalog
    #x = catalog[k, 0] 
    #y = catalog[k, 1] 
    #mag = catalog[k, 4] 
    #half_light_radius = catalog[k, 6] 
    #nsersic = catalog[k, 5] 
    #ells1 = catalog[k, 8] 
    #ells2 = catalog[k, 9]
    
    #generate galaxy parameters
    x,y,mag,half_light_radius,nsersic,ells1,ells2 = generateaUDFgal(seed*1000000000+i,image_size,sigma_eps)
        
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
    ud = galsim.UniformDeviate(seed+i)
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

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Generate gal properties.',
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
    #path = '/net/GECO/nas12c/users/nmartinet/PISCO/PIX2SHEAR/IMAGESIMS/'
    
    # TODO(pisco_euclid): Returns the Dict[split names, Iterator[Key, Example]]
    return {
    #    'train': self._generate_examples('ingal_1_b24.5_1000000.npy'),
    #    'train': self._generate_examples('/net/GECO/nas12c/users/nmartinet/PISCO/PIX2SHEAR/PROGRAMS_GIT/PISCO/pisco/datasets/pisco_euclid/ingal_1_b24.5_1000000.npy'),
        'train': self._generate_examples(),
    }

  #def _generate_examples(self, path):
  def _generate_examples(self):
    """Yields examples."""
    from multiprocessing import Pool, cpu_count
    # get the number of logical cpu cores
    n_cores = cpu_count()
    pool = Pool(processes=n_cores)
    ntrial = 50_000

    # Generate all images at once 
    results = pool.map(_get_image, np.arange(ntrial))

    # Done, closing pool
    pool.close()

    for trial, result in results:
      yield int(trial), result
