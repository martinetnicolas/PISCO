
ShapeParameters:
  - ellipse_a : Profile RMS along major axis pixel
  - ellipse_b : Profile RMS along minor axis pixel
  - ellipse_theta : Isophotal image position angle (CCW/x) deg
  - ellipse_cxx : Isophotal image Cxx ellipse parameter pixel^{-2}
  - ellipse_cyy : Isophotal image Cyy ellipse parameter pixel^{-2}
  - ellipse_cxy : Isophotal image Cxy ellipse parameter pixel^{-2}
  - area : Isophotal area (filtered) above detection threshold pixel
  - elongation : The object elongation (a_image / b_image)
  - ellipticity : The object ellipticity (1 - b_image / a_image)



##########################################################################################################
##########################################################################################################


zmin=$1
zmax=$2
rout=$3
xc=$4
rin=0.4

tpcfyn=0
shearmapyn=0
massmapyn=0
peaksyn=1
plotsyn=0

PIXPIXLENS='/net/GECO/nas12c/users/nmartinet/PISCO/SHEAR2COSMO'
SLURMCONF='#!/bin/bash
#SBATCH -J mapmapor
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=5GB
#SBATCH -t 0:10:00
#SBATCH -o SLURM/Out.txt
#SBATCH -e SLURM/Err.txt
##SBATCH --mail-type=END
##SBATCH --mail-user=nicolas.martinet@lam.fr\n'


###
#Observations
for i in $(seq 1 5)
do
    #for shape in $(seq 1001 1010)
    for shape in $(seq 1001 1005)
    do

    echo -e "${SLURMCONF}python ${PIXPIXLENS}/PROGRAMS/mapmapor_fft.py COSMOSLICS/fid_a/ GalCatalog_LOS_cone${i}.fits_s1_zmin${zmin}_zmax${zmax}.fits ${shape} 0 ${tpcfyn} ${shearmapyn} ${rout} ${rin} ${xc} ${massmapyn} ${peaksyn} ${plotsyn}" > SLURM/mapmapor_slurm.sh
    sbatch -p batch -e SLURM/map_fid_a_cone${i}_zmin${zmin}_zmax${zmax}_rout${rout}.err -o SLURM/map_fid_a_cone${i}_zmin${zmin}_zmax${zmax}_rout${rout}.out SLURM/mapmapor_slurm.sh

    echo -e "${SLURMCONF}python ${PIXPIXLENS}/PROGRAMS/mapmapor_fft.py COSMOSLICS/fid_f/ GalCatalog_LOS_cone${i}.fits_s1_zmin${zmin}_zmax${zmax}.fits ${shape} 0 ${tpcfyn} ${shearmapyn} ${rout} ${rin} ${xc} ${massmapyn} ${peaksyn} ${plotsyn}" > SLURM/mapmapor_slurm.sh
    sbatch -p batch -e SLURM/map_fid_f_cone${i}_zmin${zmin}_zmax${zmax}_rout${rout}.err -o SLURM/map_fid_f_cone${i}_zmin${zmin}_zmax${zmax}_rout${rout}.out SLURM/mapmapor_slurm.sh

    done
done 


OU
#SBATCH --ntasks-per-core 1
#SBATCH --nodes 1         # -N
#SBATCH --ntasks 24       # -n
#SBATCH --cpus-per-task 1 # -c
#SBATCH -J HAGN_SE++
#SBATCH -e logs/HAGN_%A_%a.log # stderr
#SBATCH -o logs/HAGN_%A_%a.log # stdout
#SBATCH --array 0-99%8  #
#SBATCH --mail-type=END  # 
#SBATCH --mail-user=romain.brunet@lam.fr




##########################################################################################################
##########################################################################################################


#!/bin/bash -norc
#SBATCH --time 06:00:00
#SBATCH --mem 8G
#SBATCH --ntasks-per-core 1
#SBATCH --nodes 1         # -N
#SBATCH --ntasks 24       # -n
#SBATCH --cpus-per-task 1 # -c
#SBATCH -J HAGN_SE++
#SBATCH -e logs/HAGN_%A_%a.log # stderr
#SBATCH -o logs/HAGN_%A_%a.log # stdout
#SBATCH --array 0-99%8  #
#SBATCH --mail-type=END  # 
#SBATCH --mail-user=raphael.gavazzi@lam.fr

export MKL_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"

source /cvmfs/euclid-dev.in2p3.fr/CentOS7/EDEN-3.0/bin/activate

Nth=$SLURM_NTASKS
test -z $SLURM_NTASKS && Nth=30  ## fallback if not launched with PBS !

emcd=/net/GECO/nas12c/euclid/SEpp  
pysc=$emcd/mod3band1.py


detec_min_area=9
bg_cell_s=64					      
smoothing_box_size=3
cleaning_min_area=30
detection_tresh=1.5
thre=0.0001
bv=""
#bv=" --background-value=0.0"
gpalg=split
#gpalg=moffat

#opfx=all_nf_nsp
opfx=vis_aggbgsub
#opfx=g-vis-J_fixedbg
#opfx=g_fixedbg
export wdir=$emcd/wd-$opfx
mkdir -p $wdir
db=VIS
#db=VIS_deep
#db=g_LSST
di=$emcd/$db.fits
wi=$emcd/$db.rms.fits
pi=$emcd/$db.psf.fits

echo "preparing subsection image for Array id=$SLURM_ARRAY_TASK_ID"
export IV=$SLURM_ARRAY_TASK_ID
#export IV=50
julia >/dev/null <<EOF
using FITSIO
nblock=10
i=parse(Int,ENV["IV"])
j,k=divrem(i,nblock)
@info "running for chunk \$i  (\$j,\$k)"
io=FITS("$di")
gs=size(io[1])
dx=div(gs[1],nblock) ; dy=div(gs[2],nblock)
x1=max(1,1+j*dx) ; x2=min(gs[1],x1+dx-1)
y1=max(1,1+k*dy) ; y2=min(gs[2],y1+dy-1)
@info "fraction of $di being extracted",x1:x2,y1:y2
oio=FITS(joinpath(ENV["wdir"],"co-\$i.fits"),"w")
copy_section(io[1], oio, x1:x2, y1:y2)
close(oio); close(io)
io=FITS("$wi")
oio=FITS(joinpath(ENV["wdir"],"co-\$i.rms.fits"),"w")
copy_section(io[1], oio, x1:x2,y1:y2)
close(oio); close(io)
EOF



di=$wdir/co-$IV.fits
wi=$wdir/co-$IV.rms.fits
cf=$wdir/${opfx}$q-$IV.cat
#chk=" --check-image-model-fitting $wdir/${opfx}-${IV}_model.fits --check-image-residual $wdir/${opfx}-${IV}_diff.fits  --check-image-grouping $wdir/${opfx}-${IV}_grp.fits --check-image-background $wdir/${opfx}-${IV}_bg.fits"
chk=""

echo "Job $0 running on $HOSTNAME with $Nth cores."
echo "Results will be stored in $cf"
echo " "
E-Run SourceXtractorPlusPlus 0.20 sourcextractor++ --version
echo " "

E-Run SourceXtractorPlusPlus 0.20 sourcextractor++ --detection-image $di  \
		 --weight-type=rms \
		 --weight-absolute 1 \
		 --weight-image=$wi \
		 --python-config-file $pysc \
		 --output-catalog-filename $cf \
		 --output-properties=PixelCentroid,WorldCentroid,SourceIDs,GroupInfo,SourceFlags,FlexibleModelFitting,AperturePhotometry,AutoPhotometry,FluxRadius,SNRRatio,ShapeParameters,Vignet \
		 --vignet-size 41,41 \
		 --magnitude-zero-point 33.885 \
		 --segmentation-algorithm LUTZ \
		 --segmentation-lutz-window-size 512 \
		 --thread-count $Nth \
		 --sampling-scale-factor 1 \
		 --background-cell-size $bg_cell_s \
	         --smoothing-box-size $smoothing_box_size \
		 $bv \
		 --core-minimum-area 10 \
		 --partition-corethreshold 1 \
		 --segmentation-filter $emcd/gauss2.4.fits \
		 --detection-minimum-area $detec_min_area \
		 $chk \
		 --detection-image-interpolation=0 \
                 --detection-threshold $detection_tresh \
		 --detection-image-saturation 0 \
		 --detection-image-gain 0 \
		 --use-cleaning yes \
		 --cleaning-minimum-area $cleaning_min_area \
		 --grouping-algorithm $gpalg \
                 --grouping-moffat-threshold 0.02 \
                 --grouping-moffat-max-distance 300 \
		 --partition-multithreshold yes \
		 --partition-minimum-area 10 \
		 --partition-minimum-contrast $thre \
		 --output-flush-size 1000

echo "Job completed!"




##########################################################################################################
##########################################################################################################


#!/bin/bash -norc
#SBATCH --time 03:00:00
#SBATCH --mem 16G
#SBATCH --ntasks-per-core 1
#SBATCH --nodes 1         # -N
#SBATCH --ntasks 24       # -n
#SBATCH --cpus-per-task 1 # -c
#SBATCH -J HAGN_SE++
#SBATCH -e logs/HAGN_%A_%a.log # stderr
#SBATCH -o logs/HAGN_%A_%a.log # stdout
#SBATCH --mail-type=END  # 
#SBATCH --mail-user=raphael.gavazzi@lam.fr


export MKL_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"

source /cvmfs/euclid-dev.in2p3.fr/CentOS7/EDEN-3.0/bin/activate

Nth=$SLURM_NTASKS
test -z $SLURM_NTASKS && Nth=30  ## fallback if not launched with PBS !

emcd=/net/GECO/nas12c/euclid/SEpp  
pysc=$emcd/mod3band1.py

q=""
#opfx=all_nf_nsp
opfx=test_single

thre=0.001
bv=""
#bv=" --background-value=0.0"
gpalg=split

chk=" --check-image-model-fitting ${opfx}_model.fits --check-image-residual ${opfx}_diff.fits  --check-image-grouping ${opfx}_grp.fits  --check-image-background ${opfx}_bg.fits"
#chk=""

#di=$emcd/VIS.fits
#wi=$emcd/VIS.rms.fits
#pi=$emcd/VIS.psf.fits
#cf=${opfx}$q.cat


export IV=50 
di=$emcd/co-$IV.fits
wi=$emcd/co-$IV.rms.fits
pi=$emcd/VIS.psf.fits
cf=${opfx}$q-$IV.cat

echo "Job $0 running on $HOSTNAME with $Nth cores."
echo "Results will be stored in $cf"
echo " "
E-Run SourceXtractorPlusPlus 0.20 sourcextractor++ --version
echo " "

E-Run SourceXtractorPlusPlus 0.20 sourcextractor++ --detection-image $di  \
		 --weight-type=rms \
		 --weight-absolute 1 \
		 --weight-image=$wi \
		 --python-config-file $pysc \
		 --output-catalog-filename $cf \
		 --output-properties=PixelCentroid,WorldCentroid,SourceIDs,GroupInfo,SourceFlags,FlexibleModelFitting,AperturePhotometry,AutoPhotometry,FluxRadius,SNRRatio \
		 --magnitude-zero-point 33.885 \
		 --segmentation-algorithm LUTZ \
		 --segmentation-lutz-window-size 512 \
		 --thread-count $Nth \
		 --sampling-scale-factor 1 \
		 --background-cell-size 128 $bv \
		 --core-minimum-area 10 \
		 --partition-corethreshold 1 \
		 --segmentation-filter $emcd/gauss2.4.fits \
		 --detection-minimum-area 5 \
		 $chk \
		 --detection-image-interpolation=0 \
		 --detection-image-saturation 0 \
		 --detection-image-gain 0 \
		 --use-cleaning yes \
		 --cleaning-minimum-area 3 \
		 --grouping-algorithm $gpalg \
                 --grouping-moffat-threshold 0.02 \
                 --grouping-moffat-max-distance 300 \
		 --partition-multithreshold yes \
		 --partition-minimum-area 10 \
		 --partition-minimum-contrast $thre \
		 --output-flush-size 1000

echo "Job completed!"




#####################################################################################################################
#####################################################################################################################


from glob import glob
import numpy as np
from sourcextractor.config import *
import os
#from pathlib import Path

#########################################
#########      LOAD IMAGES   ############
#########################################
meas_imags=[]
band_map={}
datadir='/net/GECO/nas12c/euclid/SEpp'
d='' # '_deep'

#for b in [ 'VIS', 'Y_EUC', 'J_EUC', 'H_EUC',  'u_LSST', 'g_LSST', 'r_LSST', 'i_LSST', 'z_LSST', 'y_LSST' ]:
for b in [ 'VIS', ]:
#for b in [ 'VIS', 'g_LSST', 'J_EUC']:
#for b in [ 'g_LSST', ]:
    ima,wht,psf=(datadir+'/'+f'{b}{d}.fits',datadir+'/'+f'{b}{d}.rms.fits',datadir+'/'+f'{b}.psf.fits')
    #ima,wht,psf=(f'co-50.fits',f'co-50.rms.fits',f'{b}.psf.fits')
    band_map[os.path.basename(ima)]=b
    #band_map[ima]=b
    meas_imags.append( MeasurementImage(fits_file=ima,
                                        weight_file=wht, 
                                        psf_file=psf,
                                        constant_background=0.0,
                                        weight_type='rms',
                                        weight_absolute=1,
                                        weight_threshold=1.e-06) )

top = ImageGroup(images=meas_imags)
top.split(ByKeyword('IMAGE_FILENAME'))
mesgroup = MeasurementGroup(top)

print("#####################")
print(band_map)
print("#####################")
print(mesgroup)
print("#####################")


#########################################
########    DEFINE APERTURES   ##########
#########################################
all_apertures = []
for band,img in mesgroup:
    b=band_map[os.path.basename(band)]
    all_apertures.extend(add_aperture_photometry(img, [10, 30, 60] ) )
    add_output_column('aper_'+str(b), all_apertures)


#########################################
#########     DEFINE MODEL   ############
#########################################
set_engine('levmar')
use_iterative_fitting(True)
set_meta_iterations(3)
set_max_iterations(150)
set_deblend_factor(0.9)
set_meta_iteration_stop(0.0001)  ## increase -> faster!


x,y = get_pos_parameters()
ra,dec = get_world_position_parameters(x, y)

r_b = FreeParameter(lambda o: o.radius*0.5, Range(lambda v,o: (0.01*v,1.5*v), RangeType.EXPONENTIAL))
r_d = FreeParameter(lambda o: o.radius*1.5, Range(lambda v,o: (0.01*v,2.0*v), RangeType.EXPONENTIAL))
### from ground truth 
###       log10(r_d) ~ N(0.33,0.22)
###       log10(r_b) ~ 1.14 * log10 r_d -1.2 + N(0,0.38)  # captures the mild covariance between d and b radii
lrd =  DependentParameter( lambda y : np.log10(y), r_d )
add_prior( lrd, 0.33, 0.25 ) ## log10(rd) in pixels 
#rel_size = DependentParameter( lambda x,y : np.log10(y)-(1.14*np.log10(x)-1.0), r_d, r_b )
rel_size = DependentParameter( lambda x,y : np.log10(y)-(1.9*np.log10(x)-1.3), r_d, r_b )
add_prior( rel_size, 0.0, 0.4 )

angle = FreeParameter( lambda o: o.angle )
ratio_d = FreeParameter( 0.5, Range( (0.01, 1.0), RangeType.LINEAR) )
ratio_b = FreeParameter( 0.6, Range( (0.01, 1.0), RangeType.LINEAR) )
X_rd = DependentParameter( lambda q: np.log( (q+0.02)/(1.02-q) ), ratio_d ) ; add_prior( X_rd, 0.03, 1.0 )
X_rb = DependentParameter( lambda q: np.log( (q+0.02)/(1.02-q) ), ratio_b ) ; add_prior( X_rb, 0.50, 1.1 )

add_output_column('x',x)
add_output_column('y',y)
add_output_column('disk_effR_px',r_d)
add_output_column('bulge_effR_px',r_b)
add_output_column('angle',angle)
add_output_column('disk_axr',ratio_d)
add_output_column('bulge_axr',ratio_b)
add_output_column('ra',ra)
add_output_column('dec',dec)
add_output_column('rel_s',rel_size)

i = 0
flux1 = {}
flux2 = {}
flux = {}
bt = {}
X_bt= {}
mag = {}
X_bt_med = { 'u_LSST':-3.23, 'g_LSST': -3.03 , 'r_LSST': -2.86, 'i_LSST':-2.74, 'z_LSST':-2.67, 'y_LSST':-2.53, 'Y_EUC':-2.53, 'J_EUC':-2.44, 'H_EUC':-2.36, 'VIS':-2.75}
X_bt_wid = { 'u_LSST':1.74, 'g_LSST':1.89, 'r_LSST':2.08, 'i_LSST':2.44, 'z_LSST':2.45, 'y_LSST':2.62, 'Y_EUC':-2.53, 'J_EUC':2.81, 'H_EUC':2.94, 'VIS':2.28 }
ZPs={'u_LSST':33.223, "g_LSST":33.820, "r_LSST":33.600, "i_LSST":33.290, "z_LSST":33.653, "y_LSST":32.872, "VIS":33.885, "Y_EUC":31.668, "J_EUC":31.888, "H_EUC":31.838 }
#ZP=23.9  

for band,group in mesgroup:
    b=band_map[os.path.basename(band)]
    flux[b] = get_flux_parameter()
    mag[b] = DependentParameter(lambda f, zp=ZPs[b]: zp - 2.5*np.log10(f), flux[b] )
    bt[b] = FreeParameter(0.1, Range((0.0,1.0), RangeType.LINEAR))
    flux1[b] = DependentParameter(lambda f, r: f*r, flux[b], bt[b] )
    flux2[b] = DependentParameter(lambda f, r: f*(1.0-r), flux[b], bt[b] )
    X_bt[b] = DependentParameter(lambda r: np.log( (r+0.01)/(1.01-r) ), bt[b] )
    add_prior( X_bt[b], X_bt_med[b], X_bt_wid[b] )
    add_model( group, ExponentialModel( x, y, flux2[b], r_d, ratio_d, angle) )
    add_model( group, DeVaucouleursModel( x, y, flux1[b], r_b, ratio_b, angle) )
    add_output_column('mag_'+b,mag[b])
    add_output_column('bt_'+b,bt[b])
    add_output_column('X_bt_'+b,X_bt[b])
