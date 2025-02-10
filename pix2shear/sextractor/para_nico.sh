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
