#!/bin/bash

# Script to rotate and clip TRMM 3B43 data, and subset, reproject and resample
# MODIS MCD12C1 data.
# Tested with GDAL 2.1.0

cd /projectnb/landsat/users/parevalo/ge529/3B43

# TRMM
for i in $(find . -type f -name "3B43*.HDF"); do

    # Assign GCP to TRMM for rotation

    fname=$(basename $i | awk -F ".HDF" '{print $1}')
    gdal_translate -a_srs EPSG:4326 -gcp 0 0 -180 -50 -gcp 0 1440 180 -50 \
     -gcp 400 0 -180 50 -gcp 400 1440 180 50 \
      HDF4_SDS:UNKNOWN:$fname".HDF:0" $fname"_GCP.tif"

    # Apply GCP to make them permament (i.e rotate)

    gdalwarp -t_srs EPSG:4326 $fname"_GCP.tif" $fname"_warp.tif"
    
    # Extract amazon region from TRMM

    gdal_translate -projwin -80 10 -45 -20 -projwin_srs EPSG:4326 \
     $fname"_warp.tif" $fname"_clip.tif"

done

cd /projectnb/landsat/users/parevalo/ge529/MCD12C1.051 

# MODIS
for i in $(find . -type f -name "MCD12C1*.hdf"); do

    fname=$(basename $i | awk -F ".hdf" '{print $1}')
    
    # Extract MODIS  and resample. Notice that -te input has a different order than -projwin

    gdalwarp -te -80 -20 -45 10 -t_srs EPSG:4326 -tr 0.25 0.25 \
    HDF4_EOS:EOS_GRID:$fname".hdf:MOD12C1:Majority_Land_Cover_Type_1" \
    $fname"_clip.tif"

done
