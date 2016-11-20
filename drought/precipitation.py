import os
import numpy as np
import gdal
import scipy.io as sio
from scipy.interpolate import UnivariateSpline
from sklearn import preprocessing
from matplotlib import pyplot as plt
from datetime import datetime


# Script to plot the time series of mean dry-season precipitation
# and anomalies

#path = '/projectnb/landsat/users/parevalo/ge529/3B43'
path = '/home/paulo/ge529/raster/data/'

years = range(2002, 2012)

def get_image(image):
    """ Read raster file into array """
    img_open = gdal.Open(image)
    img_array = img_open.GetRasterBand(1).ReadAsArray()
    return img_array

def get_trmm(dir, year, month):
    """ Read TRMM as array"""

    if year < 2011:
        image = dir + '3B43.' + str(year) + str(month).zfill(2) + '01.7A_clip.tif'
    else:
        image = dir + '3B43.' + str(year) + str(month).zfill(2) + '01.7_clip.tif'
    
    img_array = get_image(image)
    
    return img_array

def get_modis(dir, year):
    """ Read MCD12C1 as array from tif files"""

    import fnmatch
    for file in os.listdir(dir):
        string = 'MCD12C1*' + str(year) + '*.tif'
        if fnmatch.fnmatch(file, string):
            img_array = get_image(dir + file)

    new_array = np.ones(img_array.shape)
    new_array * 4  # Non vegetation
    new_array[(img_array < 15) & (img_array != 13)] = 3  # Other vegetation
    new_array[(img_array > 7) & (img_array < 10)] = 2  # Savannas
    new_array[(img_array > 0) & (img_array < 6)] = 1  # Forests
    new_array[img_array == 0] = 0  # Water

    return new_array


def calc_mean(a1, a2, a3):
    """ Calculate mean for dry season months"""
    dry_months = np.array((a1, a2, a3))
    mean = np.mean(dry_months, axis=0)
    return mean
   
def calc_anomaly(period, year):
    """ calculates anomaly for a given year in the period"""
    index = year-2002
    new_period = np.delete(period, index, axis=2)
    mean = new_period.mean(axis=2)
    std = new_period.std(axis=2)
    anomaly = (new_period[:,:,index] - mean) / std
    
    return anomaly

def save_img(array, year, suffix, title):
    """ Save image of year precipitation or anomaly"""
    outfile = '/home/paulo/ge529/drought/figures/' + str(year) + suffix
    fig, ax = plt.subplots()
    cax = ax.imshow(array, cmap=plt.get_cmap('viridis'), extent=[-80, -45, -20, 10])
    cbar = fig.colorbar(cax, orientation='vertical')
    ax.set_title(title + " - " + str(year))
    plt.savefig(outfile)
    plt.close(fig)

# Get images, calculate mean and save images
precip = np.zeros((120, 140, 10))
for i, y in enumerate(years):
    jul = get_trmm(path, y, 7)
    ago = get_trmm(path, y, 8)
    sept = get_trmm(path, y, 9)
    precip[:, :, i] = calc_mean(jul, ago, sept)
    save_img(precip[:, :, i], y, '_mean.png', 'Precipitation (mm $h^{-1}$)')

# Calculate and save anomalies
anomaly_08 = calc_anomaly(precip, 2008)
anomaly_10 = calc_anomaly(precip, 2010)

save_img(anomaly_08, 2008, '_anomaly.png', 'Precipitation anomaly')
save_img(anomaly_10, 2010, '_anomaly.png', 'Precipitation anomaly')

# Get MODIS LC data
lc_08= get_modis(path, 2008)
lc_10= get_modis(path, 2010)

# Load weights to calculate histogram and TS
wt = sio.loadmat('./drought/data/trmm_cell_size.mat')['amazon_cellwt']

# Mask precipitation outside forest
masked_precip08 = precip[:, :, 6][lc_08 == 1]
masked_weights08 = wt[lc_08 == 1].reshape(-1, 1)  # To avoid a deprecation warning

masked_precip10 = precip[:, :, 8][lc_10 == 1]
masked_weights10 = wt[lc_10 == 1].reshape(-1, 1)  # To avoid a deprecation warning

# Scale weights
minmax_scaler = preprocessing.MinMaxScaler()
scaled_wt_08 = minmax_scaler.fit_transform(masked_weights08)
scaled_wt_10 = minmax_scaler.fit_transform(masked_weights10)

# Create weighted histogram for 2008 and 2010

plt.figure(0)
plt.hist(masked_precip08, weights=scaled_wt_08)
plt.ylabel('Area weighted frequency')
plt.xlabel('Precipitation (mm $h^{-1}$)')
plt.title("Area weighted histogram for precipitation in dry months, 2008")
plt.savefig('/home/paulo/ge529/drought/figures/precip_hist_08.png')

plt.figure(1)
plt.hist(masked_precip10, weights=scaled_wt_10)
plt.ylabel('Area weighted frequency')
plt.xlabel('Precipitation (mm $h^{-1}$)')
plt.title("Area weighted histogram for precipitation in dry months, 2008")
plt.savefig('/home/paulo/ge529/drought/figures/precip_hist_10.png')

# Plot TS for entire region?
scaled_wt = minmax_scaler.fit_transform(wt)
annual_mn_precip = np.zeros(10)
for i in range(10):
    annual_mn_precip[i] = np.average(precip[:, :, i], weights=scaled_wt)

plt.plot(annual_mn_precip)  # With weighting
plt.plot(precip.mean(axis=0).mean(axis=0))  # Without weighting

###########################################################################################################
# Plot annual profile of LAI, par and precipitation for the region between 0 and -10 lat, -60 and -70 long.

# Set paths to the data
LAI_PAR ='/home/paulo/ge529/raster/data/MODIS_LAI_PAR/'
MODIS_TRMM ='/home/paulo/ge529/raster/data/'

# Load LAI files, assumed to be covering the region of interest only
yrs = range(2000, 2009)
months = range(1, 13)
lai_cube = np.zeros((9, 12, 1200, 1200))  # 9 years, 12 months, 1200 rows and columns of data

for i, y in enumerate(yrs):
    for j, m in enumerate(months):
        filename = LAI_PAR + 'BU_LAI_' + str(y) + '_' + str(m).zfill(2) + '.mat'
        lai_cube[i, j, :, :] = sio.loadmat(filename)['LAI_BU']

# Remove period from Jan-00 to May-00, June-05 to May-06 and June-08 to Dec-08(with NaN)
lai_cube[0, 0:5, :, :] = np.nan
lai_cube[5, 5:12, :, :] = np.nan
lai_cube[6, 0:5, :, :] = np.nan
lai_cube[8, 5:, :, :] = np.nan

# Calculate mean LAI over the whole area
mean_lai = np.zeros(12)
for i, m in enumerate(months):
    mean_lai[i] = np.nanmean(lai_cube[:, i, :, :])

plt.plot(mean_lai)

# PAR files have bounding coordinates (10, -20) lat,(-45, -80) long and pixel size 1° (~110 km at equator)
# PAR files already exclude the June-05 May-06 period

# Month starts in June!
yrs = range(2000, 2008)  # If we read from June to June each year, we only need 8 years
par_cube = np.zeros((8, 12, 120, 140))
count = 0
for i, y in enumerate(yrs):
    for j, m in enumerate(months):
        count += 1  # Ugly but does the job for now
        filename = LAI_PAR + "PAR_total_" + str(count) + ".clip.tif"
        par_cube[i, j, :, :] = get_image(filename)

# Remove period from June 2005 to May 2006 (4th year) (with NaN)
par_cube[5, :, :, :] = np.nan

# Calculate mean PAR over the whole area
mean_par = np.zeros(12)
for i, m in enumerate(months):
    mean_par[i] = np.nanmean(par_cube[:, i, :, :])

plt.plot(mean_par)

# This is in case the files exclude the period that we need to filter out
# Add NaN to missing months to be able to index months much more easily. There's probably an easier way to do this...
# par_cube = np.zeros((9, 12, 120, 140))  # 9 years, 12 months, rows and cols
# par_cube[0:5, :, :, :] = par_cube_temp[0:60, :, :].reshape((5, 12, 120, 140))  # Assign data up to 2004 (60 months)
# par_cube[5, 0:5, :, :] = par_cube_temp[60:65, :, :]  # Assign 2005-01 to 2005-05
# par_cube[6, 5:, :, :] = par_cube_temp[65:72, :, :]  # Assign 2006-06 to 2006-12
# par_cube[7:, :, :, :] = par_cube_temp[72:, :, :].reshape((2, 12, 120, 140))  # Assign 2007 until end

# Get precipitation
precip_cube = np.zeros((9, 12, 120, 140))  # 9 years, 12 months, rows and columns of data

for i, y in enumerate(yrs):
    for j, m in enumerate(months):
        precip_cube[i, j, :, :] = get_trmm(path + 'TRMM_Full/', y, m)

# Remove period from Jan-00 to May-00, June-05 to May-06 and June-08 to Dec-08(with NaN)
precip_cube[0, 0:5, :, :] = np.nan
precip_cube[5, 5:12, :, :] = np.nan
precip_cube[6, 0:5, :, :] = np.nan
precip_cube[8, 5:, :, :] = np.nan

# Calculate mean LAI over the whole area
mean_precip = np.zeros(12)
for i, m in enumerate(months):
    mean_precip[i] = np.nanmean(precip_cube[:, i, :, :])

plt.bar(months, mean_precip*24*30)

#Plot LAI, PAR and precipitation together
fig, ax = plt.subplots()
# Twin x axis to make independent y axes
axes = [ax, ax.twinx(), ax.twinx()]
# Make space on the right side
fig.subplots_adjust(right=0.75)
# Move the last y-axis spine over to the right by 20% of the width of the axes
axes[-1].spines['right'].set_position(('axes', 1.2))

# To make the border of the right-most axis visible, we need to turn the frame
# on. This hides the other plots, however, so we need to turn its fill off.
axes[-1].set_frame_on(True)
axes[-1].patch.set_visible(False)

# Get splines from LAI and PAR
lai_spline = UnivariateSpline(months, np.roll(mean_lai, 7))
par_spline = UnivariateSpline(months, mean_par)

# Create months labels
xlabels = ["Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dec", 'Jan', 'Feb', "Mar", "Apr", "May"]

# And finally we get to plot things...
colors = ('Red', 'Green', 'Black')

axes[0].plot(months, mean_par, marker='o', linestyle='none', color=colors[0])
axes[0].plot(months, par_spline(months), color=colors[0])
axes[1].plot(months, np.roll(mean_lai, 7), marker='s', linestyle='none', color=colors[1])
axes[1].plot(months, lai_spline(months), color=colors[1])
axes[2].bar(months, np.roll(mean_precip, 7)*24*30, fill=False, align='center')

# Set ticks and labels
axes[0].set_xticks(np.arange(1,13))
axes[0].set_xticklabels(xlabels)
axes[0].set_ylabel('PAR ($W m^{-2}$)', color=colors[0])
axes[1].set_ylabel('LAI', color=colors[1])
axes[2].set_ylabel('Precipitation (mm/month)', color=colors[2])
axes[2].set_ylim(0, 800)

for ax, color in zip(axes, colors):
    ax.tick_params(axis='y', colors=color)

plt.savefig('/home/paulo/ge529/drought/figures/precip_lai_par.png')

