import numpy as np
import gdal
import scipy.io as sio
from sklearn import preprocessing
from matplotlib import pyplot as plt

# Script to plot the time series of mean dry-season precipitation
# and anomalies

#path = '/projectnb/landsat/users/parevalo/ge529/3B43'
path = '/home/paulo/ge529/raster/data/'

years = range(2002, 2012)

def get_trmm(dir, year, month):
    """ Read TRMM as array"""

    if year < 2011:
        image = dir + '3B43.' + str(year) + '0' + str(month) + '01.7A_clip.tif'
    else:
        image = dir + '3B43.' + str(year) + '0' + str(month) + '01.7_clip.tif'
    
    img_open = gdal.Open(image)
    img_array = img_open.GetRasterBand(1).ReadAsArray()
    
    return img_array

def get_modis(dir, year):
    """ Read MCD12C1 as array from tif files"""

    import fnmatch
    for file in os.listdir(dir):
        string = 'MCD12C1*' + str(year) + '*.tif'
        if fnmatch.fnmatch(file, string):
            img_open = gdal.Open(dir + file)
            img_array = img_open.GetRasterBand(1).ReadAsArray()

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
    cax = ax.imshow(array, cmap=plt.get_cmap('viridis'))
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

plt.figure(1)
plt.hist(masked_precip10, weights=scaled_wt_10)
plt.ylabel('Area weighted frequency')
plt.xlabel('Precipitation (mm $h^{-1}$)')
plt.title("Area weighted histogram for precipitation in dry months, 2008")

# Plot TS for entire region?
scaled_wt = minmax_scaler.fit_transform(wt)
annual_mn_precip = np.zeros(10)
for i in range(10):
    annual_mn_precip[i] = np.average(precip[:, :, i], weights=scaled_wt)

plt.plot(annual_mn_precip)  # With weighting
plt.plot(precip.mean(axis=0).mean(axis=0))  # Without weighting

