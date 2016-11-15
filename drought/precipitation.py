import numpy as np
import gdal
from matplotlib import pyplot as plt

# Script to plot the time series of mean dry-season precipitation
# and anomalies

#path = '/projectnb/landsat/users/parevalo/ge529/3B43'
path = '/home/paulo/ge529/raster/data'

years = range(2002, 2012)

def get_image(dir, year, month):
    """ Read images as array"""

    if year < 2011:
        image = dir + '/3B43.' + str(year) + '0' + str(month) + '01.7A_clip.tif'
    else:
        image = dir + '/3B43.' + str(year) + '0' + str(month) + '01.7_clip.tif'
    
    img_open = gdal.Open(image)
    img_array = img_open.GetRasterBand(1).ReadAsArray()
    
    return img_array

def calc_mean(a1, a2, a3):
    """ Calculate mean for dry season months"""
    dry_months = np.array((a1, a2, a3))
    mean = np.mean(dry_months, axis=0)
    return mean
   
def calc_anomaly(period, year):
    """ calculates anomaly for a given year in the period"""
    index = year-2002
    new_period = np.delete(period, index, axis=2)
    mean = np.mean(new_period, axis=2)
    std = np.std(new_period, axis=2)
    anomaly = (new_period[:,:,index] - mean) / std
    
    return anomaly

def save_img(array, year, suffix):
    """ Save image of year precipitation or anomaly"""
    outfile = '/home/paulo/ge529/drought/figures/' + str(year) + suffix
    plt.imshow(array)
    plt.savefig(outfile)

# Get images and do the calculations

precip = np.zeros((120, 140, 10))
for i, y in enumerate(years):
    jul = get_image(path, y, 7)
    ago = get_image(path, y, 8)
    sept = get_image(path, y, 9)

    precip[:, :, i] = calc_mean(jul, ago, sept)

anomaly_08 = calc_anomaly(precip, 2008)
anomaly_10 = calc_anomaly(precip, 2010)

save_img(anomaly_08, 2008, '_anomaly.png')
save_img(anomaly_10, 2010, '_anomaly.png')


plt.imshow(anomaly_08)

