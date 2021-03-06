#!/miniconda3/bin/python

# This script is a simple example for PEM, modified from Chi Chen (2014/10/13)
# by Paulo Arevalo. It's used to plot daily changing of NPP and GPP
# among three kinds of vegetation, also including regression of NPP and GPP,
# density distribution of NPP and GPP.

import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn import datasets, linear_model


def get_daily_gpp(swrad, fpar, e):
    """ Get daily GPP from short wave radiation SWRAD, FPAR and efficiency of
    APAR conversion to GPP """
    # Convert ipar from J/m2/s to MJ/m2/day
    ipar = 0.45 * swrad * 24 * 60 * 60 / 1000000
    apar = ipar * fpar /100.0
    daily_gpp = e * apar

    return daily_gpp


def get_epsilon(lut, tmin, vpd): #passed as vectors
    """ Get efficiency of APAR conversion to GPP
        Tmin_min is tmin_stop in the lut, same for others
    """
    emax = lut.iloc[0]
    tmin_min = lut.iloc[2]
    tmin_max = lut.iloc[1]
    vpd_min = lut.iloc[4]
    vpd_max = lut.iloc[3]

    e = np.zeros(365)

    for i in range(0, 365):
        # Temperature attenuation scalar
        if tmin[i] < tmin_min:
            tmin_scalar = 0
        elif tmin[i] > tmin_max:
            tmin_scalar = 1
        else:
            _range = tmin_max - tmin_min
            tmin_scalar = (tmin[i] - tmin_min) / _range

        # VPC attenuation scalar
        if vpd[i] > vpd_max:
            vpd_scalar = 0
        elif vpd[i] < vpd_min:
            vpd_scalar = 1
        else:
            _range = vpd_max - vpd_min
            vpd_scalar = (vpd_max - vpd[i]) / _range

        e[i] = emax * tmin_scalar * vpd_scalar

    return e

def get_leaf_mr(lut, lai, tavg):
    """ Get daily leaf maintenance respiration """
    sla = lut[5]
    leaf_mr_base = lut[9]
    q10_mr = lut[6]
    leaf_mass = lai/10/sla
    leaf_mr = leaf_mass * leaf_mr_base * q10_mr ** ((tavg - 20) / 10)

    return leaf_mr

def get_froot_mr(lut, lai, tavg):
    """ Get daily fine root maintenance respiration """
    sla = lut[5]
    froot_mr_base = lut[10]
    q10_mr = lut[6]
    froot_leaf_ratio = lut[7]
    leaf_mass = lai/sla
    fine_root_mass = leaf_mass * froot_leaf_ratio
    froot_mr = fine_root_mass * froot_mr_base * q10_mr ** ((tavg - 20) / 10)

    return froot_mr


def get_live_wood_mr(lut, lai, tavg):
    """ Get live wood maintenance respiration """
    sla = lut[5]
    ann_leaf_mass_max = max(lai / sla)
    livewood_leaf_ratio = lut[8]
    livewood_mr_base = lut[11]
    q10_mr = lut[7]
    annsum_mr_index = sum(q10_mr ** ((tavg - 20) / 10))
    livewood_mass = ann_leaf_mass_max * livewood_leaf_ratio
    livewood_mr = livewood_mass * livewood_mr_base * annsum_mr_index
    
    return livewood_mr


def get_leaf_gr(lut, lai):
    sla = lut[5]
    ann_leaf_mass_max = max(lai / sla)
    ann_turnover_proportion = lut[16]
    leaf_gr_base = lut[12]
    leaf_gr = ann_leaf_mass_max * ann_turnover_proportion * leaf_gr_base

    return leaf_gr


def get_annual_gr(lut, leaf_gr):
    froot_leaf_gr_ratio = lut[13]
    livewood_leaf_gr_ratio = lut[14]
    deadwood_leaf_gr_ratio = lut[15]
    deadwood_gr = leaf_gr * deadwood_leaf_gr_ratio
    livewood_gr = leaf_gr * livewood_leaf_gr_ratio
    froot_gr = leaf_gr * froot_leaf_gr_ratio
    annual_gr = leaf_gr + froot_gr + livewood_gr + deadwood_gr
    return annual_gr


def pem(veg_file, lut_file):
    """ PEM function """
    biome = int(veg_file.iloc[0, 0])
    input = veg_file.iloc[:, 1:7]

    # Get LUT for biome type
    lut = lut_file.iloc[:, biome-1]

    # Get epsilon, and daily GPP to calculate annual GPP
    e = get_epsilon(lut, input.iloc[:, 2], input.iloc[:, 3])
    daily_gpp = get_daily_gpp(input.iloc[:, 0], input.iloc[:, 4], e)
    annual_gpp = sum(daily_gpp)

    # Get Annual Autotrophic Respiration
    # There are two kind of Autotrophic Respiration:1.Maintenance Respiration
    #                                              2.Growth respiration
    # Each respiration consisits of : 1. leaf- daily MR and annual GR
    #                                2. Fine roots - daily MR and annual GR
    #                                3. Live wood - annual MR and GR
    #                                4. Dead wood - no MR, only annual GR

    # 1. Maintenance Respiration
    leaf_mr = get_leaf_mr(lut, input.iloc[:, 5], input.iloc[:, 1])
    froot_mr = get_froot_mr(lut, input.iloc[:, 5], input.iloc[:, 1])
    livewood_mr = get_live_wood_mr(lut, input.iloc[:, 5], input.iloc[:, 1])
    daily_mr = leaf_mr + froot_mr + livewood_mr / 365.0
    annual_mr = sum(leaf_mr) + sum(froot_mr) + livewood_mr

    # 2. Growth respiration
    leaf_gr = get_leaf_gr(lut, input.iloc[:, 5])
    annual_gr = get_annual_gr(lut, leaf_gr)

    # Get daily and annual NPP
    daily_npp = daily_gpp - daily_mr - annual_gr/365.0
    annual_npp = annual_gpp - annual_mr + annual_gr

    return {'daily_npp': daily_npp, 'annual_npp': annual_npp, 'daily_gpp': daily_gpp,
            'annual_gpp': annual_gpp, 'daily_mr': daily_mr, 'annual_mr': annual_mr,
            'annual_gr': annual_gr, 'input': input, 'lut': lut}

################################

os.chdir("/home/paulo/ge529")
lutfile = pd.read_table('./data/LUT.dat', header=0, index_col=0)
# Get biome names, not used yet but could be useful
biome_names = lutfile.columns

#dtypes = {'Biome': np.float, 'SWRad': np.float, 'TAvg' : np.float, 'Tmin': np.float, 'VPD' : np.float,
#        'Fpar' : str, 'LAI' : str} Not working yet
vegtype2 = pd.read_table('./data/r94c116-2.dat', header=0, index_col=False) # dtype=dtypes)
vegtype9 = pd.read_table('./data/r115c310-9.dat', header=0, index_col=False)
vegtype11 = pd.read_table('./data/r40c220-11.dat', header=0, index_col=False)


def fix_tables(vegfile):
    """ Fix dtypes and missing values"""
    # Fix dtype for LAI and Fpar
    vegfile['LAI'] = pd.to_numeric(vegfile['LAI'], errors='coerce')
    vegfile['Fpar'] = pd.to_numeric(vegfile['Fpar'], errors='coerce')
    # Replicate monthly mean to daily
    month = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    # fpar and lai are monthly, make them daily. Reset index, needed for updating
    fpar_lai = vegfile.iloc[:12, 5:7] # Separate table (temporary to make it work)
    input = vegfile.iloc[:, 0:5] # Separate table (temporary to make it work)
    fpar_lai_rep = fpar_lai.loc[np.repeat(fpar_lai.index.values, month)].reset_index(drop=True)
    # # Join data with replicated fpar_lai
    input = input.join(fpar_lai_rep)
    # # Scale LAI
    input['LAI'] /= 10

    return input

vegtype2 = fix_tables(vegtype2)
vegtype9 = fix_tables(vegtype9)
vegtype11 = fix_tables(vegtype11)

# Get results

EBF = pem(vegtype2, lutfile)
OSL = pem(vegtype9, lutfile)
CL = pem(vegtype11, lutfile)

# PLOTS 1) Density comparison

def kde_sklearn(x, x_grid, bandwidth, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

# Obtain absolute min and max for the plot x axis (NPP)
npp_min_list = np.array((EBF['daily_npp'].min(), OSL['daily_npp'].min(), CL['daily_npp'].min()))
npp_max_list = np.array((EBF['daily_npp'].max(), OSL['daily_npp'].max(), CL['daily_npp'].max()))
npp_min = npp_min_list.min()
npp_max = npp_max_list.max()

extra_space = (npp_max - npp_min) * 0.2
x_grid = np.linspace(npp_min - extra_space, npp_max + extra_space, 1000)
bandwidth = 0.0001

# Density comparison (GPP)
fig, ax = plt.subplots()
ax.plot(x_grid, kde_sklearn(EBF['daily_npp'], x_grid, bandwidth=bandwidth),
        label='EBF', linewidth=3, alpha=0.5)
#ax.hist(EBF['daily_npp'], 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.plot(x_grid, kde_sklearn(OSL['daily_npp'], x_grid, bandwidth=bandwidth),
        label='OSL', linewidth=3, alpha=0.5)
#ax.hist(OSL['daily_npp'], 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.plot(x_grid, kde_sklearn(CL['daily_npp'], x_grid, bandwidth=bandwidth),
        label='CL', linewidth=3, alpha=0.5)
#ax.hist(CL['daily_npp'], 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.legend(loc='upper right')
ax.set_ylabel('Modeled Density')
ax.set_xlabel('Daily NPP (kg C/m2/day)')
plt.title('Daily NPP distribution by vegetation types')

# OR USE SEABORN AND AVOID ALL THAT CODE, sigh.....!!

ax = sns.distplot(EBF['daily_npp'])
sns.distplot(OSL['daily_npp'])
sns.distplot(CL['daily_npp'])
ax.set_ylabel('Modeled Density')
ax.set_xlabel('Daily NPP (kg C/m2/day)')
plt.title('Daily NPP distribution by vegetation types')
plt.legend(['EBF', 'OSL', 'CL'])
plt.savefig('./figures/density_compare_npp.png', bbox_inches='tight', dpi=300)


# Obtain absolute min and max for the plot x axis (GPP)
gpp_min_list = np.array((EBF['daily_gpp'].min(), OSL['daily_gpp'].min(), CL['daily_gpp'].min()))
gpp_max_list = np.array((EBF['daily_gpp'].max(), OSL['daily_gpp'].max(), CL['daily_gpp'].max()))
gpp_min = gpp_min_list.min()
gpp_max = gpp_max_list.max()

extra_space = (gpp_max - gpp_min) * 0.2
x_grid = np.linspace(gpp_min - extra_space, gpp_max + extra_space, 1000)
bandwidth = 0.0001

# Density comparison (GPP)
fig, ax = plt.subplots()
ax.plot(x_grid, kde_sklearn(EBF['daily_gpp'], x_grid, bandwidth=bandwidth),
        label='EBF', linewidth=3, alpha=0.5)
#ax.hist(EBF['daily_gpp'], 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.plot(x_grid, kde_sklearn(OSL['daily_gpp'], x_grid, bandwidth=bandwidth),
        label='OSL', linewidth=3, alpha=0.5)
#ax.hist(OSL['daily_gpp'], 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.plot(x_grid, kde_sklearn(CL['daily_gpp'], x_grid, bandwidth=bandwidth),
        label='CL', linewidth=3, alpha=0.5)
#ax.hist(CL['daily_gpp'], 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.legend(loc='upper right')
ax.set_ylabel('Modeled Density')
ax.set_xlabel('Daily GPP (kg C/m2/day)')
plt.title('Daily GPP distribution by vegetation types')

# Or use seaborn
ax = sns.distplot(EBF['daily_gpp'])
sns.distplot(OSL['daily_gpp'])
sns.distplot(CL['daily_gpp'])
ax.set_ylabel('Modeled Density')
ax.set_xlabel('Daily GPP (kg C/m2/day)')
plt.title('Daily GPP distribution by vegetation types')
plt.legend(['EBF', 'OSL', 'CL'])
plt.savefig('./figures/density_compare_gpp.png', bbox_inches='tight', dpi=300)


# PLOTS 2) Daily values (MR|NPP|GPP) vs Date

def daily_plotter(veg1, veg2, veg3, feature, feature_label):
    """ Plot daily values of MR, NPP or GPP for the three vegetation types"""
    fig, ax = plt.subplots()
    ax.plot(eval(veg1)[feature], label=veg1)
    ax.plot(eval(veg2)[feature], label=veg2)
    ax.plot(eval(veg3)[feature], label=veg3)
    ax.set_xlim(0,365)
    ax.legend(loc='upper right')
    ax.set_ylabel('Daily ' + feature_label + ' (kg C/m2/day)')
    ax.set_xlabel('Day of year')
    plt.savefig('./figures/' + feature + '.png', bbox_inches='tight', dpi=300)

daily_plotter('EBF', 'OSL', 'CL', 'daily_mr', 'MR')
daily_plotter('EBF', 'OSL', 'CL', 'daily_npp', 'NPP')
daily_plotter('EBF', 'OSL', 'CL', 'daily_gpp', 'GPP')

# PLOTS 3) GPP VS NPP
regr = linear_model.LinearRegression()
regr.fit(EBF['daily_npp'].reshape(365, 1), EBF['daily_gpp'].reshape(365, 1))
fig, ax = plt.subplots()
ax.scatter(EBF['daily_npp'], EBF['daily_gpp'])
ax.plot(EBF['daily_npp'], regr.predict(EBF['daily_npp'].reshape(365, 1)), color='blue',
         linewidth=3)
ax.set_xlim(npp_min_list[0], npp_max_list[0])
ax.set_ylim(gpp_min_list[0], gpp_max_list[0])
ax.set_ylabel('Daily GPP (kg C/m2/day)')
ax.set_xlabel('Daily NPP (kg C/m2/day)')
plt.title('NPP vs GPP, EBF')

# OR USING SEABORN...Regression plot, or joint plot
sns.regplot(EBF['daily_npp'], EBF['daily_gpp'])

sns.jointplot(EBF['daily_npp'], EBF['daily_gpp'], kind='reg')
plt.savefig('./figures/npp_vs_gpp_EBF.png', bbox_inches='tight', dpi=300)
sns.jointplot(OSL['daily_npp'], OSL['daily_gpp'], kind='reg')
plt.savefig('./figures/npp_vs_gpp_OSL.png', bbox_inches='tight', dpi=300)
sns.jointplot(CL['daily_npp'], CL['daily_gpp'], kind='reg')
plt.savefig('./figures/npp_vs_gpp_CL.png', bbox_inches='tight', dpi=300)

# Changing some variables to compare the results

def change_input(input, lut_file):
    """ Change input to get results with a range of values"""

    # Create arrays to store results
    annual_npp = np.zeros((101, 5))
    annual_gpp = np.zeros((101, 5))

    # Loop over different values, find a more efficient way to do this
    for i in range(1, 102):
        # LAI
        input2 = input.copy() # NEEDED in pandas to avoid modifying the original
        input2['LAI'] *= (1 + (i - 1) / 100)
        pem_output = pem(input2, lut_file)
        annual_npp[i - 1, 4] = pem_output['annual_npp']
        annual_gpp[i - 1, 4] = pem_output['annual_gpp']
        # Fpar
        input2 = input.copy()
        input2['Fpar'] *= (1 + (i - 1) / 100)
        pem_output = pem(input2, lut_file)
        annual_npp[i - 1, 3] = pem_output['annual_npp']
        annual_gpp[i - 1, 3] = pem_output['annual_gpp']
        # VPD
        input2 = input.copy()
        input2['VPD'] *= (1 + (i - 1) / 100)
        pem_output = pem(input2, lut_file)
        annual_npp[i - 1, 2] = pem_output['annual_npp']
        annual_gpp[i - 1, 2] = pem_output['annual_gpp']
        # Tavg
        input2 = input.copy()
        input2['Tavg'] *= (1 + (i - 1) / 100)
        pem_output = pem(input2, lut_file)
        annual_npp[i - 1, 1] = pem_output['annual_npp']
        annual_gpp[i - 1, 1] = pem_output['annual_gpp']
        # SWRad
        input2 = input.copy()
        input2['SWRad'] *= (1 + (i - 1) / 100)
        pem_output = pem(input2, lut_file)
        annual_npp[i - 1, 0] = pem_output['annual_npp']
        annual_gpp[i - 1, 0] = pem_output['annual_gpp']

    return {'annual_npp': annual_npp, 'annual_gpp': annual_gpp, 'input': input, 'input2': input2}


EBF2 = change_input(vegtype2, lutfile)
OSL2 = change_input(vegtype9, lutfile)
CL2 = change_input(vegtype11, lutfile)

# Make plots of those results

def change_input_plotter(input, value, vegname):
    """ Plots the trajectories of GPP and NPP for multiple values of input variables, for
        each of the types of vegetation analyzed"""
    fig, ax = plt.subplots()
    for i in range(0, 5):
        ax.plot(input['annual_' + value][:, i])

    ax.set_ylabel('Annual ' + value + ' (kg C/m2/day)')
    ax.set_xlabel('Percent of a given parameter')
    ax.legend(['SWRad', 'Tavg', 'VPD', 'Fpar', 'LAI'], loc='upper left')
    plt.title('Annual ' + value + ' variation with a range of parameter values - ' + vegname)
    plt.savefig('./figures/' + value + '_param_variations_' + vegname + '.png', bbox_inches='tight', dpi=300)

change_input_plotter(EBF2, 'npp', 'EBF')
change_input_plotter(OSL2, 'npp', 'OSL')
change_input_plotter(CL2, 'npp', 'CL')
change_input_plotter(EBF2, 'gpp', 'EBF')
change_input_plotter(OSL2, 'gpp', 'OSL')
change_input_plotter(CL2, 'gpp', 'CL')

# Zoom into VPD curve
ax.set_ylabel('Annual GPP (kg C/m2/day)')
ax.set_xlabel('Percent of VPD')
plt.plot(EBF2['annual_gpp'][:, 2])
plt.title('Zoom into VPD curve - annual GPP - EBF')
plt.savefig('./figures/vpd_zoom_EBF.png', bbox_inches='tight', dpi=300)

