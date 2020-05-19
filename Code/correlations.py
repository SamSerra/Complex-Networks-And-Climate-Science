'''
This file is used for calculating correlations. It's bascially just a copy-paste of the first 6 sections from the 'Recreating Tsonis et Al' notebooks. 

'''

from netCDF4 import Dataset
import numpy as np
import datetime as dt
from scipy.stats import pearsonr
import sys
import time
import utilities as utils
import dask
import os

root_dir = '/home/samserra/Projects/ComplexNetworksAndClimateScience'
file_name = input("Filename to write to in 'Output' Directory: \n")

#####################################
## 1. Import and filter data

# Most recent NCAR Reanalysis File (as of 2020-02-29)
reanalysis = Dataset(root_dir + '/Data/NCEP-NCAR Reanalysis (R1)/Monthly Surface Temperatures/air.mon.mean.nc', mode='r')

# extract all longitudes, latitudes, and times
lons = reanalysis.variables['lon'][:]
lats = reanalysis.variables['lat'][:]
times = reanalysis.variables['time'][:]

# date format is hours since 1800-01-01, so convert to normal date format
dates_orig = np.array(
    [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in times])

# Tsonis et al only uses data from Nov-Mar to avoid seasonal variability. However, Donges et al does not, claiming it does not affect their results.
# If the bool 'nov_to_march' is 'True',only data from Nov-Mar will be used. Else all data will be used
nov_to_march = True
if nov_to_march:
    # create mask for nov-march date range
    mask = []
    for d in dates_orig:
        # true if d.month is Jan-Mar or Nov-Dec
        mask.append((d.month <= 3) | (d.month >= 11))
    air = reanalysis.variables['air'][mask, :, :]
    dates = dates_orig[mask]
else:
    air = reanalysis.variables['air'][:]

#####################################
## 2. Produce Anomaly Values

# look at the month of the first 12 entries of 'dates' to get set of all valid months
valid_months = []
for d in dates[:12]:
    valid_months.append(d.month)
valid_months = set(valid_months)

clim_averages = np.zeros(air.shape)
for mon in valid_months:  # for each month,
    # create month mask
    mask = []
    for d in dates:
        mask.append(d.month == mon)

    # produce climatalogical averages for every lat/lon pair for the current month
    clim_averages[mask, :, :] = np.mean(air[mask, :, :], axis=0)

# produce anomaly values
air_anom = air - clim_averages

#####################################
## 3. Classify years as El Nino or Not 

soi = np.loadtxt(root_dir + "/Data/soi.txt", skiprows=88,
                 max_rows=70, usecols=np.arange(1, 13))
# each column is a month jan-dec, so flatten to get 1-d list of all dates
soi = np.ravel(soi)
# last 10 months are nonsense, and reanalysis only goes to february, so chop last 11 months off
soi = soi[:-11]
# finaly, mask data to match previous filtering
mask = []
for d in dates_orig[3*12:]:
    # true if d.month is Jan-Mar or Nov-Dec
    mask.append((d.month <= 3) | (d.month >= 11))
soi = soi[mask]

# generate mask for El Nino months
elnino_months = soi > 1
# generate mask for La Nina months
lanina_months = soi < -1
# generate mask for normal months
normal_months = (1-(elnino_months+lanina_months)
                 ).astype('bool')  # not(elnino or lanina)

# cut off first 3 years of SAT data to match soi
air_anom_trimed = air_anom[3*len(valid_months):, :, :]
dates_trimed = dates[3*len(valid_months):]

# subtract off 14 years of valid months plus Jan 2020 (second test)

air_anom_trimed = air_anom_trimed[:-(14*len(valid_months)+1)]
dates_trimed = dates_trimed[:-(14*len(valid_months)+1)]
elnino_months = elnino_months[:-(14*len(valid_months)+1)]
lanina_months = lanina_months[:-(14*len(valid_months)+1)]
normal_months = normal_months[:-(14*len(valid_months)+1)]


#####################################
## 4. Produce time series

# seperate SAT into El Nino, La Nina, and Normal
air_anom_elnino = air_anom_trimed[elnino_months, :, :]
air_anom_lanina = air_anom_trimed[lanina_months, :, :]
air_anom_normal = air_anom_trimed[normal_months, :, :]
# and the dates
dates_elnino = dates_trimed[elnino_months]
dates_lanina = dates_trimed[lanina_months]
dates_normal = dates_trimed[normal_months]

# make all objects delayed
air_anom_elnino = dask.delayed(air_anom_elnino)
air_anom_lanina = dask.delayed(air_anom_lanina)
air_anom_normal = dask.delayed(air_anom_normal)
dates_elnino    = dask.delayed(dates_elnino)
dates_lanina    = dask.delayed(dates_lanina)
dates_normal    = dask.delayed(dates_normal)


#####################################
## 5. Correlations!

# First create netcdf file to store correlations in so we aren't storing them in memory

# number of nodes
N = (len(lats)-2)*len(lons)

try:
    tsonCorRec = Dataset(
        root_dir+"/Output/" + file_name + ".nc", mode='r+', format='NETCDF4')
    cors_elnino = tsonCorRec.groups['elNino'].variables['correlations']
    cors_lanina = tsonCorRec.groups['laNina'].variables['correlations']
    adj_elnino = tsonCorRec.groups['elNino'].variables['adjacency']
    adj_lanina = tsonCorRec.groups['laNina'].variables['adjacency']
    
except FileNotFoundError:
    tsonCorRec = Dataset(
        root_dir + "/Output/" + file_name + ".nc", mode='w', format='NETCDF4')
    tsonCorRec.createGroup("laNina")
    tsonCorRec.createGroup("elNino")
    
    tsonCorRec.createDimension("Number of Nodes", N)
    
    cors_lanina = tsonCorRec.createVariable(
        "laNina/correlations", "f4", dimensions=("Number of Nodes", "Number of Nodes"), zlib=True, least_significant_digit=3)
    cors_elnino = tsonCorRec.createVariable(
        "elNino/correlations", "f4", dimensions=("Number of Nodes", "Number of Nodes"), zlib=True, least_significant_digit=3)
    adj_elnino = tsonCorRec.createVariable('elNino/adjacency', 'i', dimensions=(
    "Number of Nodes", "Number of Nodes"), zlib=True, least_significant_digit=0)
    adj_lanina = tsonCorRec.createVariable('laNina/adjacency', 'i', dimensions=(
    "Number of Nodes", "Number of Nodes"), zlib=True, least_significant_digit=0)


# create list of lat/lon indicies
indicies = [(lat_idx, lon_idx) for lat_idx in np.arange(1, len(lats)-1)
            for lon_idx in np.arange(len(lons))]

indicies = dask.delayed(indicies)

## subroutines for computing correlations/adj matricies to avoid large memory usage

#need to parallize with dask
def calc_correlations(SATdata_array):#, output_array):
    '''
    Takes in SAT air data, calculates pairwise pearson correlations for the upper triangle, and saves temp file to disk.

    Will update to be delayed. The output will be a python list which I will need to reshape using a different function. The list will be temporarily stored on disk to avoid memory overflow (shouldn't be an issue with file size)
    '''
    
    @dask.delayed
    def newpearson(*args):
        c, _ = pearsonr(args)
        return c
    
    # create matrix for storing computations in memory before writing them to netcdf file
    temp_data_array = []#np.zeros((N, N))

    count = 0
    prev_time = time.time()
    print('Creating delayed list')
    for i in np.arange(N):
        for j in np.arange(i+1):
            i_lat, i_lon = indicies[i][0], indicies[i][1]
            j_lat, j_lon = indicies[j][0], indicies[j][1]
            temp_data_array.append(newpearson(SATdata_array[:, i_lat, i_lon], SATdata_array[:, j_lat, j_lon]))
            
            # ETA bar: update every 100,000 iterations
            count += 1
            if count % 100000 == 0:
                prev_time = utils.eta_counter(N*(N-1)*.5, count, prev_time, every=100000, system=False)
            
    print('Finished delayed list. Computing...')
    # save list to disk
    temp_data_array = dask.compute(temp_data_array)
    print('Finished computing. Saving...')
    np.save(root_dir + '/Output/' + file_name + '-temp', temp_data_array)
    print('Done saving.')
    
def list_to_array(output_array):
    '''
    Reads list from disk, converts to uppper triangular array, symmetrizes it, and deletes the list from disk.
    '''
    temp_list = np.load(root_dir + file_name + '-temp.npy')

    # set UT of temp_array to list
    temp_array = np.zeros((N,N))
    indices_ut = np.triu_indicies(N)
    temp_array[indices_ut] = temp_list
    
    # symmetrize matrix 
    temp_data_array[:,:] = (np.transpose(temp_data_array) + temp_data_array)
    
    # fix doubling of diagonal 
    for i in np.arange(N):
        temp_data_array[i,i] /= 2
        
     # save temp_data_array to output_array
    output_array[:,:] = temp_data_array

    # delete temp list from disk
    try:
        os.remove(root_dir + file_name + '-temp.npy')
    except:
        print('Cannot delete temporary list file')
        
def calc_adj_matrix(corr_array_ref, output_array, sig=.5):
    '''
    corr_array_ref is a reference to a netcdf variable
    '''
    cors = corr_array_ref[:,:]
    output_array[:,:] = (np.abs(cors) > sig).astype('int')
    

# correlations, make array, then calc adjacency matricies

#elnino
calc_correlations(air_anom_elnino)
list_to_array(cors_elnino)
calc_adj_matrix(cors_elnino, adj_elnino)

#lanina
'''
calc_correlations(air_anom_lanina)
list_to_array(cors_lanina)
calc_adj_matrix(cors_lanina, adj_lanina)
'''

reanalysis.close()
