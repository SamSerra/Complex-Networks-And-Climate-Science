'''
This is a test to see if python's multiprocessing library can handle simultaneous reads from a netcdf4 file. 
'''
from netCDF4 import Dataset
import numpy as np
import multiprocessing
import xarray as xr
import dask
import dask.array as da
from dask.distributed import Client
import time

#client = Client(n_workers=4, processes=True)

# create a netcdf file that consists of a 4x4 array of numbers
'''
test = np.array([[1,0,1,1],[0,0,0,0],[1,1,1,1],[1,0,1,1]])
nc_file = Dataset('/home/samserra/Projects/ComplexNetworksAndClimateScience/Data/parallelprocesstest.nc', mode='w', format='NETCDF4')
nc_file.createDimension('4',4)
nc_file.createDimension('2',2)
data = nc_file.createVariable('data','i',dimensions=('4','4'))
data[:,:]=test
data_sum = nc_file.createVariable('sum', 'i1', dimensions=('2','2')) # array to store sum of quad    
'''
# get netcdf file
nc_file = Dataset('/home/samserra/Projects/ComplexNetworksAndClimateScience/Data/parallelprocesstest.nc', mode='r')
#nc_file = xr.open_dataset('/home/samserra/Projects/ComplexNetworksAndClimateScience/Data/parallelprocesstest.nc',chunks={})
data = nc_file.variables['data'][:]
data = dask.delayed(data)

# parallel processing
def quadrant_sum(array, dim0, dim1):
    '''
    Takes in a 2d array, returns 2x2 array where each entry is the sum of that quadrant
    0,0 |0,1
    --------
    1,0 |1,1
    '''
    ndim = int(array.shape[0]/2)
    quad_sum = np.sum(array[dim0*ndim:ndim+dim0*ndim,dim1*ndim:ndim+dim1*ndim])
    #quad_sum = array[dim0,dim1] + array[dim1,dim0]
    time.sleep(1)
    return quad_sum

results = []
for i in np.arange(2):
    for j in np.arange(2):
        results.append(dask.delayed(quadrant_sum)(data, i, j))

results = dask.compute(results,  scheduler='processes')
np.save('/home/samserra/Projects/ComplexNetworksAndClimateScience/Data/parallelprocesstestsum', results)
print(results)

nc_file.close()
