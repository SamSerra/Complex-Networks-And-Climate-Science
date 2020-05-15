'''
This is a test to see if python's multiprocessing library can handle simultaneous reads from a netcdf4 file. 
'''
from netCDF4 import Dataset
import numpy as np
import multiprocessing

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
nc_file = Dataset('/home/samserra/Projects/ComplexNetworksAndClimateScience/Data/parallelprocesstest.nc', mode='r+')
data = nc_file.variables['data']
data_sum = nc_file.variables['sum']

# parallel processing
def quadrant_sum(array, dim0, dim1):
    '''
    Takes in a 2d array, returns 2x2 array where each entry is the sum of that quadrant
    0,0 |0,1
    --------
    1,0 |1,1
    '''
    ndim = int(array.shape[0]/2)
    quad_sum = np.sum(array[dim0*2:ndim+dim0*2,dim1*2:ndim+dim1*2])
    print(quad_sum)
    return quad_sum
    #data_sum[dim0, dim1] = quad_sum


if __name__ == '__main__':
    p = multiprocessing.Pool(4)
    results = [p.apply_async(quadrant_sum, args=(data, i,j)) for i in [0,1] for j in [0,1]]
    for res in results:
       print(res.get())
nc_file.close()
