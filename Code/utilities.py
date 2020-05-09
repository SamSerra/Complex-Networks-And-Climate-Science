"""
This file is for all utility functions that end up being useful
"""

import time
import sys
import numpy as np

########################################
# NetCDF Tools

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim )
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars

########################################
# Misc

def eta_counter(total_loops, current_loop, prev_time, every=1, system=False):
    '''
    This counts the estimated time to completion for long loops.
    
    Input:
    --------
    total_loops: int
        The total number of loops the loop makes
    current_loop: int
        The current loop index
    every: int
        This function is called every "every" iterations of the loop
    prev_time: float
        The unix time of the last time this function was called
    system: bool
        If true, outputs eta to stdout. Else prints to consol.
        
    Returns:
    --------
    cur_time: float
        Current unix time
    '''
    import time 
    
    cur_time = time.time()
    eta = ((cur_time - prev_time)/every)*(total_loops-current_loop)
    mins, secs = divmod(eta, 60)
    hours, mins = divmod(mins, 60)
    
    if system:
        # std out
        sys.stdout.flush()
        sys.stdout.write("\r Current count: {:n}  Estimate Time Remaining: {:.0f} h {:.0f} m {:.0f} s ".format(current_loop, hours, mins, secs))
    else:
        print("\r", "Current count: {:n} \n Estimate Time Remaining: {:.0f} h {:.0f} m {:.0f} s ".format(current_loop, hours, mins, secs), end="")
        
    return cur_time
    

#########################################################
# Network Analysis Tools

def onion_decomp(A):
    '''
    Finds the onion decomposition of an undirected graph

    Inputs
    --------
    A: (N,N) ndarray 
        The adjacency matrix of the graph.

    Returns
    --------
    cores: (N,) ndarray
        Core number for each vertex as a 1D ndarray.
    layers: (N,) ndarray
        Layer number for each vertex as a 1D ndarray.

    # optional, maybe add later 
    num_cores: int
        Number of cores in the onion decomposition.
    num_layers: int
        Number of layers in the onion decomposition.
    '''
    # make sure A is square
    N = np.sqrt(np.size(A))
    assert(N-np.floor(N) == 0), 'Adjacency matrix must be square!'
    N = int(N)

    # produce diction of vertex:degree pairs
    # (technically the out-degree; axis=0 is the in-degree)
    degrees = np.sum(A, axis=1)
    verticies = dict(enumerate(degrees))
    # list of cores and layers to be filled in
    cores, layers = np.zeros(N), np.zeros(N)

    # first, remove all nodes of degree zero (they are the zeroth layer) to avoid problems with empty lists
    isolated_nodes = [v for v, d in verticies.items() if d == 0]
    for v in isolated_nodes:
        verticies.pop(v)

    curr_core, curr_layer = 1, 1
    while True:
        this_layer = [v for v, d in verticies.items() if d <= curr_layer]
        for v in this_layer:
            # update core/layer of each vertex
            cores[v], layers[v] = curr_core, curr_layer
            # decrease degree of each neighbour of vertex by 1
            neighbours, = np.nonzero(A[v])
            for neighbour in neighbours:
                if neighbour in verticies.keys():
                    verticies[neighbour] -= 1
            # remove current verticies from graph
            verticies.pop(v)

        # update current layer/core if verticies is non-empty
        if len(verticies) == 0:
            break
        curr_layer += 1
        min_deg = min(verticies.values())
        if min_deg >= (curr_core+1):
            curr_core = min_deg
    return cores, layers

def onion_spectrum(layers):
    '''
    Takes in the layers of the onion decomposition and produces the spectrum.

    Inputs
    -------
    layers: 1D ndarray of length N
        An array of layer numbers for each of the N verticies.

    Returns
    --------
    frac_in_layer: 1D ndarray
        For an onion decomposition resulting in m layers, a list of length m whose m-th entry is the fraction of all nodes in layer m. 
    '''
    N = len(layers)
    total_layers = np.arange(np.max(layers)+1)
    frac_in_layer = [(np.sum(layers == layer)/N) for layer in total_layers]

    return total_layers, frac_in_layer
