# Functions for reading in a folder of VTK files.
# WARNING: Use with care! Many functions contain custom code for one dataset and are not universal!

import os
import math

import numpy as np
import torch
from skimage.transform import resize

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .normalization import *



# MAIN FUNCTION
def load_dataset(path, path_geometry=None, bounds=None, rotate=False, scaling=False, masking=True, dim=3, spheres=False):
    """
    Loads a dataset from a folder of vti files.
    Please use this function only to create H5PY datasets.

    Some datasets have their geometries separate from the simulation results,
    thus a function needs to be provided to calculate the geometry path from the simulation file name.
    
    Args:
        path (str): Path to the dataset.
        path_geometry (str, optional): Path to the geometry data.
        bounds (tuple, optional): Bounds for normalization.
        rotate (bool, optional): Whether to rotate the dataset.
        scaling (bool, optional): Whether to scale the dataset.
        masking (bool, optional): Whether to create a mask.
        dim (int, optional): Dimensionality of the dataset (2 or 3).
        spheres (bool, optional): Whether the dataset contains spheres.
    
    Returns:
        tuple: (invar, outvar, mask, names, bounds)
    """

    spheres = True

    size = len(os.listdir(path))
    if rotate:
        size = 4 * size

    if scaling:
        if spheres:
            shape = (size, 1, 128, 64, 64)
        else:
            shape = (size, 1, 96, 64, 64)
    else:
        if spheres:
            shape = (size, 1, 256, 128, 128)
        else:
            shape = (size, 1, 190, 126, 126)
    if dim == 2:
        shape = (size, 1, 320, 256)

    invar = {"fill": np.empty(shape, dtype=np.float32)}
    outvar = {"p" : np.empty(shape, dtype=np.float32)}


    # iterate every file in folder
    names = []
    counter = 0
    for file in os.listdir(path):

        if dim == 2:
            name = os.path.basename(file).split('_')
            sample = int(name[2])
        
            '''if sample < 2000:
                continue
            if counter >= size:
                continue'''

        names.append(os.path.basename(file) + str(0))


        if not file.endswith('.vti'):
            continue

        #calculate path
        if not spheres:
            geometry_file, density_key = get_geometry_path(file, dim)
            dict_geometry = load_VTI(path_geometry + "/" + geometry_file, scaling, cells=True, swapZX=(dim==3), shape=shape[2:])
        dict_simulation = load_VTI(path + "/" + file, scaling, cells=True, swapZX=(dim==3 and not spheres), shape=shape[2:])
        print("loaded file: ", (counter + 1), "/", size)

        if spheres:
            invar["fill"][counter] = dict_simulation["OverlapFraction"]
            outvar["p"][counter] = dict_simulation["Density"]
        else:
            invar["fill"][counter] = dict_geometry['NoSlip']
            outvar["p"][counter] = dict_simulation[density_key]
        
        counter = counter + 1

        # rotate for increased dataset
        if rotate:
            for i in range(3):
                if spheres:
                    invar["fill"][counter] = np.rot90(dict_simulation["OverlapFraction"], k=1, axes=(-2,-1))
                    outvar["p"][counter] = np.rot90(dict_simulation["Density"], k=1, axes=(-2,-1))
                else:
                    invar["fill"][counter] = np.rot90(dict_geometry['NoSlip'], k=1, axes=(-2,-1))
                    outvar["p"][counter] = np.rot90(dict_simulation[density_key], k=1, axes=(-2,-1))
                counter = counter + 1
                names.append(os.path.basename(file) + str(i))

    
    # calculate mask for lambda weighting using fill
    if masking:
        mask = create_mask(invar["fill"])
    else:
        mask = None


    # normalization
    if spheres:
        pow = 1
        scale = 150
        offset = 1
        outvar["p"], min_pow, max_pow = normalize_old(outvar["p"], scale, offset, pow, True)    # scale default: 150
    else:
        pow = 0.5
        scale = 1
        offset = 0
        outvar["p"], min_pow, max_pow = normalize_new(outvar["p"], pow)

    invar["fill"], _, _ = normalize_new(invar["fill"], 1)
    bounds = [offset, scale, pow, min_pow, max_pow]

    return (invar, outvar, mask, np.array(names), bounds)



# HELPER FUNCTIONS
def apped_to_dict(dict, invar, outvar):
    """
    Appends data to the input and output dictionaries.
    
    Args:
        dict (dict): Dictionary containing data to be appended.
        invar (dict): Input dictionary.
        outvar (dict): Output dictionary.
    
    Returns:
        tuple: (invar, outvar)
    """

    invar["fill"] = np.append(invar["fill"], dict["OverlapFraction"], axis=0)
    outvar["p"] = np.append(outvar["p"], dict["Density"], axis=0)

    return (invar, outvar)


def scale_grid(grid, shape):
    """
    Scales the input grid to the specified shape with anti-aliasing.
    
    Args:
        grid (numpy.ndarray): Input grid.
        shape (tuple): Target shape.
    
    Returns:
        numpy.ndarray: Scaled grid.
    """
    new_shape = (grid.shape[0], grid.shape[1], *shape)
    new_grid = resize(grid, new_shape, anti_aliasing=True, preserve_range=True)

    return new_grid


def create_mask(fill):
    """
    Creates a mask based on the input 'fill' data.
    The first 12 layers are highlighted, the rest is in range [0.1,0.9] dependent on fluid domain.
    
    Args:
        fill (numpy.ndarray): Input 'fill' data.
    
    Returns:
        dict: Mask dictionary.
    """

    mask = {"p": np.empty(fill.shape, dtype=np.float32)}

    mask["p"] = (1. - fill[:]) * 0.8 + 0.1

    for i in range(fill.shape[0]):
        mask["p"][i][0][0:12][:][:] = 1

    return mask



# IO
def load_VTI(path, scaling, cells=True, swapZX=False, shape=(96, 64, 64)):
    """
    Loads a VTI file and returns a dictionary of the data.
    
    Args:
        path (str): Path to the VTI file.
        scaling (bool): Whether to scale the data.
        cells (bool, optional): Whether to use cell data or point data.
        swapZX (bool, optional): Whether to swap the Z and X axes.
        shape (tuple, optional): Target shape for the data.
    
    Returns:
        dict: Dictionary of the loaded data.
    """

    if not path.endswith(".vti"):
        raise Exception(
            ".vti file required"
        )
    
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()

    dims = data.GetDimensions()
    if cells:
        dims = tuple((x - 1) for x in dims)
    if len(shape) == 2:
        dims = dims[0:2]
    array_dict = {}

    if cells:
        cell_data = data.GetCellData()
    else:
        cell_data = data.GetPointData()


    for i in range(cell_data.GetNumberOfArrays()):
        # get matadata
        array_name = cell_data.GetArrayName(i)
        vtk_array = cell_data.GetArray(i)

        #load np array
        numpy_array = vtk_to_numpy(vtk_array)

        # reshape in dimension
        dims_current = dims
        if len(numpy_array.shape) != 1:
            numpy_array = np.swapaxes(numpy_array, 0, 1)
            dims_current = (1, 3,) + dims
        else:
            dims_current = (1, 1,) + dims
            

        # Using Fortran ordering
        numpy_array = numpy_array.reshape(dims_current, order='F')

        if swapZX:
            numpy_array = np.swapaxes(numpy_array, 4, 2)

        if scaling is True:
            array_dict[array_name] = scale_grid(numpy_array, shape)
        else:
            array_dict[array_name] = numpy_array

    return array_dict


def get_geometry_path(filename, dim):
    """
    Retrieves the path of the geometry VTI file from the simulation result file name.
    
    Args:
        filename (str): Name of the simulation result file.
        dim (int): Dimensionality of the dataset (2 or 3).
    
    Returns:
        tuple: (geometry_name, density_name)
    """
    name = filename.split('_')

    if dim == 3:
        number = int(name[1])

        rest = False
        if name[2] == "rest":
            rest = True
        reverse = False
        if name[2] == "reverse" or (len(name) > 3 and name[3] == "reverse"):
            reverse = True

        #calculate geometry name
        if not reverse:
            geometry_name = "geoemtry_" + str(number)
            if rest:
                geometry_name += "_rest"
        else:
            geometry_name = "geometry" + str(number)
            if rest:
                geometry_name += "_rest"
            geometry_name += "_reverse"
        geometry_name += ".vti"


        density_name = "porous_" + str(number)
        if rest:
            density_name += "_rest"
        if reverse:
            density_name += "_reverse"
        density_name += "_density"
        return geometry_name, density_name
    
    else:
        number = int(name[2])
        geometry_name = "2D_geometry_" + str(number) + ".vti"
        density_name = "2D_porous" + str(number) + "_density"
        return geometry_name, density_name



if __name__ == "__main__":

    from dataWriter import saveDictToVTK, saveH5PY
    import h5py
    import numpy as np

    rotate = True

    invar, outvar, _, names, bounds = load_dataset(
        "/home/vault/unrz/unrz109h/porous_media_data/spheres/simulation_files/validation/shifted",
        None,
        rotate=rotate, scaling=True, masking=False,
        dim=3,
        spheres=True
    )

    saveH5PY(invar, outvar, names, bounds, "/home/vault/unrz/unrz109h/porous_media_data/spheres/h5_datasets/std_validation.h5")

    '''file = h5py.File("/home/woody/iwia/iwia057h/external/spheres/unnorm_validation_new.h5", 'r')
    outputs = {}
    outputs["p"], _, _ = normalize_old(np.array(file["output"]["p"]), 150, 0, 1, True)

    names = []
    for name in file["name"]:
        names.append(name.decode("ascii"))
    saveH5PY(file["input"], outputs, names, file["bounds"], "/home/woody/iwia/iwia057h/external/spheres/150_validation_150_new.h5")'''


    '''for input, output, name in zip(invar.values(), outvar.values(), names):
        data_dict = {"fill":    input,
                     "p":       output }
        saveDictToVTK(data_dict, ("/home/vault/iwia/iwia057h/data/scaled/shifted/test/scaled" + name + ".vti"))'''
    
    
