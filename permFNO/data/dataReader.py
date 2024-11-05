# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math

import numpy as np
import torch
from skimage.transform import resize

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .normalization import *

#from skfmm import distance



# MAIN FUNCTION
def load_dataset(path, path_geometry=None, bounds=None, rotate=False, scaling=False, masking=True, calc_p_in=False, dim=3, spheres=False):
    "Loads a FNO dataset"

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

    if calc_p_in:
        if dim == 3:
            outvar["p_in"] = np.empty((size, 1, 1, 1, 1))
        else:
            outvar["p_in"] = np.empty((size, 1, 1, 1))


    # iterate every file in folder
    names = []
    counter = 0
    for file in os.listdir(path):

        if dim == 2:
            name = os.path.basename(file).split('_')
            sample = int(name[2])
        
            #if sample < 2000:
                #continue

            #if counter >= size:
                #continue

        names.append(os.path.basename(file))


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
            for _ in range(3):
                if spheres:
                    invar["fill"][counter] = np.rot90(dict_simulation["OverlapFraction"], k=1, axes=(-2,-1))
                    outvar["p"][counter] = np.rot90(dict_simulation["Density"], k=1, axes=(-2,-1))
                else:
                    invar["fill"][counter] = np.rot90(dict_geometry['NoSlip'], k=1, axes=(-2,-1))
                    outvar["p"][counter] = np.rot90(dict_simulation[density_key], k=1, axes=(-2,-1))
                counter = counter + 1
    print(len(names))
    
    # calculate mask for lambda weighting using fill
    if masking:
        mask = create_mask(invar["fill"], calc_p_in)
    else:
        mask = None


    # calculate signed distance field
    '''if sdf:
        invar["fill"] = create_sdf(mask["p"], invar["fill"], scaling)'''


    # get bounds for normalization

    if spheres:
        pow = 1
        scale = 1
        offset = 1
        outvar["p"], min_pow, max_pow = normalize_old(outvar["p"], scale, offset, pow, True)    # changed from external:12 old: 135 (125)
    else:
        pow = 0.5
        scale = 1
        offset = 0
        outvar["p"], min_pow, max_pow = normalize_new(outvar["p"], pow)

    invar["fill"], _, _ = normalize_new(invar["fill"], 1)
    bounds = [offset, scale, pow, min_pow, max_pow]
    #bounds = [0, 1, 1, 0, 1]

    # create p_in
    if calc_p_in:
        outvar["p_in"] = p_to_p_in(outvar["p"])


    return (invar, outvar, mask, np.array(names), bounds)


def load_dataset_fast(path, self_rotate=False, masking=True):
    "Loads a FNO dataset"

    size = len(os.listdir(path))
    shape = (size, 1, 128, 64, 64)

    invar = {"fill": np.empty(shape, dtype=np.float32)}
    outvar = {"p" : np.empty(shape, dtype=np.float32)}


    # iterate every file in folder
    counter = 0
    names = []
    for file in os.listdir(path):
        names.append(os.path.basename(file))

        dict = load_VTI(path + "/" + file, scaling=False, cells=False)
        print("loaded file: ", (counter + 1), "/", size)


        invar["fill"][counter] = dict["fill"]
        outvar["p"][counter] = dict["p"]


        if self_rotate:
            for _ in range(3):

                invar["fill"] = np.append(invar["fill"],
                    np.rot90(dict["fill"], k=1, axes=(-2,-1)), axis=0)
                outvar["p"] = np.append(outvar["p"],
                    np.rot90(dict["p"], k=1, axes=(-2,-1)), axis=0)

        counter = counter + 1
            

    # calculate mask for lambda weighting using fill
    if masking:
        mask = create_mask(invar["fill"], False)
    else:
        mask = None

    return (invar, outvar, mask, np.array(names), (0, 1, 1, 0, 1))


def apped_to_dict(dict, invar, outvar):
    invar["fill"] = np.append(invar["fill"], dict["OverlapFraction"], axis=0)
    outvar["p"] = np.append(outvar["p"], dict["Density"], axis=0)

    return (invar, outvar)



# GRID TRANSFORMATIONS
def scale_grid(grid, shape):
    new_shape = (grid.shape[0], grid.shape[1], *shape)
    new_grid = resize(grid, new_shape, anti_aliasing=True, preserve_range=True)

    return new_grid




# GRID CREATION
def create_mask(fill, calc_p_in):

    mask = {"p": np.empty(fill.shape, dtype=np.float32)}

    if calc_p_in:
        mask["p_in"] = np.ones((fill.shape[0], 1, 1, 1, 1))

    mask["p"] = (1. - fill[:]) * 0.8 + 0.1

    for i in range(fill.shape[0]):
        mask["p"][i][0][0:12][:][:] = 1

    return mask


'''def create_sdf(mask, fill, scaling):

    sdf = np.empty_like(mask)

    for b in range(mask.shape[0]):
        # Compute the distance from the boundary
        distance_mask = distance(mask[b], dx=1.0)
        distance_fill = distance(fill[b], dx=1.0)


        # Create the signed distance field
        sdf[b] = distance_mask
        sdf[b][fill[b] > 0.5] = -distance_fill[fill[b] > 0.5]

        if scaling:
            sdf[b] = np.clip(sdf[b], -9, 9)
            sdf[b] = 1/9 * sdf[b]
        else:
            sdf[b] = np.clip(sdf[b], -18, 18)
            sdf[b] = 1/18 * sdf[b]

    return sdf'''


def p_to_p_in(p):

    p_in = np.empty_like(p)

    shape = (1, 1, 1, 1)
    for i in range(p.shape[0]):
        value = p[i][0][1].mean()
        p_in[i] = np.reshape(value, (1, 1, 1, 1))

    return p_in



# IO
def load_VTI(path, scaling, cells=True, swapZX=False, shape=(96, 64, 64)):

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

    rotate = False

    invar, outvar, _, names, bounds = load_dataset(
        "/home/woody/iwia/iwia057h/external/ownSimulation/simulation",
        "/home/woody/iwia/iwia057h/external/ownSimulation/geometries",
        rotate=rotate, scaling=True, masking=False, calc_p_in=False,
        dim=3,
        spheres=False
    )

    # change names with rotation
    new_names = []
    if rotate:
        for pos, name in enumerate(names):
            new_names.append(name + "0")
            new_names.append(name + "1")
            new_names.append(name + "2")
            new_names.append(name + "3")
    else:
        new_names = names

    res = np.array(new_names)

    saveH5PY(invar, outvar, res, bounds, "/home/woody/iwia/iwia057h/external/new.h5")

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
    
    
