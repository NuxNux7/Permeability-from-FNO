# Used for comparison of the simulated and provided permeabilities

import os
import math

import numpy as np

import csv

import vtk
from vtk.util.numpy_support import vtk_to_numpy


def load_VTI(path):
    if not path.endswith(".vti"):
        raise Exception(
            ".vti file required"
        )
    
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()

    dims = data.GetDimensions()
    dims = tuple((x - 1) for x in dims)
    dims = dims[0:2]
    array_dict = {}

    cell_data = data.GetCellData()


    for i in range(cell_data.GetNumberOfArrays()):
        # get matadata
        array_name = cell_data.GetArrayName(i)
        vtk_array = cell_data.GetArray(i)

        #load np array
        numpy_array = vtk_to_numpy(vtk_array)

        # reshape in dimension
        dims_current = dims
        dims_current = (1, 1,) + dims
            
        # Using Fortran ordering
        array_dict[array_name] = numpy_array.reshape(dims_current, order='F')

    return array_dict


def load_csv(path):

    with open(path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]

    return data


def compare_permeabilities():

    # factor used for conversion delta p to permeability (estimated)
    factor = 0.001

    # read in given permeabilities from file
    path_permeabilities = 'data/2D/permeability.csv'
    perms = load_csv(path_permeabilities)

    # work with simulation data
    path_simulations = '2D/simulation/'
    for file in os.listdir(path_simulations):

        # get sample number from filename
        filename = os.path.basename(file)
        name = filename.split('_')
        sample = int(name[2])

        # calculate preassure differrence from simulation
        simulation = load_VTI(file)
        density_name = "2D_porous" + str(sample) + "_density"
        p = simulation[density_name]
        p_inlet = p[2].mean()
        calc_permeability = factor / p_inlet

        # compare with given values
        given_permeability = perms[sample]
        difference = math.abs(calc_permeability - given_permeability)

        print("calculated permeability: ", calc_permeability)
        print("calculated permeability: ", given_permeability)
        print("difference: ", difference)


if __name__ == "__main__":
    
    compare_permeabilities()