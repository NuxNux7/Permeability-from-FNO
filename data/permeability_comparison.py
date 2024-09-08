import os
import numpy as np
import csv
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from matplotlib import pyplot as plt


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
        if len(numpy_array.shape) != 1:
            numpy_array = np.swapaxes(numpy_array, 0, 1)
            dims_current = (1, 3,) + dims
        else:
            dims_current = (1, 1,) + dims
            
        # Using Fortran ordering
        array_dict[array_name] = numpy_array.reshape(dims_current, order='F')

    return array_dict


def load_csv(path):

    with open(path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]

    perms = [float(d['permeability (mD)']) for d in data]
    return perms


if __name__ == "__main__":

    # factor used for conversion delta p to permeability (estimated)
    factor = 9.472969732538735
    factors = []

    # read in given permeabilities from file
    path_permeabilities = '/home/vault/iwia/iwia057h/data/2D/permeability.csv'
    given_perms = load_csv(path_permeabilities)

    # work with simulation data
    est_perms = []
    differences = []
    relative_differences = []
    samples = []

    path_simulations = '/home/woody/iwia/iwia057h/2D/simulation/'
    for file in os.listdir(path_simulations):

        # get sample number from filename
        filename = os.path.basename(file)
        name = filename.split('_')
        sample = int(name[2])
        samples.append(sample)

        # calculate preassure differrence from simulation
        simulation = load_VTI(path_simulations + '/' + file)
        density_name = "2D_porous" + str(sample) + "_density"
        p = simulation[density_name]
        p_inlet = p[0, 0, 2].mean() - 1
        calc_permeability = factor / p_inlet
        est_perms.append(calc_permeability)

        # compare with given values
        given_permeability = given_perms[sample - 1]
        difference = abs(calc_permeability - given_permeability)
        differences.append(difference)
        relative_difference = difference / given_permeability
        relative_differences.append(relative_difference)

        # estimate factor
        factors.append(given_permeability * p_inlet)

        print("sample:                  ", sample)
        print("calculated permeability: ", calc_permeability)
        print("given permeability:      ", given_permeability)
        print("difference:              ", difference)
        print()

    mean_difference = np.array(differences).mean()
    print("mean difference: ", mean_difference)
    mean_realtive_difference = np.array(relative_differences).mean()
    print("mean relative difference: ", mean_realtive_difference)

    estimated_factor = np.array(factors).mean()
    print("estimated factor: ", estimated_factor)


    # plot p_inlets
    data_points = []
    data_points_diff = []
    for i in range(len(samples)):
        data_points.append((samples[i], given_perms[samples[i]-1], est_perms[i]))
        data_points_diff.append((samples[i], differences[i], given_perms[samples[i]-1]))

    data_points_sorted = sorted(data_points, key=lambda tup: tup[1])
    data_points_diff_sorted = sorted(data_points_diff, key=lambda tup: tup[1])

    sample_names, given_values, calc_values = zip(*data_points_sorted)

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(given_values, calc_values)
    
    # Add labels for each point
    #for i, name in enumerate(sample_names):
        #plt.annotate(name, (est_values[i], calc_values[i]), xytext=(5, 5), textcoords='offset points')
    
    # Set labels and title
    plt.xlabel('Given Value')
    plt.ylabel('Calculated Value')
    plt.title('Given vs Calculated Values')
    
    # Add a diagonal line for reference
    min_val = min(min(given_values), min(calc_values))
    max_val = max(max(given_values), max(calc_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('perm_comparison.png')
