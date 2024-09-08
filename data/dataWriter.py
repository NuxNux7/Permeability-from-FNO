import numpy as np

import matplotlib.pyplot as plt

import vtk
from vtk.util import numpy_support
import csv
import h5py


def savePNG(input_data, output_data, target_data, mask, filename):
    slice_index = input_data.shape[-1] // 2

    diff_data = mask * np.abs(output_data - target_data)

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    if len(input_data.shape) == 3:
        input_slice = input_data[:, :, slice_index]
        output_slice = output_data[:, :, slice_index]
        diff_slice = diff_data[:, :, slice_index]
    else:
        input_slice = input_data
        output_slice = output_data
        diff_slice = diff_data

    im1 = ax1.imshow(input_slice, cmap='viridis')
    ax1.set_title('Input')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(output_slice, cmap='viridis')
    ax2.set_title('Output')
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(diff_slice, cmap='bwr')
    ax3.set_title('Difference')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def saveDictToVTK(data_dict, filename):
     # Create vtkStructuredPoints
    data = vtk.vtkStructuredPoints()
    
    # Set dimensions
    shape = next(iter(data_dict.values())).shape
    if len(shape) == 2:
        shape = (*shape, 1)
    data.SetDimensions(shape[0], shape[1], shape[2])  # VTK uses (x, y, z) order
    
    # Create a vtkPointData object to store the arrays
    point_data = data.GetPointData()
    
    # Add each array to the dataset
    for key, value in data_dict.items():
        if len(value.shape) == 2:
            value = np.expand_dims(value, -1)
        vtk_array = numpy_support.numpy_to_vtk(value.flatten(order='F').astype('float32') , deep=True)
        vtk_array.SetName(key)
        point_data.AddArray(vtk_array)

    
    # Write the data to a file
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Write()


def saveArraysToVTK(input_data, output_data, target_data, mask, filename):
    diff_data = mask * np.abs(output_data - target_data)

    data_dict = {"input":       input_data,
                 "output":      output_data,
                 "target":           target_data,
                 "difference":  diff_data }
    
    saveDictToVTK(data_dict, filename)


def saveCSV(dict, filename, first: bool = True):

    if first:
        mode = 'w'
    else:
        mode = 'a'

    with open(filename, mode, newline='') as csvfile:
        if first:
            writer = csv.writer(csvfile)
        writer.writerow(['Sample', 'MAE p', 'MARE p inlet'])
        for key, values in dict.items():
            writer.writerow([key] + values)


def saveH5PY(input_data, output_data, name_data, bounds, filename):
    with h5py.File(filename, "w") as f:
        input = f.create_group("input")
        for key, value in input_data.items():
            input.create_dataset(key, data=value)

        output = f.create_group("output")
        for key, value in output_data.items():
            output.create_dataset(key, data=value)

        asciiList = [n.encode("ascii", "ignore") for n in name_data]
        f.create_dataset("name", data=asciiList)

        # create bounds with order [offset, scale, power, min_power, max_power]
        f.create_dataset("bounds", data=bounds)


def visualize(input_data, output_data, target_data, mask, output_folder):

    savePNG(input_data, output_data, target_data, mask, f"{output_folder}/slice.png")
    saveArraysToVTK(input_data, output_data, target_data, mask, f"{output_folder}/sample.vtk")