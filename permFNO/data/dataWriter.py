# Functions for saving and visualizing data in various formats (PNG, VTK, CSV, H5PY) 
# and creating comparison plots between predicted and target values.


import numpy as np

import matplotlib.pyplot as plt

import vtk
from vtk.util import numpy_support
import csv
import h5py


def savePNG(input_data, output_data, target_data, mask, filename):
    """
    Creates and saves a figure with three subplots showing input data, output data,
    and their masked difference. For 3D data, displays the middle slice.
    Saves the resulting visualization as a PNG file.
    """
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
    """
    Converts a dictionary of numpy arrays into a VTK structured points dataset.
    Handles both 2D and 3D data, automatically expanding 2D data to 3D.
    Saves the resulting dataset as a VTK file.
    """
        
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
    """
    Creates a VTK file containing input data, output data, target data, and their masked difference.
    Serves as a convenience wrapper around saveDictToVTK for specific data types.
    """
    diff_data = mask * np.abs(output_data - target_data)

    data_dict = {"input":       input_data,
                 "output":      output_data,
                 "target":           target_data,
                 "difference":  diff_data }
    
    saveDictToVTK(data_dict, filename)


def saveCSV(dict, filename, first: bool = True):
    """
    Saves a dictionary of data to a CSV file with columns for Sample, MAE p, and MAPE p inlet.
    Can either create a new file or append to an existing one based on the 'first' parameter.
    """
    if first:
        mode = 'w'
    else:
        mode = 'a'

    with open(filename, mode, newline='') as csvfile:
        if first:
            writer = csv.writer(csvfile)
        writer.writerow(['Sample', 'MAE p', 'MAPE p inlet'])
        for key, values in dict.items():
            writer.writerow([key] + values)


def saveH5PY(input_data, output_data, name_data, bounds, filename):
    """
    Saves multiple datasets to an HDF5 file with specific group structure.
    Creates groups for input and output data, and datasets for names and bounds.
    Handles ASCII encoding for name data and specific bounds format.
    """
        
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


def saveErrorPlot(outputs, targets, path):
    """
    Creates and saves a scatter plot comparing predicted outputs against target values.
    Includes a diagonal line for perfect predictions and uses equal axis scaling.
    Useful for visualizing model performance and prediction accuracy.
    """

    plt.figure(figsize=(10,10))
    plt.scatter(targets, outputs, c='crimson')
    #plt.yscale('log')
    #plt.xscale('log')

    p1 = max(max(targets), max(outputs))
    p2 = min(min(targets), min(outputs))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')

    plt.savefig(path)


def visualize(input_data, output_data, target_data, mask, output_folder):
    """
    Convenience function that creates both PNG and VTK visualizations of the data.
    Saves a 2D slice visualization as PNG and full 3D data as VTK file in the specified folder.
    """
    savePNG(input_data, output_data, target_data, mask, f"{output_folder}/slice.png")
    saveArraysToVTK(input_data, output_data, target_data, mask, f"{output_folder}/sample.vtk")