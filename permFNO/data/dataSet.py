# DictDataset and custom function for dataset modification and analysis
# WARNING: Use filter_dataset, split_dataset and analyse_dataset with care! They contain dataset custom content and are not universal!

import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import h5py

from matplotlib import pyplot as plt
from scipy import ndimage
import random


from .dataReader import load_dataset, create_mask
from .dataWriter import saveH5PY, saveErrorPlot
from .normalization import denormalize_new, normalize_new

SDF = False


class DictDataset(Dataset):
    """
    A custom Dataset class for handling dictionary-based datasets, supporting both HDF5 and a folder of vtk file as formats.
    Handles geometric data with optional rotation, masking, and scaling features.

    It stores three dicts: input, output, and lables.
    Input needs to have the entry "fill" for indicating the geometry.
    Output and target need the entry "p" for the pressure field.

    Training on multiple input and output fields needs to be added.
    
    Args:
        path (str): Path to the dataset file
        rotate (bool): Whether to apply rotation to the input data
        masking (bool): Whether to apply masking to the input data
        scaling (bool): Whether to apply scaling to the input data
        h5 (bool): Whether the input file is in HDF5 format
    """
    def __init__(self, path, rotate=False, masking=True, scaling=True, h5=False):
        self.h5 = h5
        if h5:
            self.h5f = h5py.File(path, 'r')
            self.inputs = self.h5f["input"]

            # Calculate the signed distance field
            if SDF:
                field = np.array(self.inputs["fill"])
                outside = ndimage.distance_transform_edt(1 - field).astype(np.float32)
                inside = ndimage.distance_transform_edt(field).astype(np.float32)
                sdf = outside - inside
                print(sdf.max(), sdf.min())
                sdf = sdf / 20.952328
                print(sdf.max(), sdf.min())
                self.inputs = {"fill": sdf}

            self.target = self.h5f["output"]
            self.names = self.h5f["name"]
            if "bounds" in self.h5f:
                self.bounds = self.h5f["bounds"]            #[offset, scale, power, min_power, max_power] [1.001, 1.0000] - 1 * 1000
            else:
                self.bounds = [1., 1., 1., 0., 1.]
            self.masks = None
            if masking:
                self.masks = create_mask(self.inputs["fill"])   
        else:
            (self.inputs, self.target, self.masks, self.names, self.bounds) = load_dataset(path, None, None, rotate, scaling, masking)

        self.size = self.inputs["fill"].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Returns a tuple of (input, target, mask, name) for the given index
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: (input_data, label_data, mask_data, name)
        """
        return (self.inputs["fill"][idx], self.target["p"][idx], self.masks["p"][idx], self.names[idx])

    
    def getBounds(self):
        """
        Returns the bounds parameters [offset, scale, power, min, max].

        Min-max scaling uses power, min and max
        Custom scaling uses offset and scale.

        """
        return self.bounds

    def close(self):
        if self.h5:
            self.h5f.close()
            np.allclose(self.inputs, self.lables)

    def estimate_by_formula(self):
        """
        Estimates permeability using the Kozeny–Carman equation and compares with actual values.
        Generates an error plot and calculates R-squared value for the estimations.
        
        Returns:
            None: Saves plot to 'estimated_plot.png' and prints R-squared value
        """
        estimations = np.zeros([self.size])
        targets = np.zeros([self.size])

        for i in range(self.size):
            estimations[i] = estimate_permeability_equation(self.inputs["fill"][i], self.names[i])
            targets[i] = self.lables["p"][i][0:12, :, :].mean()

        print("factor:", targets.mean() / estimations.mean())

        saveErrorPlot(estimations, targets, "estimated_plot.png")

        error = estimations - targets
        error = np.power(error, 2)

        mean_target = targets.mean()
        mean_distance = targets - mean_target
        mean_distance = np.power(mean_distance, 2)

        r2 = 1 - (error.sum() / mean_distance.sum())

        print("R2:", r2)



# CHANGE EXISTING DATASET
def filter_dataset(path, percentile, threshold=None):
    """
    Filters a dataset based on maximum pressure values, removing samples below a certain threshold.
    
    Args:
        path (str): Path to the HDF5 dataset file
        percentile (float): Percentile value to use for threshold calculation
        threshold (float, optional): Manual threshold value. If None, calculated from percentile
        
    Returns:
        None: Creates a new filtered HDF5 file with '_filtered_[percentile]' suffix
    """
     
    file = h5py.File(path, 'r')
    size = file["input"]["fill"].shape[0]

    rotate = (len(file["input"]["fill"].shape) == 5)
    if rotate:
        size = int(size / 4.)


    # entnormalize p
    p = file["output"]["p"]
    _, _, pow, min_val, max_val = file["bounds"]
    p = denormalize_new(p, pow, min_val, max_val)

    # find maximum preassures of fiels
    p_max = {}
    for i in range(size):
        index = i
        if rotate:
            index = 4 * i
        result = p[index]
        p_max[index] = result.max()

    # filter out lower values
    values = np.array(list(p_max.values()))
    if threshold is None:
        threshold = np.percentile(values, percentile)

    print("threshold: ", threshold)

    p_max_filtered = {k: v for k, v in p_max.items() if v > threshold}
    print("remove: ", len(p_max_filtered), "/", size)

    indices_remove = np.array(list(p_max_filtered.keys()))
    if rotate:
        indices_full = indices_remove.copy()
        indices_full = np.append(indices_full, indices_remove + 1)
        indices_full = np.append(indices_full, indices_remove + 2)
        indices_full = np.append(indices_full, indices_remove + 3)
        indices_remove = np.sort(indices_full)

    p_new = np.delete(p, indices_remove, axis=0)

    print(p_new.max())

    # normalize
    p_new, min_val, max_val = normalize_new(p_new, pow)
    bounds_new = (0, 1, pow, min_val, max_val)
    print(bounds_new)

    # write new h5
    input = file["input"]["fill"]
    input_new = {"fill": np.delete(input, indices_remove, axis=0)}

    output_new = {"p": p_new}

    names = file["name"]
    names_b = np.delete(names, indices_remove, axis=0)
    names_new = []
    for name in names_b:
        names_new.append(name.decode('ascii'))

    new_path = path.removesuffix('.h5') + "_filtered_" + str(percentile) + ".h5"
    saveH5PY(input_new, output_new, names_new, bounds_new, new_path)


def split_dataset(path, split, random_split=True):
    """
    Splits an HDF5 dataset into training and testing sets, handling rotated and reversed data appropriately.
    
    Args:
        path (str): Path to the HDF5 dataset file
        split (float): Fraction of data to use for training (0-1)
        random_split (bool): Whether to split randomly or use a predetermined pattern
        
    Returns:
        None: Creates two new HDF5 files with '_rocks' and '_test' suffixes
    """

    file = h5py.File(path, 'r')
    rotation = len(file["input"]["fill"].shape) == 5

    if rotation:
        size = int(file["input"]["fill"].shape[0] / 4.)
    else:
        size = file["input"]["fill"].shape[0]
    train_size = int(size * split)
    test_size = size - train_size

    indices = range(0, size)


    # train
    if random_split:
        train_indices = random.sample(indices, train_size)
    else:
        train_indices = list(set(indices) - set(range(2, size, 10)))
    if rotation:
        shape = (len(train_indices) * 4, 1, file["input"]["fill"].shape[2], file["input"]["fill"].shape[3], file["input"]["fill"].shape[4])
    else:
        shape = (len(train_indices), 1, file["input"]["fill"].shape[2], file["input"]["fill"].shape[3])
    inputs = {"fill": np.zeros(shape, dtype=np.float32)}
    outputs = {"p": np.zeros(shape, dtype=np.float32)}
    names = []

    counter = 0
    for i in train_indices:
        if rotation:
            num_rot = 4
        else:
            num_rot = 1

        for rot in range(num_rot):
            if rotation:
                position = (i * 4) + rot
            else:
                position = i

            inputs["fill"][counter] = file["input"]["fill"][position][0]
            outputs["p"][counter] = file["output"]["p"][position][0]
            names.append(file["name"][position].decode("ascii"))

            counter += 1
    
    new_name = path.removesuffix('.h5') + "_train.h5"

    
    # remove all non rocks sample 2D
    '''offset = 0
    for index in set(train_indices):
        name = file["name"][index].decode("ascii")
        name = name.split('_')
        name_pos = int(name[2])

        print(name_pos)


        if name_pos > 1200 or name_pos <= 1000:
            inputs["fill"] = np.delete(inputs["fill"], index-offset, axis=0)
            outputs["p"] = np.delete(outputs["p"], index-offset, axis=0)
            names.pop(index-offset)
            print("removed")
            offset += 1'''

    
    # remove all non noise sample 2D
    '''offset = 0
    for index in set(train_indices):
        name = file["name"][index].decode("ascii")
        name = name.split('_')
        name_pos = int(name[2])

        print(name_pos)


        if name_pos < 2000:
            inputs["fill"] = np.delete(inputs["fill"], index-offset, axis=0)
            outputs["p"] = np.delete(outputs["p"], index-offset, axis=0)
            names.pop(index-offset)
            print("removed")
            offset += 1'''

    # remove the test sets reverse and rotated simulations
    if rotation:
        test_indices = list(set(train_indices))
        for index in test_indices:
            name = file["name"][4 * index].decode("ascii")
            name = name.split('_')
            name_pos = int(name[1])

            print()
            print("looking at ", name)

            counter = 0
            for name_train in names:
                if ("_" + str(name_pos) + "_") in name_train:
                    rm = [counter, counter+1, counter+2, counter+3]
                    inputs["fill"] = np.delete(inputs["fill"], rm, axis=0)
                    outputs["p"] = np.delete(outputs["p"], rm, axis=0)
                    print("removed ", name_train)
                    break
                
                counter += 1

            names.pop(counter)
            names.pop(counter)
            names.pop(counter)
            names.pop(counter)

    print(len(names), inputs["fill"].shape[0])
    saveH5PY(inputs, outputs, names, file["bounds"], new_name)


    # test
    test_indices = list(set(indices) - set(train_indices))
    if rotation:
        shape = (test_size, 1, file["input"]["fill"].shape[2], file["input"]["fill"].shape[3], file["input"]["fill"].shape[4])
    else:
        shape = (test_size, 1, file["input"]["fill"].shape[2], file["input"]["fill"].shape[3])
    inputs = {"fill": np.zeros(shape, dtype=np.float32)}
    outputs = {"p": np.zeros(shape, dtype=np.float32)}
    names = []

    counter = 0
    for i in test_indices:
        if rotation:
            position = 4 * i
        else:
            position = i

        inputs["fill"][counter] = file["input"]["fill"][position][0]
        outputs["p"][counter] = file["output"]["p"][position][0]
        names.append(file["name"][position].decode("ascii"))

        counter += 1

    new_name = path.removesuffix('.h5') + "_test.h5"
    saveH5PY(inputs, outputs, names, file["bounds"], new_name)



# ANALYZE DATASET
def analyse_dataset(dataset, dataset2=None):
    """
    Analyzes pressure distributions in dataset(s) and generates histogram visualization.
    
    Args:
        dataset: Primary dataset to analyze
        dataset2 (optional): Secondary dataset to include in analysis
        
    Returns:
        dict: Dictionary containing maximum pressure values
    
    Notes:
        Saves histogram visualization to 'histogram.png'
    """

    p_inlet_max = {}
    p_inlet = {}
    p_max = {}
    p_inlet_scale = {}


    p = dataset.lables["p"]
    if dataset2 is not None:
        p = np.append(p, dataset2.lables["p"], axis=0)
        #p = p / 150
        p = p - 1
        print(p.shape)

    else:
        a, b, pow, min_val, max_val = dataset.bounds
        print(a, b, pow, min_val, max_val)
        p_scale = np.array(p)
        p = entnormalize_new(p, pow, 0, 1)
        #p = np.divide(p, 150)

        print(p.shape)

    if len(p.shape) == 5:
        skip = 4
    else:
        skip = 1
    skip = 1

    for i in range(0, p.shape[0], skip):
        #name = dataset.names[i]
        result = p[i]
        p_inlet_max[i] = result.max()
        p_inlet[i] = result[0,0:12].mean()
        p_inlet_scale[i] = p_scale[i, 0, 0:12].mean()

    p_inlet = dict(sorted(p_inlet.items(), key=lambda item: item[1], reverse=True))
    p_inlet_max = dict(sorted(p_inlet_max.items(), key=lambda item: item[1], reverse=True))


    # plot histogram
    pressure_values = list(p_inlet.values())
    pressure_scale = list(p_inlet_scale.values())

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean and std for annotations
    mean = np.mean(pressure_values)
    std = np.std(pressure_values)
    
    # Create histogram
    n, bins, patches = ax.hist(pressure_values, 
                             bins='auto',  # automatically determine number of bins
                             alpha=0.7,    # slight transparency
                             color='blue')
    
    
    # Add vertical line for mean
    #ax.axvline(mean, color='red', linestyle='dashed', alpha=0.5)
    
    # Add text box with statistics
    stats_text = f'Mean: {mean:.4f}\nStd Dev: {std:.4f}'
    '''ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))'''
    
    # Customize the plot
    ax.set_title('Distribution of Pressure Values', fontsize=12, pad=15)
    ax.set_xlabel('Pressure')
    ax.set_ylabel('Number of Samples')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('histogram.png')

    return p_max


def estimate_permeability_equation(geometry, name):
    """
    Estimates permeability using physical equations based on geometry and sample parameters.
    
    Args:
        geometry: Geometric data of the sample
        name (str): Sample name containing diameter information
        
    Returns:
        float: Estimated pressure difference (delta_p)
        
    Notes:
        Uses porosity and diameter to calculate permeability using a modified version
        of the Kozeny-Carman equation
    """

    print(name)
    name_arr = name.split('_')
    diameter = int(name_arr[0]) + 0.1 * int(name_arr[1])
    print("diameter:", diameter)

    porousity = 1 - (geometry.sum() / (0.8 * geometry.shape[1] * geometry.shape[2] * geometry.shape[3]))
    print("porousity:", porousity)
    factor_scaling = 0.0015632130795060562

    delta_p = (150 / pow(diameter, 2)) * ((1 - pow(porousity, 2)) / pow(porousity, 3)) * factor_scaling

    return delta_p



if __name__ == "__main__":
    import os

    basefile = '/home/woody/iwia/iwia057h/external/std'
    path = basefile + '.h5'

    #filter_dataset(path, 90)

    path = basefile + '_filtered_90.h5'
    split_dataset(path, 0.9, random_split=False)

    '''path = basefile + '_train.h5'
    split_full_dataset(path, (8/9), random_split=False)

    old = basefile + '_train_train.h5'
    new = basefile + '_train.h5'
    os.rename(old, new)

    old = basefile + '_train_test.h5'
    new = basefile + '_validation.h5'
    os.rename(old, new)'''