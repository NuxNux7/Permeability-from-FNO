
import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import h5py

from matplotlib import pyplot as plt
from scipy import stats

import random
import math
import csv

from .dataReader import load_dataset, load_dataset_fast, create_mask
from .dataWriter import saveH5PY, saveErrorPlot
from .normalization import entnormalize_new, normalize_new


class DictDataset(Dataset):
    def __init__(self, path, fast=True, rotate=False, masking=True, scaling=True, h5=False):
        self.h5 = h5
        if h5:
            self.h5f = h5py.File(path, 'r')
            self.inputs = self.h5f["input"]
            self.lables = self.h5f["output"]
            self.names = self.h5f["name"]
            if "bounds" in self.h5f:
                self.bounds = self.h5f["bounds"]            #[offset, scale, power, min_power, max_power] [1.001, 1.0000] - 1 * 1000
            else:
                self.bounds = [1., 12., 1., 0., 1.]
            self.masks = None
            if masking:
                self.masks = create_mask(self.inputs["fill"], False)   
        else:
            if fast:
                (self.inputs, self.lables, self.masks, self.names, self.bounds) = load_dataset_fast(path, rotate, masking)
            else:
                (self.inputs, self.lables, self.masks, self.names, self.bounds) = load_dataset(path, None, rotate, scaling, masking)

        self.size = self.inputs["fill"].shape[0]

        # shape input: [batch, 1, x, y, z]
        # shape output: [batch, 1, x, y, z] -> [batch, 1]

        # transformation: self.output[:, 0, 0:12, :, :].mean()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.inputs["fill"][idx], self.lables["p"][idx], self.masks["p"][idx], self.names[idx])
    
    def getBounds(self):
        "[offset, scale, power, min_power, max_power]"
        return self.bounds

    def close(self):
        if self.h5:
            self.h5f.close()
            np.allclose(self.inputs, self.lables)

    def estimate_by_formula(self):
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

def split_full_dataset(path, split, random_split=True):
    "split h5py dataset in two parts, separating independent of rotation, but depending on flow direction"

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
    
    new_name = path.removesuffix('.h5') + "_rocks.h5"

    
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

    
    # remove all non rocks sample 2D
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
        test_indixes = list( - set(train_indices))
        for index in test_indixes:
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
    test_indixes = list(set(indices) - set(train_indices))
    if rotation:
        shape = (test_size, 1, file["input"]["fill"].shape[2], file["input"]["fill"].shape[3], file["input"]["fill"].shape[4])
    else:
        shape = (test_size, 1, file["input"]["fill"].shape[2], file["input"]["fill"].shape[3])
    inputs = {"fill": np.zeros(shape, dtype=np.float32)}
    outputs = {"p": np.zeros(shape, dtype=np.float32)}
    names = []

    counter = 0
    for i in test_indixes:
        if rotation:
            position = 4 * i
        else:
            position = i

        inputs["fill"][counter] = file["input"]["fill"][position][0]
        outputs["p"][counter] = file["output"]["p"][position][0]
        names.append(file["name"][position].decode("ascii"))

        counter += 1

    new_name = path.removesuffix('.h5') + "_test.h5"
    #saveH5PY(inputs, outputs, names, file["bounds"], new_name)


def filter_dataset(path, percentile, threashold=None):
     
    file = h5py.File(path, 'r')
    size = file["input"]["fill"].shape[0]

    rotate = (len(file["input"]["fill"].shape) == 5)
    if rotate:
        size = int(size / 4.)


    # entnormalize p
    p = file["output"]["p"]
    _, _, pow, min_val, max_val = file["bounds"]
    p = entnormalize_new(p, pow, min_val, max_val)

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
    if threashold is None:
        threashold = np.percentile(values, percentile)

    print("threashold: ", threashold)

    p_max_filtered = {k: v for k, v in p_max.items() if v > threashold}
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

def estimate_permeability_equation(geometry, name):

    print(name)
    name_arr = name.split('_')
    diameter = int(name_arr[0]) + 0.1 * int(name_arr[1])
    print("diameter:", diameter)

    porousity = 1 - (geometry.sum() / (0.8 * geometry.shape[1] * geometry.shape[2] * geometry.shape[3]))
    print("porousity:", porousity)
    factor_scaling = 0.0015632130795060562

    delta_p = (150 / pow(diameter, 2)) * ((1 - pow(porousity, 2)) / pow(porousity, 3)) * factor_scaling

    return delta_p

def analyse_dataset(dataset, dataset2=None):
    p_inlet = {}
    p_inlet2 = {}
    p_max = {}


    p = dataset.lables["p"]
    if dataset2 is not None:
        p = np.append(p, dataset2.lables["p"], axis=0)
        #p = p / 150
        p = p - 1
        print(p.shape)

    else:
        _, _, pow, min_val, max_val = dataset.bounds
        p = entnormalize_new(p, pow, 0, 1)
        #p = p - 1
        p2 = dataset.lables["p"]

    if len(p.shape) == 5:
        skip = 4
    else:
        skip = 1
    skip = 1

    for i in range(0, p.shape[0], skip):
        #name = dataset.names[i]
        result = p[i]
        p_inlet[i] = result.max()
        p_inlet2[i] = p2[i][0,4].mean()
        p_max[i] = result.max()

    p_inlet = dict(sorted(p_inlet.items(), key=lambda item: item[1], reverse=True))
    p_inlet2 = dict(sorted(p_inlet2.items(), key=lambda item: item[1], reverse=True))
    p_max = dict(sorted(p_max.items(), key=lambda item: item[1], reverse=True))

    print(p_inlet)


    # plot histogram
    pressure_values = list(p_inlet.values())
    '''kde = stats.gaussian_kde(p_inlet_list)
    x_range = np.linspace(min(p_inlet_list), max(p_inlet_list), 1000)

    plt.figure(figsize=(10, 6))
    #plt.hist(p_inlet_list, bins=len(p_inlet_list), edgecolor='black')
    plt.plot(x_range, kde(x_range), 'b-', label='Pressure Distribution')
    plt.title('Distribution of Pressure Samples')
    plt.xlabel('Pressure')
    plt.ylabel('Density')
    plt.savefig("histogram.png")'''

    '''kde = stats.gaussian_kde(pressure_values)
    x_range = np.linspace(min(pressure_values), max(pressure_values), 1000)
    y_kde = kde(x_range)
    plt.plot(x_range, y_kde, 'b-', label='Pressure Distribution')

    plt.title('Distribution of Pressure Differnece Samples')
    plt.xlabel('Pressure')
    plt.ylabel('Density')

    mean_pressure = np.mean(pressure_values)
    plt.axvline(mean_pressure, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_pressure:.5f}')

    plt.legend()
    plt.savefig("histogram.png")'''

    # plot p_inlets
    x = list(p_inlet.keys())
    x = range(len(p_inlet.keys()))
    y = list(p_inlet.values())

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='blue', linewidth=3)

    x = list(p_inlet2.keys())
    x = range(len(p_inlet2.keys()))
    y = list(p_inlet2.values())

    plt.plot(x, y, color='orange', linewidth=3)

    plt.xlabel('sample')
    plt.ylabel('p_diff')

    plt.savefig('p_inlets.png')

    # plot p_maxes
    '''x = list(p_max.keys())
    y = list(p_max.values())

    plt.figure(figsize=(20, 10))
    plt.plot(x, y, color='blue', marker='x')

    plt.xlabel('geometry')
    plt.ylabel('p_max')

    plt.savefig('p_maxes.png')'''

    #p_inlet = dict(sorted(p_inlet.items(), key=lambda item: item[1], reverse=False))
    #for p, name in p_inlet.items():
        #print(name, ": ", p)


    '''with open("p_inlets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, p_inlet.keys())
        w.writeheader()
        w.writerow(p_inlet)

    with open("p_maxes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, p_max.keys())
        w.writeheader()
        w.writerow(p_max)'''

    return p_max


if __name__ == "__main__":
    import os

    basefile = '/home/woody/iwia/iwia057h/external/5Scaling_Interpol'
    path = basefile + '.h5'

    #filter_dataset(path, 90)

    path = basefile + '_filtered_90.h5'
    split_full_dataset(path, 0.9, random_split=False)

    '''path = basefile + '_train.h5'
    split_full_dataset(path, (8/9), random_split=False)

    old = basefile + '_train_train.h5'
    new = basefile + '_train.h5'
    os.rename(old, new)

    old = basefile + '_train_test.h5'
    new = basefile + '_validation.h5'
    os.rename(old, new)'''