import numpy as np
import torch
import torch.nn as nn

def normalize_old(grid, scale, offset, pow, verbose=False):

    # normalization
    grid[:][:][:][:][:] = (grid[:][:][:][:][:] - offset) * scale

    min_val = np.min(grid)
    max_val = np.max(grid)

    print(min_val)
    print(max_val)

    if verbose:
        big = len(grid[grid > 1.1])
        print("number of outlaws > 1.1: ", big, " with max: ", max_val)

        big = len(grid[grid < -0.1])
        print("number of outlaws < -0.1: ", big, " with min: ", min_val)

    if pow != 1:
        grid = np.clip(grid, 0., 20.)
        grid = np.power(grid, pow)

        min_val = np.min(grid)
        max_val = np.max(grid)

        grid = (grid - min_val) / (max_val - min_val)
    
    return grid, min_val, max_val


def normalize_new(grid, pow, verbose=False):

    # normalization
    min_val = np.min(grid)
    max_val = np.max(grid)

    grid = (grid - min_val) / (max_val - min_val)

    grid = np.power(grid, pow)

    return grid, min_val, max_val


def entnormalize_new(grid, pow, min_val, max_val):
    grid = np.clip(grid, 0, None)
    grid = np.power(grid, (1/pow))

    grid = (grid * (max_val - min_val)) + min_val

    return grid


class Entnormalizer(nn.Module):
    def __init__(self, offset=0, scale=1, pow=1, min_val=0, max_val=1):
        super(Entnormalizer, self).__init__()
        self.offset = offset
        self.scale = scale
        self.pow = pow
        self.min_val = min_val
        self.max_val = max_val


    def forwardOld(self, grid):
        if self.pow != 1:
            grid[:] = (grid[:] * (self.max_val - self.min_val)) + self.min_val
            grid = torch.clip(grid, 0, None)
            grid = torch.pow(grid, (1/self.pow))

        grid = (grid / self.scale) + self.offset

        return grid
    
    
    def forward(self, grid):
        grid = torch.clip(grid, 0, None)
        grid = torch.pow(grid, (1/self.pow))

        #grid = (grid * (self.max_val - self.min_val)) + self.min_val

        return grid
