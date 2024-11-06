# Functions used for normalization and their reversion
# WARNING: Use with care! The old normalization is not used combined with a power transformation.
#          This code is used costom to datasets and not universal!

import numpy as np
import torch
import torch.nn as nn

def normalize_old(grid, scale, offset, pow, verbose=False):
    """
    Legacy normalization function that applies scaling, offset, and power transformation
    with optional clipping and verbose output.

    Legacy means fixed scale!
    """

    # normalization
    grid[:][:][:][:][:] = (grid[:][:][:][:][:] - offset) * scale

    min_val = np.min(grid)
    max_val = np.max(grid)

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

    grid = np.clip(grid, -0.1, 1.1)
    
    return grid, min_val, max_val


def normalize_new(grid, pow, verbose=False):
    """
    Simplified normalization that first normalizes to [0,1] range
    then applies power transformation.

    New mean Min-Max normalization
    """

    # normalization
    min_val = np.min(grid)
    max_val = np.max(grid)

    grid = (grid - min_val) / (max_val - min_val)

    grid = np.power(grid, pow)

    return grid, min_val, max_val


def denormalize_new(grid, pow, min_val, max_val):
    """
    Inverse normalization function that undoes power transformation
    and rescales back to original range.
    """

    grid = np.clip(grid, 0, None)
    grid = np.power(grid, (1/pow))

    grid = (grid * (max_val - min_val)) + min_val

    return grid


class Denormalizer(nn.Module):
    """
    PyTorch module for inverse normalization with configurable parameters.
    Includes both old and new implementations.
    """
    def __init__(self, offset=0, scale=1, pow=1, min_val=0, max_val=1):
        super(Denormalizer, self).__init__()
        self.offset = offset
        self.scale = scale
        self.pow = pow
        self.min_val = min_val
        self.max_val = max_val


    def forwardOld(self, grid):
        """Legacy inverse normalization method"""
        if self.pow != 1:
            grid[:] = (grid[:] * (self.max_val - self.min_val)) + self.min_val
            grid = torch.clip(grid, 0, None)
            grid = torch.pow(grid, (1/self.pow))

        grid = (grid / self.scale) + self.offset

        return grid
    
    
    def forward(self, grid):
        """
        Simplified inverse normalization method focusing on
        power transformation only.

        Does not revert Min-Max scaling!
        """
        grid = torch.clip(grid, 0, None)
        grid = torch.pow(grid, (1/self.pow))

        #grid = (grid * (self.max_val - self.min_val)) + self.min_val

        return grid
