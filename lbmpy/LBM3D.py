# Code used for simulating the geometries with the LBM solver lbmpy in 2D

from pystencils import Target
from lbmpy.session import *
from lbmpy.methods import create_trt_with_magic_number
from lbmpy.relaxationrates import relaxation_rate_from_magic_number
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity

from skimage.transform import resize
import h5py
import numpy as np
import os

import time


def load_geometry_from_file(path, domain_size, sample, flip=False):
    file = h5py.File(path, 'r')
    print(file["name"][sample])
    geometry = resize(file["input"]['fill'][sample], (domain_size[0]+2, domain_size[0]+2, domain_size[0]+2), anti_aliasing=True)
    array = np.array(geometry, dtype=int)

    if flip:
        array = np.flip(array, axis=-1)

    return array

def create_mask_from_geometry(grometry, domain_size, buffer):
    mask = np.zeros((domain_size[0]+2, domain_size[1]+2, domain_size[2]+2), dtype=bool)
    start_geo = buffer
    end_geo = start_geo + domain_size[0] + 2
    mask[:, :, start_geo:end_geo] = geometry

    return mask


def simulate_medium(geometry, domain_size, sample_name, buffer=32):

    # calculate TRT relaxation rates
    '''reference_length = 0.05
    maximal_velocity = 0.00008
    reynolds_number = 50
    kinematic_vicosity = (reference_length * maximal_velocity) / reynolds_number
    initial_velocity=(maximal_velocity, 0)
    omega = relaxation_rate_from_lattice_viscosity(kinematic_vicosity)'''
    omega = 1.0
    rr_odd = relaxation_rate_from_magic_number(omega, (3/16))

    stencil = LBStencil(Stencil.D3Q19)
    lbm_config = LBMConfig(stencil=stencil, method=Method.TRT, relaxation_rates=[omega, rr_odd])

    config = CreateKernelConfig(target=Target.GPU, gpu_indexing_params={'block_size': (128, 1, 1)})

    sc1 = LatticeBoltzmannStep(domain_size=domain_size,
                               periodicity=(False, False, False),
                               initial_velocity=(0, 0, 0.00008),
                               lbm_config=lbm_config,
                               config=config,
                               name=sample_name)

    # inflow: left (z-)
    inflow = UBB((0, 0, 0.00008), dim=sc1.method.dim)
    sc1.boundary_handling.set_boundary(inflow, make_slice[:, :, 0])

    # outflow: right (z+)
    stencil = LBStencil(Stencil.D3Q19)
    outflow = FixedDensity(1.0)
    sc1.boundary_handling.set_boundary(outflow, make_slice[:, :, -1])

    # medium
    mask = np.zeros((domain_size[0]+2, domain_size[1]+2, domain_size[2]+2), dtype=bool)
    start_geo = buffer
    end_geo = start_geo + domain_size[0] + 2
    mask[:, :, start_geo:end_geo] = geometry

    wall = NoSlip()
    
    sc1.boundary_handling.set_boundary_from_mask(wall, mask=mask, ghost_layers=True, inner_ghost_layers=True)

    # walls: top, bottom, front, back (x+-, y+-)
    sc1.boundary_handling.set_boundary(wall, make_slice[0, :, :])
    sc1.boundary_handling.set_boundary(wall, make_slice[-1, :, :])
    sc1.boundary_handling.set_boundary(wall, make_slice[:, 0, :])
    sc1.boundary_handling.set_boundary(wall, make_slice[:, -1, :])

    sc1.boundary_handling.geometry_to_vtk()

    # start simulation
    oldnorm = 1000
    i = 0
    stime = time.time()
    for i in range(50):
        start = time.time()
        sc1.run(5000)
        end = time.time()

        norm = np.sum(np.square(sc1.density[:, :, :] - 1))

        performance = (np.product(domain_size) * 1e-6 * 5000) / (end-start)
        print("norm: ", norm, ", iteration: ", (5000*(i+1)), ", MLUPS: ", performance)

        # abort if norm does not change
        if abs(oldnorm - norm) < 0.001:
            break
        elif norm >= 3000 and i == 9:
            print("Simulation ", str(sample_name), " failed!")
            i = -1
            break
        oldnorm = norm

    etime = time.time()

    print("time: ", etime-stime)

    sc1.write_vtk()

    return i


if __name__ == "__main__":

    path = 'external/ownSimulations/geometries.hp5'

    domain = 128
    buffer = 32
    flip = True
    domain_size = (domain - 2,
                   domain - 2,
                   domain - 2 + 2 * buffer)

    failed_sims = []

    for i in [12, 15, 16, 19, 37, 46, 48, 57, 61, 116, 128]:
        print("start simulation nbr.: ", str(i))

        geometry = load_geometry_from_file(path, domain_size, i, flip)
        sample_name = "porous_" + str(200+i) + "_reverse"
        result = simulate_medium(geometry, domain_size, sample_name, buffer)

        old_name = 'geometry_00000001.vti'
        new_name = "geometry" + str(200+i) + "_reverse.vti"
        os.rename(old_name, new_name)

        print()
        if result == -1:
            failed_sims.append(i)

    print(failed_sims)