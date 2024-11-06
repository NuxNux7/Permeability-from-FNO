from setuptools import setup, find_packages

setup(
    name="permFNO",
    version="0.1",
    packages=find_packages(''),
    install_requires=[
        'torch',
        'numpy<2.0.0',
        'scikit-image',
        'vtk',
        'h5py',
        'tensorboard',
        'tabulate',
        'opt_einsum',
    ],
    author='Lukas Schröder',
    author_email='lukas.ls.schroeder@fau.de',
    description='This package uses the Fourier neural operator for predicting the permeability of porous media.'
)