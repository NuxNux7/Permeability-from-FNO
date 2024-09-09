from setuptools import setup, find_packages

setup(
    name="permFNO",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy<2.0.0',
        'scikit-image',
        'vtk',
        'h5py',
        'tensorboard',
        'tabulate',
    ],
)