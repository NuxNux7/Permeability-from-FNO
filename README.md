Estimating the Permeability of Porous Media with Fourier Neural Operators
============
Master Thesis Code by Lukas Schröder
==============

Abstract
--------------
Rapid and accurate permeability prediction in porous media is a critical yet computationally demanding task in fields like geoscience and materials engineering. To address this, a surrogate model based on Fourier Neural Operators, which are neural networks specialized in learning partial differential equations, is introduced. The core concept is to first predict the entire physical pressure field from the input geometry, from which permeability is then derived. This approach grounds the model in the underlying flow physics. FNOs leverage the frequency domain to apply their weights and are independent of input resolution. Additionally, the improved factorized Fourier neural operators (FFNO) with separate convolutions for every dimension and skip connections was compared to the original and a hybrid approach in regard to quality and performance. In a 3D dataset consisting of the flow around randomly shifted spheres an $R^2$ score of 0.99 was achieved, demonstrating the capabilities of the concept. On a second dataset of complicated and diverse geometries, it achieved an $R^2$ score of 0.93 with an increasing error for smaller permeabilities. A 2D benchmark comparison revealed a key trade-off: the FNO model was more computationally efficient and achieved a slightly higher peak accuracy, while the Swin transformer demonstrated superior robustness across more challenging datasets. Although the training process is prone to instabilities, the results underline that all tested FNOs are adequate for the task on the tested cases and that the difference between their versions are small, but noticeable. If the effort of creating and training a dataset is worth it, depends on the number of times it is used afterwards, but in general, FNO based surrogate models bridge the gap of fast, but limited analytical solutions and universal, but slower numerical simulations.


Structure
--------------

- permFNO:       Python package for the Fourier neural operator permeability prediction
    - data:          Handles the datasets, I/O and transformations
    - models:        Contains the FNO and FFNO together with its sub-parts
    - learning:      Methods needed for neural network training
    - main.py:       The main function is used for setup and running the training process

- lbmpy:         Code used for LBM simulation with the package lbmpy
    - LBM2D.py: Code for simulating the 2D geometries
    - LBM3D.py: Code for simulating the 3D geometries
    - permeability_comparison.py: Comparison method for given and calculated values


Data
-----------
The datasets used in this research is present at https://data.mendeley.com/preview/29m6gyhy2r?a=ba112c32-49fa-460f-9d76-b2f3ad8818cb


Work
-----------
The thesis and its presentation slides are available as PDFs.
