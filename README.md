Estimating the Permeability of Porous Media with Fourier Neural Operators
============
Master Thesis Code by Lukas Schröder
==============

Structure
--------------

- data:          Handles the datasets, I/O and transformations
- models:        Contains the FNO and FFNO together with its sub-parts
- learning:      Methods needed for neural network training
- main.py        The main function is used for setup and running the training process


Abstract
--------------
This work introduces and investigates a surrogate model for predicting the permeability of porous media based on Fourier neural operators (FNOs). The core concept is to predict the entire pressure field created by the input geometry and then to extract the resulting pressure difference afterwards to allow the network a closer proximity to the flow phenomena in physics. FNOs leverage the frequency domain to apply their weights, are independent of input resolution and thus provide good results in a short training time compared to other neural networks. Techniques like weight normalization, cosine learning rate scheduling with warmups and the Adam optimizer with weight decay are applied for improved stability and performance. Additionally, the improved factorized Fourier neural operators (FFNO) with separate weights and convolutions for every dimension and skip connections for deeper and smaller models is compared to the original approach. In a 3D dataset consisting of the flow around randomly shifted spheres simulated by the Lattice-Boltzmann method an R2-score of 99.91 with a 4-layer FFNO was achieved, proving the capabilities of the concept. To show how the model can handle small datasets consisting of complicated and diverse geometries, a second experiment was conducted on a different set. It achieved an R2-score of 88.54 with an increasing error for smaller permeabilities. When these extreme cases are filtered out, the model scores again above 90, showing that it is also able to represent more complex geometries. The Swin transformer served as a comparison on a 2D dataset, where it achieved a better result in a fraction of its time. Analyzing the results highlighted that when speed is important the 4-layer original FNO approach is optimal and that the 8-layer FFNO provides the best accuracy. If the effort of creating and training a dataset is worth it, depends on the number of times it is used afterwards, but in general, FNO based surrogate models bridge the gap of fast, but limited analytical solutions and universal, but slower numerical simulations.
