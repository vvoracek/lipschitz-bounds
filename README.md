# lipschitz-bounds
This repository implements the Gram iteration method, power iteration, and accelerated power iteration and provides a simple speed comparison.
The provided script runs numpy on CPU and the relative speed of the methods is similar to the one reported in the paper. 

The speed strongly depends on the distribution (here we use gaussian) of the matrix entries and in the framework used. for implementation.

Delattre et al. : Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration
https://arxiv.org/pdf/2305.16173.pdf

De Sa et al.
: Accelerated Stochastic Power Iteration
https://arxiv.org/pdf/1707.02670.pdf
