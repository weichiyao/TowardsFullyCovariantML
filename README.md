# TowardsFullyCovariantML

## Authors:
TowardsFullyCovariantML contains the codes to reproduce the experiments in the Transactions on Machine Learning Research paper (to appear) [Towards fully covariant machine learning](https://arxiv.org/abs/2301.13724).

Some contributor names appear to the right of this GitHub page because we imported their codes from their public GitHub repository [equivariant-MLP](https://github.com/mfinzi/equivariant-MLP.git).

## Introduction
This GitHub repository links to codes for the following three experiments in the paper.

### Passive symmetry of units covariance on black-body radiation
In this experiment, we generate noisy samples of intensities as a function of wavelength and temperature, 
and the learning task is to predict the intensity for different values of wavelengths and temperatures.
We found that the units-covariant model with an extra-dimensional constant can outperform 
the baseline multi-layer perceptron model, suggesting passive symmetry brings new capabilities.

The codes are provided in [this repository](https://github.com/davidwhogg/LearnDimensionalConstant). 

### Passive *O*(3) symmetry example on springy double pendulum
In this example, we consider the springy double pendulum, a system that is subject to 
a passive *O*(3) symmetry (equivariance w.r.t. orthogonal coordinate
transformations), an active O(2) symmetry (equivariance w.r.t. rotations and reflections in the
2D plane normal to the gravity), and an active time-translation symmetry, arising from the fact that the
total energy is conserved.
The learning task is to predict the dynamics of the double pendulum using *O*(3)-equivariant models.
In this experiment, we consider the three models: (Known-g) an *O*(3)-equivariant model where the gravity is an input to the model, 
; (No-g) an *O*(3)-equivariant model where the gravity is not given, and 
(learned-g) an *O*(3)-equivariant model that uses the position and momenta as well as an unknown vector that the model learns. 
The results show that *O*(3)-equivariance permits the learning of the gravity vector from data with only minimal impact on performance.

The codes are provided in [this repository](https://github.com/weichiyao/ScalarEMLP/tree/learn-g).


### Covariant vs non-covariant data normalization example on springy double pendulum
To make contemporary neural network models numerically stable, it is conventional to normalize the input
data, and possibly also layers of the network with either layer normalization or batch normalization. This
normalization usually involves shifting and scaling the features or latent variables to bring them closer to
zero mean and unit variance (or something akin to these).
However, naïve normalization will in general break the passive symmetries of geometry and units. 
In the paper, we again use the springy double pendulum example 
and empirically show that non-covariant data normalization can lead to severely downgraded performance, 
whereas covariant data normalization does not.

The codes are provided in [this repository](https://github.com/weichiyao/ScalarEMLP/tree/normalization).


