# TowardsFullyCovariantML

## Authors:
TowardsFullyCovariantML contains the codes to reproduce the experiments in the Transactions on Machine Learning Research paper (to appear) [Towards fully covariant machine learning](https://arxiv.org/abs/2301.13724).

Some contributor names appear to the right of this GitHub page because we imported their codes from their public GitHub repository [equivariant-MLP](https://github.com/mfinzi/equivariant-MLP.git).

## Introduction
This GitHub repository links to codes for the following three experiments in the paper.

### Passive symmetry of units covariance on black-body radiation
In this example, we generate noisy samples of electromagnetic radiation intensity $B_{\lambda}(\lambda)$ 
as a function of wavelength and temperature according to
$$B_{\lambda}(\lambda) = \frac{2 h c^2}{\lambda^5}\left[\exp\frac{h c}{\lambda k T}-1\right]^{-1},$$
where $h$ is Planck’s constant, $c$ is the speed of light, $\lambda$ is the wavelength of the electromagnetic radiation, 
$k$ is Boltzmann’s constant, and $T$ is the temperature. 
The learning task is to predict the intensity for different values of wavelengths $\lambda$ and temperatures $T$.
We conducted three experiments, (A) a units-covariant regression using only $\lambda$, $T$, $c$, $k$; 
(B) a units covariant regression with an extra dimensional constant found by cross-validation; and 
(C) a standard multi-layer perceptron regression (MLP) with no units constraints
We found that the units-covariant model that learns an extra constant (with units consistent with $h$)
can outperform the baseline MLP with no units constraints, suggesting passive symmetry brings new capabilities.

The codes are provided in [this repository](https://github.com/davidwhogg/LearnDimensionalConstant). 

### Passive *O*(3) symmetry example on springy double pendulum
In this example, we consider the dissipationless spherical double pendulum with springs, with a pivot $o$ and two
masses connected by springs. The kinetic energy $\mathcal{T}$ and potential energy $\mathcal{U}$ of the system are given by

$$
\begin{align}
KE =&\frac{|\mathbf{p}_1|^2}{2m_1} +\frac{|\mathbf{p}_2|^2}{2m_2}, \\
PE =&\frac12 k_1(|\mathbf{q}_1-\mathbf{q}_o|-l_1)^2 + \frac12 k_2(|\mathbf{q}_2-\mathbf{q}_1|-l_2)^2 
    -m_1\,\mathbf{g}\cdot (\mathbf{q}_1-\mathbf{q}_o)- m_2 \,\mathbf{g}\cdot  (\mathbf{q}_2-\mathbf{q}_o),  
\end{align}
$$

where $\mathbf{q}_1, \mathbf{p}_1$ are the position and momentum vectors for mass $m_1$, similarly $\mathbf{q}_2, \mathbf{p}_2$ for mass $m_2$, and a position $\mathbf{q}_o$ for the pivot. The springs have scalar spring constants $k_1$, $k_2$, and natural lengths $l_1$, $l_2$. The gravitational acceleration vector is $\mathbf{g}$. 

This system is subject to a passive *O*(3) symmetry (equivariance w.r.t. orthogonal coordinate transformations), 
an active *O*(2) symmetry (equivariance w.r.t. rotations and reflections in the 2D plane normal to the gravity), 
and an active time-translation symmetry, arising from the fact that the total energy $\mathcal{T}+\mathcal{U}$ is conserved.
The learning task is to predict the dynamics of the double pendulum using *O*(3)-equivariant models.
In this experiment, we consider the three models: 
(Known-g) an *O*(3)-equivariant model where the gravity is an input to the model, 
(No-g) an *O*(3)-equivariant model where the gravity is not given, 
and (learned-g) an *O*(3)-equivariant model that uses the position and momenta as well as an unknown vector that the model learns. 
The results show that *O*(3)-equivariance permits the learning of the gravity vector from data with only minimal impact on performance.

The codes are provided in [this repository](https://github.com/weichiyao/ScalarEMLP/tree/learn-g).


### Covariant vs non-covariant data normalization example on springy double pendulum
To make contemporary neural network models numerically stable, it is conventional to normalize the input
data, and possibly also layers of the network with either layer normalization or batch normalization. This
normalization usually involves shifting and scaling the features or latent variables to bring them closer to
zero mean and unit variance (or something akin to these).
However, naïve normalization will in general break the passive symmetries of geometry and units. 
We again use the springy double pendulum as an example 
and empirically show that non-covariant data normalization can lead to severely downgraded performance, 
whereas covariant data normalization does not.

The codes are provided in [this repository](https://github.com/weichiyao/ScalarEMLP/tree/normalization).


