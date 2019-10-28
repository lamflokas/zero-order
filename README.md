# Efficiently avoiding saddle points with zero order methods: No gradients required
Lampros Flokas, Emmanouil V. Vlatakis-Gkaragkounis, Georgios Piliouras

33rd Annual Conference on Neural Information Processing Systems (NeurIPS 2019)

## Abstract
We consider the case of derivative-free algorithms for non-convex optimization, also known as zero order algorithms, that use only function evaluations rather than gradients. For a wide variety of gradient approximators based on finite differences, we establish asymptotic convergence to second order stationary points using a carefully tailored application of the Stable Manifold Theorem.  Regarding efficiency, we introduce a noisy zero-order method that converges to second order stationary points, i.e avoids saddle points. Our algorithm uses only $\tilde{\mathcal{O}}(1 / \epsilon^2)$ approximate gradient calculations and, thus, it matches the converge rate guarantees of their exact gradient counterparts up to constants. In contrast to previous work, our convergence rate analysis avoids imposing additional dimension dependent slowdowns in the number of iterations required for non-convex zero order optimization.

## Code
The code is uisng a combination of Jupyter notebooks and python scripts.

You can install the dependencies via conda
```shell
conda env create -f environment.yml
```
The outputs are in the figures folder.
