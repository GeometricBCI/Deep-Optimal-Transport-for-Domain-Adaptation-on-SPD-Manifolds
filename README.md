# Deep-Optimal-Transport-for-Domain-Adaptation-on-SPD-Manifolds

This is the official GitHub repository for "Deep Optimal Transport for Domain Adaptation on SPD Manifolds".

This work aims to establish a framework based on optimal transport on the SPD manifold for the domain transfer problem on the SPD manifold. As many different kinds of neural signals, such as EEG, ECoG, fMRI, etc., are multichannel time series, many relevant engineering methods consider these problems from the perspective of the signal covariance matrix. Therefore, this research will directly benefit to related engineering methods, especially to the **Geometric BCI classifier** that we proposed.


[<img src="[https://arxiv.org/abs/2201.05745]"></img>](https://arxiv.org/abs/2201.05745)

![Illustration of Deep Optimal Transport](DOT.png)

### Usages

The DOT.py contained in the utils directory represents the principal contribution of this research, encompassing methodologies addressing discrepancies in marginal distribution and conditional distribution, as well as the DJDOT approach. Other baselines, along with the touched upon visualization results, can respectively be found within EMD_OT.py, RPA.py, and visualization.py files. Additionally, DOT_KU.py serves as an experimental script, applying the DOT method to the KU dataset. The hyperparameters may not necessarily be optimal, and should therefore be tailored to suit varying BCI tasks.

### Related Repositories

We extend our gratitude to the open-source community, which facilitates the wider dissemination of the work of other researchers as well as our own. The coding style in this repo is relatively rough. We welcome anyone to refactor it to make it more effective. The repositories related to our work are enumerated below:
[<img src="https://img.shields.io/badge/GitHub-Geometric BCI Classifier-b31b1b"></img>](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet)
[<img src="https://img.shields.io/badge/GitHub-Riemannian Procrustes Analysis-b31b1b"></img>](https://github.com/plcrodrigues/RPA)
[<img src="https://img.shields.io/badge/GitHub-Sliced Wasserstein-b31b1b"></img>](https://github.com/clbonet/SPDSW)
[<img src="https://img.shields.io/badge/GitHub-pyRiemann-b31b1b"></img>](https://github.com/pyRiemann/pyRiemann)
[<img src="https://img.shields.io/badge/GitHub-JDOT-b31b1b"></img>](https://github.com/rflamary/JDOT)
[<img src="https://img.shields.io/badge/GitHub-DeepJDOT-b31b1b"></img>](https://github.com/bbdamodaran/deepJDOT)

### Data Availability

All of this data can be accessed through the [**MOABB**] package (https://github.com/NeuroTechX/moabb). This package includes a benchmark dataset for advanced decoding algorithms, which comprises 12 open-access datasets and covers over 250 subjects.

### License and Attribution

Copyright 2022 S-Lab. All rights reserved.

Please refer to the LICENSE file for the licensing of our code.
