# SIDM Gravothermal Fluid Solver

A one-dimensional Lagrangian fully-implicit coupled gravothermal fluid solver for
self-interacting dark matter halos. The code and associated tests will be 
described in Dutta Chowdhury and Croton (in prep).

## Features

- Spherically symmetric halo evolution
- Hydrostatic equilibrium
- Conductive heat transport
- Fully implicit time integration
- Coupled hydrostatic and energy residuals
- Newton iterations
- Analytical Jacobian
- NFW initial conditions (with optional truncation)

## Citation

If you use this software in a publication, please cite:

```bibtex
@article{duttachowdhury26,
  title = {A Fully-Implicit Coupled Gravothermal Fluid Solver for SIDM Halos},
  author = {Dutta Chowdhury, Dhruba and Croton, Darren J.},
  year = 2026,
  journal = {In prep},
}
```

## Installation

```bash
pip install sidm-gtf
