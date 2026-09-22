# SIDM Gravothermal Fluid Solver

A one-dimensional Lagrangian fully-implicit coupled gravothermal fluid solver for
self-interacting dark matter halos. The code and associated tests will be 
described in Dutta Chowdhury and Croton (in prep).

## Key Features

- Spherically symmetric halo evolution
- Hydrostatic equilibrium and energy conservation
- Conductive heat transport
- Fully implicit time integration
- Coupled hydrostatic and energy residuals
- Analytical Jacobian
- Newton iterations for root finding
- Currently supports NFW initial conditions (with optional truncation)

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

### Install from PyPI

The latest released version can be installed using `pip`:

```bash
pip install sidm-gtf
```

### Install from GitHub

To obtain the latest released version directly from GitHub:

```bash
git clone https://github.com/dhrubadc/sidm-gtf.git
cd sidm-gtf
pip install .
```

In both cases, you can then import the package in Python:

```python
import gtf
```





