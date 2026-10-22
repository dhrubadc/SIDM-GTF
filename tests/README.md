# SIDM-GTF tests

Run the fast unit suite from the repository root after installing the package and test extra:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The suite covers package imports; geometric identities; conductivity formulae; initial-grid and mass bookkeeping; hydrostatic initialization; luminosity boundaries; state shapes and positivity; residual dimensions and basic identities; analytic-Jacobian shape and finite-difference agreement; energy diagnostics; Newton helper functions; output round trips; and mocked timestep-control branches.

## Scope and limitations

- These are unit and focused control-flow tests, not a validation of the physical conductivity model against an external reference.
- The Jacobian finite-difference test is a numerical consistency check. If it fails, investigate the analytic derivative and finite-difference sensitivity rather than loosening tolerance automatically.
- The mocked time-evolution tests do not establish convergence of a real nonlinear solve or correctness of a long gravothermal evolution.
- No test here claims global energy conservation, spatial or temporal convergence, restart correctness, or high-density stability. Those require separate integration/validation tests.
- The `settings` module contains mutable global parameters; fixtures monkeypatch the small test configuration so values are restored after each test.
