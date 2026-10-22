"""Package import smoke tests."""


def test_package_imports():
    import gtf
    from gtf import conductivity, energy, geometry, initial_conditions
    from gtf import jacobian, luminosity, newton, output, residual, settings, state
    from gtf import time_evolution

    assert gtf.__version__
    assert all(module is not None for module in (
        conductivity, energy, geometry, initial_conditions, jacobian,
        luminosity, newton, output, residual, settings, state, time_evolution
    ))
