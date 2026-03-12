import numpy as np
import matplotlib.pyplot as plt
import pytest

def test_example():
    # Given...
    num1 = 1
    num2 = 1

    # When...
    result = num1 + num2

    # Expect...
    assert result == 2


def gaussian(x, x0, sigma):
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def test_fdtd_solves_one_wave():
    x = np.linspace(-1, 1, 100)
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    fdtd = FDTD1D(initial_e)
    fdtd.load_initial_field(initial_e)

    fdtd.run_until(t_final)

    e_solved = get_e()

    e_expected = 0.5 * (gaussian(x - c * t_final, x0, sigma) + gaussian(x + c * t_final, x0, sigma))

    assert np.allclose(e_solved, e_expected)


if __name__ == "__main__":
    pytest.main([__file__])
