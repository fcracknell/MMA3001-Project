"""Tests for the RTN frame transformation."""

import numpy as np
import pytest

from sgp4corr.frames import rtn_basis, to_rtn

RNG = np.random.default_rng(1)
R_REF = RNG.normal(size=(50, 3)) * 7000.0
V_REF = RNG.normal(size=(50, 3)) * 7.5


def test_basis_is_orthonormal_and_right_handed():
    R, T, N = rtn_basis(R_REF, V_REF)
    for a in (R, T, N):
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    for a, b in ((R, T), (T, N), (R, N)):
        assert np.allclose(np.sum(a * b, axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.cross(R, T), N)


def test_circular_orbit_axes():
    R, T, N = rtn_basis([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    assert np.allclose(R, [1, 0, 0])
    assert np.allclose(T, [0, 1, 0])      # along the velocity
    assert np.allclose(N, [0, 0, 1])


def test_shift_along_each_axis_lands_in_that_component():
    R, T, N = rtn_basis(R_REF, V_REF)
    for k, axis in enumerate((R, T, N)):
        expected = np.zeros((50, 3))
        expected[:, k] = 2.5
        assert np.allclose(to_rtn(2.5 * axis, R_REF, V_REF), expected, atol=1e-9)


def test_rotation_preserves_length():
    delta = RNG.normal(size=(50, 3))
    rtn = to_rtn(delta, R_REF, V_REF)
    assert np.allclose(np.linalg.norm(rtn, axis=1), np.linalg.norm(delta, axis=1))


def test_single_vector_input():
    assert to_rtn([0.0, 3.0, 0.0], [7000.0, 0.0, 0.0], [0.0, 7.5, 0.0]).shape == (3,)


@pytest.mark.parametrize("r, v", [
    ([0.0, 0.0, 0.0], [0.0, 7.5, 0.0]),          # zero position
    ([7000.0, 0.0, 0.0], [1.0, 0.0, 0.0]),       # velocity parallel to position
    ([7000.0, 0.0], [0.0, 7.5]),                 # wrong dimension
])
def test_invalid_reference_state_raises(r, v):
    with pytest.raises(ValueError):
        rtn_basis(r, v)