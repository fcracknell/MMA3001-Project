"""Coordinate-frame helpers: the RTN (radial, along-track, cross-track) frame.

SGP4 returns positions in TEME, an Earth-centred inertial frame. A position
error expressed in TEME x/y/z has no physical meaning for the satellite, so
errors are rotated into RTN, a frame that moves with the reference satellite:

* R (radial)      - from the Earth's centre out through the satellite.
* T (along-track) - in the orbit plane, perpendicular to R, in the direction of travel.
* N (normal)      - perpendicular to the orbit plane (along the angular momentum).

All functions accept either a single vector of shape (3,) or a stack of
vectors of shape (n, 3).
"""

from __future__ import annotations

import numpy as np


def rtn_basis(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the RTN unit vectors from a reference position and velocity.

    Args:
        r: Reference position(s) in an inertial frame, shape (3,) or (n, 3).
        v: Reference velocity(ies) in the same frame, same shape as ``r``.

    Returns:
        Tuple ``(R, T, N)`` of unit vectors, each the same shape as ``r``.

    Raises:
        ValueError: If shapes differ, or if ``r`` and ``v`` are zero or
            parallel (the orbit plane is then undefined).
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    if r.shape != v.shape or r.shape[-1] != 3:
        raise ValueError(f"r and v must share shape (3,) or (n, 3); got {r.shape}, {v.shape}.")

    r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
    h = np.cross(r, v)                                   # orbital angular momentum
    h_norm = np.linalg.norm(h, axis=-1, keepdims=True)
    if np.any(r_norm == 0) or np.any(h_norm == 0):
        raise ValueError("r and v must be non-zero and non-parallel.")

    R = r / r_norm
    N = h / h_norm
    T = np.cross(N, R)
    return R, T, N


def to_rtn(delta: np.ndarray, r_ref: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
    """Express a position difference in the RTN frame of a reference state.

    This is a straight-line projection. For errors that are small compared
    with the orbit radius (a few km against ~7000 km) it is effectively exact;
    for errors of hundreds of km the orbit's curvature moves a little of the
    along-track error into the radial component.

    Args:
        delta: Position difference(s), e.g. predicted minus reference, shape (3,) or (n, 3).
        r_ref: Reference position(s), same shape as ``delta``.
        v_ref: Reference velocity(ies), same shape as ``delta``.

    Returns:
        Array of the same shape as ``delta`` holding the (R, T, N) components.

    Raises:
        ValueError: If the shapes are inconsistent or the reference state is degenerate.
    """
    delta = np.asarray(delta, dtype=float)
    R, T, N = rtn_basis(r_ref, v_ref)
    if delta.shape != R.shape:
        raise ValueError(f"delta shape {delta.shape} does not match reference shape {R.shape}.")
    return np.stack(
        [np.sum(delta * R, axis=-1), np.sum(delta * T, axis=-1), np.sum(delta * N, axis=-1)],
        axis=-1,
    )