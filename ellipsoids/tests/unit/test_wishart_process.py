"""
Unit tests for core.wishart_process — first-principles checks.

These tests use synthetic W matrices (no pkl required) and verify
mathematical properties that must hold regardless of the specific
coefficient values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.wishart_process import WishartProcessModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEGREE = 3
NUM_DIMS = 2
EXTRA_DIMS = 1
VARIANCE_SCALE = 1e-3
DECAY_RATE = 0.5
DIAG_TERM = 1e-4


@pytest.fixture(scope="module")
def model():
    return WishartProcessModel(
        degree=DEGREE,
        num_dims=NUM_DIMS,
        extra_dims=EXTRA_DIMS,
        variance_scale=VARIANCE_SCALE,
        decay_rate=DECAY_RATE,
        diag_term=DIAG_TERM,
    )


@pytest.fixture(scope="module")
def W(model):
    """Small random W with fixed seed."""
    key = jax.random.PRNGKey(42)
    shape = (DEGREE, DEGREE, NUM_DIMS, NUM_DIMS + EXTRA_DIMS)
    return jax.random.normal(key, shape=shape) * 0.1


@pytest.fixture(scope="module")
def x_grid():
    """5×5 grid of stimulus points in [-0.7, 0.7]^2."""
    pts = np.linspace(-0.7, 0.7, 5)
    g = np.stack(np.meshgrid(pts, pts, indexing="ij"), axis=-1)  # (5,5,2)
    return jnp.array(g.reshape(-1, 2))  # (25, 2)


# ---------------------------------------------------------------------------
# compute_U
# ---------------------------------------------------------------------------


class TestComputeU:
    def test_output_shape(self, model, W, x_grid):
        """U should have shape (N, num_dims_cov, num_dims + extra_dims)."""
        U = model.compute_U(W, x_grid)
        N = x_grid.shape[0]
        assert U.shape == (N, NUM_DIMS, NUM_DIMS + EXTRA_DIMS)

    def test_output_finite(self, model, W, x_grid):
        """All U values must be finite."""
        U = model.compute_U(W, x_grid)
        assert jnp.all(jnp.isfinite(U))

    def test_single_point(self, model, W):
        """compute_U with a single point should return shape (1, ndims_cov, ndims_extra)."""
        x = jnp.array([[0.0, 0.0]])
        U = model.compute_U(W, x)
        assert U.shape == (1, NUM_DIMS, NUM_DIMS + EXTRA_DIMS)

    def test_clips_to_unit_interval(self, model, W):
        """Points outside [-1, 1] should be clipped, not raise an error."""
        x_oob = jnp.array([[2.0, -3.0], [-1.5, 1.5]])
        U = model.compute_U(W, x_oob)
        assert jnp.all(jnp.isfinite(U))


# ---------------------------------------------------------------------------
# compute_Sigmas
# ---------------------------------------------------------------------------


class TestComputeSigmas:
    def test_output_shape(self, model, W, x_grid):
        """Sigma should have shape (N, num_dims_cov, num_dims_cov)."""
        U = model.compute_U(W, x_grid)
        Sigma = model.compute_Sigmas(U)
        N = x_grid.shape[0]
        assert Sigma.shape == (N, NUM_DIMS, NUM_DIMS)

    def test_symmetric(self, model, W, x_grid):
        """Every Sigma_i must be symmetric."""
        U = model.compute_U(W, x_grid)
        Sigma = np.array(model.compute_Sigmas(U))
        np.testing.assert_allclose(Sigma, Sigma.transpose(0, 2, 1), atol=1e-12)

    def test_positive_definite(self, model, W, x_grid):
        """Every Sigma_i must be positive definite (all eigenvalues > 0)."""
        U = model.compute_U(W, x_grid)
        Sigma = np.array(model.compute_Sigmas(U))
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues found: {eigvals.min()}"

    def test_consistent_with_compute_U(self, model, W, x_grid):
        """Sigma should equal U @ U.T + diag_term * I."""
        U = np.array(model.compute_U(W, x_grid))
        Sigma_direct = np.array(model.compute_Sigmas(jnp.array(U)))
        Sigma_from_U = U @ U.transpose(0, 2, 1) + DIAG_TERM * np.eye(NUM_DIMS)
        np.testing.assert_allclose(Sigma_direct, Sigma_from_U, atol=1e-10)

    def test_diag_term_sets_minimum_eigenvalue(self, model, W, x_grid):
        """Minimum eigenvalue of Sigma must be >= diag_term."""
        U = model.compute_U(W, x_grid)
        Sigma = np.array(model.compute_Sigmas(U))
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals >= DIAG_TERM - 1e-12)
