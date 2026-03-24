"""
Unit tests for analysis.ellipses_tools — first-principles checks.

Verifies round-trip consistency and geometric properties using
simple known-answer inputs (axis-aligned ellipses, circles).
"""

import numpy as np
import pytest

from analysis.ellipses_tools import (
    covMat_to_ellParamsQ,
    ellParamsQ_to_covMat,
    fit_2d_isothreshold_contour,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ellipse_points(a, b, theta_deg, xc=0.0, yc=0.0, n=200):
    """Generate (2, n) points on an ellipse with semi-axes a, b, rotation theta."""
    theta_rad = np.deg2rad(theta_deg)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_local = a * np.cos(t)
    y_local = b * np.sin(t)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    x = cos_t * x_local - sin_t * y_local + xc
    y = sin_t * x_local + cos_t * y_local + yc
    return np.vstack([x, y])


# ---------------------------------------------------------------------------
# ellParamsQ_to_covMat / covMat_to_ellParamsQ  round-trip
# ---------------------------------------------------------------------------


class TestEllParamsRoundTrip:
    @pytest.mark.parametrize(
        "a, b, theta_deg",
        [
            (0.10, 0.10, 0.0),  # circle
            (0.15, 0.08, 0.0),  # axis-aligned
            (0.15, 0.08, 45.0),  # rotated
            (0.20, 0.05, 30.0),  # elongated, rotated
        ],
    )
    def test_round_trip(self, a, b, theta_deg):
        """params → Sigma → params → Sigma should recover the original matrix."""
        # ellParamsQ_to_covMat takes theta in degrees
        Sigma = np.array(ellParamsQ_to_covMat(a, b, theta_deg))
        # covMat_to_ellParamsQ returns (eigvals, eigvecs, axes_lengths, theta_deg)
        _, _, axes_lengths, theta_deg2 = covMat_to_ellParamsQ(Sigma)
        a2, b2 = axes_lengths[0], axes_lengths[1]
        Sigma2 = np.array(ellParamsQ_to_covMat(a2, b2, theta_deg2))
        np.testing.assert_allclose(Sigma2, Sigma, rtol=1e-5, atol=1e-10)


# ---------------------------------------------------------------------------
# fit_2d_isothreshold_contour
# ---------------------------------------------------------------------------


class TestFit2dIsothresholdContour:
    def test_circle_recovers_isotropic_sigma(self):
        """
        Threshold points lying on a circle should yield an isotropic Sigma.
        Eigenvalue ratio of recovered Sigma must be close to 1.
        """
        r = 0.12
        w_ref = np.array([0.0, 0.0])
        w_comp = make_ellipse_points(r, r, 0.0, *w_ref)  # (2, 200)

        # params_ell = [xc, yc, a, b, theta_deg]
        _, _, params_ell, _ = fit_2d_isothreshold_contour(w_ref, w_comp, nTheta=200, flag_force_centered_ref=True)
        Sigma = np.array(ellParamsQ_to_covMat(*params_ell[2:]))
        eigvals = np.linalg.eigvalsh(Sigma)

        assert eigvals.min() > 0, "Sigma not positive definite"
        ratio = eigvals.max() / eigvals.min()
        assert ratio < 1.1, f"Expected near-isotropic Sigma, eigenvalue ratio={ratio:.4f}"

    @pytest.mark.parametrize(
        "a, b, theta_deg",
        [
            (0.15, 0.08, 0.0),
            (0.15, 0.08, 45.0),
        ],
    )
    def test_ellipse_recovers_correct_axes(self, a, b, theta_deg):
        """
        Threshold points on a known ellipse should recover semi-axes within 5%.
        params_ell[2] = major axis, params_ell[3] = minor axis (fit output).
        """
        w_ref = np.array([0.0, 0.0])
        w_comp = make_ellipse_points(a, b, theta_deg, *w_ref)

        _, _, params_ell, _ = fit_2d_isothreshold_contour(w_ref, w_comp, nTheta=200, flag_force_centered_ref=True)
        # params_ell = [xc, yc, a_fit, b_fit, theta_deg_fit]
        a_fit, b_fit = params_ell[2], params_ell[3]  # already sorted major, minor
        a_ref, b_ref = max(a, b), min(a, b)

        assert abs(a_fit - a_ref) / a_ref < 0.05, f"Major axis off: {a_fit:.4f} vs {a_ref:.4f}"
        assert abs(b_fit - b_ref) / b_ref < 0.05, f"Minor axis off: {b_fit:.4f} vs {b_ref:.4f}"
