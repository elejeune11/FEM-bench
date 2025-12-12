def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    import pytest
    invalid_values = [0, 2, 3, 5, 6, 7, 8, 10, -1]
    for val in invalid_values:
        with pytest.raises(ValueError):
            fcn(val)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    import numpy as np
    for n in (1, 4, 9):
        (points, weights) = fcn(n)
        points = np.asarray(points)
        weights = np.asarray(weights)
        assert points.shape == (n, 2)
        assert weights.shape == (n,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(np.sum(weights), 4.0, rtol=0, atol=1e-12)
        assert np.all(points[:, 0] >= -1.0 - 1e-12)
        assert np.all(points[:, 0] <= 1.0 + 1e-12)
        assert np.all(points[:, 1] >= -1.0 - 1e-12)
        assert np.all(points[:, 1] <= 1.0 + 1e-12)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check.
    """
    import numpy as np
    (points, weights) = fcn(1)
    points = np.asarray(points)
    weights = np.asarray(weights)
    xi = points[:, 0]
    eta = points[:, 1]

    def analytic_integral(i, j):

        def one_dim(k):
            if k % 2 == 1:
                return 0.0
            return 2.0 / (k + 1)
        return one_dim(i) * one_dim(j)
    tol = 1e-12
    for i in (0, 1):
        for j in (0, 1):
            vals = xi ** i * eta ** j
            q = np.dot(weights, vals)
            a = analytic_integral(i, j)
            assert np.isclose(q, a, rtol=0, atol=tol)
    for (i, j) in ((2, 0), (0, 2), (2, 1), (1, 2)):
        vals = xi ** i * eta ** j
        q = np.dot(weights, vals)
        a = analytic_integral(i, j)
        assert not np.isclose(q, a, rtol=0, atol=tol)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness.
    """
    import numpy as np
    (points, weights) = fcn(4)
    points = np.asarray(points)
    weights = np.asarray(weights)
    xi = points[:, 0]
    eta = points[:, 1]

    def analytic_integral(i, j):

        def one_dim(k):
            if k % 2 == 1:
                return 0.0
            return 2.0 / (k + 1)
        return one_dim(i) * one_dim(j)
    tol = 1e-12
    for i in range(0, 4):
        for j in range(0, 4):
            vals = xi ** i * eta ** j
            q = np.dot(weights, vals)
            a = analytic_integral(i, j)
            assert np.isclose(q, a, rtol=0, atol=tol)
    (i, j) = (4, 0)
    vals = xi ** i * eta ** j
    q = np.dot(weights, vals)
    a = analytic_integral(i, j)
    assert not np.isclose(q, a, rtol=0, atol=tol)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness.
    """
    import numpy as np
    (points, weights) = fcn(9)
    points = np.asarray(points)
    weights = np.asarray(weights)
    xi = points[:, 0]
    eta = points[:, 1]

    def analytic_integral(i, j):

        def one_dim(k):
            if k % 2 == 1:
                return 0.0
            return 2.0 / (k + 1)
        return one_dim(i) * one_dim(j)
    tol = 1e-12
    for i in range(0, 6):
        for j in range(0, 6):
            vals = xi ** i * eta ** j
            q = np.dot(weights, vals)
            a = analytic_integral(i, j)
            assert np.isclose(q, a, rtol=0, atol=tol)
    (i, j) = (6, 0)
    vals = xi ** i * eta ** j
    q = np.dot(weights, vals)
    a = analytic_integral(i, j)
    assert not np.isclose(q, a, rtol=0, atol=tol)