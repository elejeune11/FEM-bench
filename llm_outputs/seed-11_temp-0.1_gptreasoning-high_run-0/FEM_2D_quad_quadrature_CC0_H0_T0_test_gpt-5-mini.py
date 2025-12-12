def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    import pytest
    invalid_values = [0, 2, 3, 5, 6, 7, 8, 10, -1]
    for n in invalid_values:
        with pytest.raises(ValueError):
            fcn(n)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    import numpy as np
    tol = 1e-12
    for n in (1, 4, 9):
        (points, weights) = fcn(n)
        assert isinstance(points, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert points.shape == (n, 2)
        assert weights.shape == (n,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.allclose(weights.sum(), 4.0, atol=tol, rtol=1e-12)
        assert np.all(points[:, 0] >= -1 - tol)
        assert np.all(points[:, 0] <= 1 + tol)
        assert np.all(points[:, 1] >= -1 - tol)
        assert np.all(points[:, 1] <= 1 + tol)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    The tensor-product 1-point rule places a single node at the center (0,0) with
    weight 4. This integrates exactly any polynomial that is at most degree 1
    in each variable, i.e. constants and linear terms in x or y.
    For higher-degree terms, the rule is no longer guaranteed to be exact.
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check
    (i.e. the quadrature does not reproduce the analytic integrals).
    """
    import numpy as np
    (points, weights) = fcn(1)

    def analytic(i, j):
        ax = 0.0 if i % 2 == 1 else 2.0 / (i + 1)
        ay = 0.0 if j % 2 == 1 else 2.0 / (j + 1)
        return ax * ay
    tol = 1e-12
    for i in range(0, 2):
        for j in range(0, 2):
            vals = points[:, 0] ** i * points[:, 1] ** j
            q = float((weights * vals).sum())
            a = analytic(i, j)
            assert np.allclose(q, a, atol=tol, rtol=1e-12)
    for (i, j) in [(2, 0), (0, 2), (2, 2)]:
        vals = points[:, 0] ** i * points[:, 1] ** j
        q = float((weights * vals).sum())
        a = analytic(i, j)
        assert not np.allclose(q, a, atol=tol, rtol=1e-12)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
    Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
    any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
    in x and y).
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    import numpy as np
    (points, weights) = fcn(4)

    def analytic(i, j):
        ax = 0.0 if i % 2 == 1 else 2.0 / (i + 1)
        ay = 0.0 if j % 2 == 1 else 2.0 / (j + 1)
        return ax * ay
    tol = 1e-12
    for i in range(0, 4):
        for j in range(0, 4):
            vals = points[:, 0] ** i * points[:, 1] ** j
            q = float((weights * vals).sum())
            a = analytic(i, j)
            assert np.allclose(q, a, atol=tol, rtol=1e-12)
    vals = points[:, 0] ** 4 * points[:, 1] ** 0
    q = float((weights * vals).sum())
    a = analytic(4, 0)
    assert not np.allclose(q, a, atol=tol, rtol=1e-12)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
    Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
    for any polynomial where the degree in each variable is ≤ 5.
    The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    import numpy as np
    (points, weights) = fcn(9)

    def analytic(i, j):
        ax = 0.0 if i % 2 == 1 else 2.0 / (i + 1)
        ay = 0.0 if j % 2 == 1 else 2.0 / (j + 1)
        return ax * ay
    tol = 1e-12
    for i in range(0, 6):
        for j in range(0, 6):
            vals = points[:, 0] ** i * points[:, 1] ** j
            q = float((weights * vals).sum())
            a = analytic(i, j)
            assert np.allclose(q, a, atol=tol, rtol=1e-12)
    vals = points[:, 0] ** 6 * points[:, 1] ** 0
    q = float((weights * vals).sum())
    a = analytic(6, 0)
    assert not np.allclose(q, a, atol=tol, rtol=1e-12)