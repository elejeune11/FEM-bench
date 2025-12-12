def test_quad_quadrature_2D_invalid_inputs(fcn):
    """
    Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(3)
    with pytest.raises(ValueError):
        fcn(5)

def test_quad_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(np.sum(weights), 4.0)
        assert np.all(-1 <= points[:, 0]) and np.all(points[:, 0] <= 1)
        assert np.all(-1 <= points[:, 1]) and np.all(points[:, 1] <= 1)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """
    Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    The tensor-product 1-point rule places a single node at the center (0,0) with
    weight 4. This integrates exactly any polynomial that is at most degree 1
    in each variable, i.e. constants and linear terms in x or y.
    For higher-degree terms, the rule is no longer guaranteed to be exact.
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check
    (i.e. the quadrature does not reproduce the analytic integrals).
    """
    (points, weights) = fcn(1)
    (x, y) = points[0]
    assert np.isclose(x, 0) and np.isclose(y, 0)
    assert np.isclose(weights[0], 4.0)
    assert np.isclose(fcn(1)[1][0] * 1, 4)
    assert np.isclose(fcn(1)[1][0] * x, 0)
    assert np.isclose(fcn(1)[1][0] * y, 0)
    assert not np.isclose(fcn(1)[1][0] * x ** 2, 4 / 3)
    assert not np.isclose(fcn(1)[1][0] * y ** 2, 4 / 3)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """
    Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
    Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
    any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
    in x and y).
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(4)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 0), 4)
    assert np.isclose(np.sum(weights * points[:, 0] ** 1 * points[:, 1] ** 0), 0)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 1), 0)
    assert np.isclose(np.sum(weights * points[:, 0] ** 2 * points[:, 1] ** 0), 4 / 3)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 2), 4 / 3)
    assert np.isclose(np.sum(weights * points[:, 0] ** 3 * points[:, 1] ** 0), 0)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 3), 0)
    assert not np.isclose(np.sum(weights * points[:, 0] ** 4 * points[:, 1] ** 0), 4 / 5)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """
    Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
    Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
    for any polynomial where the degree in each variable is ≤ 5.
    The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(9)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 0), 4)
    assert np.isclose(np.sum(weights * points[:, 0] ** 1 * points[:, 1] ** 0), 0)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 1), 0)
    assert np.isclose(np.sum(weights * points[:, 0] ** 2 * points[:, 1] ** 0), 4 / 3)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 2), 4 / 3)
    assert np.isclose(np.sum(weights * points[:, 0] ** 3 * points[:, 1] ** 0), 0)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 3), 0)
    assert np.isclose(np.sum(weights * points[:, 0] ** 4 * points[:, 1] ** 0), 4 / 5)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 4), 4 / 5)
    assert np.isclose(np.sum(weights * points[:, 0] ** 5 * points[:, 1] ** 0), 0)
    assert np.isclose(np.sum(weights * points[:, 0] ** 0 * points[:, 1] ** 5), 0)
    assert not np.isclose(np.sum(weights * points[:, 0] ** 6 * points[:, 1] ** 0), 4 / 7)