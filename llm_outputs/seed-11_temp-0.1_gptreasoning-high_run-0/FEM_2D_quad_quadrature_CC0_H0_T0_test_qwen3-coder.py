def test_quad_quadrature_2D_invalid_inputs(fcn: Callable) -> None:
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    for num_pts in [-1, 0, 2, 3, 5, 10]:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_quad_quadrature_2D_basics(fcn: Callable) -> None:
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert abs(np.sum(weights) - 4.0) < 1e-14
        assert np.all(points[:, 0] >= -1.0) and np.all(points[:, 0] <= 1.0)
        assert np.all(points[:, 1] >= -1.0) and np.all(points[:, 1] <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn: Callable) -> None:
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
    (points, weights) = fcn(1)
    assert np.allclose(points, [[0.0, 0.0]])
    assert np.allclose(weights, [4.0])
    integrand = np.ones_like(points[:, 0])
    quad_result = np.sum(weights * integrand)
    analytic_result = 4.0
    assert abs(quad_result - analytic_result) < 1e-14
    integrand = points[:, 0]
    quad_result = np.sum(weights * integrand)
    analytic_result = 0.0
    assert abs(quad_result - analytic_result) < 1e-14
    integrand = points[:, 1]
    quad_result = np.sum(weights * integrand)
    analytic_result = 0.0
    assert abs(quad_result - analytic_result) < 1e-14
    integrand = points[:, 0] * points[:, 1]
    quad_result = np.sum(weights * integrand)
    analytic_result = 0.0
    assert abs(quad_result - analytic_result) < 1e-14
    integrand = points[:, 0] ** 2
    quad_result = np.sum(weights * integrand)
    analytic_result = 4.0 / 3.0
    assert abs(quad_result - analytic_result) > 1e-14

def test_quad_quadrature_2D_degree_exactness_2x2(fcn: Callable) -> None:
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
    Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
    any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
    in x and y).
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(4)
    expected_points = np.array([[-1 / np.sqrt(3), -1 / np.sqrt(3)], [1 / np.sqrt(3), -1 / np.sqrt(3)], [-1 / np.sqrt(3), 1 / np.sqrt(3)], [1 / np.sqrt(3), 1 / np.sqrt(3)]])
    assert np.allclose(points, expected_points)
    expected_weights = np.array([1.0, 1.0, 1.0, 1.0])
    assert np.allclose(weights, expected_weights)
    monomials_exact = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (3, 0), (0, 3), (3, 1), (1, 3), (2, 2)]
    for (i, j) in monomials_exact:
        integrand = points[:, 0] ** i * points[:, 1] ** j
        quad_result = np.sum(weights * integrand)
        analytic_x = 2 / (i + 1) if i % 2 == 0 else 0
        analytic_y = 2 / (j + 1) if j % 2 == 0 else 0
        analytic_result = analytic_x * analytic_y
        assert abs(quad_result - analytic_result) < 1e-12
    integrand = points[:, 0] ** 4
    quad_result = np.sum(weights * integrand)
    analytic_result = 4.0 / 5.0
    assert abs(quad_result - analytic_result) > 1e-12

def test_quad_quadrature_2D_degree_exactness_3x3(fcn: Callable) -> None:
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
    Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
    for any polynomial where the degree in each variable is ≤ 5.
    The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(9)
    monomials_exact = []
    for i in range(6):
        for j in range(6):
            monomials_exact.append((i, j))
    for (i, j) in monomials_exact:
        integrand = points[:, 0] ** i * points[:, 1] ** j
        quad_result = np.sum(weights * integrand)
        analytic_x = 2 / (i + 1) if i % 2 == 0 else 0
        analytic_y = 2 / (j + 1) if j % 2 == 0 else 0
        analytic_result = analytic_x * analytic_y
        assert abs(quad_result - analytic_result) < 1e-10, f'Failed for x^{i} * y^{j}'
    integrand = points[:, 0] ** 6
    quad_result = np.sum(weights * integrand)
    analytic_result = 4.0 / 7.0
    assert abs(quad_result - analytic_result) > 1e-10