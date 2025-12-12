def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError."""
    invalid_inputs = [0, 2, 3, 5, 6, 7, 8, 10, 16, 25, -1, 100]
    for num_pts in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1."""
    supported_pts = [1, 4, 9]
    for num_pts in supported_pts:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2), f'Points shape mismatch for {num_pts} points'
        assert weights.shape == (num_pts,), f'Weights shape mismatch for {num_pts} points'
        assert points.dtype == np.float64, f'Points dtype should be float64 for {num_pts} points'
        assert weights.dtype == np.float64, f'Weights dtype should be float64 for {num_pts} points'
        assert np.isclose(np.sum(weights), 4.0), f'Weights should sum to 4.0 for {num_pts} points'
        assert np.all(points >= -1.0), f'All points should have coordinates >= -1 for {num_pts} points'
        assert np.all(points <= 1.0), f'All points should have coordinates <= 1 for {num_pts} points'

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    The tensor-product 1-point rule places a single node at the center (0,0) with
    weight 4. This integrates exactly any polynomial that is at most degree 1
    in each variable, i.e. constants and linear terms in x or y.
    For higher-degree terms, the rule is no longer guaranteed to be exact.
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check
    (i.e. the quadrature does not reproduce the analytic integrals)."""
    (points, weights) = fcn(1)
    xi = points[:, 0]
    eta = points[:, 1]

    def analytic_integral(m, n):

        def int_1d(p):
            if p % 2 == 1:
                return 0.0
            else:
                return 2.0 / (p + 1)
        return int_1d(m) * int_1d(n)
    test_cases_exact = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for (m, n) in test_cases_exact:
        quad_result = np.sum(weights * xi ** m * eta ** n)
        exact_result = analytic_integral(m, n)
        assert np.isclose(quad_result, exact_result), f'1pt rule should be exact for x^{m}*y^{n}'
    quad_result_x2 = np.sum(weights * xi ** 2)
    exact_result_x2 = analytic_integral(2, 0)
    assert not np.isclose(quad_result_x2, exact_result_x2), '1pt rule should NOT be exact for x^2'

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
    Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
    any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
    in x and y).
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test."""
    (points, weights) = fcn(4)
    xi = points[:, 0]
    eta = points[:, 1]

    def analytic_integral(m, n):

        def int_1d(p):
            if p % 2 == 1:
                return 0.0
            else:
                return 2.0 / (p + 1)
        return int_1d(m) * int_1d(n)
    for m in range(4):
        for n in range(4):
            quad_result = np.sum(weights * xi ** m * eta ** n)
            exact_result = analytic_integral(m, n)
            assert np.isclose(quad_result, exact_result), f'2x2 rule should be exact for x^{m}*y^{n}'
    quad_result_x4 = np.sum(weights * xi ** 4)
    exact_result_x4 = analytic_integral(4, 0)
    assert not np.isclose(quad_result_x4, exact_result_x4), '2x2 rule should NOT be exact for x^4'

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
    Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
    for any polynomial where the degree in each variable is ≤ 5.
    The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test."""
    (points, weights) = fcn(9)
    xi = points[:, 0]
    eta = points[:, 1]

    def analytic_integral(m, n):

        def int_1d(p):
            if p % 2 == 1:
                return 0.0
            else:
                return 2.0 / (p + 1)
        return int_1d(m) * int_1d(n)
    for m in range(6):
        for n in range(6):
            quad_result = np.sum(weights * xi ** m * eta ** n)
            exact_result = analytic_integral(m, n)
            assert np.isclose(quad_result, exact_result), f'3x3 rule should be exact for x^{m}*y^{n}'
    quad_result_x6 = np.sum(weights * xi ** 6)
    exact_result_x6 = analytic_integral(6, 0)
    assert not np.isclose(quad_result_x6, exact_result_x6), '3x3 rule should NOT be exact for x^6'