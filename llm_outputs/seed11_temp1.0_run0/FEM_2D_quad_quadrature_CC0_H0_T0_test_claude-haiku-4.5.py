def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 3, 5, 6, 7, 8, 10, 16, -1, 100]
    for num_pts in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2), f'Points shape mismatch for {num_pts} points'
        assert weights.shape == (num_pts,), f'Weights shape mismatch for {num_pts} points'
        assert points.dtype == np.float64, f'Points dtype should be float64, got {points.dtype}'
        assert weights.dtype == np.float64, f'Weights dtype should be float64, got {weights.dtype}'
        assert np.isclose(np.sum(weights), 4.0), f'Weights sum to {np.sum(weights)} instead of 4.0 for {num_pts} points'
        assert np.all(points >= -1.0) and np.all(points <= 1.0), f'Not all points lie in [-1,1]×[-1,1] for {num_pts} points'

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule.
    The tensor-product 1-point rule places a single node at the center (0,0) with
    weight 4. This integrates exactly any polynomial that is at most degree 1
    in each variable, i.e. constants and linear terms in x or y.
    For higher-degree terms, the rule is no longer guaranteed to be exact.
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check
    (i.e. the quadrature does not reproduce the analytic integrals).
    """
    (points, weights) = fcn(1)

    def integrate_monomial(xi_pow: int, eta_pow: int) -> float:
        """Compute quadrature approximation of ∫∫ ξ^xi_pow η^eta_pow dξ dη over [-1,1]^2."""
        result = 0.0
        for i in range(len(weights)):
            (xi, eta) = points[i]
            result += weights[i] * xi ** xi_pow * eta ** eta_pow
        return result

    def analytical_integral(xi_pow: int, eta_pow: int) -> float:
        """Compute analytical integral of ξ^xi_pow η^eta_pow over [-1,1]^2."""

        def integral_1d(n):
            return (1 - (-1) ** (n + 1)) / (n + 1) if n >= 0 else 2.0
        return integral_1d(xi_pow) * integral_1d(eta_pow)
    for xi_pow in [0, 1]:
        for eta_pow in [0, 1]:
            quad = integrate_monomial(xi_pow, eta_pow)
            analytical = analytical_integral(xi_pow, eta_pow)
            assert np.isclose(quad, analytical), f'Exactness failed for ξ^{xi_pow} η^{eta_pow}: quad={quad}, analytical={analytical}'
    non_exact_found = False
    for xi_pow in [0, 1, 2]:
        for eta_pow in [0, 1, 2]:
            if xi_pow + eta_pow <= 2 and (xi_pow == 2 or eta_pow == 2):
                quad = integrate_monomial(xi_pow, eta_pow)
                analytical = analytical_integral(xi_pow, eta_pow)
                if not np.isclose(quad, analytical):
                    non_exact_found = True
                    break
        if non_exact_found:
            break
    assert non_exact_found, 'Expected non-exactness for at least one quadratic term'

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule.
    The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
    Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
    any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
    in x and y).
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(4)

    def integrate_monomial(xi_pow: int, eta_pow: int) -> float:
        """Compute quadrature approximation of ∫∫ ξ^xi_pow η^eta_pow dξ dη over [-1,1]^2."""
        result = 0.0
        for i in range(len(weights)):
            (xi, eta) = points[i]
            result += weights[i] * xi ** xi_pow * eta ** eta_pow
        return result

    def analytical_integral(xi_pow: int, eta_pow: int) -> float:
        """Compute analytical integral of ξ^xi_pow η^eta_pow over [-1,1]^2."""

        def integral_1d(n):
            return (1 - (-1) ** (n + 1)) / (n + 1) if n >= 0 else 2.0
        return integral_1d(xi_pow) * integral_1d(eta_pow)
    for xi_pow in range(4):
        for eta_pow in range(4):
            quad = integrate_monomial(xi_pow, eta_pow)
            analytical = analytical_integral(xi_pow, eta_pow)
            assert np.isclose(quad, analytical, atol=1e-14), f'Exactness failed for ξ^{xi_pow} η^{eta_pow}: quad={quad}, analytical={analytical}'
    non_exact_found = False
    for xi_pow in range(5):
        for eta_pow in range(5):
            if xi_pow == 4 or eta_pow == 4:
                quad = integrate_monomial(xi_pow, eta_pow)
                analytical = analytical_integral(xi_pow, eta_pow)
                if not np.isclose(quad, analytical, atol=1e-14):
                    non_exact_found = True
                    break
        if non_exact_found:
            break
    assert non_exact_found, 'Expected non-exactness for at least one quartic term'

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule.
    The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
    Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
    for any polynomial where the degree in each variable is ≤ 5.
    The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(9)

    def integrate_monomial(xi_pow: int, eta_pow: int) -> float:
        """Compute quadrature approximation of ∫∫ ξ^xi_pow η^eta_pow dξ dη over [-1,1]^2."""
        result = 0.0
        for i in range(len(weights)):
            (xi, eta) = points[i]
            result += weights[i] * xi ** xi_pow * eta ** eta_pow
        return result

    def analytical_integral(xi_pow: int, eta_pow: int) -> float:
        """Compute analytical integral of ξ^xi_pow η^eta_pow over [-1,1]^2."""

        def integral_1d(n):
            return (1 - (-1) ** (n + 1)) / (n + 1) if n >= 0 else 2.0
        return integral_1d(xi_pow) * integral_1d(eta_pow)
    for xi_pow in range(6):
        for eta_pow in range(6):
            quad = integrate_monomial(xi_pow, eta_pow)
            analytical = analytical_integral(xi_pow, eta_pow)
            assert np.isclose(quad, analytical, atol=1e-14), f'Exactness failed for ξ^{xi_pow} η^{eta_pow}: quad={quad}, analytical={analytical}'
    non_exact_found = False
    for xi_pow in range(7):
        for eta_pow in range(7):
            if xi_pow == 6 or eta_pow == 6:
                quad = integrate_monomial(xi_pow, eta_pow)
                analytical = analytical_integral(xi_pow, eta_pow)
                if not np.isclose(quad, analytical, atol=1e-14):
                    non_exact_found = True
                    break
        if non_exact_found:
            break
    assert non_exact_found, 'Expected non-exactness for at least one degree-6 term'