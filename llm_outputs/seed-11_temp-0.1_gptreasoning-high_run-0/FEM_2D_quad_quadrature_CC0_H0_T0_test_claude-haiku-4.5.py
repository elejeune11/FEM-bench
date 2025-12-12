def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 3, 5, 6, 7, 8, 10, 16, 25, -1, 100]
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
        assert np.isclose(np.sum(weights), 4.0), f'Weights sum to {np.sum(weights)}, expected 4.0'
        assert np.all(points >= -1.0) and np.all(points <= 1.0), f'Some quadrature points lie outside [-1,1]×[-1,1]'

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule.
    The tensor-product 1-point rule places a single node at the center (0,0) with
    weight 4. This integrates exactly any polynomial that is at most degree 1
    in each variable, i.e. constants and linear terms in x or y.
    For higher-degree terms, the rule is no longer guaranteed to be exact.
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check.
    """
    (points, weights) = fcn(1)

    def integrate_monomial(xi_pow, eta_pow):
        """Compute quadrature approximation of x^xi_pow * y^eta_pow over [-1,1]^2."""
        quad_approx = np.sum(weights * points[:, 0] ** xi_pow * points[:, 1] ** eta_pow)
        return quad_approx

    def exact_integral_monomial(xi_pow, eta_pow):
        """Compute exact integral of x^xi_pow * y^eta_pow over [-1,1]^2."""
        if xi_pow % 2 == 1 or eta_pow % 2 == 1:
            return 0.0
        xi_integral = 2.0 / (xi_pow + 1)
        eta_integral = 2.0 / (eta_pow + 1)
        return xi_integral * eta_integral
    for xi_pow in range(2):
        for eta_pow in range(2):
            quad_val = integrate_monomial(xi_pow, eta_pow)
            exact_val = exact_integral_monomial(xi_pow, eta_pow)
            assert np.isclose(quad_val, exact_val), f'1-point rule not exact for x^{xi_pow}*y^{eta_pow}: quad={quad_val}, exact={exact_val}'
    quad_val = integrate_monomial(2, 0)
    exact_val = exact_integral_monomial(2, 0)
    assert not np.isclose(quad_val, exact_val), f'1-point rule should not be exact for x^2: quad={quad_val}, exact={exact_val}'

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule.
    The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
    Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
    any polynomial with degree ≤ 3 in each variable separately.
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness.
    """
    (points, weights) = fcn(4)

    def integrate_monomial(xi_pow, eta_pow):
        """Compute quadrature approximation of x^xi_pow * y^eta_pow over [-1,1]^2."""
        quad_approx = np.sum(weights * points[:, 0] ** xi_pow * points[:, 1] ** eta_pow)
        return quad_approx

    def exact_integral_monomial(xi_pow, eta_pow):
        """Compute exact integral of x^xi_pow * y^eta_pow over [-1,1]^2."""
        if xi_pow % 2 == 1 or eta_pow % 2 == 1:
            return 0.0
        xi_integral = 2.0 / (xi_pow + 1)
        eta_integral = 2.0 / (eta_pow + 1)
        return xi_integral * eta_integral
    for xi_pow in range(4):
        for eta_pow in range(4):
            quad_val = integrate_monomial(xi_pow, eta_pow)
            exact_val = exact_integral_monomial(xi_pow, eta_pow)
            assert np.isclose(quad_val, exact_val), f'2×2 rule not exact for x^{xi_pow}*y^{eta_pow}: quad={quad_val}, exact={exact_val}'
    quad_val = integrate_monomial(4, 0)
    exact_val = exact_integral_monomial(4, 0)
    assert not np.isclose(quad_val, exact_val), f'2×2 rule should not be exact for x^4: quad={quad_val}, exact={exact_val}'

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule.
    The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
    Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
    for any polynomial where the degree in each variable is ≤ 5.
    The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness.
    """
    (points, weights) = fcn(9)

    def integrate_monomial(xi_pow, eta_pow):
        """Compute quadrature approximation of x^xi_pow * y^eta_pow over [-1,1]^2."""
        quad_approx = np.sum(weights * points[:, 0] ** xi_pow * points[:, 1] ** eta_pow)
        return quad_approx

    def exact_integral_monomial(xi_pow, eta_pow):
        """Compute exact integral of x^xi_pow * y^eta_pow over [-1,1]^2."""
        if xi_pow % 2 == 1 or eta_pow % 2 == 1:
            return 0.0
        xi_integral = 2.0 / (xi_pow + 1)
        eta_integral = 2.0 / (eta_pow + 1)
        return xi_integral * eta_integral
    for xi_pow in range(6):
        for eta_pow in range(6):
            quad_val = integrate_monomial(xi_pow, eta_pow)
            exact_val = exact_integral_monomial(xi_pow, eta_pow)
            assert np.isclose(quad_val, exact_val), f'3×3 rule not exact for x^{xi_pow}*y^{eta_pow}: quad={quad_val}, exact={exact_val}'
    quad_val = integrate_monomial(6, 0)
    exact_val = exact_integral_monomial(6, 0)
    assert not np.isclose(quad_val, exact_val), f'3×3 rule should not be exact for x^6: quad={quad_val}, exact={exact_val}'