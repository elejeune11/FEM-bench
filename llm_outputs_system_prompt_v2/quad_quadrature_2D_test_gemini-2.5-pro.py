def test_quad_quadrature_2D_invalid_inputs(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 3, 5, 8, 10, 16, -1, 1.0]
    for num_pts in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_quad_quadrature_2D_basics(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert isinstance(points, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(np.sum(weights), 4.0)
        assert np.all(points >= -1.0)
        assert np.all(points <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    Background
    ----------
    • The tensor-product 1-point rule places a single node at the center (0,0) with
      weight 4. This integrates exactly any polynomial that is at most degree 1
      in each variable, i.e. constants and linear terms in x or y.
    • For higher-degree terms, the rule is no longer guaranteed to be exact.
    What this test does
    -------------------
    1. Constructs a random affine polynomial P(x,y) = a00 + a10·x + a01·y.
    2. Adds quadratic contributions (x², y², xy).
         confirming that the 1-point rule is not exact for degree ≥ 2.
    Expected outcome
    ----------------
    • Exactness assertions for monomials of degree ≤ 1 should pass.
    • Non-exactness assertions for quadratics should fail the exactness check
      (i.e. the quadrature does not reproduce the analytic integrals).
    """
    (points, weights) = fcn(1)
    rng = np.random.default_rng(seed=101)
    (a, b, c) = rng.random(3)
    analytic_integral = 4.0 * a
    poly_values = a + b * points[:, 0] + c * points[:, 1]
    numerical_integral = np.dot(weights, poly_values)
    assert np.isclose(numerical_integral, analytic_integral)
    (d, e, f) = rng.random(3) + 0.1
    analytic_integral_quad = analytic_integral + d * (4.0 / 3.0) + e * (4.0 / 3.0)
    poly_values_quad = poly_values + d * points[:, 0] ** 2 + e * points[:, 1] ** 2 + f * points[:, 0] * points[:, 1]
    numerical_integral_quad = np.dot(weights, poly_values_quad)
    assert not np.isclose(numerical_integral_quad, analytic_integral_quad)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Background
    ----------
    • The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
      Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
      any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
      in x and y).
    • It is not guaranteed to integrate quartic terms (degree 4 in either variable) exactly.
    What this test does
    -------------------
    1. Builds a random polynomial P(x,y) = Σ a_ij x^i y^j for 0 ≤ i,j ≤ 3, i.e. all terms up
       to cubic in x and cubic in y. The quadrature approximation is compared to the analytic
       integral, the two results must match.
    2. Extends the polynomial with quartic terms x^4 and y^4. For these terms the 2×2 rule
       is not exact, so the quadrature result is expected to deviate from the analytic integral.
    Expected outcome
    ----------------
    • Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    • Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(4)
    rng = np.random.default_rng(seed=202)
    max_deg = 3
    coeffs = rng.random((max_deg + 1, max_deg + 1))
    analytic_integral = 0.0
    poly_values = np.zeros(4)
    for i in range(max_deg + 1):
        for j in range(max_deg + 1):
            poly_values += coeffs[i, j] * points[:, 0] ** i * points[:, 1] ** j
            if i % 2 == 0 and j % 2 == 0:
                integral_term = 2.0 / (i + 1.0) * (2.0 / (j + 1.0))
                analytic_integral += coeffs[i, j] * integral_term
    numerical_integral = np.dot(weights, poly_values)
    assert np.isclose(numerical_integral, analytic_integral)
    a40 = rng.random() + 0.1
    a04 = rng.random() + 0.1
    analytic_integral_quartic = analytic_integral + a40 * (4.0 / 5.0) + a04 * (4.0 / 5.0)
    poly_values_quartic = poly_values + a40 * points[:, 0] ** 4 + a04 * points[:, 1] ** 4
    numerical_integral_quartic = np.dot(weights, poly_values_quartic)
    assert not np.isclose(numerical_integral_quartic, analytic_integral_quartic)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Background
    ----------
    • The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
      Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
      for any polynomial where the degree in each variable is ≤ 5.
    • The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    What this test does
    -------------------
    1. Constructs a random polynomial P(x,y) = Σ a_ij x^i y^j with 0 ≤ i,j ≤ 5 (up to quintic
       terms in both variables). The quadrature approximation is compared to the analytic
       integral, the two results must match within tight tolerance.
    2. Extends the polynomial with degree-6 contributions (x^6, y^6, x^4y^2). These exceed the
       rule’s guaranteed exactness, so the quadrature result is expected to deviate from the
       analytic integral.
    Expected outcome
    ----------------
    • Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    • Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(9)
    rng = np.random.default_rng(seed=303)
    max_deg = 5
    coeffs = rng.random((max_deg + 1, max_deg + 1))
    analytic_integral = 0.0
    poly_values = np.zeros(9)
    for i in range(max_deg + 1):
        for j in range(max_deg + 1):
            poly_values += coeffs[i, j] * points[:, 0] ** i * points[:, 1] ** j
            if i % 2 == 0 and j % 2 == 0:
                integral_term = 2.0 / (i + 1.0) * (2.0 / (j + 1.0))
                analytic_integral += coeffs[i, j] * integral_term
    numerical_integral = np.dot(weights, poly_values)
    assert np.isclose(numerical_integral, analytic_integral)
    a60 = rng.random() + 0.1
    a06 = rng.random() + 0.1
    analytic_integral_sextic = analytic_integral + a60 * (4.0 / 7.0) + a06 * (4.0 / 7.0)
    poly_values_sextic = poly_values + a60 * points[:, 0] ** 6 + a06 * points[:, 1] ** 6
    numerical_integral_sextic = np.dot(weights, poly_values_sextic)
    assert not np.isclose(numerical_integral_sextic, analytic_integral_sextic)