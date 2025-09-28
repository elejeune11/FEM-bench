def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError."""
    for invalid_num in [-1, 0, 2, 3, 5, 8, 10, 16]:
        with pytest.raises(ValueError):
            fcn(invalid_num)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1."""
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.abs(np.sum(weights) - 4.0) < 1e-14
        assert np.all(points >= -1.0) and np.all(points <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
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
      (i.e. the quadrature does not reproduce the analytic integrals)."""
    (points, weights) = fcn(1)
    np.random.seed(42)
    (a00, a10, a01) = np.random.rand(3)

    def linear_poly(x, y):
        return a00 + a10 * x + a01 * y

    def quadratic_poly(x, y):
        return linear_poly(x, y) + 0.5 * x * y + 0.3 * x ** 2 + 0.7 * y ** 2
    quadrature_linear = np.sum(weights * linear_poly(points[:, 0], points[:, 1]))
    analytic_linear = 4 * a00
    assert np.abs(quadrature_linear - analytic_linear) < 1e-14
    quadrature_quad = np.sum(weights * quadratic_poly(points[:, 0], points[:, 1]))
    analytic_quad = analytic_linear + 4 / 3 * (0.3 + 0.7)
    assert np.abs(quadrature_quad - analytic_quad) > 0.01

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
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
    • Adding quartic terms should break exactness, and the mismatch is detected by the test."""
    (points, weights) = fcn(4)
    np.random.seed(42)
    coeffs_cubic = np.random.rand(4, 4)

    def cubic_poly(x, y):
        result = np.zeros_like(x)
        for i in range(4):
            for j in range(4):
                result += coeffs_cubic[i, j] * x ** i * y ** j
        return result

    def quartic_poly(x, y):
        return cubic_poly(x, y) + 0.5 * x ** 4 + 0.3 * y ** 4
    quadrature_cubic = np.sum(weights * cubic_poly(points[:, 0], points[:, 1]))
    analytic_cubic = 0.0
    for i in range(4):
        for j in range(4):
            x_int = (1 - (-1) ** (i + 1)) / (i + 1) if i != 0 else 2.0
            y_int = (1 - (-1) ** (j + 1)) / (j + 1) if j != 0 else 2.0
            analytic_cubic += coeffs_cubic[i, j] * x_int * y_int
    assert np.abs(quadrature_cubic - analytic_cubic) < 1e-14
    quadrature_quartic = np.sum(weights * quartic_poly(points[:, 0], points[:, 1]))
    analytic_quartic = analytic_cubic + 0.5 * (2 / 5) + 0.3 * (2 / 5)
    assert np.abs(quadrature_quartic - analytic_quartic) > 0.01

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Background
    ----------
    • The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
      Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
      for any polynomial where the degree in each variable is ≤ 5.
    • The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    What this test does
    --------------------
    1. Constructs a random polynomial P(x,y) = Σ a_ij x^i y^j with 0 ≤ i,j ≤ 5 (up to quintic
       terms in both variables). The quadrature approximation is compared to the analytic
       integral, the two results must match within tight tolerance.
    2. Extends the polynomial with degree-6 contributions (x^6, y^6, x^4y^2). These exceed the
       rule's guaranteed exactness, so the quadrature result is expected to deviate from the
       analytic integral.
    Expected outcome
    ----------------
    • Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    • Adding degree-6 terms should break exactness, and the mismatch is detected by the test."""
    (points, weights) = fcn(9)
    np.random.seed(42)
    coeffs_quintic = np.random.rand(6, 6)

    def quintic_poly(x, y):
        result = np.zeros_like(x)
        for i in range(6):
            for j in range(6):
                result += coeffs_quintic[i, j] * x ** i * y ** j
        return result

    def degree6_poly(x, y):
        return quintic_poly(x, y) + 0.2 * x ** 6 + 0.4 * y ** 6 + 0.1 * x ** 4 * y ** 2
    quadrature_quintic = np.sum(weights * quintic_poly(points[:, 0], points[:, 1]))
    analytic_quintic = 0.0
    for i in range(6):
        for j in range(6):
            x_int = (1 - (-1) ** (i + 1)) / (i + 1) if i != 0 else 2.0
            y_int = (1 - (-1) ** (j + 1)) / (j + 1) if j != 0 else 2.0
            analytic_quintic += coeffs_quintic[i, j] * x_int * y_int
    assert np.abs(quadrature_quintic - analytic_quintic) < 1e-14
    quadrature_degree6 = np.sum(weights * degree6_poly(points[:, 0], points[:, 1]))
    analytic_degree6 = analytic_quintic + 0.2 * (2 / 7) + 0.4 * (2 / 7) + 0.1 * (2 / 5) * (2 / 3)
    assert np.abs(quadrature_degree6 - analytic_degree6) > 0.01