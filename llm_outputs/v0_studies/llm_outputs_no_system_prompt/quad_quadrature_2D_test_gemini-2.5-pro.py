def test_quad_quadrature_2D_invalid_inputs(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 3, 5, 6, 7, 8, 10, 16, -1]
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
    rng = np.random.default_rng(seed=1234)
    coeffs = rng.random(5)
    (a00, a10, a01) = coeffs[:3]
    p_affine = lambda x, y: a00 + a10 * x + a01 * y
    quad_result_affine = np.sum(weights * p_affine(points[:, 0], points[:, 1]))
    analytic_integral_affine = 4.0 * a00
    assert np.isclose(quad_result_affine, analytic_integral_affine)
    (a20, a02) = coeffs[3:5] + 0.1
    p_quad = lambda x, y: a20 * x ** 2 + a02 * y ** 2
    quad_result_quad = np.sum(weights * p_quad(points[:, 0], points[:, 1]))
    analytic_integral_quad = 4.0 / 3.0 * (a20 + a02)
    assert not np.isclose(quad_result_quad, analytic_integral_quad)

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
    rng = np.random.default_rng(seed=5678)

    def analytic_integral_monomial(i, j):
        if i % 2 != 0 or j % 2 != 0:
            return 0.0
        return 2.0 / (i + 1.0) * (2.0 / (j + 1.0))
    max_degree = 3
    coeffs = rng.random((max_degree + 1, max_degree + 1))
    quad_result_exact = 0.0
    analytic_integral_exact = 0.0
    for i in range(max_degree + 1):
        for j in range(max_degree + 1):
            p_ij = points[:, 0] ** i * points[:, 1] ** j
            quad_result_exact += coeffs[i, j] * np.sum(weights * p_ij)
            analytic_integral_exact += coeffs[i, j] * analytic_integral_monomial(i, j)
    assert np.isclose(quad_result_exact, analytic_integral_exact)
    (a40, a04) = rng.random(2) + 0.1
    p_quartic = lambda x, y: a40 * x ** 4 + a04 * y ** 4
    quad_result_quartic = np.sum(weights * p_quartic(points[:, 0], points[:, 1]))
    analytic_integral_quartic = a40 * analytic_integral_monomial(4, 0) + a04 * analytic_integral_monomial(0, 4)
    assert not np.isclose(quad_result_quartic, analytic_integral_quartic)

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
    rng = np.random.default_rng(seed=91011)

    def analytic_integral_monomial(i, j):
        if i % 2 != 0 or j % 2 != 0:
            return 0.0
        return 2.0 / (i + 1.0) * (2.0 / (j + 1.0))
    max_degree = 5
    coeffs = rng.random((max_degree + 1, max_degree + 1))
    quad_result_exact = 0.0
    analytic_integral_exact = 0.0
    for i in range(max_degree + 1):
        for j in range(max_degree + 1):
            p_ij = points[:, 0] ** i * points[:, 1] ** j
            quad_result_exact += coeffs[i, j] * np.sum(weights * p_ij)
            analytic_integral_exact += coeffs[i, j] * analytic_integral_monomial(i, j)
    assert np.isclose(quad_result_exact, analytic_integral_exact)
    (a60, a06, a42) = rng.random(3) + 0.1
    p_deg6 = lambda x, y: a60 * x ** 6 + a06 * y ** 6 + a42 * x ** 4 * y ** 2
    quad_result_deg6 = np.sum(weights * p_deg6(points[:, 0], points[:, 1]))
    analytic_integral_deg6 = a60 * analytic_integral_monomial(6, 0) + a06 * analytic_integral_monomial(0, 6) + a42 * analytic_integral_monomial(4, 2)
    assert not np.isclose(quad_result_deg6, analytic_integral_deg6)