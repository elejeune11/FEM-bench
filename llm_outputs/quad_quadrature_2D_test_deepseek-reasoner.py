def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
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
    with pytest.raises(ValueError):
        fcn(10)

def test_quad_quadrature_2D_basics(fcn):
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
        assert np.all(points >= -1) and np.all(points <= 1)
        assert np.sum(weights) == pytest.approx(4.0)

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
      (i.e. the quadrature does not reproduce the analytic integrals).
    """
    (points, weights) = fcn(1)
    rng = np.random.RandomState(42)
    (a00, a10, a01) = rng.uniform(-1, 1, 3)

    def linear_poly(x, y):
        return a00 + a10 * x + a01 * y
    analytic = 4 * a00
    quadrature_result = np.sum(weights * linear_poly(points[:, 0], points[:, 1]))
    assert quadrature_result == pytest.approx(analytic)
    (a20, a02, a11) = rng.uniform(-1, 1, 3)

    def quadratic_poly(x, y):
        return linear_poly(x, y) + a20 * x ** 2 + a02 * y ** 2 + a11 * x * y
    analytic_quad = analytic + 4 / 3 * a20 + 4 / 3 * a02
    quadrature_result_quad = np.sum(weights * quadratic_poly(points[:, 0], points[:, 1]))
    assert quadrature_result_quad != pytest.approx(analytic_quad)

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
    • Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(4)
    rng = np.random.RandomState(42)
    coeffs = rng.uniform(-1, 1, (4, 4))

    def poly_up_to_cubic(x, y):
        result = 0
        for i in range(4):
            for j in range(4):
                result += coeffs[i, j] * x ** i * y ** j
        return result
    analytic = 0
    for i in range(4):
        for j in range(4):
            int_x = 0 if i % 2 == 1 else 2 / (i + 1)
            int_y = 0 if j % 2 == 1 else 2 / (j + 1)
            analytic += coeffs[i, j] * int_x * int_y
    quadrature_result = np.sum(weights * poly_up_to_cubic(points[:, 0], points[:, 1]))
    assert quadrature_result == pytest.approx(analytic, rel=1e-10)
    (a40, a04) = rng.uniform(-1, 1, 2)

    def poly_with_quartic(x, y):
        return poly_up_to_cubic(x, y) + a40 * x ** 4 + a04 * y ** 4
    analytic_quartic = analytic + a40 * (2 / 5) * 2 + a04 * (2 / 5) * 2
    quadrature_result_quartic = np.sum(weights * poly_with_quartic(points[:, 0], points[:, 1]))
    assert quadrature_result_quartic != pytest.approx(analytic_quartic, rel=1e-10)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
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
       rule's guaranteed exactness, so the quadrature result is expected to deviate from the
       analytic integral.
    Expected outcome
    ----------------
    • Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    • Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(9)
    rng = np.random.RandomState(42)
    coeffs = rng.uniform(-1, 1, (6, 6))

    def poly_up_to_quintic(x, y):
        result = 0
        for i in range(6):
            for j in range(6):
                result += coeffs[i, j] * x ** i * y ** j
        return result
    analytic = 0
    for i in range(6):
        for j in range(6):
            int_x = 0 if i % 2 == 1 else 2 / (i + 1)
            int_y = 0 if j % 2 == 1 else 2 / (j + 1)
            analytic += coeffs[i, j] * int_x * int_y
    quadrature_result = np.sum(weights * poly_up_to_quintic(points[:, 0], points[:, 1]))
    assert quadrature_result == pytest.approx(analytic, rel=1e-12)
    (a60, a06, a42) = rng.uniform(-1, 1, 3)

    def poly_with_degree6(x, y):
        return poly_up_to_quintic(x, y) + a60 * x ** 6 + a06 * y ** 6 + a42 * x ** 4 * y ** 2
    analytic_degree6 = analytic + a60 * (2 / 7) * 2 + a06 * (2 / 7) * 2 + a42 * (2 / 5) * (2 / 3) * 2
    quadrature_result_degree6 = np.sum(weights * poly_with_degree6(points[:, 0], points[:, 1]))
    assert quadrature_result_degree6 != pytest.approx(analytic_degree6, rel=1e-12)