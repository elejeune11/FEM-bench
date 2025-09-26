def test_quad_quadrature_2D_invalid_inputs(fcn):
    """
    Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_values = [0, 2, 3, 5, 6, 7, 8, 10, -1, 16, 12]
    for v in invalid_values:
        with pytest.raises(ValueError):
            fcn(v)

def test_quad_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for n in (1, 4, 9):
        (pts, wts) = fcn(n)
        assert isinstance(pts, np.ndarray) and isinstance(wts, np.ndarray)
        assert pts.shape == (n, 2)
        assert wts.shape == (n,)
        assert pts.dtype == np.float64
        assert wts.dtype == np.float64
        assert np.isclose(np.sum(wts), 4.0, atol=1e-14)
        assert pts.min() >= -1.0 - 1e-15
        assert pts.max() <= 1.0 + 1e-15

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """
    Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
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

    def analytic_monomial_integral(i, j):
        ix = 0.0 if i % 2 == 1 else 2.0 / (i + 1)
        iy = 0.0 if j % 2 == 1 else 2.0 / (j + 1)
        return ix * iy

    def analytic_poly_integral(coeffs):
        s = 0.0
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                s += coeffs[i, j] * analytic_monomial_integral(i, j)
        return s

    def quad_integral(points, weights, coeffs):
        vals = np.zeros(points.shape[0], dtype=np.float64)
        x = points[:, 0]
        y = points[:, 1]
        for i in range(coeffs.shape[0]):
            xpow = x ** i
            for j in range(coeffs.shape[1]):
                if coeffs[i, j] != 0.0:
                    vals += coeffs[i, j] * xpow * y ** j
        return float(np.dot(weights, vals))
    rng = np.random.default_rng(123)
    (pts, wts) = fcn(1)
    coeffs_affine = np.zeros((2, 2), dtype=np.float64)
    coeffs_affine[0, 0] = float(rng.normal())
    coeffs_affine[1, 0] = float(rng.normal())
    coeffs_affine[0, 1] = float(rng.normal())
    exact_affine = analytic_poly_integral(coeffs_affine)
    quad_affine = quad_integral(pts, wts, coeffs_affine)
    assert np.isclose(quad_affine, exact_affine, atol=1e-14, rtol=0.0)
    coeffs_quad = np.zeros((3, 3), dtype=np.float64)
    coeffs_quad[:2, :2] = coeffs_affine
    coeffs_quad[2, 0] = 0.7
    coeffs_quad[0, 2] = -0.5
    coeffs_quad[1, 1] = 1.2
    exact_quad = analytic_poly_integral(coeffs_quad)
    quad_quad = quad_integral(pts, wts, coeffs_quad)
    assert abs(quad_quad - exact_quad) > 1e-08

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """
    Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
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

    def analytic_monomial_integral(i, j):
        ix = 0.0 if i % 2 == 1 else 2.0 / (i + 1)
        iy = 0.0 if j % 2 == 1 else 2.0 / (j + 1)
        return ix * iy

    def analytic_poly_integral(coeffs):
        s = 0.0
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                s += coeffs[i, j] * analytic_monomial_integral(i, j)
        return s

    def quad_integral(points, weights, coeffs):
        vals = np.zeros(points.shape[0], dtype=np.float64)
        x = points[:, 0]
        y = points[:, 1]
        for i in range(coeffs.shape[0]):
            xpow = x ** i
            for j in range(coeffs.shape[1]):
                if coeffs[i, j] != 0.0:
                    vals += coeffs[i, j] * xpow * y ** j
        return float(np.dot(weights, vals))
    rng = np.random.default_rng(321)
    (pts, wts) = fcn(4)
    coeffs_cubic = rng.normal(size=(4, 4)).astype(np.float64)
    exact_cubic = analytic_poly_integral(coeffs_cubic)
    quad_cubic = quad_integral(pts, wts, coeffs_cubic)
    assert np.isclose(quad_cubic, exact_cubic, atol=1e-12, rtol=0.0)
    coeffs_x4 = np.zeros((5, 5), dtype=np.float64)
    coeffs_x4[4, 0] = 1.0
    exact_x4 = analytic_poly_integral(coeffs_x4)
    quad_x4 = quad_integral(pts, wts, coeffs_x4)
    assert abs(quad_x4 - exact_x4) > 1e-08
    coeffs_y4 = np.zeros((5, 5), dtype=np.float64)
    coeffs_y4[0, 4] = 1.0
    exact_y4 = analytic_poly_integral(coeffs_y4)
    quad_y4 = quad_integral(pts, wts, coeffs_y4)
    assert abs(quad_y4 - exact_y4) > 1e-08

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """
    Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
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

    def analytic_monomial_integral(i, j):
        ix = 0.0 if i % 2 == 1 else 2.0 / (i + 1)
        iy = 0.0 if j % 2 == 1 else 2.0 / (j + 1)
        return ix * iy

    def analytic_poly_integral(coeffs):
        s = 0.0
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                s += coeffs[i, j] * analytic_monomial_integral(i, j)
        return s

    def quad_integral(points, weights, coeffs):
        vals = np.zeros(points.shape[0], dtype=np.float64)
        x = points[:, 0]
        y = points[:, 1]
        for i in range(coeffs.shape[0]):
            xpow = x ** i
            for j in range(coeffs.shape[1]):
                if coeffs[i, j] != 0.0:
                    vals += coeffs[i, j] * xpow * y ** j
        return float(np.dot(weights, vals))
    rng = np.random.default_rng(999)
    (pts, wts) = fcn(9)
    coeffs_quintic = rng.normal(size=(6, 6)).astype(np.float64)
    exact_quintic = analytic_poly_integral(coeffs_quintic)
    quad_quintic = quad_integral(pts, wts, coeffs_quintic)
    assert np.isclose(quad_quintic, exact_quintic, atol=1e-12, rtol=0.0)
    coeffs_x6 = np.zeros((7, 7), dtype=np.float64)
    coeffs_x6[6, 0] = 1.0
    exact_x6 = analytic_poly_integral(coeffs_x6)
    quad_x6 = quad_integral(pts, wts, coeffs_x6)
    assert abs(quad_x6 - exact_x6) > 1e-08
    coeffs_y6 = np.zeros((7, 7), dtype=np.float64)
    coeffs_y6[0, 6] = 1.0
    exact_y6 = analytic_poly_integral(coeffs_y6)
    quad_y6 = quad_integral(pts, wts, coeffs_y6)
    assert abs(quad_y6 - exact_y6) > 1e-08
    coeffs_x4y2 = np.zeros((7, 7), dtype=np.float64)
    coeffs_x4y2[4, 2] = 1.0
    exact_x4y2 = analytic_poly_integral(coeffs_x4y2)
    quad_x4y2 = quad_integral(pts, wts, coeffs_x4y2)
    assert abs(quad_x4y2 - exact_x4y2) >= 0.0