def test_quad_quadrature_2D_invalid_inputs(fcn):
    """
    Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_values = [-3, -1, 0, 2, 3, 5, 6, 7, 8, 10, 16]
    for val in invalid_values:
        with pytest.raises(ValueError):
            fcn(val)

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
        assert np.isclose(wts.sum(), 4.0, rtol=0, atol=1e-14)
        assert np.all(np.isfinite(pts)) and np.all(np.isfinite(wts))
        eps = 1e-14
        assert np.all(pts >= -1.0 - eps)
        assert np.all(pts <= 1.0 + eps)

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
    (pts, wts) = fcn(1)

    def I1D(k: int) -> float:
        return 0.0 if k % 2 == 1 else 2.0 / (k + 1)

    def integ(terms):
        s = 0.0
        for (c, i, j) in terms:
            s += c * I1D(i) * I1D(j)
        return s

    def quad_eval(terms):
        x = pts[:, 0]
        y = pts[:, 1]
        vals = np.zeros_like(wts)
        for (c, i, j) in terms:
            vals += c * x ** i * y ** j
        return float(np.dot(wts, vals))
    rng = np.random.default_rng(12345)
    (a00, a10, a01) = rng.uniform(-1, 1, size=3)
    affine_terms = [(a00, 0, 0), (a10, 1, 0), (a01, 0, 1)]
    exact_affine = integ(affine_terms)
    quad_affine = quad_eval(affine_terms)
    assert np.isclose(quad_affine, exact_affine, rtol=0, atol=1e-14)
    (c20, c02, c11) = (1.2345, -0.5678, 0.321)
    quad_terms = affine_terms + [(c20, 2, 0), (c02, 0, 2), (c11, 1, 1)]
    exact_quad = integ(quad_terms)
    quad_quad = quad_eval(quad_terms)
    assert not np.isclose(quad_quad, exact_quad, rtol=1e-12, atol=1e-12)

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
    (pts, wts) = fcn(4)

    def I1D(k: int) -> float:
        return 0.0 if k % 2 == 1 else 2.0 / (k + 1)

    def eval_poly_terms(terms):
        x = pts[:, 0]
        y = pts[:, 1]
        vals = np.zeros_like(wts)
        for (c, i, j) in terms:
            vals += c * x ** i * y ** j
        return float(np.dot(wts, vals))

    def exact_integral(terms):
        s = 0.0
        for (c, i, j) in terms:
            s += c * I1D(i) * I1D(j)
        return s
    rng = np.random.default_rng(2024)
    coeffs = rng.uniform(-1, 1, size=(4, 4))
    cubic_terms = []
    for i in range(4):
        for j in range(4):
            cubic_terms.append((float(coeffs[i, j]), i, j))
    quad_cubic = eval_poly_terms(cubic_terms)
    exact_cubic = exact_integral(cubic_terms)
    assert np.isclose(quad_cubic, exact_cubic, rtol=1e-13, atol=1e-13)
    c40 = 0.75
    c04 = -0.35
    extended_terms = list(cubic_terms) + [(c40, 4, 0), (c04, 0, 4)]
    quad_ext = eval_poly_terms(extended_terms)
    exact_ext = exact_integral(extended_terms)
    assert not np.isclose(quad_ext, exact_ext, rtol=1e-12, atol=1e-12)

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
    (pts, wts) = fcn(9)

    def I1D(k: int) -> float:
        return 0.0 if k % 2 == 1 else 2.0 / (k + 1)

    def eval_poly_terms(terms):
        x = pts[:, 0]
        y = pts[:, 1]
        vals = np.zeros_like(wts)
        for (c, i, j) in terms:
            vals += c * x ** i * y ** j
        return float(np.dot(wts, vals))

    def exact_integral(terms):
        s = 0.0
        for (c, i, j) in terms:
            s += c * I1D(i) * I1D(j)
        return s
    rng = np.random.default_rng(7)
    coeffs = rng.uniform(-1, 1, size=(6, 6))
    quintic_terms = []
    for i in range(6):
        for j in range(6):
            quintic_terms.append((float(coeffs[i, j]), i, j))
    quad_quintic = eval_poly_terms(quintic_terms)
    exact_quintic = exact_integral(quintic_terms)
    assert np.isclose(quad_quintic, exact_quintic, rtol=1e-13, atol=1e-13)
    c60 = 0.42
    c06 = -0.27
    c42 = 0.33
    extended_terms = list(quintic_terms) + [(c60, 6, 0), (c06, 0, 6), (c42, 4, 2)]
    quad_ext = eval_poly_terms(extended_terms)
    exact_ext = exact_integral(extended_terms)
    assert not np.isclose(quad_ext, exact_ext, rtol=1e-12, atol=1e-12)