def test_quad_quadrature_2D_invalid_inputs(fcn):
    """
    Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_values = [0, 2, 3, 5, 6, 7, 8, 10, -1, 16, 25]
    for val in invalid_values:
        with pytest.raises(ValueError):
            fcn(val)

def test_quad_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    eps = 1e-14
    for n in (1, 4, 9):
        (pts, w) = fcn(n)
        assert isinstance(pts, np.ndarray) and isinstance(w, np.ndarray)
        assert pts.shape == (n, 2)
        assert w.shape == (n,)
        assert pts.dtype == np.float64
        assert w.dtype == np.float64
        assert np.isclose(w.sum(), 4.0, rtol=0.0, atol=1e-14)
        assert np.all(pts[:, 0] >= -1.0 - eps) and np.all(pts[:, 0] <= 1.0 + eps)
        assert np.all(pts[:, 1] >= -1.0 - eps) and np.all(pts[:, 1] <= 1.0 + eps)

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
    rng = np.random.default_rng(12345)
    (pts, w) = fcn(1)
    (x, y) = (pts[:, 0], pts[:, 1])
    (a00, a10, a01) = rng.standard_normal(3)
    P_affine = a00 + a10 * x + a01 * y
    quad_affine = float(np.dot(w, P_affine))
    exact_affine = 4.0 * a00
    assert np.isclose(quad_affine, exact_affine, rtol=1e-14, atol=1e-14)
    P_quad = P_affine + x ** 2 + y ** 2
    quad_with_quad = float(np.dot(w, P_quad))
    exact_with_quad = exact_affine + 8.0 / 3.0
    assert not np.isclose(quad_with_quad, exact_with_quad, rtol=1e-12, atol=1e-12)

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
    rng = np.random.default_rng(202)
    (pts, w) = fcn(4)

    def I(n: int) -> float:
        return 0.0 if n % 2 == 1 else 2.0 / (n + 1)
    coeffs = rng.standard_normal((4, 4))
    values = []
    for (xi, eta) in pts:
        val = 0.0
        xp = 1.0
        for i in range(4):
            yp = 1.0
            for j in range(4):
                val += coeffs[i, j] * xp * yp
                yp *= eta
            xp *= xi
        values.append(val)
    quad_val = float(np.dot(w, np.array(values)))
    exact_val = 0.0
    for i in range(4):
        for j in range(4):
            exact_val += coeffs[i, j] * I(i) * I(j)
    assert np.isclose(quad_val, exact_val, rtol=1e-13, atol=1e-13)

    def poly_with_quartic(xi, eta):
        val = 0.0
        xp = 1.0
        for i in range(4):
            yp = 1.0
            for j in range(4):
                val += coeffs[i, j] * xp * yp
                yp *= eta
            xp *= xi
        val += xi ** 4 + eta ** 4
        return val
    quad_with_q = float(np.dot(w, np.array([poly_with_quartic(xi, eta) for (xi, eta) in pts])))
    exact_with_q = exact_val + I(4) * I(0) + I(0) * I(4)
    assert not np.isclose(quad_with_q, exact_with_q, rtol=1e-12, atol=1e-12)

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
    rng = np.random.default_rng(909)
    (pts, w) = fcn(9)

    def I(n: int) -> float:
        return 0.0 if n % 2 == 1 else 2.0 / (n + 1)
    coeffs = rng.standard_normal((6, 6))
    values = []
    for (xi, eta) in pts:
        val = 0.0
        xp = 1.0
        for i in range(6):
            yp = 1.0
            for j in range(6):
                val += coeffs[i, j] * xp * yp
                yp *= eta
            xp *= xi
        values.append(val)
    quad_val = float(np.dot(w, np.array(values)))
    exact_val = 0.0
    for i in range(6):
        for j in range(6):
            exact_val += coeffs[i, j] * I(i) * I(j)
    assert np.isclose(quad_val, exact_val, rtol=1e-13, atol=1e-13)

    def poly_with_deg6(xi, eta):
        val = 0.0
        xp = 1.0
        for i in range(6):
            yp = 1.0
            for j in range(6):
                val += coeffs[i, j] * xp * yp
                yp *= eta
            xp *= xi
        val += xi ** 6 + eta ** 6 + xi ** 4 * eta ** 2
        return val
    quad_with_deg6 = float(np.dot(w, np.array([poly_with_deg6(xi, eta) for (xi, eta) in pts])))
    exact_with_deg6 = exact_val + I(6) * I(0) + I(0) * I(6) + I(4) * I(2)
    assert not np.isclose(quad_with_deg6, exact_with_deg6, rtol=1e-12, atol=1e-12)