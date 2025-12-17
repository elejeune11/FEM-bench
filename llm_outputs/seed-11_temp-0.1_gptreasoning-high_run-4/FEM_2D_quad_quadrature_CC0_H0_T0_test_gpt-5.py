def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_values = [-3, -1, 0, 2, 3, 5, 6, 7, 8, 10, 16, 100]
    for n in invalid_values:
        with pytest.raises(ValueError):
            fcn(n)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for n in (1, 4, 9):
        pts, w = fcn(n)
        assert isinstance(pts, np.ndarray)
        assert isinstance(w, np.ndarray)
        assert pts.shape == (n, 2)
        assert w.shape == (n,)
        assert pts.dtype == np.float64
        assert w.dtype == np.float64
        assert np.isclose(w.sum(), 4.0, rtol=1e-14, atol=1e-14)
        tol = 1e-14
        assert np.all(pts[:, 0] >= -1.0 - tol)
        assert np.all(pts[:, 0] <= 1.0 + tol)
        assert np.all(pts[:, 1] >= -1.0 - tol)
        assert np.all(pts[:, 1] <= 1.0 + tol)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    The tensor-product 1-point rule places a single node at the center (0,0) with
    weight 4. This integrates exactly any polynomial that is at most degree 1
    in each variable, i.e. constants and linear terms in x or y.
    For higher-degree terms, the rule is no longer guaranteed to be exact.
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check
    (i.e. the quadrature does not reproduce the analytic integrals).
    """
    pts, w = fcn(1)
    x = pts[:, 0]
    y = pts[:, 1]
    tol = 1e-12

    def exact_monomial(a, b):

        def one_dim(n):
            return 0.0 if n % 2 == 1 else 2.0 / (n + 1)
        return one_dim(a) * one_dim(b)

    def approx(a, b):
        return np.sum(w * x ** a * y ** b)
    assert np.isclose(approx(0, 0), exact_monomial(0, 0), rtol=tol, atol=tol)
    assert np.isclose(approx(1, 0), exact_monomial(1, 0), rtol=tol, atol=tol)
    assert np.isclose(approx(0, 1), exact_monomial(0, 1), rtol=tol, atol=tol)
    assert np.isclose(approx(1, 1), exact_monomial(1, 1), rtol=tol, atol=tol)
    approx_linear_combo = np.sum(w * (x + 2 * y + 3))
    exact_linear_combo = 0.0 + 0.0 + 3.0 * 4.0
    assert np.isclose(approx_linear_combo, exact_linear_combo, rtol=tol, atol=tol)
    assert not np.isclose(approx(2, 0), exact_monomial(2, 0), rtol=1e-13, atol=1e-13)
    assert not np.isclose(approx(0, 2), exact_monomial(0, 2), rtol=1e-13, atol=1e-13)
    assert not np.isclose(approx(2, 2), exact_monomial(2, 2), rtol=1e-13, atol=1e-13)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
    Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
    any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
    in x and y).
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    pts, w = fcn(4)
    x = pts[:, 0]
    y = pts[:, 1]
    tol = 1e-12

    def exact_monomial(a, b):

        def one_dim(n):
            return 0.0 if n % 2 == 1 else 2.0 / (n + 1)
        return one_dim(a) * one_dim(b)

    def approx(a, b):
        return np.sum(w * x ** a * y ** b)
    for a in range(0, 4):
        for b in range(0, 4):
            assert np.isclose(approx(a, b), exact_monomial(a, b), rtol=tol, atol=tol)
    for a, b in [(4, 0), (0, 4), (4, 4)]:
        assert not np.isclose(approx(a, b), exact_monomial(a, b), rtol=1e-13, atol=1e-13)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
    Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
    for any polynomial where the degree in each variable is ≤ 5.
    The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    pts, w = fcn(9)
    x = pts[:, 0]
    y = pts[:, 1]
    tol = 1e-12

    def exact_monomial(a, b):

        def one_dim(n):
            return 0.0 if n % 2 == 1 else 2.0 / (n + 1)
        return one_dim(a) * one_dim(b)

    def approx(a, b):
        return np.sum(w * x ** a * y ** b)
    for a in range(0, 6):
        for b in range(0, 6):
            assert np.isclose(approx(a, b), exact_monomial(a, b), rtol=tol, atol=tol)
    for a, b in [(6, 0), (0, 6), (6, 6)]:
        assert not np.isclose(approx(a, b), exact_monomial(a, b), rtol=1e-13, atol=1e-13)