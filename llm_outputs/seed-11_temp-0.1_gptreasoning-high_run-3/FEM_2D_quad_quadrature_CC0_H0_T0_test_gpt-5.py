def test_quad_quadrature_2D_invalid_inputs(fcn):
    """
    Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid = [-3, -1, 0, 2, 3, 5, 6, 7, 8, 10, 12, 16, 25]
    for n in invalid:
        with pytest.raises(ValueError):
            fcn(n)

def test_quad_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    tol = 1e-14
    for n in (1, 4, 9):
        pts, wts = fcn(n)
        assert isinstance(pts, np.ndarray)
        assert isinstance(wts, np.ndarray)
        assert pts.shape == (n, 2)
        assert wts.shape == (n,)
        assert pts.dtype == np.float64
        assert wts.dtype == np.float64
        assert np.isfinite(pts).all()
        assert np.isfinite(wts).all()
        assert np.isclose(wts.sum(), 4.0, rtol=0, atol=1e-14)
        assert (pts[:, 0] <= 1.0 + tol).all() and (pts[:, 0] >= -1.0 - tol).all()
        assert (pts[:, 1] <= 1.0 + tol).all() and (pts[:, 1] >= -1.0 - tol).all()

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """
    Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    The tensor-product 1-point rule places a single node at the center (0,0) with
    weight 4. This integrates exactly any polynomial that is at most degree 1
    in each variable, i.e. constants and linear terms in x or y.
    For higher-degree terms, the rule is no longer guaranteed to be exact.
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check
    (i.e. the quadrature does not reproduce the analytic integrals).
    """
    pts, wts = fcn(1)

    def exact_int(i, j):

        def oned(n):
            return 0.0 if n % 2 == 1 else 2.0 / (n + 1)
        return oned(i) * oned(j)

    def quad_int(i, j):
        x = pts[:, 0]
        y = pts[:, 1]
        return float(np.dot(wts, x ** i * y ** j))
    for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        q = quad_int(i, j)
        e = exact_int(i, j)
        assert np.isclose(q, e, rtol=0, atol=1e-14)
    for i, j in [(2, 0), (0, 2), (2, 2)]:
        q = quad_int(i, j)
        e = exact_int(i, j)
        assert not np.isclose(q, e, rtol=0, atol=1e-12)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """
    Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
    Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
    any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
    in x and y).
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    pts, wts = fcn(4)

    def exact_int(i, j):

        def oned(n):
            return 0.0 if n % 2 == 1 else 2.0 / (n + 1)
        return oned(i) * oned(j)

    def quad_int(i, j):
        x = pts[:, 0]
        y = pts[:, 1]
        return float(np.dot(wts, x ** i * y ** j))
    for i in range(0, 4):
        for j in range(0, 4):
            q = quad_int(i, j)
            e = exact_int(i, j)
            assert np.isclose(q, e, rtol=0, atol=1e-13)
    for i, j in [(4, 0), (0, 4), (4, 2), (2, 4)]:
        q = quad_int(i, j)
        e = exact_int(i, j)
        assert not np.isclose(q, e, rtol=0, atol=1e-12)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """
    Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
    Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
    for any polynomial where the degree in each variable is ≤ 5.
    The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    pts, wts = fcn(9)

    def exact_int(i, j):

        def oned(n):
            return 0.0 if n % 2 == 1 else 2.0 / (n + 1)
        return oned(i) * oned(j)

    def quad_int(i, j):
        x = pts[:, 0]
        y = pts[:, 1]
        return float(np.dot(wts, x ** i * y ** j))
    for i in range(0, 6):
        for j in range(0, 6):
            q = quad_int(i, j)
            e = exact_int(i, j)
            assert np.isclose(q, e, rtol=0, atol=1e-13)
    for i, j in [(6, 0), (0, 6), (6, 2), (2, 6)]:
        q = quad_int(i, j)
        e = exact_int(i, j)
        assert not np.isclose(q, e, rtol=0, atol=1e-12)