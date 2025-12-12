def test_quad_quadrature_2D_invalid_inputs(fcn):
    """
    Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    for bad in [-1, 0, 2, 3, 5, 6, 7, 8, 10, 12, 16, 25]:
        with pytest.raises(ValueError):
            fcn(bad)

def test_quad_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for n in (1, 4, 9):
        (pts, wts) = fcn(n)
        assert isinstance(pts, np.ndarray)
        assert isinstance(wts, np.ndarray)
        assert pts.shape == (n, 2)
        assert wts.shape == (n,)
        assert pts.dtype == np.float64
        assert wts.dtype == np.float64
        assert np.isclose(np.sum(wts), 4.0, rtol=0.0, atol=1e-14)
        tol = 1e-14
        assert np.all(pts[:, 0] >= -1.0 - tol)
        assert np.all(pts[:, 0] <= 1.0 + tol)
        assert np.all(pts[:, 1] >= -1.0 - tol)
        assert np.all(pts[:, 1] <= 1.0 + tol)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """
    Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check.
    """
    (pts, wts) = fcn(1)

    def exact_1d(p):
        return 0.0 if p % 2 == 1 else 2.0 / (p + 1)

    def exact_2d(i, j):
        return exact_1d(i) * exact_1d(j)

    def quad(i, j):
        vals = pts[:, 0] ** i * pts[:, 1] ** j
        return float(np.dot(wts, vals))
    for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        assert np.isclose(quad(i, j), exact_2d(i, j), rtol=0.0, atol=1e-14)
    for (i, j) in [(2, 0), (0, 2), (2, 2)]:
        assert not np.isclose(quad(i, j), exact_2d(i, j), rtol=0.0, atol=1e-12)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """
    Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    (pts, wts) = fcn(4)

    def exact_1d(p):
        return 0.0 if p % 2 == 1 else 2.0 / (p + 1)

    def exact_2d(i, j):
        return exact_1d(i) * exact_1d(j)

    def quad(i, j):
        vals = pts[:, 0] ** i * pts[:, 1] ** j
        return float(np.dot(wts, vals))
    for i in range(0, 4):
        for j in range(0, 4):
            assert np.isclose(quad(i, j), exact_2d(i, j), rtol=0.0, atol=1e-13)
    for (i, j) in [(4, 0), (0, 4), (4, 2), (2, 4), (4, 4)]:
        assert not np.isclose(quad(i, j), exact_2d(i, j), rtol=0.0, atol=1e-12)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """
    Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    (pts, wts) = fcn(9)

    def exact_1d(p):
        return 0.0 if p % 2 == 1 else 2.0 / (p + 1)

    def exact_2d(i, j):
        return exact_1d(i) * exact_1d(j)

    def quad(i, j):
        vals = pts[:, 0] ** i * pts[:, 1] ** j
        return float(np.dot(wts, vals))
    for i in range(0, 6):
        for j in range(0, 6):
            assert np.isclose(quad(i, j), exact_2d(i, j), rtol=0.0, atol=1e-12)
    for (i, j) in [(6, 0), (0, 6), (6, 2), (2, 6), (6, 6)]:
        assert not np.isclose(quad(i, j), exact_2d(i, j), rtol=0.0, atol=1e-12)