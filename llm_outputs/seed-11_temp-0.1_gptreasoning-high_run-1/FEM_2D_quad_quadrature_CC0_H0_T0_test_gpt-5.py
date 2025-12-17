def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_values = [-5, -1, 0, 2, 3, 5, 6, 7, 8, 10, 16, 25]
    for n in invalid_values:
        with pytest.raises(ValueError):
            fcn(n)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
    """
    for n in (1, 4, 9):
        pts, wts = fcn(n)
        assert isinstance(pts, np.ndarray)
        assert isinstance(wts, np.ndarray)
        assert pts.shape == (n, 2)
        assert wts.shape == (n,)
        assert pts.dtype == np.float64
        assert wts.dtype == np.float64
        assert np.isclose(np.sum(wts), 4.0, rtol=0.0, atol=1e-14)
        eps = 1e-14
        assert np.all(pts[:, 0] >= -1.0 - eps)
        assert np.all(pts[:, 0] <= 1.0 + eps)
        assert np.all(pts[:, 1] >= -1.0 - eps)
        assert np.all(pts[:, 1] <= 1.0 + eps)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the reference square.
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check.
    """
    pts, wts = fcn(1)

    def exact_int_2d(i, j):

        def I1D(k):
            return 0.0 if k % 2 == 1 else 2.0 / (k + 1)
        return I1D(i) * I1D(j)
    for i in (0, 1):
        for j in (0, 1):
            approx = np.sum(wts * pts[:, 0] ** i * pts[:, 1] ** j)
            exact = exact_int_2d(i, j)
            assert np.isclose(approx, exact, rtol=1e-12, atol=1e-12)
    for i, j in [(2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]:
        approx = np.sum(wts * pts[:, 0] ** i * pts[:, 1] ** j)
        exact = exact_int_2d(i, j)
        assert not np.isclose(approx, exact, rtol=1e-12, atol=1e-12)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    pts, wts = fcn(4)

    def exact_int_2d(i, j):

        def I1D(k):
            return 0.0 if k % 2 == 1 else 2.0 / (k + 1)
        return I1D(i) * I1D(j)
    for i in range(0, 4):
        for j in range(0, 4):
            approx = np.sum(wts * pts[:, 0] ** i * pts[:, 1] ** j)
            exact = exact_int_2d(i, j)
            assert np.isclose(approx, exact, rtol=1e-12, atol=1e-12)
    for i, j in [(4, 0), (0, 4), (4, 2), (2, 4), (4, 4)]:
        approx = np.sum(wts * pts[:, 0] ** i * pts[:, 1] ** j)
        exact = exact_int_2d(i, j)
        assert not np.isclose(approx, exact, rtol=1e-12, atol=1e-12)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    pts, wts = fcn(9)

    def exact_int_2d(i, j):

        def I1D(k):
            return 0.0 if k % 2 == 1 else 2.0 / (k + 1)
        return I1D(i) * I1D(j)
    for i in range(0, 6):
        for j in range(0, 6):
            approx = np.sum(wts * pts[:, 0] ** i * pts[:, 1] ** j)
            exact = exact_int_2d(i, j)
            assert np.isclose(approx, exact, rtol=1e-12, atol=1e-12)
    for i, j in [(6, 0), (0, 6), (6, 2), (2, 6), (6, 6)]:
        approx = np.sum(wts * pts[:, 0] ** i * pts[:, 1] ** j)
        exact = exact_int_2d(i, j)
        assert not np.isclose(approx, exact, rtol=1e-12, atol=1e-12)