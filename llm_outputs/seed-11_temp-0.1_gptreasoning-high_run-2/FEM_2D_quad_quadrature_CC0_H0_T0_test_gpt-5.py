def test_quad_quadrature_2D_invalid_inputs(fcn):
    """
    Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_nums = [-5, -1, 0, 2, 3, 5, 6, 7, 8, 10, 100]
    for n in invalid_nums:
        with pytest.raises(ValueError):
            fcn(n)

def test_quad_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for n in (1, 4, 9):
        pts, w = fcn(n)
        assert isinstance(pts, np.ndarray) and isinstance(w, np.ndarray)
        assert pts.shape == (n, 2)
        assert w.shape == (n,)
        assert pts.dtype == np.float64
        assert w.dtype == np.float64
        assert np.isclose(np.sum(w), 4.0, rtol=1e-14, atol=1e-14)
        assert np.all(pts[:, 0] <= 1.0 + 1e-15)
        assert np.all(pts[:, 0] >= -1.0 - 1e-15)
        assert np.all(pts[:, 1] <= 1.0 + 1e-15)
        assert np.all(pts[:, 1] >= -1.0 - 1e-15)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """
    Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check
    (i.e. the quadrature does not reproduce the analytic integrals).
    """
    pts, w = fcn(1)

    def analytic_integral(a, b):
        ix = 0.0 if a % 2 == 1 else 2.0 / (a + 1)
        iy = 0.0 if b % 2 == 1 else 2.0 / (b + 1)
        return ix * iy
    for ax in range(0, 2):
        for ay in range(0, 2):
            vals = pts[:, 0] ** ax * pts[:, 1] ** ay
            approx = np.dot(w, vals)
            exact = analytic_integral(ax, ay)
            assert np.isclose(approx, exact, rtol=1e-14, atol=1e-14)
    for ax, ay in [(2, 0), (0, 2), (2, 2), (1, 2), (2, 1)]:
        vals = pts[:, 0] ** ax * pts[:, 1] ** ay
        approx = np.dot(w, vals)
        exact = analytic_integral(ax, ay)
        assert not np.isclose(approx, exact, rtol=1e-12, atol=1e-12)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """
    Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    pts, w = fcn(4)

    def analytic_integral(a, b):
        ix = 0.0 if a % 2 == 1 else 2.0 / (a + 1)
        iy = 0.0 if b % 2 == 1 else 2.0 / (b + 1)
        return ix * iy
    for ax in range(0, 4):
        for ay in range(0, 4):
            vals = pts[:, 0] ** ax * pts[:, 1] ** ay
            approx = np.dot(w, vals)
            exact = analytic_integral(ax, ay)
            assert np.isclose(approx, exact, rtol=1e-12, atol=1e-12)
    for ax, ay in [(4, 0), (0, 4), (4, 2), (2, 4), (4, 4), (5, 0), (0, 5)]:
        vals = pts[:, 0] ** ax * pts[:, 1] ** ay
        approx = np.dot(w, vals)
        exact = analytic_integral(ax, ay)
        assert not np.isclose(approx, exact, rtol=1e-12, atol=1e-12)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """
    Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    pts, w = fcn(9)

    def analytic_integral(a, b):
        ix = 0.0 if a % 2 == 1 else 2.0 / (a + 1)
        iy = 0.0 if b % 2 == 1 else 2.0 / (b + 1)
        return ix * iy
    for ax in range(0, 6):
        for ay in range(0, 6):
            vals = pts[:, 0] ** ax * pts[:, 1] ** ay
            approx = np.dot(w, vals)
            exact = analytic_integral(ax, ay)
            assert np.isclose(approx, exact, rtol=1e-12, atol=1e-12)
    for ax, ay in [(6, 0), (0, 6), (6, 2), (2, 6), (6, 6)]:
        vals = pts[:, 0] ** ax * pts[:, 1] ** ay
        approx = np.dot(w, vals)
        exact = analytic_integral(ax, ay)
        assert not np.isclose(approx, exact, rtol=1e-12, atol=1e-12)