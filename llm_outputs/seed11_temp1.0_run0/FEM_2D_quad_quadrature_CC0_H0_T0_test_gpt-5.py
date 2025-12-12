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
    tol = 1e-14
    for n in (1, 4, 9):
        (pts, w) = fcn(n)
        assert isinstance(pts, np.ndarray) and isinstance(w, np.ndarray)
        assert pts.shape == (n, 2)
        assert w.shape == (n,)
        assert pts.dtype == np.float64
        assert w.dtype == np.float64
        assert abs(np.sum(w) - 4.0) < tol
        assert np.all(pts[:, 0] >= -1 - tol)
        assert np.all(pts[:, 0] <= 1 + tol)
        assert np.all(pts[:, 1] >= -1 - tol)
        assert np.all(pts[:, 1] <= 1 + tol)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the reference square.
    Exactness for monomials with per-variable degree ≤ 1 should pass.
    Quadratic terms should not be integrated exactly, and the mismatch is detected.
    """
    (pts, w) = fcn(1)
    x = pts[:, 0]
    y = pts[:, 1]
    tol = 1e-13

    def exact_monomial_integral(i, j):

        def one_dim(n):
            if n % 2 == 1:
                return 0.0
            return 2.0 / (n + 1)
        return one_dim(i) * one_dim(j)
    for i in range(0, 2):
        for j in range(0, 2):
            approx = np.sum(w * x ** i * y ** j)
            exact = exact_monomial_integral(i, j)
            assert abs(approx - exact) < tol
    for (i, j) in [(2, 0), (0, 2), (2, 2)]:
        approx = np.sum(w * x ** i * y ** j)
        exact = exact_monomial_integral(i, j)
        assert abs(approx - exact) > 1e-06

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness for all monomials with per-variable degree ≤ 3 should pass.
    Quartic terms should break exactness, and the mismatch is detected.
    """
    (pts, w) = fcn(4)
    x = pts[:, 0]
    y = pts[:, 1]
    tol = 1e-13

    def exact_monomial_integral(i, j):

        def one_dim(n):
            if n % 2 == 1:
                return 0.0
            return 2.0 / (n + 1)
        return one_dim(i) * one_dim(j)
    for i in range(0, 4):
        for j in range(0, 4):
            approx = np.sum(w * x ** i * y ** j)
            exact = exact_monomial_integral(i, j)
            assert abs(approx - exact) < tol
    for (i, j) in [(4, 0), (0, 4), (4, 2), (2, 4), (4, 4)]:
        approx = np.sum(w * x ** i * y ** j)
        exact = exact_monomial_integral(i, j)
        assert abs(approx - exact) > 1e-10

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Exactness for all monomials with per-variable degree ≤ 5 should pass.
    Degree-6 terms should break exactness, and the mismatch is detected.
    """
    (pts, w) = fcn(9)
    x = pts[:, 0]
    y = pts[:, 1]
    tol = 1e-13

    def exact_monomial_integral(i, j):

        def one_dim(n):
            if n % 2 == 1:
                return 0.0
            return 2.0 / (n + 1)
        return one_dim(i) * one_dim(j)
    for i in range(0, 6):
        for j in range(0, 6):
            approx = np.sum(w * x ** i * y ** j)
            exact = exact_monomial_integral(i, j)
            assert abs(approx - exact) < tol
    for (i, j) in [(6, 0), (0, 6), (6, 2), (2, 6), (6, 6)]:
        approx = np.sum(w * x ** i * y ** j)
        exact = exact_monomial_integral(i, j)
        assert abs(approx - exact) > 1e-10