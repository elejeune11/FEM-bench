def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    for n in [0, 2, 3, 5, 6, 7, 8, 10, -1, 12, 16]:
        with pytest.raises(ValueError):
            fcn(n)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    tol = 1e-12
    for n in (1, 4, 9):
        (pts, w) = fcn(n)
        assert isinstance(pts, np.ndarray)
        assert isinstance(w, np.ndarray)
        assert pts.dtype == np.float64
        assert w.dtype == np.float64
        assert pts.shape == (n, 2)
        assert w.shape == (n,)
        assert abs(w.sum() - 4.0) <= tol
        assert np.all(pts[:, 0] <= 1.0 + tol) and np.all(pts[:, 0] >= -1.0 - tol)
        assert np.all(pts[:, 1] <= 1.0 + tol) and np.all(pts[:, 1] >= -1.0 - tol)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the reference square.
    1. Constructs a random affine polynomial P(x,y) = a00 + a10·x + a01·y and checks exactness.
    2. Adds quadratic contributions (x², y², xy) and verifies the rule is no longer exact.
    """
    rng = np.random.default_rng(12345)
    (pts, w) = fcn(1)
    xi = pts[:, 0]
    eta = pts[:, 1]

    def I1D(k):
        return 0.0 if k % 2 == 1 else 2.0 / (k + 1)
    (a00, a10, a01) = rng.standard_normal(3)
    vals_affine = a00 + a10 * xi + a01 * eta
    quad_affine = float(np.dot(w, vals_affine))
    exact_affine = a00 * I1D(0) * I1D(0) + a10 * I1D(1) * I1D(0) + a01 * I1D(0) * I1D(1)
    assert abs(quad_affine - exact_affine) <= 1e-12
    (a20, a02, a11) = (0.5, 0.25, -0.3)
    vals_quad = vals_affine + a20 * xi ** 2 + a02 * eta ** 2 + a11 * xi * eta
    quad_quad = float(np.dot(w, vals_quad))
    exact_quad = a00 * I1D(0) * I1D(0) + a10 * I1D(1) * I1D(0) + a01 * I1D(0) * I1D(1) + a20 * I1D(2) * I1D(0) + a02 * I1D(0) * I1D(2) + a11 * I1D(1) * I1D(1)
    assert abs(quad_quad - exact_quad) > 1e-10

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    1. Random polynomial with per-variable degree ≤ 3: quadrature equals analytic integral.
    2. Add a quartic term (degree 4 in one variable): quadrature deviates from analytic integral.
    """
    rng = np.random.default_rng(2024)
    (pts, w) = fcn(4)
    xi = pts[:, 0]
    eta = pts[:, 1]

    def I1D(k):
        return 0.0 if k % 2 == 1 else 2.0 / (k + 1)
    coeffs = rng.standard_normal((4, 4))
    vals = np.zeros_like(w)
    exact = 0.0
    for i in range(4):
        for j in range(4):
            a = coeffs[i, j]
            vals += a * xi ** i * eta ** j
            exact += a * I1D(i) * I1D(j)
    quad = float(np.dot(w, vals))
    assert abs(quad - exact) <= 1e-12
    a40 = 0.75
    vals2 = vals + a40 * xi ** 4
    exact2 = exact + a40 * I1D(4) * I1D(0)
    quad2 = float(np.dot(w, vals2))
    assert abs(quad2 - exact2) > 1e-10

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    1. Random polynomial with per-variable degree ≤ 5: quadrature equals analytic integral.
    2. Add degree-6 terms in either variable: quadrature deviates from analytic integral.
    """
    rng = np.random.default_rng(99)
    (pts, w) = fcn(9)
    xi = pts[:, 0]
    eta = pts[:, 1]

    def I1D(k):
        return 0.0 if k % 2 == 1 else 2.0 / (k + 1)
    coeffs = rng.standard_normal((6, 6))
    vals = np.zeros_like(w)
    exact = 0.0
    for i in range(6):
        for j in range(6):
            a = coeffs[i, j]
            vals += a * xi ** i * eta ** j
            exact += a * I1D(i) * I1D(j)
    quad = float(np.dot(w, vals))
    assert abs(quad - exact) <= 1e-12
    (b60, b06) = (0.8, 0.3)
    vals2 = vals + b60 * xi ** 6 + b06 * eta ** 6
    exact2 = exact + b60 * I1D(6) * I1D(0) + b06 * I1D(0) * I1D(6)
    quad2 = float(np.dot(w, vals2))
    assert abs(quad2 - exact2) > 1e-10