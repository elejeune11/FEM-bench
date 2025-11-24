def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points."""
    for n in [-1, 0, 2, 5, 10]:
        with pytest.raises(ValueError):
            fcn(n)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule."""
    for n in [1, 3, 4]:
        (points, weights) = fcn(n)
        assert points.shape == (n, 2)
        assert weights.shape == (n,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert abs(np.sum(weights) - 0.5) < 1e-14
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 1] >= 0)
        assert np.all(points[:, 0] + points[:, 1] <= 1 + 1e-14)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule."""
    (points, weights) = fcn(1)

    def exact_integral(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)

    def quad_integral(p, q):
        return np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
    for (p, q) in [(0, 0), (1, 0), (0, 1)]:
        assert abs(quad_integral(p, q) - exact_integral(p, q)) < 1e-14
    assert abs(quad_integral(2, 0) - exact_integral(2, 0)) > 1e-10
    assert abs(quad_integral(1, 1) - exact_integral(1, 1)) > 1e-10
    assert abs(quad_integral(0, 2) - exact_integral(0, 2)) > 1e-10

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule."""
    (points, weights) = fcn(3)

    def exact_integral(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)

    def quad_integral(p, q):
        return np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
    for (p, q) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
        assert abs(quad_integral(p, q) - exact_integral(p, q)) < 1e-14
    for (p, q) in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        assert abs(quad_integral(p, q) - exact_integral(p, q)) > 1e-10

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule."""
    (points, weights) = fcn(4)

    def exact_integral(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)

    def quad_integral(p, q):
        return np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
    for p in range(4):
        for q in range(4 - p):
            assert abs(quad_integral(p, q) - exact_integral(p, q)) < 1e-14
    for (p, q) in [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        assert abs(quad_integral(p, q) - exact_integral(p, q)) > 1e-10