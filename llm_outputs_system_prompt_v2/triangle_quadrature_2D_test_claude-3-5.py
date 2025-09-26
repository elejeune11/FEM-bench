def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points."""
    for n in [-1, 0, 2, 5, 10]:
        with pytest.raises(ValueError):
            fcn(n)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule."""
    for n in [1, 3, 4]:
        (points, weights) = fcn(n)
        assert isinstance(points, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert points.shape == (n, 2)
        assert weights.shape == (n,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.abs(np.sum(weights) - 0.5) < 1e-14
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 1] >= 0)
        assert np.all(points[:, 0] + points[:, 1] <= 1 + 1e-14)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule."""
    (points, weights) = fcn(1)

    def integrate(p, q):
        return np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)

    def exact(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
    assert np.abs(integrate(0, 0) - exact(0, 0)) < 1e-14
    assert np.abs(integrate(1, 0) - exact(1, 0)) < 1e-14
    assert np.abs(integrate(0, 1) - exact(0, 1)) < 1e-14
    assert np.abs(integrate(2, 0) - exact(2, 0)) > 1e-10
    assert np.abs(integrate(1, 1) - exact(1, 1)) > 1e-10
    assert np.abs(integrate(0, 2) - exact(0, 2)) > 1e-10

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule."""
    (points, weights) = fcn(3)

    def integrate(p, q):
        return np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)

    def exact(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
    for p in range(3):
        for q in range(3 - p):
            assert np.abs(integrate(p, q) - exact(p, q)) < 1e-14
    assert np.abs(integrate(3, 0) - exact(3, 0)) > 1e-10
    assert np.abs(integrate(2, 1) - exact(2, 1)) > 1e-10
    assert np.abs(integrate(1, 2) - exact(1, 2)) > 1e-10
    assert np.abs(integrate(0, 3) - exact(0, 3)) > 1e-10

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule."""
    (points, weights) = fcn(4)

    def integrate(p, q):
        return np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)

    def exact(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
    for p in range(4):
        for q in range(4 - p):
            assert np.abs(integrate(p, q) - exact(p, q)) < 1e-14
    assert np.abs(integrate(4, 0) - exact(4, 0)) > 1e-10
    assert np.abs(integrate(3, 1) - exact(3, 1)) > 1e-10
    assert np.abs(integrate(2, 2) - exact(2, 2)) > 1e-10
    assert np.abs(integrate(1, 3) - exact(1, 3)) > 1e-10
    assert np.abs(integrate(0, 4) - exact(0, 4)) > 1e-10