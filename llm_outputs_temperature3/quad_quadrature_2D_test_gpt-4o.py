def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(3)
    with pytest.raises(ValueError):
        fcn(5)
    with pytest.raises(ValueError):
        fcn(6)
    with pytest.raises(ValueError):
        fcn(7)
    with pytest.raises(ValueError):
        fcn(8)
    with pytest.raises(ValueError):
        fcn(10)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(weights.sum(), 4.0)
        assert np.all((-1 <= points) & (points <= 1))

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    """
    (points, weights) = fcn(1)
    (xi, eta) = points[0]
    weight = weights[0]

    def P(x, y):
        return 2 + 3 * x + 4 * y
    integral_exact = 4 * (2 + 0 + 0)
    integral_quad = weight * P(xi, eta)
    assert np.isclose(integral_quad, integral_exact)

    def Q(x, y):
        return x ** 2 + y ** 2 + x * y
    integral_exact = 4 * (1 / 3 + 1 / 3 + 0)
    integral_quad = weight * Q(xi, eta)
    assert not np.isclose(integral_quad, integral_exact)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(4)

    def P(x, y):
        return 1 + x + y + x ** 2 + y ** 2 + x * y + x ** 3 + y ** 3
    integral_exact = 4 * (1 + 0 + 0 + 1 / 3 + 1 / 3 + 0 + 0 + 0)
    integral_quad = sum((w * P(xi, eta) for ((xi, eta), w) in zip(points, weights)))
    assert np.isclose(integral_quad, integral_exact)

    def Q(x, y):
        return x ** 4 + y ** 4
    integral_exact = 4 * (2 / 5 + 2 / 5)
    integral_quad = sum((w * Q(xi, eta) for ((xi, eta), w) in zip(points, weights)))
    assert not np.isclose(integral_quad, integral_exact)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(9)

    def P(x, y):
        return 1 + x + y + x ** 2 + y ** 2 + x * y + x ** 3 + y ** 3 + x ** 4 + y ** 4 + x ** 5 + y ** 5
    integral_exact = 4 * (1 + 0 + 0 + 1 / 3 + 1 / 3 + 0 + 0 + 0 + 2 / 5 + 2 / 5 + 0 + 0)
    integral_quad = sum((w * P(xi, eta) for ((xi, eta), w) in zip(points, weights)))
    assert np.isclose(integral_quad, integral_exact)

    def Q(x, y):
        return x ** 6 + y ** 6 + x ** 4 * y ** 2
    integral_exact = 4 * (2 / 7 + 2 / 7 + 4 / 15)
    integral_quad = sum((w * Q(xi, eta) for ((xi, eta), w) in zip(points, weights)))
    assert not np.isclose(integral_quad, integral_exact)