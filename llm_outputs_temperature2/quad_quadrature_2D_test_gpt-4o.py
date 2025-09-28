def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(5)
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
        assert np.all(points >= -1) and np.all(points <= 1)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    """
    (points, weights) = fcn(1)
    (xi, eta) = points[0]
    weight = weights[0]
    (a00, a10, a01) = np.random.rand(3)
    P = lambda x, y: a00 + a10 * x + a01 * y
    integral_exact = 4 * a00
    integral_quad = weight * P(xi, eta)
    assert np.isclose(integral_quad, integral_exact)
    P_quad = lambda x, y: x ** 2 + y ** 2 + x * y
    integral_quad = weight * P_quad(xi, eta)
    integral_exact = 4 * (2 / 3)
    assert not np.isclose(integral_quad, integral_exact)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(4)
    a = np.random.rand(4, 4)
    P = lambda x, y: sum((a[i, j] * x ** i * y ** j for i in range(4) for j in range(4) if i <= 3 and j <= 3))
    integral_quad = sum((weight * P(xi, eta) for ((xi, eta), weight) in zip(points, weights)))
    integral_exact = 4 * sum((a[i, j] * (2 / (i + 1)) * (2 / (j + 1)) for i in range(4) for j in range(4) if i <= 3 and j <= 3))
    assert np.isclose(integral_quad, integral_exact)
    P_quartic = lambda x, y: x ** 4 + y ** 4
    integral_quad = sum((weight * P_quartic(xi, eta) for ((xi, eta), weight) in zip(points, weights)))
    integral_exact = 4 * (2 / 5)
    assert not np.isclose(integral_quad, integral_exact)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(9)
    a = np.random.rand(6, 6)
    P = lambda x, y: sum((a[i, j] * x ** i * y ** j for i in range(6) for j in range(6) if i <= 5 and j <= 5))
    integral_quad = sum((weight * P(xi, eta) for ((xi, eta), weight) in zip(points, weights)))
    integral_exact = 4 * sum((a[i, j] * (2 / (i + 1)) * (2 / (j + 1)) for i in range(6) for j in range(6) if i <= 5 and j <= 5))
    assert np.isclose(integral_quad, integral_exact, atol=1e-12)
    P_degree6 = lambda x, y: x ** 6 + y ** 6 + x ** 4 * y ** 2
    integral_quad = sum((weight * P_degree6(xi, eta) for ((xi, eta), weight) in zip(points, weights)))
    integral_exact = 4 * (2 / 7)
    assert not np.isclose(integral_quad, integral_exact)