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
    integral_analytic = 4 * (a00 + a10 * 0 + a01 * 0)
    integral_quad = weight * P(xi, eta)
    assert np.isclose(integral_quad, integral_analytic)
    (a20, a02, a11) = np.random.rand(3)
    P = lambda x, y: a00 + a10 * x + a01 * y + a20 * x ** 2 + a02 * y ** 2 + a11 * x * y
    integral_quad = weight * P(xi, eta)
    assert not np.isclose(integral_quad, integral_analytic)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(4)
    coeffs = np.random.rand(4, 4)
    P = lambda x, y: sum((coeffs[i, j] * x ** i * y ** j for i in range(4) for j in range(4) if i <= 3 and j <= 3))
    integral_analytic = 4 * sum((coeffs[i, j] * (2 / (i + 1)) * (2 / (j + 1)) for i in range(4) for j in range(4) if i <= 3 and j <= 3))
    integral_quad = sum((weights[k] * P(points[k, 0], points[k, 1]) for k in range(4)))
    assert np.isclose(integral_quad, integral_analytic)
    (coeffs[4, 0], coeffs[0, 4]) = np.random.rand(2)
    P = lambda x, y: sum((coeffs[i, j] * x ** i * y ** j for i in range(5) for j in range(5)))
    integral_quad = sum((weights[k] * P(points[k, 0], points[k, 1]) for k in range(4)))
    assert not np.isclose(integral_quad, integral_analytic)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(9)
    coeffs = np.random.rand(6, 6)
    P = lambda x, y: sum((coeffs[i, j] * x ** i * y ** j for i in range(6) for j in range(6) if i <= 5 and j <= 5))
    integral_analytic = 4 * sum((coeffs[i, j] * (2 / (i + 1)) * (2 / (j + 1)) for i in range(6) for j in range(6) if i <= 5 and j <= 5))
    integral_quad = sum((weights[k] * P(points[k, 0], points[k, 1]) for k in range(9)))
    assert np.isclose(integral_quad, integral_analytic, atol=1e-12)
    (coeffs[6, 0], coeffs[0, 6], coeffs[4, 2]) = np.random.rand(3)
    P = lambda x, y: sum((coeffs[i, j] * x ** i * y ** j for i in range(7) for j in range(7)))
    integral_quad = sum((weights[k] * P(points[k, 0], points[k, 1]) for k in range(9)))
    assert not np.isclose(integral_quad, integral_analytic, atol=1e-12)