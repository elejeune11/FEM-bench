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
        fcn(3)
    with pytest.raises(ValueError):
        fcn(5)
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
        assert np.isclose(np.sum(weights), 4.0)
        assert np.all(points >= -1) and np.all(points <= 1)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    """
    (points, weights) = fcn(1)
    (xi, eta) = points[0]
    weight = weights[0]
    assert np.isclose(weight, 4.0)
    assert np.isclose(weight * xi, 0.0)
    assert np.isclose(weight * eta, 0.0)
    assert not np.isclose(weight * xi ** 2, 4.0 / 3.0)
    assert not np.isclose(weight * eta ** 2, 4.0 / 3.0)
    assert not np.isclose(weight * xi * eta, 0.0)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(4)
    exact_integral = lambda f: sum((w * f(xi, eta) for ((xi, eta), w) in zip(points, weights)))
    assert np.isclose(exact_integral(lambda x, y: 1), 4.0)
    assert np.isclose(exact_integral(lambda x, y: x), 0.0)
    assert np.isclose(exact_integral(lambda x, y: y), 0.0)
    assert np.isclose(exact_integral(lambda x, y: x ** 2), 4.0 / 3.0)
    assert np.isclose(exact_integral(lambda x, y: y ** 2), 4.0 / 3.0)
    assert np.isclose(exact_integral(lambda x, y: x * y), 0.0)
    assert np.isclose(exact_integral(lambda x, y: x ** 3), 0.0)
    assert np.isclose(exact_integral(lambda x, y: y ** 3), 0.0)
    assert not np.isclose(exact_integral(lambda x, y: x ** 4), 4.0 / 5.0)
    assert not np.isclose(exact_integral(lambda x, y: y ** 4), 4.0 / 5.0)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(9)
    exact_integral = lambda f: sum((w * f(xi, eta) for ((xi, eta), w) in zip(points, weights)))
    assert np.isclose(exact_integral(lambda x, y: 1), 4.0)
    assert np.isclose(exact_integral(lambda x, y: x), 0.0)
    assert np.isclose(exact_integral(lambda x, y: y), 0.0)
    assert np.isclose(exact_integral(lambda x, y: x ** 2), 4.0 / 3.0)
    assert np.isclose(exact_integral(lambda x, y: y ** 2), 4.0 / 3.0)
    assert np.isclose(exact_integral(lambda x, y: x * y), 0.0)
    assert np.isclose(exact_integral(lambda x, y: x ** 3), 0.0)
    assert np.isclose(exact_integral(lambda x, y: y ** 3), 0.0)
    assert np.isclose(exact_integral(lambda x, y: x ** 4), 4.0 / 5.0)
    assert np.isclose(exact_integral(lambda x, y: y ** 4), 4.0 / 5.0)
    assert np.isclose(exact_integral(lambda x, y: x ** 5), 0.0)
    assert np.isclose(exact_integral(lambda x, y: y ** 5), 0.0)
    assert not np.isclose(exact_integral(lambda x, y: x ** 6), 4.0 / 7.0)
    assert not np.isclose(exact_integral(lambda x, y: y ** 6), 4.0 / 7.0)
    assert not np.isclose(exact_integral(lambda x, y: x ** 4 * y ** 2), 4.0 / 15.0)